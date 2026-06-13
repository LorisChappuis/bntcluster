"""
bnt_framework
=============

User-facing framework for building **cluster-adapted BNT** (Bernardeau–
Nishimichi–Taruya) shear-nulling schemes, optimised for a given lens redshift
and a given survey.

This module is the single entry point used by ``paper_notebook.ipynb``.
It wraps the low-level toolkit (``bntcluster`` → ``BNTNuller``,
``NFWMockGenerator``) behind three small objects:

``SurveySetup``
    Everything that describes the survey: source redshift distribution
    ``n(z)``, source density, shape noise, photo-z error model, field size and
    map resolution.

``CostFunction``
    Everything that describes *what the bin-edge optimiser tries to achieve*:
    the objective (maximise the detection-S/N ratio ``R``, or minimise the
    shape-noise amplification ``alpha``), and the fine-tuning of the penalty
    terms (kernel-peak weight, foreground-leakage weight and tolerance).

``ClusterBNT``
    Binds a survey to a cost function and exposes the high-level methods:
    ``optimise_bin_edges``, ``build_scheme``, ``scan_lens_redshifts``,
    ``standard_tomography``, plus the analytic diagnostics (``alpha``,
    leakage ``L``, predicted ratio ``R``).

A typical session::

    import bntcluster as bnt
    from bntcluster import SurveySetup, CostFunction, ClusterBNT

    survey = SurveySetup(n_of_z=bnt.nz_euclid, n_gal_per_arcmin2=30.0,
                         shape_noise=0.26, photoz_sigma=0.05,
                         photoz_outlier_fraction=0.10,
                         field_deg=5.0, npix=512)
    cost   = CostFunction(objective="max_R", peak_weight=200.0,
                          leakage_weight=300.0, leakage_tolerance=0.35)
    cbnt   = ClusterBNT(survey, cost)

    scheme = cbnt.build_scheme(z_lens=0.6, strategy="3-free")
    scheme.summary()              # alpha, R, leakage, peak — true z & photo-z
    scheme.bin_edges              # the optimised tomographic binning
    scheme.nulled_kernel_true     # the BNT-weighted observable's kernel

The numerical conventions (grids, optimiser tolerances, seeds of the published
analysis) are ported **unchanged** from the notebook so that results are
reproducible to machine precision.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import bntcluster as bnt
from bntcluster import BNTNuller, NFWMockGenerator


# ════════════════════════════════════════════════════════════════════════════
#  1 · Survey description
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SurveySetup:
    """Describes the survey for which the BNT framework is built.

    Parameters
    ----------
    n_of_z : callable
        Un-normalised source redshift distribution ``n(z)`` (e.g.
        ``bntcluster.nz_euclid``).  Used both for the analytic lensing
        kernels and to sample mock source catalogues.
    n_gal_per_arcmin2 : float
        Source surface density [galaxies / arcmin^2].
    shape_noise : float
        Intrinsic ellipticity dispersion per shear component
        (sigma_epsilon, e.g. 0.26).
    photoz_sigma : float
        Photometric-redshift scatter prefactor: sigma_z = photoz_sigma * (1+z).
    photoz_outlier_fraction : float
        Fraction of catastrophic photo-z outliers, redistributed uniformly
        over [z_min_sources, z_max_sources].
    field_deg : float
        Side of the square simulated field [degrees].
    npix : int
        Pixel resolution of the (square) convergence map.
    z_min_sources, z_max_sources : float
        Redshift range over which sources exist (sampling range of the mock
        catalogues and support of the analytic n(z) grid).
    n_z_grid : int
        Number of points of the redshift grid used for kernels / binning.
    n_z_slices_maps : int
        Number of redshift slices used by the map generator when integrating
        the NFW convergence over the source distribution.
    cosmo_params : dict
        Cosmological parameters forwarded to astropy / colossus
        (defaults to the Planck 2015 values used in the paper).
    """
    n_of_z: Callable = bnt.nz_euclid
    n_gal_per_arcmin2: float = 30.0
    shape_noise: float = 0.26
    photoz_sigma: float = 0.05
    photoz_outlier_fraction: float = 0.10
    field_deg: float = 5.0
    npix: int = 512
    z_min_sources: float = 0.001
    z_max_sources: float = 3.0
    n_z_grid: int = 1000
    n_z_slices_maps: int = 60
    cosmo_params: dict = field(default_factory=lambda: dict(bnt.DEFAULT_COSMO_PARAMS))

    # ── derived quantities ──────────────────────────────────────────────────
    @cached_property
    def z_grid(self) -> np.ndarray:
        """Redshift grid shared by all analytic computations."""
        return np.linspace(self.z_min_sources, self.z_max_sources, self.n_z_grid)

    @cached_property
    def nuller(self) -> BNTNuller:
        """The low-level BNT engine built on this survey's n(z)."""
        return BNTNuller(z=self.z_grid, n_of_z=self.n_of_z)

    @property
    def pixel_arcmin(self) -> float:
        """Pixel scale of the convergence map [arcmin]."""
        return self.field_deg * 60.0 / self.npix

    @property
    def n_sources_total(self) -> int:
        """Total number of source galaxies implied by density × area."""
        return int(self.n_gal_per_arcmin2 * (self.field_deg * 60.0) ** 2)

    # ── mock catalogues ─────────────────────────────────────────────────────
    def draw_source_catalog(self, rng: np.random.Generator,
                            n_gal_per_arcmin2: Optional[float] = None
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a mock source catalogue for this survey.

        Draws true redshifts from ``n_of_z`` by inverse-CDF sampling, then
        produces the photometric redshifts with Gaussian scatter
        ``photoz_sigma * (1+z)`` plus ``photoz_outlier_fraction`` catastrophic
        outliers.  Both draws consume the *same* generator ``rng`` in sequence,
        so a fixed seed reproduces the catalogue exactly.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random generator (pass ``np.random.default_rng(seed)`` for
            reproducibility).
        n_gal_per_arcmin2 : float, optional
            Override the survey density for this draw (e.g. a denser variant).

        Returns
        -------
        (z_true, z_phot) : tuple of ndarray
            True and photometric redshifts of the sources.
        """
        density = self.n_gal_per_arcmin2 if n_gal_per_arcmin2 is None else float(n_gal_per_arcmin2)
        n_total = int(density * (self.field_deg * 60.0) ** 2)
        z_true = bnt.sample_nz(self.n_of_z, n_total,
                               z_min=self.z_min_sources, z_max=self.z_max_sources,
                               rng=rng)
        z_phot = bnt.apply_photoz_errors(z_true, sigma_factor=self.photoz_sigma,
                                         outlier_fraction=self.photoz_outlier_fraction,
                                         z_max=self.z_max_sources, rng=rng)
        return z_true, z_phot


# ════════════════════════════════════════════════════════════════════════════
#  2 · Cost-function description
# ════════════════════════════════════════════════════════════════════════════

#: Strategy aliases accepted everywhere a ``strategy`` argument appears.
STRATEGIES = ("equipopulated", "1-free", "2-free", "3-free")

#: Display labels used in figures / summaries.
STRATEGY_LABELS = {
    "equipopulated": "equipopulated",
    "1-free": "1 free (z_fg)",
    "2-free": "2 free (z_fg, z_bg)",
    "3-free": "3 free (z_min, z_fg, z_bg)",
}


def _normalise_strategy(strategy) -> str:
    """Map user input ('3-free', 3, '3 free (z_min, z_fg, z_bg)') to a key."""
    if isinstance(strategy, (int, np.integer)):
        key = f"{int(strategy)}-free"
    else:
        s = str(strategy).strip().lower()
        if s.startswith("equipop"):
            key = "equipopulated"
        else:
            key = s.replace(" free", "-free").split("(")[0].strip()
            key = key if key.endswith("-free") else s
    if key not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; use one of {STRATEGIES}.")
    return key


@dataclass
class CostFunction:
    """Configuration of the bin-edge optimisation cost.

    The optimiser minimises

        C(theta) = base(theta) * [ 1 + peak_weight   * (z_peak - z_lens)^2
                                     + leakage_weight * max(0, L - leakage_tolerance)^2 ]

    where ``theta`` are the free bin edges, ``z_peak`` is the redshift at which
    the nulled kernel peaks, and ``L`` is the foreground leakage
    (mean |w~| below z_lens, relative to the kernel peak).

    Parameters
    ----------
    objective : {"max_R", "min_alpha"}
        Choice of the base term:

        * ``"max_R"``   — base = −log R, with R = w~(z_lens) / (alpha ·
          w_single(z_lens) · sqrt(f_single)) the predicted BNT/unweighted
          detection-S/N ratio.  R < 1 for these schemes, so −log R > 0 and
          minimising C maximises R.  **This is the objective of the current
          analysis.**
        * ``"min_alpha"`` — base = alpha, the shape-noise amplification.
          This reproduces the earlier (alpha-minimising) analysis.

    peak_weight : float
        lambda_pk — weight of the kernel-peak-position penalty.
    leakage_weight : float
        lambda_nl — weight of the foreground-leakage penalty.
    leakage_tolerance : float
        epsilon — maximum tolerated foreground leakage L before the leakage
        penalty activates (one-sided constraint).
    """
    objective: str = "max_R"
    peak_weight: float = 200.0
    leakage_weight: float = 300.0
    leakage_tolerance: float = 0.35

    def __post_init__(self):
        if self.objective not in ("max_R", "min_alpha"):
            raise ValueError("objective must be 'max_R' or 'min_alpha', "
                             f"got {self.objective!r}")


# ════════════════════════════════════════════════════════════════════════════
#  3 · Result containers
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterBNTScheme:
    """All outputs of one cluster-adapted BNT scheme at one lens redshift.

    Every quantity exists in a *true-z* and a *photo-z* version: the photo-z
    one uses the true-redshift distribution of the galaxies that land in each
    *photometric* bin (Gaussian scatter + catastrophic outliers), so it shows
    how the scheme degrades under realistic redshift errors.

    Attributes
    ----------
    z_lens : float
        Target lens redshift the scheme was optimised for.
    strategy : str
        One of ``STRATEGIES``.
    bin_edges : ndarray, shape (4,)
        Tomographic bin edges [z_min, z_fg, z_bg, z_max].
    z_grid : ndarray
        Redshift grid on which kernels / distributions are tabulated.
    bin_distributions_true / bin_distributions_photoz : ndarray, (3, n_z)
        Normalised per-bin source distributions n_i(z).
    bin_fractions_true / bin_fractions_photoz : ndarray, (3,)
        Fraction of the source sample in each bin (f_i).
    kernels_true / kernels_photoz : ndarray, (3, n_z)
        Ordinary (un-nulled) tomographic lensing kernels W_i(z).
    bnt_matrix_true / bnt_matrix_photoz : ndarray, (3, 3)
        BNT mixing matrix M; the *last row* is the nulled combination.
    nulled_kernel_true / nulled_kernel_photoz : ndarray, (n_z,)
        BNT-weighted kernel w~(z) of the nulled observable (last BNT row).
    alpha_true / alpha_photoz : float
        Shape-noise amplification sqrt(sum_j M_j^2 / f_j).
    z_peak_true / z_peak_photoz : float
        Redshift at which the nulled kernel peaks.
    leakage_true / leakage_photoz : float
        Foreground leakage L (mean |w~| below z_lens relative to the peak).
    R_pred_true / R_pred_photoz : float
        Predicted BNT/unweighted detection-S/N ratio.
    """
    z_lens: float
    strategy: str
    bin_edges: np.ndarray
    z_grid: np.ndarray
    # true-z quantities
    bin_distributions_true: np.ndarray
    bin_fractions_true: np.ndarray
    kernels_true: np.ndarray
    bnt_matrix_true: np.ndarray
    nulled_kernel_true: np.ndarray
    alpha_true: float
    z_peak_true: float
    leakage_true: float
    R_pred_true: float
    # photo-z quantities
    bin_distributions_photoz: np.ndarray
    bin_fractions_photoz: np.ndarray
    kernels_photoz: np.ndarray
    bnt_matrix_photoz: np.ndarray
    nulled_kernel_photoz: np.ndarray
    alpha_photoz: float
    z_peak_photoz: float
    leakage_photoz: float
    R_pred_photoz: float

    @property
    def bnt_weights_true(self) -> np.ndarray:
        """Weights of the nulled combination (last BNT row), true z."""
        return self.bnt_matrix_true[-1]

    @property
    def bnt_weights_photoz(self) -> np.ndarray:
        """Weights of the nulled combination (last BNT row), photo-z."""
        return self.bnt_matrix_photoz[-1]

    def summary(self) -> None:
        """Print a compact human-readable summary of the scheme."""
        print(f"Cluster-adapted BNT scheme — strategy: {self.strategy!r}, "
              f"z_lens = {self.z_lens}")
        print(f"  bin edges [z_min, z_fg, z_bg, z_max]: "
              f"{np.round(self.bin_edges, 4).tolist()}")
        print(f"  bin fractions (true z):  "
              f"{np.round(self.bin_fractions_true, 4).tolist()}")
        for tag, a, p, L, R in [
            ("true z ", self.alpha_true, self.z_peak_true,
             self.leakage_true, self.R_pred_true),
            ("photo-z", self.alpha_photoz, self.z_peak_photoz,
             self.leakage_photoz, self.R_pred_photoz),
        ]:
            print(f"  {tag}:  alpha = {a:6.3f}   kernel peak = {p:5.3f}   "
                  f"leakage L = {100*L:4.1f}%   R_pred = {R:5.3f}")

    def as_dict(self) -> dict:
        """Return all fields as a plain dictionary (e.g. for np.savez)."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class SchemeScan:
    """Results of ``ClusterBNT.scan_lens_redshifts`` over a z_lens grid.

    Attributes are arrays indexed like ``z_lens_grid``; ``schemes`` holds the
    full per-point :class:`ClusterBNTScheme` objects for anything not
    pre-extracted here.
    """
    strategy: str
    z_lens_grid: np.ndarray
    schemes: list      # list[ClusterBNTScheme]

    @property
    def bin_edges(self) -> np.ndarray:
        """(n_z_lens, 4) array of bin edges."""
        return np.array([s.bin_edges for s in self.schemes])

    def _vec(self, attr: str) -> np.ndarray:
        return np.array([getattr(s, attr) for s in self.schemes])

    @property
    def alpha_true(self): return self._vec("alpha_true")
    @property
    def alpha_photoz(self): return self._vec("alpha_photoz")
    @property
    def leakage_true(self): return self._vec("leakage_true")
    @property
    def leakage_photoz(self): return self._vec("leakage_photoz")
    @property
    def z_peak_true(self): return self._vec("z_peak_true")
    @property
    def z_peak_photoz(self): return self._vec("z_peak_photoz")
    @property
    def R_pred_true(self): return self._vec("R_pred_true")
    @property
    def R_pred_photoz(self): return self._vec("R_pred_photoz")

    def as_legacy_dict(self) -> dict:
        """Return the dict layout used by the notebook's figure cells:
        keys ``edges, kern_tz, kern_pz, alpha_tz, alpha_pz, peak_tz, peak_pz,
        leak_tz, leak_pz`` (lists indexed like ``z_lens_grid``)."""
        return dict(
            edges=[s.bin_edges for s in self.schemes],
            kern_tz=[s.nulled_kernel_true for s in self.schemes],
            kern_pz=[s.nulled_kernel_photoz for s in self.schemes],
            alpha_tz=[s.alpha_true for s in self.schemes],
            alpha_pz=[s.alpha_photoz for s in self.schemes],
            peak_tz=[s.z_peak_true for s in self.schemes],
            peak_pz=[s.z_peak_photoz for s in self.schemes],
            leak_tz=[s.leakage_true for s in self.schemes],
            leak_pz=[s.leakage_photoz for s in self.schemes],
        )


@dataclass
class StandardBNT:
    """Outputs of the standard ('normal') many-bin BNT tomography.

    All arrays are indexed by BNT row (= tomographic bin) ``i``; rows with
    ``i >= 2`` satisfy both nulling conditions, rows 0 and 1 are the identity
    and a plain difference, respectively.
    """
    n_bins: int
    bin_edges: np.ndarray
    z_grid: np.ndarray
    # true z
    bin_distributions_true: np.ndarray
    bin_fractions_true: np.ndarray
    kernels_true: np.ndarray
    bnt_matrix_true: np.ndarray
    nulled_kernels_true: np.ndarray
    alpha_true: np.ndarray
    # photo-z
    bin_distributions_photoz: np.ndarray
    bin_fractions_photoz: np.ndarray
    kernels_photoz: np.ndarray
    bnt_matrix_photoz: np.ndarray
    nulled_kernels_photoz: np.ndarray
    alpha_photoz: np.ndarray

    @property
    def z_peak(self) -> np.ndarray:
        """Peak redshift of each true-z BNT-nulled kernel."""
        return np.array([self.z_grid[np.argmax(k)] for k in self.nulled_kernels_true])

    def summary(self) -> None:
        """Print bin edges, fractions, per-row alpha and kernel peaks."""
        print(f"Standard BNT tomography — {self.n_bins} equal-count bins")
        print(f"edges:   {np.round(self.bin_edges, 3)}")
        print(f"fracs:   {np.round(self.bin_fractions_true, 3)} "
              f"(sum = {self.bin_fractions_true.sum():.3f})")
        print(f"alpha (true z):  {np.round(self.alpha_true, 2)}")
        print(f"alpha (photo-z): {np.round(self.alpha_photoz, 2)}")
        print(f"BNT kernel peak: {np.round(self.z_peak, 3)}")


# ════════════════════════════════════════════════════════════════════════════
#  4 · The framework
# ════════════════════════════════════════════════════════════════════════════

class ClusterBNT:
    """Cluster-adapted BNT framework for one survey and one cost function.

    Parameters
    ----------
    survey : SurveySetup
        Survey description (n(z), density, noise, photo-z model, geometry).
    cost : CostFunction
        Optimisation objective and penalty fine-tuning.

    Notes
    -----
    Numerical-convention constants (kept identical to the published analysis):

    * ``Z_EPS = 1e-3`` — lower bound used instead of exactly 0 for z_min, to
      avoid an empty first foreground bin;
    * ``MIN_BIN_DZ = 0.01`` — minimum allowed bin width;
    * Brent tolerance ``xatol = 1e-6`` (1 free edge);
    * L-BFGS-B ``ftol = 1e-12``, ``maxiter = 500`` (2–3 free edges).
    """

    Z_EPS = 0.001
    MIN_BIN_DZ = 0.01

    def __init__(self, survey: SurveySetup, cost: CostFunction):
        self.survey = survey
        self.cost = cost

    # ── basic accessors ─────────────────────────────────────────────────────
    @property
    def nuller(self) -> BNTNuller:
        """The low-level BNT engine (kernels, bin distributions, BNT matrix)."""
        return self.survey.nuller

    @property
    def z_grid(self) -> np.ndarray:
        return self.nuller.z

    @property
    def z_max_source(self) -> float:
        """Upper edge used for the background bin (survey ceiling on the grid)."""
        return float(min(self.nuller.z_max_source, self.nuller.z[-1]))

    # ── per-binning diagnostics (analytic, no maps) ─────────────────────────
    def kernel_diagnostics_true(self, bin_edges):
        """(nulled_kernel, z_peak, alpha, bin_fractions) for the last BNT row,
        assuming perfect (true) source redshifts."""
        n_i, frac = self.nuller.bin_distributions(bin_edges)
        M = self.nuller.compute_bnt_matrix(n_i)
        kern = (M @ self.nuller.compute_lensing_kernels(n_i))[-1]
        return (kern, float(self.nuller.z[np.argmax(kern)]),
                float(np.sqrt(np.sum(M[-1] ** 2 / frac))), frac)

    def kernel_diagnostics_photoz(self, bin_edges):
        """(nulled_kernel, z_peak, alpha, bin_fractions, n_i) for the last BNT
        row, with the survey's photo-z error model applied to the binning."""
        n_i, frac = self.nuller.photoz_smeared_bin_distributions(
            bin_edges, sigma_factor=self.survey.photoz_sigma,
            outlier_fraction=self.survey.photoz_outlier_fraction)
        M = self.nuller.compute_bnt_matrix(n_i)
        kern = (M @ self.nuller.compute_lensing_kernels(n_i))[-1]
        return (kern, float(self.nuller.z[np.argmax(kern)]),
                float(np.sqrt(np.sum(M[-1] ** 2 / frac))), frac, n_i)

    def foreground_leakage(self, nulled_kernel, z_lens) -> float:
        """Foreground leakage L: mean |w~| over [0, z_lens] relative to the
        kernel peak.  L = 0 means a perfectly nulled foreground; L ~ 0.5 means
        the foreground is barely suppressed."""
        pk = float(np.max(nulled_kernel))
        if pk <= 0:
            return 9.9
        fg = self.nuller.z <= z_lens
        return float(np.trapz(np.abs(nulled_kernel[fg]), self.nuller.z[fg])
                     / (pk * z_lens))

    # ── the unweighted single-bin reference (denominator of R) ──────────────
    def single_bin_source_fraction(self, z_lens, photoz: bool = False) -> float:
        """f_single: fraction of the source sample kept by the z_s > z_lens cut."""
        e = np.array([z_lens, self.z_max_source])
        if photoz:
            _, fr = self.nuller.photoz_smeared_bin_distributions(
                e, sigma_factor=self.survey.photoz_sigma,
                outlier_fraction=self.survey.photoz_outlier_fraction)
        else:
            _, fr = self.nuller.bin_distributions(e)
        return float(fr[0])

    def single_bin_kernel_at_lens(self, z_lens, photoz: bool = False) -> float:
        """w_single(z_lens): ordinary lensing kernel of the single z_s > z_lens
        reference bin, evaluated at the lens redshift."""
        e = np.array([z_lens, self.z_max_source])
        if photoz:
            n_i, _ = self.nuller.photoz_smeared_bin_distributions(
                e, sigma_factor=self.survey.photoz_sigma,
                outlier_fraction=self.survey.photoz_outlier_fraction)
        else:
            n_i, _ = self.nuller.bin_distributions(e)
        kern = self.nuller.compute_lensing_kernels(n_i)[0]
        return float(kern[np.argmin(np.abs(self.nuller.z - z_lens))])

    def reference_signal_scale(self, z_lens) -> float:
        """w_single(z_lens) * sqrt(f_single): signal scale of the unweighted
        single-bin reference map (true z).  This is the denominator of R up to
        the alpha factor, and depends on z_lens only — not on the bin edges —
        so the optimiser computes it once per call."""
        return (self.single_bin_kernel_at_lens(z_lens, photoz=False)
                * np.sqrt(self.single_bin_source_fraction(z_lens, photoz=False)))

    def predict_R(self, bin_edges, z_lens, photoz: bool = False) -> float:
        """Predicted BNT/unweighted detection-S/N ratio for a given binning:

            R = w~(z_lens) / ( alpha * w_single(z_lens) * sqrt(f_single) ).

        ``photoz=True`` evaluates every ingredient (nulled kernel, alpha,
        reference kernel and fraction) under the photo-z error model.
        """
        if photoz:
            kern, _, alpha, _, _ = self.kernel_diagnostics_photoz(bin_edges)
        else:
            kern, _, alpha, _ = self.kernel_diagnostics_true(bin_edges)
        w_at_lens = float(kern[np.argmin(np.abs(self.nuller.z - z_lens))])
        w_single = self.single_bin_kernel_at_lens(z_lens, photoz)
        f_single = self.single_bin_source_fraction(z_lens, photoz)
        return (w_at_lens / alpha) / (w_single * np.sqrt(f_single))

    # ── bin-edge optimisation ───────────────────────────────────────────────
    def optimise_bin_edges(self, z_lens, strategy, peak_weight=None,
                           leakage_tolerance=None, warm_starts=()):
        """Optimise the 3-bin edges for a target lens redshift.

        Minimises  C = base * [1 + peak_weight*(z_peak - z_lens)^2
        + leakage_weight*max(0, L - leakage_tolerance)^2], where the base term
        is set by ``cost.objective`` (−log R for ``"max_R"``, alpha for
        ``"min_alpha"``).

        Parameters
        ----------
        z_lens : float
            Target lens redshift.
        strategy : str or int
            ``"equipopulated"`` (closed form, no optimisation), or
            ``"1-free"`` / ``"2-free"`` / ``"3-free"`` (number of free edges).
        peak_weight, leakage_tolerance : float, optional
            Per-call overrides of the cost configuration (used e.g. by the
            Pareto sweep, which varies the tolerance epsilon).
        warm_starts : sequence of edge vectors, optional
            Extra candidate solutions (e.g. the neighbouring z_lens solution
            in a scan).  A warm start can only improve the result.

        Returns
        -------
        ndarray, shape (4,)
            Bin edges [z_min, z_fg, z_bg, z_max].

        Notes
        -----
        Optimiser strategy (identical to the published analysis):

        * 1 free edge — bounded scalar **Brent** search (the 1-D landscape is
          well behaved, so this is smooth and precise across z_lens);
        * 2–3 free edges — **coarse grid seed** over the allowed edge ranges,
          then bounded **L-BFGS-B** refinement; the 3-free search is also
          warm-started from the 2-free optimum, so it can only improve on it.
        """
        key = _normalise_strategy(strategy)
        if key == "equipopulated":
            # Closed form: two equal-count foreground bins below z_lens,
            # background bin [z_lens, z_max].  Nothing to optimise.
            return self.nuller.foreground_equal_count_edges(
                z_lens, n_foreground_bins=2)
        n_free = int(key[0])

        peak_weight = self.cost.peak_weight if peak_weight is None else peak_weight
        null_tol = (self.cost.leakage_tolerance if leakage_tolerance is None
                    else leakage_tolerance)
        leak_weight = self.cost.leakage_weight
        maximise_R = (self.cost.objective == "max_R")

        nuller = self.nuller
        Z_EPS, MIN_BIN_DZ = self.Z_EPS, self.MIN_BIN_DZ
        Z_MAX_SRC = self.z_max_source
        # Reference scale of R (depends on z_lens only -> computed once).
        ref = self.reference_signal_scale(z_lens) if maximise_R else None

        def base_term(kern, alpha):
            """Objective base: -log R (max_R) or alpha (min_alpha).
            Returns None when the configuration is invalid (R <= 0)."""
            if not maximise_R:
                return alpha
            w_at_lens = float(kern[np.argmin(np.abs(nuller.z - z_lens))])
            R = w_at_lens / (alpha * ref) if (alpha > 0 and ref > 0) else 0.0
            if not (np.isfinite(R) and R > 0.0):
                return None
            return -np.log(R)

        if n_free == 1:
            def cost1(zfg):
                edges = np.array([Z_EPS, zfg, z_lens, Z_MAX_SRC])
                if np.any(np.diff(edges) < MIN_BIN_DZ):
                    return 1e10
                try:
                    kern, zp, al, _ = self.kernel_diagnostics_true(edges)
                    base = base_term(kern, al)
                    if base is None:
                        return 1e10
                    c = base * (1.0 + peak_weight * (zp - z_lens) ** 2)
                    if null_tol is not None:
                        L = self.foreground_leakage(kern, z_lens)
                        c *= (1.0 + leak_weight * max(0.0, L - null_tol) ** 2)
                    return c
                except Exception:
                    return 1e10

            lo, hi = Z_EPS + MIN_BIN_DZ, z_lens - MIN_BIN_DZ
            res = minimize_scalar(cost1, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-6})
            best_x, best_c = float(res.x), float(res.fun)
            for ws in warm_starts:
                if ws is None:
                    continue
                try:
                    zfg = max(lo, min(hi, float(np.asarray(ws)[1])))
                    c = cost1(zfg)
                    if c < best_c:
                        best_c, best_x = c, zfg
                except Exception:
                    pass
            return np.array([Z_EPS, best_x, z_lens, Z_MAX_SRC])

        if n_free == 2:
            def build(p): return np.array([Z_EPS, p[0], p[1], Z_MAX_SRC])
            z_max_bg = min(z_lens + 0.7, Z_MAX_SRC - MIN_BIN_DZ)
            bounds = [(Z_EPS + MIN_BIN_DZ,            z_lens + 0.3),
                      (max(z_lens - 0.1, MIN_BIN_DZ), z_max_bg)]
            grid = [[qfg, zbg]
                    for qfg in np.linspace(0.05, 0.95, 12) * z_lens
                    for zbg in np.linspace(z_lens, z_max_bg, 12)]
            x0 = [0.5 * z_lens, z_lens + 0.1]
            def pack(ws): return [float(ws[1]), float(ws[2])]
        elif n_free == 3:
            def build(p): return np.array([p[0], p[1], p[2], Z_MAX_SRC])
            z_max_bg = min(z_lens + 0.7, Z_MAX_SRC - MIN_BIN_DZ)
            bounds = [(Z_EPS,                          0.5 * z_lens),
                      (Z_EPS + MIN_BIN_DZ,             z_lens + 0.3),
                      (max(z_lens - 0.1, MIN_BIN_DZ),  z_max_bg)]
            grid = [[zmn, qfg, zbg]
                    for zmn in np.linspace(0.0, 0.3 * z_lens, 4)
                    for qfg in np.linspace(0.05, 0.95, 8) * z_lens
                    for zbg in np.linspace(z_lens, z_max_bg, 8)]
            x0 = [Z_EPS, 0.5 * z_lens, z_lens + 0.1]
            def pack(ws): return [float(ws[0]), float(ws[1]), float(ws[2])]
        else:
            raise ValueError("n_free must be 1, 2 or 3")

        def cost(p):
            try:
                edges = build(p)
                if np.any(np.diff(edges) < MIN_BIN_DZ):
                    return 1e10
                kern, zp, al, _ = self.kernel_diagnostics_true(edges)
                base = base_term(kern, al)
                if base is None:
                    return 1e10
                c = base * (1.0 + peak_weight * (zp - z_lens) ** 2)
                if null_tol is not None:
                    L = self.foreground_leakage(kern, z_lens)
                    c *= (1.0 + leak_weight * max(0.0, L - null_tol) ** 2)
                return c
            except Exception:
                return 1e10

        # Coarse grid seed.
        best_x, best_c = list(x0), cost(x0)
        for g in grid:
            c = cost(g)
            if c < best_c:
                best_c, best_x = c, list(g)

        # 3-free: warm-start from the 2-free optimum (z_min prepended).
        if n_free == 3:
            e2 = self.optimise_bin_edges(z_lens, "2-free",
                                         peak_weight=peak_weight,
                                         leakage_tolerance=null_tol)
            for x_alt in ([Z_EPS, float(e2[1]), float(e2[2])],
                          [0.05 * z_lens, float(e2[1]), float(e2[2])]):
                c = cost(x_alt)
                if c < best_c:
                    best_c, best_x = c, list(x_alt)

        # Caller-supplied warm starts (e.g. previous z_lens in a scan).
        for ws in warm_starts:
            if ws is None:
                continue
            try:
                p = pack(np.asarray(ws, dtype=float))
                p = [max(b[0], min(b[1], x)) for x, b in zip(p, bounds)]
                c = cost(p)
                if c < best_c:
                    best_c, best_x = c, list(p)
            except Exception:
                pass

        res = minimize(cost, best_x, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-12, "maxiter": 500})
        return build(res.x) if cost(list(res.x)) < best_c else build(best_x)

    def evaluate_cost(self, bin_edges, z_lens, peak_weight=None,
                      leakage_tolerance=None) -> float:
        """Value of the optimiser's cost function for a given binning.

        Computes, for a full 4-edge vector,

            C = base * [ 1 + peak_weight * (z_peak - z_lens)^2
                           + leakage_weight * max(0, L - leakage_tolerance)^2 ],

        with ``base = -log R`` for ``cost.objective == "max_R"`` and
        ``base = alpha`` for ``"min_alpha"`` (true-z, last BNT row).  This is
        exactly the quantity :meth:`optimise_bin_edges` minimises, exposed so
        that callers (e.g. a scan breaking forward/backward ties) can rank
        candidate binnings on the cost itself rather than on alpha alone.

        Parameters
        ----------
        bin_edges : array-like, shape (4,)
            Tomographic bin edges [z_min, z_fg, z_bg, z_max].
        z_lens : float
            Target lens redshift the binning was built for.
        peak_weight, leakage_tolerance : float, optional
            Per-call overrides of the cost configuration (defaults taken from
            ``self.cost``), mirroring :meth:`optimise_bin_edges`.

        Returns
        -------
        float
            The cost C, or ``np.inf`` for an invalid binning (degenerate edge
            ordering, or R <= 0 under the ``max_R`` objective).
        """
        edges = np.asarray(bin_edges, dtype=float)
        if np.any(np.diff(edges) < self.MIN_BIN_DZ):
            return np.inf
        peak_weight = self.cost.peak_weight if peak_weight is None else peak_weight
        null_tol = (self.cost.leakage_tolerance if leakage_tolerance is None
                    else leakage_tolerance)
        leak_weight = self.cost.leakage_weight

        kern, z_peak, alpha, _ = self.kernel_diagnostics_true(edges)
        if self.cost.objective == "max_R":
            ref = self.reference_signal_scale(z_lens)
            w_at_lens = float(kern[np.argmin(np.abs(self.nuller.z - z_lens))])
            R = w_at_lens / (alpha * ref) if (alpha > 0 and ref > 0) else 0.0
            if not (np.isfinite(R) and R > 0.0):
                return np.inf
            base = -np.log(R)
        else:
            base = alpha
        c = base * (1.0 + peak_weight * (z_peak - z_lens) ** 2)
        if null_tol is not None:
            L = self.foreground_leakage(kern, z_lens)
            c *= (1.0 + leak_weight * max(0.0, L - null_tol) ** 2)
        return float(c)

    # ── high-level builders ─────────────────────────────────────────────────
    def build_scheme(self, z_lens, strategy, bin_edges=None,
                     **optimiser_kwargs) -> ClusterBNTScheme:
        """Build the complete cluster-adapted scheme for one lens redshift.

        Optimises the bin edges (unless ``bin_edges`` is supplied) and returns
        a :class:`ClusterBNTScheme` exposing the binning, the kernels, the BNT
        matrix, the nulled observable's kernel, and the diagnostics (alpha,
        kernel peak, leakage L, predicted R) — each in true-z and photo-z
        versions.

        Parameters
        ----------
        z_lens : float
            Target lens redshift.
        strategy : str or int
            See :meth:`optimise_bin_edges`.
        bin_edges : array-like, optional
            Skip the optimisation and evaluate this binning instead.
        **optimiser_kwargs
            Forwarded to :meth:`optimise_bin_edges` (peak_weight,
            leakage_tolerance, warm_starts).
        """
        key = _normalise_strategy(strategy)
        if bin_edges is None:
            bin_edges = self.optimise_bin_edges(z_lens, key, **optimiser_kwargs)
        bin_edges = np.asarray(bin_edges, dtype=float)

        # true z
        n_i_t, frac_t = self.nuller.bin_distributions(bin_edges)
        W_t = self.nuller.compute_lensing_kernels(n_i_t)
        M_t = self.nuller.compute_bnt_matrix(n_i_t)
        nulled_t = (M_t @ W_t)[-1]
        alpha_t = float(np.sqrt(np.sum(M_t[-1] ** 2 / frac_t)))
        peak_t = float(self.nuller.z[np.argmax(nulled_t)])
        leak_t = self.foreground_leakage(nulled_t, z_lens)
        # photo-z
        n_i_p, frac_p = self.nuller.photoz_smeared_bin_distributions(
            bin_edges, sigma_factor=self.survey.photoz_sigma,
            outlier_fraction=self.survey.photoz_outlier_fraction)
        W_p = self.nuller.compute_lensing_kernels(n_i_p)
        M_p = self.nuller.compute_bnt_matrix(n_i_p)
        nulled_p = (M_p @ W_p)[-1]
        alpha_p = float(np.sqrt(np.sum(M_p[-1] ** 2 / frac_p)))
        peak_p = float(self.nuller.z[np.argmax(nulled_p)])
        leak_p = self.foreground_leakage(nulled_p, z_lens)

        return ClusterBNTScheme(
            z_lens=float(z_lens), strategy=key, bin_edges=bin_edges,
            z_grid=self.nuller.z,
            bin_distributions_true=n_i_t, bin_fractions_true=frac_t,
            kernels_true=W_t, bnt_matrix_true=M_t, nulled_kernel_true=nulled_t,
            alpha_true=alpha_t, z_peak_true=peak_t, leakage_true=leak_t,
            R_pred_true=self.predict_R(bin_edges, z_lens, photoz=False),
            bin_distributions_photoz=n_i_p, bin_fractions_photoz=frac_p,
            kernels_photoz=W_p, bnt_matrix_photoz=M_p,
            nulled_kernel_photoz=nulled_p,
            alpha_photoz=alpha_p, z_peak_photoz=peak_p, leakage_photoz=leak_p,
            R_pred_photoz=self.predict_R(bin_edges, z_lens, photoz=True),
        )

    def scan_lens_redshifts(self, z_lens_grid, strategy,
                            continuation=True, tie_break="alpha") -> SchemeScan:
        """Optimise the scheme over a grid of lens redshifts.

        Parameters
        ----------
        z_lens_grid : array-like
            Lens redshifts to scan.
        strategy : str or int
            See :meth:`optimise_bin_edges`.
        continuation : bool
            If True (and the strategy has free edges), run a forward and a
            backward pass in which each solution warm-starts its neighbour,
            then keep, at each z_lens, the better of the two (per ``tie_break``).
            This keeps consecutive solutions in the same basin and makes the
            diagnostic curves smooth.
        tie_break : {"alpha", "cost", "R"}, default "alpha"
            Which quantity decides, at each z_lens, between the forward- and
            backward-pass solution (ties favour the forward pass):

            * ``"alpha"`` — keep the lower shape-noise amplification alpha.
              **Default, for reproducibility with the published analysis**; note
              it ranks on alpha even when ``cost.objective == "max_R"``.
            * ``"cost"`` — keep the lower value of the actual cost function
              (:meth:`evaluate_cost`).  This is the self-consistent choice: the
              kept solution is the one the optimiser itself prefers, whatever
              the objective.
            * ``"R"`` — keep the higher predicted detection-S/N ratio R
              (true-z, :meth:`predict_R`); natural for ``objective == "max_R"``.

            Only consulted when ``continuation`` is True and the strategy has
            free edges; ignored for the equipopulated (closed-form) strategy.

        Returns
        -------
        SchemeScan
        """
        key = _normalise_strategy(strategy)
        z_lens_grid = np.asarray(z_lens_grid, dtype=float)

        tie_break = str(tie_break).lower()
        if tie_break not in ("alpha", "cost", "r"):
            raise ValueError(
                f"tie_break must be 'alpha', 'cost' or 'R', got {tie_break!r}.")

        def _tie_break_score(edges, zl):
            """Lower is better: alpha, the cost, or -R (so max R wins)."""
            if tie_break == "alpha":
                return self.kernel_diagnostics_true(edges)[2]
            if tie_break == "cost":
                return self.evaluate_cost(edges, zl)
            return -self.predict_R(edges, zl, photoz=False)  # "r"

        if key == "equipopulated" or not continuation:
            edges_list = [self.optimise_bin_edges(zl, key) for zl in z_lens_grid]
        else:
            # Forward pass (low -> high z_lens).
            edges_fwd, prev = [], None
            for zl in z_lens_grid:
                warm = (prev,) if prev is not None else ()
                e = self.optimise_bin_edges(zl, key, warm_starts=warm)
                edges_fwd.append(e)
                prev = e
            # Backward pass (high -> low z_lens).
            edges_bwd = [None] * len(z_lens_grid)
            prev = None
            for idx, zl in enumerate(z_lens_grid[::-1]):
                warm = (prev,) if prev is not None else ()
                e = self.optimise_bin_edges(zl, key, warm_starts=warm)
                edges_bwd[-(idx + 1)] = e
                prev = e
            # Keep, at each z_lens, the better of the fwd/bwd solution per
            # `tie_break` (ties favour the forward pass).
            edges_list = []
            for zl, ef, eb in zip(z_lens_grid, edges_fwd, edges_bwd):
                edges_list.append(
                    ef if _tie_break_score(ef, zl) <= _tie_break_score(eb, zl)
                    else eb)

        schemes = [self.build_scheme(zl, key, bin_edges=e)
                   for zl, e in zip(z_lens_grid, edges_list)]
        return SchemeScan(strategy=key, z_lens_grid=z_lens_grid, schemes=schemes)

    def standard_tomography(self, n_bins=8) -> StandardBNT:
        """Build the standard ('normal') BNT tomography with ``n_bins``
        equal-count bins spanning the whole source redshift range, in both
        true-z and photo-z versions."""
        edges = bnt.global_equal_count_edges(self.nuller, n_bins)
        n_i_t, frac_t = self.nuller.bin_distributions(edges)
        n_i_p, frac_p = self.nuller.photoz_smeared_bin_distributions(
            edges, sigma_factor=self.survey.photoz_sigma,
            outlier_fraction=self.survey.photoz_outlier_fraction)
        W_t = self.nuller.compute_lensing_kernels(n_i_t)
        W_p = self.nuller.compute_lensing_kernels(n_i_p)
        M_t = self.nuller.compute_bnt_matrix(n_i_t)
        M_p = self.nuller.compute_bnt_matrix(n_i_p)
        return StandardBNT(
            n_bins=n_bins, bin_edges=edges, z_grid=self.nuller.z,
            bin_distributions_true=n_i_t, bin_fractions_true=frac_t,
            kernels_true=W_t, bnt_matrix_true=M_t,
            nulled_kernels_true=self.nuller.apply_bnt(W_t, M_t),
            alpha_true=np.sqrt(np.sum(M_t ** 2 / frac_t[None, :], axis=1)),
            bin_distributions_photoz=n_i_p, bin_fractions_photoz=frac_p,
            kernels_photoz=W_p, bnt_matrix_photoz=M_p,
            nulled_kernels_photoz=self.nuller.apply_bnt(W_p, M_p),
            alpha_photoz=np.sqrt(np.sum(M_p ** 2 / frac_p[None, :], axis=1)),
        )

    # ── convergence maps ────────────────────────────────────────────────────
    def build_map_generator(self, bin_edges, z_true_catalog, z_phot_catalog,
                            lens_catalog, output_dir, noise_seed):
        """Run the NFW mock generator for this survey and a given binning.

        Produces, for every tomographic bin defined by ``bin_edges``, the
        noiseless convergence map of the injected lenses and a noisy version
        with per-bin Gaussian shape noise (dispersion = survey.shape_noise,
        scaled by the per-pixel galaxy count of the bin).

        Parameters
        ----------
        bin_edges : array-like
            Tomographic bin edges (n_bins + 1 values).
        z_true_catalog, z_phot_catalog : ndarray
            Source redshifts.  Pass the *same* array twice for a perfect-
            redshift analysis; pass (true, photometric) to select sources by
            photometric redshift while lensing them at their true redshift.
        lens_catalog : astropy.table.Table
            Lens catalogue with columns z, RA, Dec, mass.
        output_dir : str
            Directory where the generator writes its FITS maps.
        noise_seed : int
            Seed of the shape-noise realisation (fixing it makes every map
            of the analysis share the same noise field).
        """
        e = np.asarray(bin_edges, dtype=float)
        gen = NFWMockGenerator(
            cosmo_params=self.survey.cosmo_params,
            npix=self.survey.npix, field_deg=self.survey.field_deg,
            shape_noise=self.survey.shape_noise,
            zmin_list=list(e[:-1]), zmax_list=list(e[1:]),
            truncation_radius=None, output_dir=output_dir, seed=noise_seed,
        )
        gen.set_source_catalogs(z_true_catalog, z_phot_catalog)
        gen.set_lens_catalog(lens_catalog)
        gen.run(disable_tqdm=True)
        return gen


# ════════════════════════════════════════════════════════════════════════════
#  5 · Map-level utilities
# ════════════════════════════════════════════════════════════════════════════

def stack_bin_maps(generator, noisy=False) -> np.ndarray:
    """Stack the per-bin maps of a generator into one (n_bins, ny, nx) array.

    ``noisy=False`` returns the noiseless convergence maps; ``noisy=True`` the
    maps with the shape-noise realisation added.
    """
    attr = "noisy_maps" if noisy else "kappa_maps"
    return np.array([getattr(generator, attr)[t] for t in generator.kappa_maps])


def bnt_weighted_map(bnt_row_weights, bin_maps) -> np.ndarray:
    """Apply one row of the BNT matrix to a stack of per-bin maps.

    Parameters
    ----------
    bnt_row_weights : ndarray, shape (n_bins,)
        Weights of the BNT combination (e.g. ``scheme.bnt_weights_true``).
    bin_maps : ndarray, shape (n_bins, ny, nx)
        Per-bin convergence maps (output of :func:`stack_bin_maps`).

    Returns
    -------
    ndarray, shape (ny, nx)
        The BNT-weighted (nulled) map  kappa~ = sum_j M_j kappa_j.
    """
    return np.einsum("j,jyx->yx", np.asarray(bnt_row_weights), bin_maps)


def rectangular_cutout(kappa_map, ra_range_deg, dec_range_deg, field_deg):
    """Cut a rectangular window out of a square full-field map.

    Returns ``(cutout, extent)`` where ``extent`` is the
    [ra_min, ra_max, dec_min, dec_max] list expected by ``plt.imshow``.
    """
    npix = kappa_map.shape[0]
    pix_per_deg = npix / field_deg
    x0 = int(round(ra_range_deg[0] * pix_per_deg))
    x1 = int(round(ra_range_deg[1] * pix_per_deg))
    y0 = int(round(dec_range_deg[0] * pix_per_deg))
    y1 = int(round(dec_range_deg[1] * pix_per_deg))
    return (kappa_map[y0:y1, x0:x1],
            [ra_range_deg[0], ra_range_deg[1], dec_range_deg[0], dec_range_deg[1]])


# ════════════════════════════════════════════════════════════════════════════
#  6 · Plot helpers (A&A figure conventions)
# ════════════════════════════════════════════════════════════════════════════

def aa_style_ax(ax):
    """Apply A&A-compatible axis cosmetics to one Matplotlib axis."""
    ax.grid(False)
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, right=True)
    if ax.legend_ is not None:
        ax.legend(frameon=False)
    return ax


def aa_style_axes(axes):
    """Apply A&A-compatible axis cosmetics to an array of axes."""
    for ax in np.ravel(axes):
        aa_style_ax(ax)
    return axes


def robust_symmetric_vmax(img, percentile=99.5, floor=1e-8) -> float:
    """Robust symmetric colour-scale limit for one displayed map."""
    vals = np.asarray(img, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return floor
    return max(float(np.percentile(np.abs(vals), percentile)), floor)


def map_inside_colorbar(ax, im, width="4.0%", height="76%", nbins=4,
                        labelsize=7.0):
    """Thin vertical colour bar drawn *inside* a map panel.

    The map cut-outs do not use their right edge, so the bar lives there with
    a light translucent backing and compact, power-of-ten-factored tick
    labels.  The colour-bar solids are rasterised so the gradient survives in
    vector PDFs (a transparent colour-bar axis would render empty there).
    """
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax = inset_axes(ax, width=width, height=height, loc="center right",
                     bbox_to_anchor=(-0.085, 0.0, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    cax.add_patch(mpatches.FancyBboxPatch(
        (-0.1, -0.08), 3.3, 1.16, transform=cax.transAxes,
        boxstyle="round,pad=0.08", facecolor="white", edgecolor="0.75",
        linewidth=0.35, alpha=0.70, zorder=-10, clip_on=False))
    cb = ax.figure.colorbar(im, cax=cax, orientation="vertical")
    cb.solids.set_rasterized(True)
    cb.solids.set_edgecolor("face")
    cb.ax.tick_params(labelsize=labelsize, direction="in", length=2.5,
                      width=0.7, pad=1.5)
    cb.outline.set_linewidth(0.6)
    vmin, vmax = im.get_clim()
    vmax_abs = max(abs(vmin), abs(vmax))
    exponent = 0 if vmax_abs == 0 else int(np.floor(np.log10(vmax_abs)))
    scale = 10.0 ** exponent

    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))

    def _fmt(x, _pos):
        y = x / scale
        return (rf"${int(round(y))}$" if np.isclose(y, round(y), atol=1e-8)
                else rf"${y:.1f}$")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    if exponent != 0:
        cb.ax.text(0.5, 1.02, rf"$\times10^{{{exponent}}}$",
                   transform=cb.ax.transAxes, ha="left", va="bottom",
                   fontsize=labelsize)
    return cb


__all__ = [
    "SurveySetup", "CostFunction", "ClusterBNT",
    "ClusterBNTScheme", "SchemeScan", "StandardBNT",
    "STRATEGIES", "STRATEGY_LABELS",
    "stack_bin_maps", "bnt_weighted_map", "rectangular_cutout",
    "aa_style_ax", "aa_style_axes", "robust_symmetric_vmax",
    "map_inside_colorbar",
]
