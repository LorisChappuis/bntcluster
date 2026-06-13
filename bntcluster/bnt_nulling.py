
"""
bnt_nulling.py

Clean BNT-based redshift-nulling framework.

This module implements the framework from the notebook `bnt.ipynb` in a reusable
class.  Instead of histogramming a catalogue of source redshifts, the class takes
an analytical source redshift distribution n(z), builds tomographic source-bin
distributions, computes ordinary lensing kernels, applies the Bernardeau-Nishimichi-
Taruya (BNT) transformation, and extracts the BNT kernel associated with a chosen
target/peak lens redshift.

The intended use is:
    nuller = BNTNuller(z_grid, n_of_z, cosmo)
    result = nuller.compute_for_peak(z_peak=0.6)
    nuller.plot_result(result)
"""

from __future__ import annotations

__version__ = "2026-05-25-equal-count-binning"

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

try:
    from astropy.cosmology import FlatLambdaCDM
except ImportError:  # pragma: no cover
    FlatLambdaCDM = None


class SimpleFlatLambdaCDM:
    """
    Lightweight fallback flat-LambdaCDM cosmology.

    This is used only when astropy is unavailable.  It provides the small subset
    of functionality needed here: h and comoving_distance(z) in Mpc.
    """

    def __init__(self, H0: float = 67.31, Om0: float = 0.31345, Ob0: float = 0.0481):
        self.H0 = float(H0)
        self.Om0 = float(Om0)
        self.Ob0 = float(Ob0)
        self.h = self.H0 / 100.0
        self.c_km_s = 299792.458

    def comoving_distance(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        flat = z.ravel()
        order = np.argsort(flat)
        z_sorted = flat[order]

        # Integrate 1 / E(z) from 0 to z on the requested grid plus z=0.
        grid = np.unique(np.concatenate([[0.0], z_sorted]))
        Ez = np.sqrt(self.Om0 * (1.0 + grid) ** 3 + (1.0 - self.Om0))
        inv_E = 1.0 / Ez
        dz = np.diff(grid)
        integ = np.zeros_like(grid)
        integ[1:] = np.cumsum(0.5 * (inv_E[:-1] + inv_E[1:]) * dz)

        dist_grid = (self.c_km_s / self.H0) * integ
        dist_sorted = np.interp(z_sorted, grid, dist_grid)

        out = np.empty_like(flat)
        out[order] = dist_sorted
        return out.reshape(z.shape)


def _default_cosmology():
    if FlatLambdaCDM is not None:
        return FlatLambdaCDM(H0=67.31, Om0=0.31345, Ob0=0.0481)
    return SimpleFlatLambdaCDM(H0=67.31, Om0=0.31345, Ob0=0.0481)


def _comoving_distance_value(cosmo: object, z: np.ndarray) -> np.ndarray:
    dist = cosmo.comoving_distance(z)
    return np.asarray(getattr(dist, "value", dist), dtype=float)


ArrayLike = Sequence[float] | np.ndarray


@dataclass
class BNTResult:
    """Container for the output of a BNT nulling calculation."""

    z_peak: float
    bin_edges: np.ndarray
    z: np.ndarray
    chi: np.ndarray
    n_i: np.ndarray
    bin_fractions: np.ndarray
    bnt_matrix: np.ndarray
    kernels: np.ndarray
    bnt_kernels: np.ndarray
    selected_kernel_index: int

    @property
    def selected_kernel(self) -> np.ndarray:
        """BNT-transformed kernel selected as the kernel localized around z_peak."""
        return self.bnt_kernels[self.selected_kernel_index]

    @property
    def ordinary_selected_kernel(self) -> np.ndarray:
        """Ordinary, non-BNT kernel for the same source bin."""
        return self.kernels[self.selected_kernel_index]

    @property
    def z_at_kernel_max(self) -> float:
        """
        Redshift at which the selected BNT kernel is maximal.

        Uses parabolic interpolation around the grid-peak to achieve
        sub-grid-spacing accuracy rather than snapping to the nearest z point.
        """
        k = self.selected_kernel
        i = int(np.argmax(k))
        if i == 0 or i == len(k) - 1:
            return float(self.z[i])
        k0, k1, k2 = k[i - 1], k[i], k[i + 1]
        denom = k0 - 2.0 * k1 + k2
        if abs(denom) < 1e-14 * abs(k1):
            return float(self.z[i])
        # Vertex of the interpolating parabola, expressed as a fraction of
        # the grid step away from z[i]
        frac = 0.5 * (k0 - k2) / denom
        dz = self.z[i + 1] - self.z[i]
        return float(self.z[i] + frac * dz)

    @property
    def peak_offset(self) -> float:
        """Difference between the kernel maximum and the requested peak redshift."""
        return self.z_at_kernel_max - self.z_peak

    @property
    def shape_noise_amplification(self) -> float:
        """
        Shape-noise amplification factor for the selected nulled kernel.

        Defined as sqrt(Σ_j M[k,j]² / f_j), where k is the selected kernel
        row, M[k,j] are BNT coefficients, and f_j = bin_fractions[j].
        Gives the noise of the nulled map relative to a single full-sample map.
        """
        k = self.selected_kernel_index
        row = self.bnt_matrix[k]
        return float(np.sqrt(np.sum(row**2 / self.bin_fractions)))


class BNTNuller:
    """
    BNT-based tomographic kernel nuller using an analytical source redshift distribution.

    Parameters
    ----------
    z : array-like
        Redshift grid used both for the lens-redshift grid and source-redshift
        distribution sampling. It should be increasing and should cover the
        full source range.
    n_of_z : callable
        Analytical source redshift distribution. It must accept an array `z`
        and return an array proportional to n(z). The class normalizes it
        internally, so the amplitude is irrelevant.
    cosmo : astropy.cosmology instance, optional
        Cosmology used to compute comoving distances. If omitted, the Euclid-like
        FlatLambdaCDM cosmology used in the original notebook is adopted.
    z_max_source : float, optional
        Maximum redshift used for the final source bin. If omitted, max(z) is used.
    eps : float
        Small value used to avoid divisions by zero.

    Notes
    -----
    The BNT transformation constructs linear combinations of tomographic lensing
    kernels. For row i >= 2, the combination of source bins i, i-1, and i-2 is
    chosen so that the low-redshift part of the kernel cancels to first order:
        M[i, i] = 1,
        M[i, i-1] and M[i, i-2] solve two moment-cancellation equations.
    """

    def __init__(
        self,
        z: ArrayLike,
        n_of_z: Callable[[np.ndarray], np.ndarray],
        cosmo: Optional[object] = None,
        z_max_source: Optional[float] = None,
        eps: float = 1.0e-12,
    ) -> None:
        self.z = np.asarray(z, dtype=float)
        if self.z.ndim != 1:
            raise ValueError("z must be a 1D array.")
        if np.any(np.diff(self.z) <= 0):
            raise ValueError("z must be strictly increasing.")

        self.n_of_z = n_of_z
        self.cosmo = cosmo or _default_cosmology()
        self.z_max_source = float(np.max(self.z) if z_max_source is None else z_max_source)
        self.eps = eps

        self.chi = _comoving_distance_value(self.cosmo, self.z)
        self.chi = np.maximum(self.chi, eps)

        raw_nz = np.asarray(self.n_of_z(self.z), dtype=float)
        raw_nz = np.where(np.isfinite(raw_nz), raw_nz, 0.0)
        raw_nz = np.clip(raw_nz, 0.0, None)

        norm = np.trapz(raw_nz, self.z)
        if norm <= 0:
            raise ValueError("The analytical n_of_z must be positive on the supplied z grid.")

        self.n_total = raw_nz / norm

    def make_peak_bin_edges(
        self,
        z_peak: float,
        offsets: Sequence[float] = (0.2, 0.1, 0.0),
        z_min: float = 0.0,
    ) -> np.ndarray:
        """
        Reproduce the notebook's target-dependent binning.

        For the default offsets, the source bins are:
            [z_min, z_peak - 0.2],
            [z_peak - 0.2, z_peak - 0.1],
            [z_peak - 0.1, z_peak],
            [z_peak, z_max_source].

        The last BNT kernel is then the localized/nulling kernel associated
        with the requested z_peak.
        """
        z_peak = float(z_peak)
        raw_edges = [z_min] + [z_peak - off for off in offsets] + [self.z_max_source]
        edges = np.asarray(raw_edges, dtype=float)

        # Keep edges inside the sampled redshift range.
        edges[0] = max(edges[0], self.z[0])
        edges[-1] = min(edges[-1], self.z[-1])

        if np.any(np.diff(edges) <= 0):
            raise ValueError(
                "Invalid target bin edges. This usually means z_peak is too close "
                "to the low-redshift edge for the requested offsets."
            )

        return edges


    def foreground_equal_count_edges(
        self,
        z_peak: float,
        n_foreground_bins: int = 3,
        z_min: float = 0.0,
    ) -> np.ndarray:
        """
        Build target-dependent bin edges with equal source counts in foreground bins.

        The interval [z_min, z_peak] is split into ``n_foreground_bins`` bins
        containing equal fractions of the analytical source distribution n(z).
        A final high-redshift source bin [z_peak, z_max_source] is appended.

        This is useful for BNT nulling because the BNT rows combine neighboring
        bins with positive and negative coefficients; very low-density foreground
        bins can therefore lead to a large shape-noise amplification. Equal-count
        foreground bins make the input tomographic maps more balanced.
        """
        z_peak = float(z_peak)
        z_min = max(float(z_min), float(self.z[0]))
        z_max = min(float(self.z_max_source), float(self.z[-1]))

        if n_foreground_bins < 1:
            raise ValueError("n_foreground_bins must be >= 1.")
        if not (z_min < z_peak < z_max):
            raise ValueError(
                "z_peak must satisfy z_min < z_peak < z_max_source for "
                "equal-count foreground binning."
            )

        fg_mask = (self.z >= z_min) & (self.z <= z_peak)
        z_fg = self.z[fg_mask]
        n_fg = self.n_total[fg_mask]

        if len(z_fg) < n_foreground_bins + 1:
            raise ValueError(
                "The z grid has too few samples below z_peak to build the "
                "requested number of foreground bins."
            )

        cumulative = np.zeros_like(z_fg)
        dz = np.diff(z_fg)
        cumulative[1:] = np.cumsum(0.5 * (n_fg[:-1] + n_fg[1:]) * dz)

        total = cumulative[-1]
        if total <= 0:
            raise ValueError("The analytical n(z) integrates to zero below z_peak.")

        cumulative /= total

        quantiles = np.linspace(0.0, 1.0, n_foreground_bins + 1)
        fg_edges = np.interp(quantiles, cumulative, z_fg)
        fg_edges[0] = z_min
        fg_edges[-1] = z_peak

        edges = np.concatenate([fg_edges, [z_max]])
        edges = np.asarray(edges, dtype=float)

        if np.any(np.diff(edges) <= 0):
            raise ValueError(
                "Equal-count foreground binning produced non-increasing edges. "
                "Try fewer foreground bins or a larger z_peak."
            )

        return edges

    def noise_optimal_bin_edges(
        self,
        z_peak: float,
        n_foreground_bins: int = 3,
        z_min: float = 0.0,
        min_bin_fraction: float = 0.0,
    ) -> np.ndarray:
        """
        Find foreground bin edges that minimise shape-noise amplification α.

        Uses scipy.optimize.minimize (L-BFGS-B) to search over the internal
        foreground bin boundaries within [z_min, z_peak].  The free parameters
        are expressed as quantiles of the foreground n(z) CDF so the search
        space is always bounded in (0, 1).  The equal-count solution is used
        as the starting point; if the optimiser finds nothing better the
        equal-count edges are returned unchanged.

        For n_foreground_bins=1 there is nothing to optimise and the result is
        identical to foreground_equal_count_edges with n_fg=1.

        Parameters
        ----------
        min_bin_fraction : float
            If > 0, constrain the optimiser so every bin holds at least this
            fraction of the *total* source population.  This avoids the
            optimiser placing a near-empty foreground bin (which, in a mock,
            translates to too few galaxies per pixel to build a usable map).
            A ValueError is raised if no feasible binning exists for the
            requested z_peak.  See bntcluster.bin_diagnostics for converting
            a "minimum galaxies per pixel" floor into a fraction.
        """
        try:
            from scipy.optimize import minimize as _scipy_min
        except ImportError:
            raise ImportError(
                "scipy is required for noise_optimal_bin_edges. "
                "Install it with: pip install scipy"
            )

        z_min_actual = max(float(z_min), float(self.z[0]))
        z_peak = float(z_peak)
        z_max = min(float(self.z_max_source), float(self.z[-1]))

        if n_foreground_bins < 1:
            raise ValueError("n_foreground_bins must be >= 1.")
        if not (z_min_actual < z_peak < z_max):
            raise ValueError("z_peak must satisfy z_min < z_peak < z_max_source.")

        # Single foreground bin: no internal edges to place
        if n_foreground_bins == 1:
            return np.array([z_min_actual, z_peak, z_max])

        # Build normalised CDF of n(z) over [z_min_actual, z_peak]
        fg_mask = (self.z >= z_min_actual) & (self.z <= z_peak)
        z_fg = self.z[fg_mask]
        n_fg = self.n_total[fg_mask]

        if len(z_fg) < n_foreground_bins + 1:
            raise ValueError(
                "Too few z-grid samples below z_peak to build the "
                "requested number of foreground bins."
            )

        cum = np.zeros_like(z_fg)
        cum[1:] = np.cumsum(0.5 * (n_fg[:-1] + n_fg[1:]) * np.diff(z_fg))
        total_fg = cum[-1]
        if total_fg <= 0:
            raise ValueError("n(z) integrates to zero below z_peak.")
        cum /= total_fg

        n_internal = n_foreground_bins - 1

        def _edges_from_quantiles(q_raw):
            q = np.sort(np.clip(q_raw, 1e-4, 1.0 - 1e-4))
            z_int = np.interp(q, cum, z_fg)
            return np.concatenate([[z_min_actual], z_int, [z_peak, z_max]])

        def _alpha(edges):
            try:
                n_i, fracs = self.bin_distributions(edges)
                if min_bin_fraction > 0.0 and np.min(fracs) < min_bin_fraction * (1.0 - 1e-9):
                    return 1e10
                M = self.compute_bnt_matrix(n_i)
                row = M[n_foreground_bins]
                return float(np.sqrt(np.sum(row ** 2 / fracs)))
            except Exception:
                return 1e10

        # Equal-count quantiles as baseline
        q0 = np.linspace(0.0, 1.0, n_foreground_bins + 1)[1:-1]
        alpha_eq = _alpha(_edges_from_quantiles(q0))

        if n_internal == 1:
            # ── 1-D case (n_foreground_bins = 2) ─────────────────────────────
            # The objective α(q₁) is strictly unimodal on (0, 1) for all
            # realistic n(z).  Brent's bounded minimisation therefore finds the
            # global minimum without a grid search.
            from scipy.optimize import minimize_scalar as _min_scalar
            q_bounds = (0.02, 0.98)
            if min_bin_fraction > 0.0:
                # Each foreground bin must hold >= min_bin_fraction of the *total*
                # population.  The foreground holds total_fg, so the internal split
                # quantile q is restricted to [r, 1 - r] with r = f_min / total_fg.
                ratio = min_bin_fraction / total_fg
                if 2.0 * ratio >= 1.0:
                    raise ValueError(
                        f"Cannot place 2 foreground bins each holding "
                        f">= {min_bin_fraction:.4g} of the population below "
                        f"z_peak={z_peak:.3f}: the foreground only holds "
                        f"{total_fg:.4g}. Lower the floor, raise the source "
                        f"density, or increase z_peak."
                    )
                q_bounds = (max(0.02, ratio), min(0.98, 1.0 - ratio))
            res1d = _min_scalar(
                lambda q: _alpha(_edges_from_quantiles(np.array([q]))),
                bounds=q_bounds,
                method="bounded",
                options={"xatol": 1e-6},
            )
            opt_edges = _edges_from_quantiles(np.array([res1d.x]))

        else:
            # ── Multi-dimensional case (n_foreground_bins ≥ 3) ───────────────
            # The landscape may be non-unimodal in higher dimensions, so use a
            # coarse grid search to seed L-BFGS-B.
            n_grid = 10
            q_grid = np.linspace(0.05, 0.95, n_grid)
            best_q, best_a = q0.copy(), alpha_eq

            if n_internal == 2:
                for qi in q_grid:
                    for qj in q_grid:
                        if qj <= qi:
                            continue
                        a = _alpha(_edges_from_quantiles(np.array([qi, qj])))
                        if a < best_a:
                            best_a, best_q = a, np.array([qi, qj])
            # For n_internal ≥ 3 the grid is too expensive; keep equal-count start.

            res = _scipy_min(
                lambda q: _alpha(_edges_from_quantiles(q)),
                best_q,
                method="L-BFGS-B",
                bounds=[(0.02, 0.98)] * n_internal,
                options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
            )
            opt_edges = _edges_from_quantiles(res.x)

        # Return the better of the optimised result and the equal-count baseline
        chosen = opt_edges if _alpha(opt_edges) < alpha_eq else _edges_from_quantiles(q0)

        if min_bin_fraction > 0.0:
            _, chosen_fracs = self.bin_distributions(chosen)
            if np.min(chosen_fracs) < min_bin_fraction * (1.0 - 1e-9):
                raise ValueError(
                    f"No feasible {n_foreground_bins}-foreground binning at "
                    f"z_peak={z_peak:.3f} keeps every bin above "
                    f"{min_bin_fraction:.4g} of the population "
                    f"(best achievable min fraction = {np.min(chosen_fracs):.4g}). "
                    f"Lower the floor, raise the source density, or change z_peak."
                )
        return chosen

    def bin_distributions(self, bin_edges: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Build normalized n_i(z) distributions from analytical n(z) and bin edges.

        Returns
        -------
        n_i : ndarray, shape (n_bins, n_z)
            Per-bin redshift distributions. Each non-empty bin is normalized to
            integral 1.
        bin_fractions : ndarray, shape (n_bins,)
            Fraction of the full source population lying in each bin. This is
            useful for shape-noise diagnostics, because narrower bins contain
            fewer galaxies.
        """
        edges = np.asarray(bin_edges, dtype=float)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges must be strictly increasing.")

        n_bins = len(edges) - 1
        n_i = np.zeros((n_bins, len(self.z)))
        fractions = np.zeros(n_bins)

        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                mask = (self.z >= lo) & (self.z <= hi)
            else:
                mask = (self.z >= lo) & (self.z < hi)

            nz_bin = np.where(mask, self.n_total, 0.0)
            frac = np.trapz(nz_bin, self.z)
            fractions[i] = frac

            if frac > 0:
                n_i[i] = nz_bin / frac

        if np.any(fractions <= 0):
            empty = np.where(fractions <= 0)[0]
            raise ValueError(f"Empty source bin(s): {empty}. Adjust bin edges or z grid.")

        return n_i, fractions

    def photoz_smeared_bin_distributions(
        self,
        bin_edges: ArrayLike,
        sigma_factor: float = 0.05,
        outlier_fraction: float = 0.1,
        z_out_max: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute true-z distributions for photometric bins with photo-z errors.

        For each photometric bin [lo_k, hi_k], the probability that a galaxy at
        true redshift z_true scatters into that bin is modelled as a Gaussian
        core plus a uniform catastrophic outlier floor:

            P(z_phot ∈ [lo, hi] | z_true) =
                (1 - f_out) · [Φ((hi - z)/σ(z)) - Φ((lo - z)/σ(z))]
                + f_out · (hi - lo) / Δz_range

        where σ(z) = sigma_factor · (1 + z) and Δz_range = z_out_max - z[0].
        The resulting per-bin n_i(z) ∝ n_total(z) · P(z_phot ∈ bin | z)
        represents the actual true-z distribution of galaxies that land in
        that photometric bin.  Returns the same (n_i, fractions) structure as
        bin_distributions, so the result feeds directly into
        compute_lensing_kernels.
        """
        try:
            from scipy.special import ndtr as _ndtr
        except ImportError:
            raise ImportError("scipy is required for photoz_smeared_bin_distributions.")

        edges = np.asarray(bin_edges, dtype=float)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges must be strictly increasing.")

        n_bins = len(edges) - 1
        if z_out_max is None:
            z_out_max = self.z_max_source
        z_range = float(z_out_max - self.z[0])

        sigma = sigma_factor * (1.0 + self.z)

        n_i = np.zeros((n_bins, len(self.z)))
        fractions = np.zeros(n_bins)

        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            gauss = _ndtr((hi - self.z) / sigma) - _ndtr((lo - self.z) / sigma)
            uniform = (hi - lo) / z_range
            p_sel = (1.0 - outlier_fraction) * gauss + outlier_fraction * uniform

            nz_w = self.n_total * p_sel
            frac = np.trapz(nz_w, self.z)
            fractions[i] = frac
            if frac > 0:
                n_i[i] = nz_w / frac

        if np.any(fractions <= 0):
            empty = np.where(fractions <= 0)[0]
            raise ValueError(
                f"Empty source bin(s) after photo-z smearing: {empty}. "
                "Adjust bin edges or z grid."
            )

        return n_i, fractions

    def compute_for_bins_photoz(
        self,
        bin_edges: ArrayLike,
        sigma_factor: float = 0.05,
        outlier_fraction: float = 0.1,
        z_peak: Optional[float] = None,
        selected_kernel_index: int = -1,
    ) -> BNTResult:
        """
        Compute BNT kernels using photo-z smeared bin distributions.

        Equivalent to compute_for_bins but replaces bin_distributions with
        photoz_smeared_bin_distributions, modelling the leakage of galaxies
        between photometric bins due to redshift measurement errors.
        """
        n_i, fractions = self.photoz_smeared_bin_distributions(
            bin_edges,
            sigma_factor=sigma_factor,
            outlier_fraction=outlier_fraction,
        )
        kernels = self.compute_lensing_kernels(n_i)
        M = self.compute_bnt_matrix(n_i)
        bnt_kernels = self.apply_bnt(kernels, M)

        if selected_kernel_index < 0:
            selected_kernel_index = bnt_kernels.shape[0] + selected_kernel_index

        if z_peak is None:
            z_peak = float(self.z[np.argmax(bnt_kernels[selected_kernel_index])])

        return BNTResult(
            z_peak=float(z_peak),
            bin_edges=np.asarray(bin_edges, dtype=float),
            z=self.z.copy(),
            chi=self.chi.copy(),
            n_i=n_i,
            bin_fractions=fractions,
            bnt_matrix=M,
            kernels=kernels,
            bnt_kernels=bnt_kernels,
            selected_kernel_index=int(selected_kernel_index),
        )

    def compute_lensing_kernels(self, n_i: np.ndarray) -> np.ndarray:
        """
        Compute ordinary lensing kernels for all source bins.

        For each source-bin distribution n_i(z_s), the kernel is

            W_i(z_l) = chi_l ∫_{z_l}^∞ dz_s n_i(z_s)
                       (chi_s - chi_l) / chi_s.

        The computation is vectorized using cumulative integrals from high to
        low redshift.
        """
        n_i = np.asarray(n_i, dtype=float)
        if n_i.ndim != 2 or n_i.shape[1] != len(self.z):
            raise ValueError("n_i must have shape (n_bins, len(z)).")

        kernels = np.zeros_like(n_i)

        for i, nz in enumerate(n_i):
            cumulative_n = self._reverse_cumulative_trapz(nz)
            cumulative_n_over_chi = self._reverse_cumulative_trapz(nz / self.chi)

            kernels[i] = self.chi * (cumulative_n - self.chi * cumulative_n_over_chi)
            kernels[i] = np.where(kernels[i] > 0, kernels[i], 0.0)

        return kernels

    def compute_bnt_matrix(self, n_i: np.ndarray) -> np.ndarray:
        """
        Compute the BNT mixing matrix.

        This follows the construction used in the notebook:
        - row 0 is unchanged;
        - row 1 is kernel_1 - kernel_0;
        - rows i >= 2 use bins i, i-1, i-2 and solve two moment-cancellation
          equations involving ∫n_i dz and ∫n_i/chi dz.
        """
        n_i = np.asarray(n_i, dtype=float)
        n_bins = n_i.shape[0]

        A = np.array([np.trapz(nz, self.z) for nz in n_i])
        B = np.array([np.trapz(nz / self.chi, self.z) for nz in n_i])

        M = np.eye(n_bins)

        if n_bins >= 2:
            M[1, 0] = -1.0

        for i in range(2, n_bins):
            mat = np.array(
                [
                    [A[i - 1], A[i - 2]],
                    [B[i - 1], B[i - 2]],
                ]
            )
            rhs = -np.array([A[i], B[i]])
            weights = np.linalg.solve(mat, rhs)

            M[i, i - 1] = weights[0]
            M[i, i - 2] = weights[1]

        return M

    def apply_bnt(self, kernels: np.ndarray, bnt_matrix: np.ndarray) -> np.ndarray:
        """Apply the BNT matrix to the ordinary tomographic kernels."""
        return bnt_matrix @ kernels

    def compute_for_bins(
        self,
        bin_edges: ArrayLike,
        z_peak: Optional[float] = None,
        selected_kernel_index: int = -1,
    ) -> BNTResult:
        """
        Compute ordinary and BNT-transformed kernels for arbitrary bin edges.
        """
        n_i, fractions = self.bin_distributions(bin_edges)
        kernels = self.compute_lensing_kernels(n_i)
        M = self.compute_bnt_matrix(n_i)
        bnt_kernels = self.apply_bnt(kernels, M)

        if selected_kernel_index < 0:
            selected_kernel_index = bnt_kernels.shape[0] + selected_kernel_index

        if z_peak is None:
            z_peak = float(self.z[np.argmax(bnt_kernels[selected_kernel_index])])

        return BNTResult(
            z_peak=float(z_peak),
            bin_edges=np.asarray(bin_edges, dtype=float),
            z=self.z.copy(),
            chi=self.chi.copy(),
            n_i=n_i,
            bin_fractions=fractions,
            bnt_matrix=M,
            kernels=kernels,
            bnt_kernels=bnt_kernels,
            selected_kernel_index=int(selected_kernel_index),
        )

    def compute_for_peak(
        self,
        z_peak: float,
        offsets: Sequence[float] = (0.2, 0.1, 0.0),
        z_min: float = 0.0,
        selected_kernel_index: int = -1,
        binning: str = "fixed_offsets",
        n_foreground_bins: int = 3,
        min_bin_fraction: float = 0.0,
    ) -> BNTResult:
        """
        Compute a localized BNT kernel for a target peak redshift.

        Parameters
        ----------
        z_peak : float
            Target redshift around which the selected BNT kernel should be localized.
        offsets : sequence of float
            Offsets used by the original fixed-offset binning. With the default
            values, the edges are [z_min, z_peak-0.2, z_peak-0.1, z_peak, z_max].
            Used only when ``binning='fixed_offsets'``.
        z_min : float
            Lower edge of the first source bin.
        selected_kernel_index : int
            BNT row/kernel to select. The default, -1, selects the last row.
        binning : {'fixed_offsets', 'equal_count_foreground', 'noise_optimal'}
            Binning strategy. ``fixed_offsets`` reproduces the original notebook.
            ``equal_count_foreground`` splits [z_min, z_peak] into foreground bins
            containing equal numbers of galaxies according to the analytical n(z),
            then appends the final high-redshift bin [z_peak, z_max_source].
            ``noise_optimal`` numerically minimises the shape-noise amplification α
            over the internal foreground bin boundaries (requires scipy).
        n_foreground_bins : int
            Number of foreground bins used for ``equal_count_foreground`` and
            ``noise_optimal`` binning strategies.
        """
        if binning == "fixed_offsets":
            edges = self.make_peak_bin_edges(z_peak=z_peak, offsets=offsets, z_min=z_min)
        elif binning == "equal_count_foreground":
            edges = self.foreground_equal_count_edges(
                z_peak=z_peak,
                n_foreground_bins=n_foreground_bins,
                z_min=z_min,
            )
        elif binning == "noise_optimal":
            edges = self.noise_optimal_bin_edges(
                z_peak=z_peak,
                n_foreground_bins=n_foreground_bins,
                z_min=z_min,
                min_bin_fraction=min_bin_fraction,
            )
        else:
            raise ValueError(
                "binning must be 'fixed_offsets', 'equal_count_foreground', "
                "or 'noise_optimal'."
            )

        # For binnings that do not optimise against it, validate the floor here.
        if min_bin_fraction > 0.0 and binning != "noise_optimal":
            _, _fracs = self.bin_distributions(edges)
            if np.min(_fracs) < min_bin_fraction * (1.0 - 1e-9):
                raise ValueError(
                    f"Binning '{binning}' at z_peak={z_peak:.3f} leaves a bin below "
                    f"the floor (min fraction {np.min(_fracs):.4g} < {min_bin_fraction:.4g}). "
                    f"Use binning='noise_optimal', lower the floor, or change z_peak."
                )

        return self.compute_for_bins(
            bin_edges=edges,
            z_peak=z_peak,
            selected_kernel_index=selected_kernel_index,
        )

    def scan_peaks(
        self,
        z_peaks: Iterable[float],
        offsets: Sequence[float] = (0.2, 0.1, 0.0),
        z_min: float = 0.0,
        selected_kernel_index: int = -1,
        binning: str = "fixed_offsets",
        n_foreground_bins: int = 3,
        exact: bool = False,
        min_bin_fraction: float = 0.0,
    ) -> list[BNTResult]:
        """
        Compute localized BNT kernels for several target redshifts.

        Parameters
        ----------
        exact : bool
            If True, adjust the bin boundary for each target so the kernel
            peak falls exactly on z_peak (requires scipy).  If False (default),
            z_peak is used directly as the bin boundary, which places the peak
            close to but not exactly at z_peak.
        min_bin_fraction : float
            Minimum fraction of the total population per bin (only with
            exact=False; see noise_optimal_bin_edges).
        """
        if exact and min_bin_fraction > 0.0:
            raise ValueError("min_bin_fraction is only supported with exact=False.")

        method = self.compute_for_peak_exact if exact else self.compute_for_peak
        kwargs = dict(
            offsets=offsets,
            z_min=z_min,
            selected_kernel_index=selected_kernel_index,
            binning=binning,
            n_foreground_bins=n_foreground_bins,
        )
        if not exact:
            kwargs["min_bin_fraction"] = min_bin_fraction
        return [method(z_peak=zp, **kwargs) for zp in z_peaks]

    def compute_for_peak_exact(
        self,
        z_peak: float,
        binning: str = "equal_count_foreground",
        n_foreground_bins: int = 3,
        z_min: float = 0.0,
        selected_kernel_index: int = -1,
        offsets: Sequence[float] = (0.2, 0.1, 0.0),
        tol: float = 5e-4,
    ) -> BNTResult:
        """
        Compute a BNT kernel whose peak is exactly at z_peak (to within tol).

        The bin boundary is treated as a free parameter and solved via Brent's
        root-finding method.  The returned BNTResult stores z_peak = z_peak
        (the desired target); bin_edges reflects the actual boundary used.

        Parameters
        ----------
        tol : float
            Tolerance on the kernel peak position in redshift units.
        """
        try:
            from scipy.optimize import brentq
        except ImportError:
            raise ImportError(
                "scipy is required for compute_for_peak_exact. "
                "Install it with: pip install scipy"
            )

        z_target = float(z_peak)

        def _peak_residual(z_boundary):
            r = self.compute_for_peak(
                z_peak=z_boundary,
                binning=binning,
                n_foreground_bins=n_foreground_bins,
                z_min=z_min,
                selected_kernel_index=selected_kernel_index,
                offsets=offsets,
            )
            return r.z_at_kernel_max - z_target

        # Evaluate at the naive boundary (z_boundary = z_target)
        f0 = _peak_residual(z_target)
        if abs(f0) < tol:
            result = self.compute_for_peak(
                z_target, binning=binning, n_foreground_bins=n_foreground_bins,
                z_min=z_min, selected_kernel_index=selected_kernel_index,
                offsets=offsets,
            )
            result.z_peak = z_target
            return result

        # The kernel typically peaks just below the bin boundary (f0 < 0).
        # Search for the other bracket by moving the boundary in the opposite
        # direction to f0 until the residual changes sign.
        search_deltas = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4]
        z_lo, z_hi = z_target, z_target
        found = False
        for delta in search_deltas:
            z_try = z_target + (delta if f0 < 0 else -delta)
            z_try = float(np.clip(z_try, self.z[0] + 0.001, self.z_max_source - 0.001))
            try:
                f_try = _peak_residual(z_try)
                if f0 * f_try <= 0:
                    z_lo = min(z_target, z_try)
                    z_hi = max(z_target, z_try)
                    found = True
                    break
            except ValueError:
                continue

        if not found:
            # Bracket not found; return closest available result
            result = self.compute_for_peak(
                z_target, binning=binning, n_foreground_bins=n_foreground_bins,
                z_min=z_min, selected_kernel_index=selected_kernel_index,
                offsets=offsets,
            )
            result.z_peak = z_target
            return result

        z_boundary_opt = brentq(_peak_residual, z_lo, z_hi, xtol=tol)
        result = self.compute_for_peak(
            z_boundary_opt, binning=binning, n_foreground_bins=n_foreground_bins,
            z_min=z_min, selected_kernel_index=selected_kernel_index, offsets=offsets,
        )
        result.z_peak = z_target
        return result

    def plot_result(
        self,
        result: BNTResult,
        ax: Optional[plt.Axes] = None,
        normalize: bool = False,
        show_ordinary: bool = True,
        multiply_by_h: bool = False,
        **plot_kwargs,
    ) -> plt.Axes:
        """
        Plot the selected BNT kernel and, optionally, the ordinary source-bin kernel.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))

        factor = getattr(self.cosmo, "h", 1.0) if multiply_by_h else 1.0

        y_bnt = result.selected_kernel * factor
        y_ord = result.ordinary_selected_kernel * factor

        if normalize:
            norm = np.nanmax(np.abs(y_bnt))
            if norm > 0:
                y_bnt = y_bnt / norm
                y_ord = y_ord / norm

        kwargs = {"lw": 2.5, "label": f"BNT kernel, target z={result.z_peak:.2f}"}
        kwargs.update(plot_kwargs)

        ax.plot(result.z, y_bnt, **kwargs)

        if show_ordinary:
            ax.plot(
                result.z,
                y_ord,
                ls="--",
                lw=1.8,
                label="Ordinary kernel, same source bin",
            )

        ax.axvline(result.z_peak, ls=":", lw=1.8, color="k", label="target redshift")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\tilde W(z)$" if not normalize else r"normalized $\tilde W(z)$")
        ax.grid(alpha=0.3)
        ax.legend()
        return ax

    def plot_scan(
        self,
        results: Sequence[BNTResult],
        ax: Optional[plt.Axes] = None,
        normalize: bool = False,
        show_targets: bool = True,
        multiply_by_h: bool = False,
    ) -> plt.Axes:
        """
        Plot the selected BNT kernels from a target-redshift scan.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        factor = getattr(self.cosmo, "h", 1.0) if multiply_by_h else 1.0

        for result in results:
            y = result.selected_kernel * factor
            if normalize:
                norm = np.nanmax(np.abs(y))
                if norm > 0:
                    y = y / norm
            ax.plot(result.z, y, lw=2, label=fr"$z_{{\rm target}}={result.z_peak:.2f}$")

            if show_targets:
                ax.axvline(result.z_peak, ls=":", lw=1, alpha=0.5)

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\tilde W(z)$" if not normalize else r"normalized $\tilde W(z)$")
        ax.grid(alpha=0.3)
        ax.legend(title="Target")
        return ax

    def plot_shape_noise_scan(
        self,
        results: Sequence[BNTResult],
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> plt.Axes:
        """
        Plot shape-noise amplification factor vs target redshift.

        Parameters
        ----------
        results : sequence of BNTResult
            Output of scan_peaks(). Each result contributes one (z_peak, α) point.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))

        z_peaks = [r.z_peak for r in results]
        alphas = [r.shape_noise_amplification for r in results]

        kwargs = {"lw": 2.5, "marker": "o", "ms": 5}
        if label:
            kwargs["label"] = label
        kwargs.update(plot_kwargs)

        ax.plot(z_peaks, alphas, **kwargs)
        ax.set_xlabel(r"target $z_{\rm peak}$")
        ax.set_ylabel(r"shape-noise amplification $\alpha$")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        if label:
            ax.legend()
        return ax

    def plot_scan_with_noise_impact(
        self,
        results: Sequence[BNTResult],
        ax: Optional[plt.Axes] = None,
        multiply_by_h: bool = False,
        show_targets: bool = True,
        cmap: str = "viridis",
    ) -> plt.Axes:
        """
        Plot BNT kernels from a scan with fill opacity encoding shape-noise amplification.

        Each kernel is normalized to its own peak amplitude so that the noise
        impact is visible rather than buried in absolute-amplitude differences.
        The filled area under each curve has opacity proportional to 1/log(α),
        so noise-degraded kernels appear washed out while low-noise kernels are opaque.
        Legend entries include z_peak and the amplification factor α.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        factor = getattr(self.cosmo, "h", 1.0) if multiply_by_h else 1.0
        n = len(results)
        colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.9, n))

        alphas = [r.shape_noise_amplification for r in results]
        log_a = np.log(np.array(alphas))
        log_a_min, log_a_max = log_a.min(), log_a.max()
        dlog = log_a_max - log_a_min if log_a_max > log_a_min else 1.0

        for result, color, amp, la in zip(results, colors, alphas, log_a):
            y = result.selected_kernel * factor
            peak = np.nanmax(np.abs(y))
            if peak > 0:
                y = y / peak

            # Opacity: best SNR (lowest α) → 0.70, worst → 0.12
            fill_opacity = 0.70 - 0.58 * (la - log_a_min) / dlog

            label = fr"$z_{{\rm t}}={result.z_peak:.1f}$,  $\alpha={amp:.1f}$"
            ax.fill_between(result.z, 0, y, color=color, alpha=fill_opacity, lw=0)
            ax.plot(result.z, y, color=color, lw=2, label=label)

            if show_targets:
                ax.axvline(result.z_peak, ls=":", lw=1, color=color, alpha=0.5)

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"normalized $\tilde W(z)$")
        ax.grid(alpha=0.3)
        ax.legend(title=r"$z_{\rm target}$,  noise ampl. $\alpha$")
        return ax

    def _reverse_cumulative_trapz(self, y: np.ndarray) -> np.ndarray:
        """
        Return I[j] = ∫_{z[j]}^{z[-1]} y(z) dz using trapezoidal integration.
        """
        y = np.asarray(y, dtype=float)
        dz = np.diff(self.z)
        segment_integrals = 0.5 * (y[:-1] + y[1:]) * dz
        out = np.zeros_like(y)
        out[:-1] = np.cumsum(segment_integrals[::-1])[::-1]
        return out


# -----------------------------------------------------------------------------
# Example analysis script converted from the notebook
# -----------------------------------------------------------------------------

def main():
    """Run the same demonstration as the notebook, but from a plain script."""
    print("BNTNuller version:", __version__)

    # Euclid-like source redshift distribution (single canonical n(z))
    from .nfw_mock_generator import nz_euclid
    z = np.linspace(0.001, 3.0, 1000)
    n_of_z = nz_euclid
    nuller = BNTNuller(z=z, n_of_z=n_of_z)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(nuller.z, nuller.n_total)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$n(z)$")
    ax.set_title("Analytical source redshift distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("./figs/bnt_source_redshift_distribution.png", dpi=200)

    # Equal-count foreground binning
    z_peak = 0.6
    result = nuller.compute_for_peak(
        z_peak=z_peak,
        binning="equal_count_foreground",
        n_foreground_bins=3,
    )

    print("\nEqual-count foreground binning")
    print("Bin edges:", result.bin_edges)
    print("Bin fractions:", result.bin_fractions)
    print("Foreground bin fractions:", result.bin_fractions[:-1])
    print(f"z at kernel max (naive):  {result.z_at_kernel_max:.5f}  "
          f"(offset {result.peak_offset:+.5f})")

    result_exact = nuller.compute_for_peak_exact(
        z_peak=z_peak,
        binning="equal_count_foreground",
        n_foreground_bins=3,
    )
    print(f"z at kernel max (exact):  {result_exact.z_at_kernel_max:.5f}  "
          f"(offset {result_exact.peak_offset:+.5f})")
    print(f"bin boundary used:        {result_exact.bin_edges[-2]:.5f}  "
          f"(vs target {z_peak})")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    nuller.plot_result(result, ax=ax, multiply_by_h=True, normalize=False)
    fig.tight_layout()
    fig.savefig("./figs/bnt_equal_count_kernel.png", dpi=200)

    # Compare with original fixed-offset binning
    result_fixed = nuller.compute_for_peak(
        z_peak=z_peak,
        binning="fixed_offsets",
        offsets=(0.2, 0.1, 0.0),
    )

    print("\nFixed-offset binning")
    print("Fixed-offset bin edges:", result_fixed.bin_edges)
    print("Fixed-offset bin fractions:", result_fixed.bin_fractions)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.z, result.selected_kernel, lw=2.5, label="Equal-count foreground")
    ax.plot(result_fixed.z, result_fixed.selected_kernel, lw=2.5, ls="--", label="Fixed offsets")
    ax.axvline(z_peak, color="k", ls=":", lw=1.5, label="target")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\tilde W(z)$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("./figs/bnt_equal_count_vs_fixed_offsets.png", dpi=200)

    # Scan over target redshifts – exact peak placement, uniform grid
    z_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    results = nuller.scan_peaks(
        z_values,
        binning="equal_count_foreground",
        n_foreground_bins=3,
        exact=True,
    )

    print("\nNoise amplification by target redshift:")
    for r in results:
        print(f"  z_peak={r.z_peak:.1f}  alpha={r.shape_noise_amplification:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    nuller.plot_scan_with_noise_impact(results, ax=ax, multiply_by_h=True)
    ax.set_title("BNT kernels – fill opacity encodes noise (transparent = noisy)")
    fig.tight_layout()
    fig.savefig("./figs/bnt_scan_equal_count_foreground.png", dpi=200)

    # Shape-noise amplification scan from z=0.05 to z=1.0
    z_scan = np.linspace(0.05, 1.0, 60)
    noise_results = []
    for zp in z_scan:
        try:
            r = nuller.compute_for_peak(
                z_peak=zp,
                binning="equal_count_foreground",
                n_foreground_bins=3,
            )
            noise_results.append(r)
        except ValueError:
            pass  # z_peak too low for the current n(z) / binning

    if noise_results:
        print(f"\nShape-noise scan: {len(noise_results)} feasible z_peaks "
              f"({noise_results[0].z_peak:.3f} – {noise_results[-1].z_peak:.3f})")
        for r in noise_results[:3]:
            print(f"  z_peak={r.z_peak:.2f}  alpha={r.shape_noise_amplification:.2f}")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        nuller.plot_shape_noise_scan(noise_results, ax=ax,
                                     label="equal-count foreground, 3 bins")
        ax.set_title("Shape-noise amplification of nulled convergence map")
        fig.tight_layout()
        fig.savefig("./figs/bnt_shape_noise_amplification.png", dpi=200)

    # ── 4-way noise comparison ───────────────────────────────────────────────
    # Each configuration is (label, binning, n_foreground_bins, linestyle, color)
    configs = [
        ("Original (equal-count, $n_{\\rm fg}=3$)",      "equal_count_foreground", 3, "-",  "C0"),
        ("Method 1 (equal-count, $n_{\\rm fg}=2$)",      "equal_count_foreground", 2, "--", "C1"),
        ("Method 2 (noise-optimal, $n_{\\rm fg}=3$)",    "noise_optimal",          3, "-",  "C2"),
        ("Method 1+2 (noise-optimal, $n_{\\rm fg}=2$)",  "noise_optimal",          2, "--", "C3"),
    ]

    z_scan_cmp = np.linspace(0.15, 1.0, 30)
    z_showcase = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    print("\n4-way comparison at z_peak=0.6:")
    for label, binning, n_fg, ls, color in configs:
        # α(z_peak) curve
        scan_results_cmp = []
        for zp in z_scan_cmp:
            try:
                r = nuller.compute_for_peak(zp, binning=binning, n_foreground_bins=n_fg)
                scan_results_cmp.append(r)
            except (ValueError, ImportError):
                pass
        if scan_results_cmp:
            axes[0].plot(
                [r.z_peak for r in scan_results_cmp],
                [r.shape_noise_amplification for r in scan_results_cmp],
                lw=2, ls=ls, color=color, label=label,
            )

        # Kernel shape at z_showcase — exact peak placement
        try:
            r_show = nuller.compute_for_peak_exact(
                z_showcase, binning=binning, n_foreground_bins=n_fg
            )
            y = r_show.selected_kernel
            y = y / np.nanmax(y)
            alpha_val = r_show.shape_noise_amplification
            print(f"  {label.replace('$', '').replace('{', '').replace('}', '')}: "
                  f"α={alpha_val:.2f}  bin_edges={np.round(r_show.bin_edges, 3)}")
            axes[1].plot(
                r_show.z, y, lw=2, ls=ls, color=color,
                label=fr"{label},  $\alpha={alpha_val:.1f}$",
            )
        except (ValueError, ImportError):
            pass

    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"target $z_{\rm peak}$")
    axes[0].set_ylabel(r"noise amplification $\alpha$")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=9)
    axes[0].set_title(r"Shape-noise amplification $\alpha(z_{\rm peak})$")

    axes[1].axvline(z_showcase, ls=":", lw=1.5, color="k",
                    label=f"target $z={z_showcase}$")
    axes[1].set_xlabel(r"$z$")
    axes[1].set_ylabel(r"normalized $\tilde W(z)$")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].set_title(
        f"BNT kernels at $z_{{\\rm target}}={z_showcase}$ "
        "(normalized to peak)"
    )

    fig.tight_layout()
    fig.savefig("./figs/bnt_noise_method_comparison.png", dpi=200)

    # ── Photo-z comparison ───────────────────────────────────────────────────
    pz_sigma = 0.05
    pz_fout = 0.05
    n_fg_pz = 2

    # Figure 1: bin distributions with and without photo-z for one z_target
    z_showcase_pz = 0.5
    edges_pz_demo = nuller.foreground_equal_count_edges(
        z_peak=z_showcase_pz, n_foreground_bins=n_fg_pz
    )
    n_i_nopz, _ = nuller.bin_distributions(edges_pz_demo)
    n_i_pz_demo, _ = nuller.photoz_smeared_bin_distributions(
        edges_pz_demo, sigma_factor=pz_sigma, outlier_fraction=pz_fout
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    ls_styles = ["-", "--", ":"]
    for i in range(len(n_i_nopz)):
        color = f"C{i}"
        ls = ls_styles[i % len(ls_styles)]
        axes[0].plot(nuller.z, n_i_nopz[i], color=color, ls=ls, lw=2,
                     label=f"bin {i + 1}")
        axes[1].plot(nuller.z, n_i_pz_demo[i], color=color, ls=ls, lw=2,
                     label=f"bin {i + 1}")
    for axi in axes:
        axi.axvline(z_showcase_pz, ls=":", color="k", lw=1.5, alpha=0.6,
                    label=rf"$z_{{\rm t}}={z_showcase_pz}$")
        axi.set_xlabel(r"$z$")
        axi.set_ylabel(r"$n_i(z)$ (normalized per bin)")
        axi.set_xlim(0.0, min(2.0, nuller.z[-1]))
        axi.grid(alpha=0.3)
        axi.legend(fontsize=9)
    axes[0].set_title("Bin distributions — no photo-z errors")
    axes[1].set_title(
        rf"Bin distributions — photo-z ($\sigma_z/(1+z)={pz_sigma}$,"
        rf" $f_{{\rm out}}={pz_fout}$)"
    )
    fig.tight_layout()
    fig.savefig("./figs/bnt_photoz_bin_distributions.png", dpi=200)

    # Figure 2: BNT kernels with and without photo-z for several z_targets
    z_targets_pz = [0.3, 0.5, 0.7, 0.9]
    colors_pz = ["C0", "C1", "C2", "C3"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for zt, color in zip(z_targets_pz, colors_pz):
        edges_zt = nuller.foreground_equal_count_edges(
            z_peak=zt, n_foreground_bins=n_fg_pz
        )

        r_nopz = nuller.compute_for_bins(edges_zt, z_peak=zt)
        k_nopz = r_nopz.selected_kernel
        mx = np.nanmax(k_nopz)
        if mx > 0:
            k_nopz = k_nopz / mx
        axes[0].plot(nuller.z, k_nopz, color=color, lw=2,
                     label=rf"$z_{{\rm t}}={zt}$")
        axes[0].axvline(zt, ls=":", lw=1, color=color, alpha=0.4)

        r_pz = nuller.compute_for_bins_photoz(
            edges_zt, sigma_factor=pz_sigma, outlier_fraction=pz_fout, z_peak=zt
        )
        k_pz = r_pz.selected_kernel
        mx = np.nanmax(k_pz)
        if mx > 0:
            k_pz = k_pz / mx
        axes[1].plot(nuller.z, k_pz, color=color, lw=2,
                     label=rf"$z_{{\rm t}}={zt}$")
        axes[1].axvline(zt, ls=":", lw=1, color=color, alpha=0.4)

    for axi in axes:
        axi.set_xlabel(r"$z$")
        axi.set_ylabel(r"normalized $\tilde W(z)$")
        axi.set_xlim(0.0, min(1.8, nuller.z[-1]))
        axi.grid(alpha=0.3)
        axi.legend(fontsize=9)
    axes[0].set_title(r"BNT kernels — no photo-z errors")
    axes[1].set_title(
        rf"BNT kernels — photo-z ($\sigma_z/(1+z)={pz_sigma}$,"
        rf" $f_{{\rm out}}={pz_fout}$)"
    )
    fig.tight_layout()
    fig.savefig("./figs/bnt_photoz_comparison.png", dpi=200)

    print("\nSaved figures:")
    print("  - bnt_source_redshift_distribution.png")
    print("  - bnt_equal_count_kernel.png")
    print("  - bnt_equal_count_vs_fixed_offsets.png")
    print("  - bnt_scan_equal_count_foreground.png")
    if noise_results:
        print("  - bnt_shape_noise_amplification.png")
    print("  - bnt_noise_method_comparison.png")
    print("  - bnt_photoz_bin_distributions.png")
    print("  - bnt_photoz_comparison.png")


if __name__ == "__main__":
    main()
