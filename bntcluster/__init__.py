"""
bntcluster
==========
Unified package facade for the BNT shear-nulling toolkit.

Wraps the package submodules:
- ``bnt_nulling``        : analytical BNT framework (BNTNuller, kernels,
  binning strategies).
- ``nfw_mock_generator`` : NFW convergence-map generator + source
  sampling / photo-z helpers.
- ``bnt_framework``      : user-facing cluster-adapted BNT interface
  (SurveySetup, CostFunction, ClusterBNT) + plot helpers.

The whole toolkit is reached through a single ``import bntcluster as bnt``.

Two small self-contained utilities are kept at this level so the notebook
need only ``import bntcluster as bnt``:

- :func:`global_equal_count_edges` — N equal-count bins spanning the full
  ``[z_min, z_max_source]`` range of a ``BNTNuller``;
- :func:`configure_maps`           — return a consistent map / sampling
  configuration dict from a few survey-geometry knobs.

The single canonical Euclid-like source distribution is
``nz_euclid`` (Laureijs et al. 2011) defined in ``nfw_mock_generator``,
along with its companions ``sample_nz`` and ``apply_photoz_errors``.
"""
from __future__ import annotations

import numpy as _np

# ── Analytical BNT framework ─────────────────────────────────────────────────
from .bnt_nulling import (
    BNTNuller,
    BNTResult,
    SimpleFlatLambdaCDM,
)

# ── NFW mock-map generator + survey utilities ────────────────────────────────
from .nfw_mock_generator import (
    NFWMockGenerator,
    LensCatalog,
    build_kappa_map_nfw,
    nz_euclid,
    sample_nz,
    apply_photoz_errors,
    _sigma_crit as sigma_crit,
)

# Default cosmology shared across the toolkit: Planck 2015 results
# (Planck Collaboration XIII 2016, A&A 594, A13; TT,TE,EE+lowP+lensing+ext).
# These match astropy's ``Planck15`` preset.
DEFAULT_COSMO_PARAMS = dict(
    H0=67.74, Om0=0.3075, Ob0=0.0486, flat=True, sigma8=0.8159, ns=0.9667,
)


def global_equal_count_edges(nuller, n_bins, z_min=0.0):
    """N equal-count tomographic bins spanning ``[z_min, z_max_source]``.

    Unlike ``BNTNuller.foreground_equal_count_edges`` (which partitions only
    the foreground below a target ``z_peak``), this partitions the *full*
    redshift range.
    """
    mask = (nuller.z >= z_min) & (nuller.z <= nuller.z_max_source)
    z_g  = nuller.z[mask]
    n_g  = nuller.n_total[mask]
    cum  = _np.zeros_like(z_g)
    cum[1:] = _np.cumsum(0.5 * (n_g[:-1] + n_g[1:]) * _np.diff(z_g))
    cum /= cum[-1]
    edges = _np.interp(_np.linspace(0.0, 1.0, n_bins + 1), cum, z_g)
    edges[0]  = max(z_min, nuller.z[0])
    edges[-1] = nuller.z_max_source
    return edges


def configure_maps(
    npix: int | None = None,
    field_deg: float | None = None,
    n_sources: int | None = None,
    n_z_bins: int | None = None,
    n_gal_per_arcmin2: float | None = None,
) -> dict:
    """Return a consistent map / sampling configuration dictionary.

    All values are derived from the supplied keyword arguments; sensible
    defaults are applied for anything left unspecified.  No module-level
    state is mutated — the returned dict is the single source of truth.

    Parameters
    ----------
    npix : int, optional
        Square map side in pixels  (default ``1024``).
    field_deg : float, optional
        Square field side in degrees  (default ``10.0``).
    n_sources : int, optional
        Total number of mock source galaxies.  Overrides any value implied
        by ``n_gal_per_arcmin2``.
    n_z_bins : int, optional
        Number of redshift slices in the lensing-efficiency integral
        (default ``60``).
    n_gal_per_arcmin2 : float, optional
        Source number density; sets ``n_sources`` from the field area
        unless ``n_sources`` is also given.

    Returns
    -------
    dict
        Keys: ``NPIX``, ``FIELD_DEG``, ``PIX_ARCMIN``, ``N_SOURCES``,
        ``N_Z_BINS``.
    """
    field_deg  = float(field_deg) if field_deg is not None else 10.0
    npix       = int(npix)        if npix      is not None else 1024
    pix_arcmin = field_deg * 60.0 / npix

    if n_gal_per_arcmin2 is not None and n_sources is None:
        n_sources = int(float(n_gal_per_arcmin2) * (field_deg * 60.0) ** 2)
    n_sources = int(n_sources) if n_sources is not None else None
    n_z_bins  = int(n_z_bins)  if n_z_bins  is not None else 60

    return dict(
        NPIX=npix,
        FIELD_DEG=field_deg,
        PIX_ARCMIN=pix_arcmin,
        N_SOURCES=n_sources,
        N_Z_BINS=n_z_bins,
    )


# ── User-facing cluster-adapted BNT framework ────────────────────────────────
# Re-exported from ``bnt_framework`` so a user has a single import path:
#     import bntcluster as bnt
#     survey = bnt.SurveySetup(...);  cost = bnt.CostFunction(...)
#     cbnt   = bnt.ClusterBNT(survey, cost)
#
# The import is intentionally placed *after* the symbols above so that
# ``bnt_framework`` (which does ``import bntcluster as bnt`` to use
# ``nz_euclid`` and ``DEFAULT_COSMO_PARAMS`` as dataclass defaults) finds them
# already populated on the partially-initialised package — i.e. no circular
# import.
from .bnt_framework import (
    SurveySetup,
    CostFunction,
    ClusterBNT,
    ClusterBNTScheme,
    SchemeScan,
    StandardBNT,
    STRATEGIES,
    STRATEGY_LABELS,
    stack_bin_maps,
    bnt_weighted_map,
    rectangular_cutout,
    aa_style_ax,
    aa_style_axes,
    robust_symmetric_vmax,
    map_inside_colorbar,
)


__all__ = [
    # analytical BNT framework (low level)
    "BNTNuller",
    "BNTResult",
    "SimpleFlatLambdaCDM",
    # NFW mock generator + utilities
    "NFWMockGenerator",
    "LensCatalog",
    "build_kappa_map_nfw",
    "nz_euclid",
    "sample_nz",
    "apply_photoz_errors",
    "sigma_crit",
    # configuration / helpers
    "DEFAULT_COSMO_PARAMS",
    "configure_maps",
    "global_equal_count_edges",
    # user-facing cluster-adapted BNT framework
    "SurveySetup",
    "CostFunction",
    "ClusterBNT",
    "ClusterBNTScheme",
    "SchemeScan",
    "StandardBNT",
    "STRATEGIES",
    "STRATEGY_LABELS",
    "stack_bin_maps",
    "bnt_weighted_map",
    "rectangular_cutout",
    "detection_snr",
    # plotting helpers
    "aa_style_ax",
    "aa_style_axes",
    "robust_symmetric_vmax",
    "map_inside_colorbar",
]
