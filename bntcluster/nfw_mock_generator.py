"""
nfw_mock_generator.py
=====================
Generate mock weak-lensing convergence maps by injecting NFW haloes onto a
pixelised sky, accounting for a (possibly photometric) galaxy source
distribution.

Two modes are supported for the source population:
  1. **Catalogue mode** – pass two astropy Tables (true-z and phot-z) directly.
  2. **Parametric mode** – pass a n(z) callable plus survey parameters; the
     class samples synthetic true- and photo-z catalogues internally, with
     Gaussian scatter and catastrophic outliers.

Quick-start examples
--------------------
**From catalogues:**

    gen = (
        NFWMockGenerator(
            cosmo_params=MY_COSMO,
            npix=1024, field_deg=10.0,
            shape_noise=0.26,
            zmin_list=[0.0, 0.3, 0.5],
        )
        .set_source_catalogs(true_z_table, phot_z_table)
        .set_lens_catalog(cluster_table)
        .run()
    )

**From a parametric n(z):**

    gen = (
        NFWMockGenerator(cosmo_params=MY_COSMO, ...)
        .set_parametric_nz(nz_func=nz_euclid, n_gal_per_arcmin2=30.0)
        .set_lens_catalog(cluster_table)
        .run()
    )
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Union

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from colossus.cosmology import cosmology as colossus_cosmo
from colossus.halo import mass_so, profile_nfw
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Parametric n(z)
# ---------------------------------------------------------------------------

def nz_euclid(
    z: np.ndarray,
    A: float = 1.8048,
    a: float = 0.4170,
    b: float = 4.8685,
    c: float = 0.7841,
) -> np.ndarray:
    """
    Un-normalised redshift distribution of Euclid-like sources.

    Parameters
    ----------
    z : array
        Redshift values (must be > 0).
    A, a, b, c : float
        Shape parameters (Euclid defaults from Laureijs et al. 2011).
    """
    return A * (z**a + z**(a * b)) / (z**b + c)


def sample_nz(
    nz_func: Callable,
    n_total: int,
    z_min: float = 0.01,
    z_max: float = 3.0,
    n_grid: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw *n_total* redshifts via inverse-CDF sampling from an arbitrary n(z).

    Parameters
    ----------
    nz_func : callable
        Un-normalised PDF n(z).
    n_total : int
        Number of galaxies to sample.
    z_min, z_max : float
        Sampling range.
    n_grid : int
        Resolution of the CDF grid.
    rng : numpy Generator

    Returns
    -------
    z_true : array (n_total,)
    """
    if rng is None:
        rng = np.random.default_rng()
    z_grid = np.linspace(z_min, z_max, n_grid)
    pz = np.maximum(nz_func(z_grid), 0.0)
    cdf = np.cumsum(pz)
    cdf /= cdf[-1]
    u_samples = rng.uniform(0.0, 1.0, n_total)
    return np.interp(u_samples, cdf, z_grid)


def apply_photoz_errors(
    z_true: np.ndarray,
    sigma_factor: float = 0.05,
    outlier_fraction: float = 0.10,
    z_min: float = 0.0,
    z_max: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Scatter true redshifts to simulate photometric redshift errors.

    Uses rejection sampling instead of clipping, to avoid artificial pile-up
    at z_min and z_max.
    """
    if rng is None:
        rng = np.random.default_rng()

    z_true = np.asarray(z_true)
    n = len(z_true)

    sigma = sigma_factor * (1.0 + z_true)
    z_phot = z_true + rng.normal(0.0, sigma, size=n)

    # Resample Gaussian-scattered galaxies that fall outside the allowed range
    bad = (z_phot < z_min) | (z_phot > z_max)

    while np.any(bad):
        z_phot[bad] = z_true[bad] + rng.normal(0.0, sigma[bad])
        bad = (z_phot < z_min) | (z_phot > z_max)

    # Catastrophic outliers
    n_outliers = int(round(outlier_fraction * n))
    if n_outliers > 0:
        idx_out = rng.choice(n, size=n_outliers, replace=False)

        # Uniform outliers are fine, but they should not be clipped afterward
        z_phot[idx_out] = rng.uniform(z_min, z_max, size=n_outliers)

    return z_phot


# ---------------------------------------------------------------------------
# Critical surface density
# ---------------------------------------------------------------------------

def _sigma_crit(
    cosmo: FlatLambdaCDM,
    zl: float,
    z_s: float,
) -> u.Quantity:
    """Critical surface density Σ_crit [M_sun / kpc²]."""
    factor = (const.c**2 / (4.0 * np.pi * const.G)).to(u.M_sun / u.kpc)
    d_s  = cosmo.angular_diameter_distance(z_s).to(u.kpc)
    d_l  = cosmo.angular_diameter_distance(zl).to(u.kpc)
    d_ls = cosmo.angular_diameter_distance_z1z2(zl, z_s).to(u.kpc)
    return (factor * d_s / (d_l * d_ls)).to(u.M_sun / u.kpc**2)


# ---------------------------------------------------------------------------
# Core NFW kappa-map builder
# ---------------------------------------------------------------------------

def build_kappa_map_nfw(
    M_arr: np.ndarray,
    c_arr: np.ndarray,
    zl_arr: np.ndarray,
    center_pix_arr: np.ndarray,
    z_cat: np.ndarray,
    cosmo: FlatLambdaCDM,
    h: float,
    npix: int = 1024,
    pix_size_arcmin: float = 0.2,
    nbins: int = 50,
    truncation_radius: Optional[float] = 2.0,
    min_r_kpc: Optional[float] = None,
    disable_tqdm: bool = False,
) -> np.ndarray:
    """
    Populate a 2-D κ map with multiple NFW haloes.

    The convergence at each pixel is

        κ(θ) = ∫ p(z_s) Σ_NFW(θ, z_l) / Σ_crit(z_l, z_s) dz_s

    where the source distribution p(z_s) is approximated by a histogram of
    `z_cat`.

    Parameters
    ----------
    M_arr : array (N,)
        Halo masses [M_sun].  Internally converted to M_sun/h for colossus.
    c_arr : array (N,)
        Concentrations.
    zl_arr : array (N,)
        Lens (halo) redshifts.
    center_pix_arr : array (N, 2)
        Halo pixel centres as rows of (y_pix, x_pix).
    z_cat : array
        True source redshifts already filtered by the tomographic cut.
    cosmo : FlatLambdaCDM
        Astropy cosmology.
    h : float
        Dimensionless Hubble parameter h = H0/100.
    npix : int
        Square map side in pixels.
    pix_size_arcmin : float
        Pixel scale [arcmin].
    nbins : int
        Number of z-bins for the source integral.
    truncation_radius : float or None
        Truncate NFW at this multiple of r200.  None → no truncation.
    min_r_kpc : float or None
        Minimum physical radius [kpc] used when a halo falls exactly on
        a pixel centre (prevents the NFW divergence that produces ±∞ pixels).
        If None, defaults to 0.5 × pixel_size_kpc, which is the natural
        sub-pixel floor.
    disable_tqdm : bool
        Suppress progress bar.

    Returns
    -------
    kappa_map : 2-D array (npix, npix)
    """
    npix = int(npix)
    kappa_map = np.zeros((npix, npix))

    # Approximate source p(z_s) with a histogram
    bins = np.linspace(0.0, float(np.max(z_cat)), nbins)
    hist, edges = np.histogram(z_cat, bins=bins, density=True)
    z_s_arr = 0.5 * (edges[:-1] + edges[1:])
    pz_arr  = hist  # shape (nbins-1,)

    # Pixel index grid (shared)
    y_grid, x_grid = np.indices((npix, npix))

    for M, c, zl, (cy, cx) in tqdm(
        list(zip(M_arr, c_arr, zl_arr, center_pix_arr)),
        total=len(M_arr),
        leave=False,
        desc="Injecting haloes",
        dynamic_ncols=True,
        disable=disable_tqdm,
    ):
        kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(zl).value
        pix_kpc        = pix_size_arcmin * kpc_per_arcmin

        # Physical angular separation [kpc] of each pixel from the halo centre
        dy_arcmin = (y_grid - cy) * pix_size_arcmin
        dx_arcmin = (x_grid - cx) * pix_size_arcmin
        theta_kpc = np.sqrt(dx_arcmin**2 + dy_arcmin**2) * kpc_per_arcmin

        # ---------------------------------------------------------------
        # FIX: avoid the NFW central divergence that turns the centre pixel
        # into ±inf when the halo lands exactly on an integer pixel position.
        # We clamp theta to at least half a pixel width in physical units.
        # ---------------------------------------------------------------
        _floor = 0.5 * pix_kpc if min_r_kpc is None else float(min_r_kpc)
        theta_kpc = np.where(theta_kpc < _floor, _floor, theta_kpc)

        # NFW surface mass density [M_sun/kpc²]
        p_nfw = profile_nfw.NFWProfile(M=M * h, c=c, z=zl, mdef='200c')
        sigma_theta = p_nfw.surfaceDensity(theta_kpc) * (u.M_sun / u.kpc**2)

        # Optional radial truncation at truncation_radius * r200
        if truncation_radius is not None:
            r200_kpc = mass_so.M_to_R(M * h, zl, '200c') / h
            sigma_theta = np.where(
                theta_kpc <= truncation_radius * r200_kpc,
                sigma_theta,
                0.0 * u.M_sun / u.kpc**2,
            )

        # Integrate κ over source redshift distribution
        # κ_i = Σ(θ) / Σ_crit(zl, z_si)   for z_si > zl
        kappa_stack = np.zeros((len(z_s_arr), npix, npix))
        for i, z_s in enumerate(z_s_arr):
            if z_s <= zl:
                continue
            sig_crit = _sigma_crit(cosmo, zl, z_s)
            kappa_stack[i] = (sigma_theta / sig_crit).value

        # Trapezoid rule: κ(θ) = ∫ p(z_s) κ(θ, z_s) dz_s
        kappa_halo = np.trapz(
            kappa_stack * pz_arr[:, np.newaxis, np.newaxis],
            z_s_arr,
            axis=0,
        )
        kappa_map += kappa_halo

    return kappa_map


# ---------------------------------------------------------------------------
# LensCatalog dataclass
# ---------------------------------------------------------------------------

class LensCatalog:
    """
    Container for a cluster / lens catalogue.

    Parameters
    ----------
    z : array (N,)
        Lens redshifts.
    RA : array (N,)
        Right ascension [degrees].
    Dec : array (N,)
        Declination [degrees].
    mass : array (N,)
        Halo masses [M_sun].
    concentration : float or array (N,)
        NFW concentration parameter(s).
    lens_id : array (N,) or None
        Optional per-halo identifiers.
    """

    def __init__(
        self,
        z: np.ndarray,
        RA: np.ndarray,
        Dec: np.ndarray,
        mass: np.ndarray,
        concentration: Union[float, np.ndarray] = 4.0,
        lens_id: Optional[np.ndarray] = None,
    ):
        self.z             = np.asarray(z,    dtype=float)
        self.RA            = np.asarray(RA,   dtype=float)
        self.Dec           = np.asarray(Dec,  dtype=float)
        self.mass          = np.asarray(mass, dtype=float)
        self.concentration = (
            np.full(len(self.z), float(concentration))
            if np.isscalar(concentration)
            else np.asarray(concentration, dtype=float)
        )
        self.lens_id = lens_id

    def __len__(self):
        return len(self.z)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NFWMockGenerator:
    """
    End-to-end generator of mock NFW weak-lensing convergence maps.

    Workflow
    --------
    1. Instantiate with cosmology, map geometry, redshift cuts, and options.
    2. Set the source population via :meth:`set_source_catalogs` **or**
       :meth:`set_parametric_nz`.
    3. Set the lens (cluster) catalogue via :meth:`set_lens_catalog`.
    4. Call :meth:`run` (or the individual ``generate_*`` methods) to produce
       noiseless κ maps, correlated noise realisations, and noisy maps.

    Parameters
    ----------
    cosmo_params : dict
        Cosmological parameters forwarded to both astropy and colossus.
        Required keys: ``H0``, ``Om0``, ``Ob0``.
        Optional (for colossus): ``flat`` (bool), ``sigma8``, ``ns``.
    npix : int
        Square map side in pixels.
    field_deg : float
        Square field side in degrees (sets the pixel scale).
    shape_noise : float
        Per-galaxy intrinsic ellipticity dispersion σ_ε used when generating
        noise maps.
    zmin_list : list of float
        Lower tomographic redshift cuts.  Galaxies with phot-z > zmin are
        included.
    zmax_list : list of float or None
        Upper tomographic cuts.  ``None`` entries (or a shorter list) leave
        the corresponding bin open-ended.
    truncation_radius : float or None
        NFW profile is set to zero beyond ``truncation_radius * r200``.
    nbins : int
        Number of redshift bins to approximate the ∫ p(z_s) dz_s integral.
    output_dir : str
        Root directory for all FITS outputs.
    seed : int
        Master random seed used throughout (noise maps, parametric sampling).
    """

    def __init__(
        self,
        cosmo_params: dict,
        npix: int = 1024,
        field_deg: float = 10.0,
        shape_noise: float = 0.26,
        zmin_list: Optional[List[float]] = None,
        zmax_list: Optional[List[float]] = None,
        truncation_radius: Optional[float] = 2.0,
        nbins: int = 50,
        output_dir: str = './maps/',
        seed: int = 42,
    ):
        self.npix             = int(npix)
        self.field_deg        = float(field_deg)
        self.pix_size_arcmin  = field_deg * 60.0 / npix
        self.shape_noise      = float(shape_noise)
        self.zmin_list        = list(zmin_list) if zmin_list is not None else [0.0]
        self.zmax_list        = list(zmax_list) if zmax_list is not None else []
        self.truncation_radius = truncation_radius
        self.nbins            = int(nbins)
        self.output_dir       = output_dir
        self.seed             = seed

        # Cosmology --------------------------------------------------------
        self._cosmo_params = cosmo_params
        self.h   = cosmo_params['H0'] / 100.0
        self.cosmo = FlatLambdaCDM(
            H0=cosmo_params['H0'],
            Om0=cosmo_params['Om0'],
            Ob0=cosmo_params['Ob0'],
        )
        colossus_cosmo.setCosmology('_nfw_mock', **cosmo_params)

        # Internal state ---------------------------------------------------
        self._true_z: Optional[np.ndarray] = None
        self._phot_z: Optional[np.ndarray] = None
        self._lens_cat: Optional[LensCatalog] = None

        # Results (populated by generate_* methods) ----------------------
        self.kappa_maps: Dict[str, np.ndarray] = {}
        self.noise_maps: Dict[str, np.ndarray] = {}
        self.noisy_maps: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Source-population setters
    # ------------------------------------------------------------------

    def set_source_catalogs(
        self,
        true_z_cat: Union[np.ndarray, Table],
        phot_z_cat: Union[np.ndarray, Table],
    ) -> "NFWMockGenerator":
        """
        Provide source redshifts from two pre-computed catalogues.

        Parameters
        ----------
        true_z_cat : array or astropy Table
            True redshifts.  If a Table, must contain a ``'redshift'`` column.
        phot_z_cat : array or astropy Table
            Photometric redshifts (same galaxy positions but scattered z).

        Returns
        -------
        self (for method chaining)
        """
        self._true_z = np.asarray(
            true_z_cat["redshift"]
            if hasattr(true_z_cat, "colnames")
            else true_z_cat,
            dtype=float,
        )
        self._phot_z = np.asarray(
            phot_z_cat["redshift"]
            if hasattr(phot_z_cat, "colnames")
            else phot_z_cat,
            dtype=float,
        )
        return self

    def set_parametric_nz(
        self,
        nz_func: Callable = nz_euclid,
        n_gal_per_arcmin2: float = 30.0,
        z_min: float = 0.01,
        z_max: float = 6.0,
        phot_sigma_factor: float = 0.05,
        outlier_fraction: float = 0.10,
        seed: Optional[int] = None,
    ) -> "NFWMockGenerator":
        """
        Generate synthetic source catalogues from a parametric n(z).

        The method samples ``n_gal_per_arcmin2 × field_area`` true redshifts
        from ``nz_func``, then smears them with Gaussian photo-z errors
        (σ_z = phot_sigma_factor × (1+z)) and replaces a fraction
        ``outlier_fraction`` of galaxies with catastrophic outliers drawn
        uniformly over [z_min, z_max].

        Parameters
        ----------
        nz_func : callable
            Un-normalised redshift distribution n(z).  Defaults to
            :func:`nz_euclid`.
        n_gal_per_arcmin2 : float
            Source number density [arcmin⁻²].
        z_min, z_max : float
            Redshift sampling range.
        phot_sigma_factor : float
            Photo-z scatter amplitude (e.g. 0.05 for Euclid).
        outlier_fraction : float
            Fraction of catastrophic outliers (e.g. 0.10 = 10 %).
        seed : int or None
            Overrides the instance seed for this call only.

        Returns
        -------
        self (for method chaining)
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        field_arcmin2 = (self.field_deg * 60.0) ** 2
        n_total = int(round(n_gal_per_arcmin2 * field_arcmin2))
        print(
            f"[NFWMockGenerator] Sampling {n_total:,} source galaxies "
            f"from parametric n(z) over z ∈ [{z_min}, {z_max}]…"
        )
        self._true_z = sample_nz(
            nz_func, n_total, z_min=z_min, z_max=z_max, rng=rng
        )
        self._phot_z = apply_photoz_errors(
            self._true_z,
            sigma_factor=phot_sigma_factor,
            outlier_fraction=outlier_fraction,
            z_min=z_min,
            z_max=z_max,
            rng=rng,
        )
        print(
            f"[NFWMockGenerator] Photo-z scatter σ_z = {phot_sigma_factor} × (1+z), "
            f"{100*outlier_fraction:.0f}% catastrophic outliers."
        )
        return self

    # ------------------------------------------------------------------
    # Lens catalogue setter
    # ------------------------------------------------------------------

    def set_lens_catalog(
        self,
        clust_cat: Union[Table, LensCatalog],
        concentration: Union[float, np.ndarray] = 4.0,
    ) -> "NFWMockGenerator":
        """
        Set the cluster (lens) catalogue.

        Parameters
        ----------
        clust_cat : astropy Table or LensCatalog
            Must contain columns ``'z'``, ``'RA'``, ``'Dec'``, ``'mass'``.
            An optional ``'concentration'`` column overrides the `concentration`
            argument on a per-halo basis.
        concentration : float or array
            Default concentration to use for all haloes (ignored if
            ``'concentration'`` is already a column of ``clust_cat``).

        Returns
        -------
        self (for method chaining)
        """
        if isinstance(clust_cat, LensCatalog):
            self._lens_cat = clust_cat
            return self

        c_col = (
            np.asarray(clust_cat["concentration"], dtype=float)
            if "concentration" in clust_cat.colnames
            else concentration  # scalar or array; LensCatalog handles both
        )
        self._lens_cat = LensCatalog(
            z=np.asarray(clust_cat["z"],    dtype=float),
            RA=np.asarray(clust_cat["RA"],  dtype=float),
            Dec=np.asarray(clust_cat["Dec"], dtype=float),
            mass=np.asarray(clust_cat["mass"], dtype=float),
            concentration=c_col,
            lens_id=(
                np.asarray(clust_cat["lens_id"])
                if "lens_id" in clust_cat.colnames
                else None
            ),
        )
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_ready(self):
        if self._true_z is None or self._phot_z is None:
            raise RuntimeError(
                "Source distribution not set. "
                "Call set_source_catalogs() or set_parametric_nz() first."
            )
        if self._lens_cat is None:
            raise RuntimeError(
                "Lens catalogue not set.  Call set_lens_catalog() first."
            )

    def _radec_to_pix(
        self, ra_deg: np.ndarray, dec_deg: np.ndarray
    ) -> np.ndarray:
        """Flat-sky RA/Dec [deg] → pixel position (y_pix, x_pix)."""
        x_pix = ra_deg  * 60.0 / self.pix_size_arcmin
        y_pix = dec_deg * 60.0 / self.pix_size_arcmin
        return np.column_stack([y_pix, x_pix])   # shape (N, 2)

    def _zcut_tag(self, zmin: float, zmax: Optional[float]) -> str:
        """Build a short string tag identifying a tomographic bin."""
        lo = f"{int(round(zmin * 10)):02d}"
        if zmax is not None:
            hi = f"{int(round(zmax * 10)):02d}"
        else:
            hi = "30"   # open-ended (≡ "to infinity")
        return f"zs_{lo}{hi}"

    def _get_zmax(self, idx: int) -> Optional[float]:
        if idx < len(self.zmax_list):
            return self.zmax_list[idx]
        return None


    # ------------------------------------------------------------------
    # Public generation methods
    # ------------------------------------------------------------------

    def generate_kappa_maps(
        self,
        disable_tqdm: bool = False,
        overwrite: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Build noiseless κ maps for every requested redshift cut.

        The tomographic selection is performed on **photometric** redshifts,
        while the convergence integral uses **true** redshifts — exactly
        as in the original notebook.

        Results are stored in ``self.kappa_maps`` and written to FITS files
        under ``{output_dir}/noiseless/{tag}/kappaE.fits``.

        Returns
        -------
        kappa_maps : dict  {tag → 2-D array}
        """
        self._check_ready()
        lc = self._lens_cat
        center_pix = self._radec_to_pix(lc.RA, lc.Dec)

        for idx, zmin in enumerate(
            tqdm(self.zmin_list, desc="Redshift cuts (noiseless)")
        ):
            zmax = self._get_zmax(idx)
            tag  = self._zcut_tag(zmin, zmax)

            # Tomographic mask on phot-z
            mask = self._phot_z > zmin
            if zmax is not None:
                mask &= self._phot_z <= zmax

            z_true_sel = self._true_z[mask]
            if len(z_true_sel) == 0:
                print(f"  [WARNING] No sources in cut {tag}; skipping.")
                continue

            kappa = build_kappa_map_nfw(
                M_arr=lc.mass,
                c_arr=lc.concentration,
                zl_arr=lc.z,
                center_pix_arr=center_pix,
                z_cat=z_true_sel,
                cosmo=self.cosmo,
                h=self.h,
                npix=self.npix,
                pix_size_arcmin=self.pix_size_arcmin,
                nbins=self.nbins,
                truncation_radius=self.truncation_radius,
                disable_tqdm=disable_tqdm,
            )
            self.kappa_maps[tag] = kappa

            # Persist to FITS
            path = os.path.join(self.output_dir, "noiseless", tag)
            os.makedirs(path, exist_ok=True)
            fits.PrimaryHDU(kappa).writeto(
                os.path.join(path, "kappaE.fits"), overwrite=overwrite
            )

        return self.kappa_maps

    def generate_noise_maps(
        self,
        z_noise_edges: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate correlated κ-noise maps for every requested redshift cut.

        Noise maps are built slice-by-slice (each slice between adjacent
        ``z_noise_edges`` is an independent Gaussian realisation), then
        cumulated with number-count weights.  Maps sharing many redshift
        slices are therefore automatically correlated, as they should be.

        Parameters
        ----------
        z_noise_edges : list of float or None
            Edges defining the non-overlapping slices used to build the
            independent noise realisations.  If None, the sorted unique
            ``zmin_list`` values plus a high-z ceiling (max(phot_z)+1%) are
            used.
        seed : int or None
            Overrides the instance seed for this call only.

        Returns
        -------
        noise_maps : dict  {tag → 2-D array}
        """
        self._check_ready()
        rng    = np.random.default_rng(seed if seed is not None else self.seed)
        n_pix  = self.npix ** 2
        phot_z = self._phot_z

        # Build default slice edges
        z_ceil = max(3.0, float(np.max(phot_z)) * 1.01)
        if z_noise_edges is None:
            z_noise_edges = sorted(set(self.zmin_list)) + [z_ceil]
        z_edges = np.asarray(z_noise_edges)

        # Count galaxies per non-overlapping slice
        N_per_slice = np.array([
            int(np.sum((phot_z >= z_edges[j]) & (phot_z < z_edges[j + 1])))
            for j in range(len(z_edges) - 1)
        ])

        # Independent Gaussian maps, one per slice
        slice_maps = np.array([
            rng.normal(
                0.0,
                self.shape_noise / np.sqrt(max(N / n_pix, 1e-12)),
                size=(self.npix, self.npix),
            )
            for N in N_per_slice
        ])
        N_pix_per_slice = N_per_slice / n_pix  # galaxy density per pixel

        # Accumulate slices for each tomographic cut
        for idx, zmin in enumerate(self.zmin_list):
            zmax = self._get_zmax(idx)
            tag  = self._zcut_tag(zmin, zmax)

            lo_edges = z_edges[:-1]
            if zmax is None:
                inc = np.where(lo_edges >= zmin)[0]
            else:
                inc = np.where((lo_edges >= zmin) & (lo_edges < zmax))[0]

            if len(inc) == 0 or N_pix_per_slice[inc].sum() < 1e-12:
                self.noise_maps[tag] = np.zeros((self.npix, self.npix))
                continue

            # Weighted combination of independent slices
            w = N_pix_per_slice[inc] / N_pix_per_slice[inc].sum()
            noise = np.tensordot(w, slice_maps[inc], axes=([0], [0]))
            self.noise_maps[tag] = noise

        return self.noise_maps

    def generate_noisy_maps(
        self,
        overwrite: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Combine noiseless κ maps with noise maps and save.

        Must be called after :meth:`generate_kappa_maps` and
        :meth:`generate_noise_maps`.

        The noisy map is mean-subtracted:
            κ_noisy = κ_noiseless + noise − mean(κ_noiseless + noise)

        Outputs:
          - ``{output_dir}/noisy/{tag}/kappaE.fits`` – noisy κ map
          - ``{output_dir}/noisy/{tag}/kappaB.fits`` – pure noise map

        Returns
        -------
        noisy_maps : dict  {tag → 2-D array}
        """
        if not self.kappa_maps:
            raise RuntimeError("Call generate_kappa_maps() first.")
        if not self.noise_maps:
            raise RuntimeError("Call generate_noise_maps() first.")

        for idx, zmin in enumerate(self.zmin_list):
            zmax = self._get_zmax(idx)
            tag  = self._zcut_tag(zmin, zmax)

            if tag not in self.kappa_maps or tag not in self.noise_maps:
                continue

            noisy = self.kappa_maps[tag] + self.noise_maps[tag]
            noisy -= noisy.mean()
            self.noisy_maps[tag] = noisy

            path = os.path.join(self.output_dir, "noisy", tag)
            os.makedirs(path, exist_ok=True)
            fits.PrimaryHDU(noisy).writeto(
                os.path.join(path, "kappaE.fits"), overwrite=overwrite
            )
            fits.PrimaryHDU(self.noise_maps[tag]).writeto(
                os.path.join(path, "kappaB.fits"), overwrite=overwrite
            )

        return self.noisy_maps

    def run(
        self,
        disable_tqdm: bool = False,
        overwrite: bool = True,
    ) -> "NFWMockGenerator":
        """
        Execute the full pipeline:
        1. Build noiseless κ maps.
        2. Generate correlated noise realisations.
        3. Combine into noisy maps 

        Returns
        -------
        self (for method chaining)
        """
        self.generate_kappa_maps(disable_tqdm=disable_tqdm, overwrite=overwrite)
        self.generate_noise_maps()
        self.generate_noisy_maps(overwrite=overwrite)
        return self

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    def __repr__(self):
        src = "not set"
        if self._true_z is not None:
            src = f"{len(self._true_z):,} galaxies"
        lens = "not set"
        if self._lens_cat is not None:
            lens = f"{len(self._lens_cat)} haloes"
        return (
            f"NFWMockGenerator(\n"
            f"  grid      : {self.npix}×{self.npix} pix, "
            f"{self.field_deg}°×{self.field_deg}° "
            f"({self.pix_size_arcmin:.3f} arcmin/pix)\n"
            f"  sources   : {src}\n"
            f"  lenses    : {lens}\n"
            f"  zmin cuts : {self.zmin_list}\n"
            f"  zmax cuts : {self.zmax_list if self.zmax_list else 'open-ended'}\n"
            f"  shape σ_ε : {self.shape_noise}\n"
            f"  output    : {self.output_dir}\n"
            f")"
        )
