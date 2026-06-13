# bntcluster

**Cluster-adapted BNT shear-nulling for galaxy-cluster weak-lensing maps.**

`bntcluster` builds foreground-nulled weak-lensing convergence observables
$\tilde\kappa(\boldsymbol\theta)$ targeted at a chosen lens redshift, by
combining three tomographic source bins with optimised edges so that the
nulled kernel $\tilde w(z)$ peaks at the target redshift and goes progressibely to ~zero
below it. The method is the
[Bernardeau–Nishimichi–Taruya (BNT) transform](https://arxiv.org/abs/1310.6286),
adapted from cosmic shear to galaxy-cluster science.

The package provides:

- the analytic BNT engine (per-bin source distributions, ordinary lensing
  kernels, BNT matrix, photo-z smearing);
- the **cluster-adapted framework** (`SurveySetup` / `CostFunction` /
  `ClusterBNT`) that optimises the bin edges for a chosen target lens
  redshift under one of four binning strategies;
- a mock NFW convergence-map generator (Colossus-backed) with shape noise
  and photo-z error model;
- map-level utilities (BNT-weighted stacking, cut-outs, simple Gaussian filtering of the maps);
- a pedagogical end-to-end **tutorial notebook**.

> **Status / honest summary.** The framework cleanly nulls the foreground
> response in the noiseless case, but the BNT recombination amplifies the
> shape noise by a factor $\alpha\sim 3$–$6$ depending on the binning. For
> cluster *detection* on Euclid-like maps this noise penalty dominates and
> the BNT-nulled S/N is below the unweighted S/N at every lens redshift
> (transfer ratio $R\lesssim 0.6$). The package is a research tool for
> exploring this trade-off.

---

## Installation

```bash
pip install bntcluster
```

The `[notebook]` extra adds `jupyter`, `ipykernel`, `nbformat` so the
tutorial runs out of the box.

**Requirements** (handled by `pip`): `numpy<2`, `scipy`, `matplotlib`,
`astropy`, `colossus`, `tqdm`. Python 3.9 – 3.12.

> `numpy<2` is required because the code uses `numpy.trapz`, removed in
> NumPy 2.0
