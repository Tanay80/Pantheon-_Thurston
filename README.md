# Thurston geometries and parameter constraints from SNIa data

Code, chains and analysis scripts for

> T. Gupta, A. Verma, S. Panda, P. K. Aluri,
> *Thurston geometries and parameter constraints from SNIa data*,
> [arXiv:2602.02936](https://arxiv.org/abs/2602.02936).

We constrain the homogeneous but anisotropic Thurston geometries R × H²/S², Nil and Solv, as extensions of flat ΛCDM, using the Pantheon+&SH0ES Type Ia supernovae (1701 light curves, full covariance matrix) and nested sampling (PolyChord through Cobaya). The sampled parameters are h, σ (shear), e² (eccentricity), Ω_m, Ω_κ, the preferred direction (l_a, b_a) in galactic coordinates, and the SNIa absolute magnitude M₀. The matter equation of state is fixed to w̄ = 0, and a cosmological constant is included.

The repository also contains the isotropic mock-catalogue tests and the z_CMB robustness check (secs. 3.4 and 4.4 of the revised manuscript).

---

## Repository layout

```
├── README.md
├── data.zip                      # zipped pplus.csv and pplus_cov.txt (Pantheon+&SH0ES, shared)
├── sky.py, sky.png               # sky map of the preferred-direction posteriors (Fig. 2d)
├── analyse_mock.py               # summarises the mock tests (table 4)
├── compare_z.py                  # z_HD vs z_CMB comparison (table 5)
├── check_table3.py               # posterior means, std, <chi2>, min chi2 and log Z of every chain
├── check3_all.py                 # minimum chi2 in every chain
├── lcdm/                         # flat LCDM (h, Omega_m, M0)
│   ├── chain.1.txt               # sampled parameters in the run
│   ├── chain.stats               # summarized chain statistics
│   ├── lcdm.yaml                 # YAML config file 
│   ├── rh2s2_pplus.py            # likelihood used for LCDM (same file as rh2s2/rh2s2_pplus.py)
│   └── lcdm.png, lcdm_out.png    # sampled outputs using getdist
├── rh2s2/                        # R x H^2/S^2
├── nil/                          # Nil
└── solv/                         # Solv
```

Each geometry folder contains:

| File | Purpose |
|---|---|
| `chain.1.txt` | Sampled parameters in the run |
| `chain.stats` | Summarized chain statistics |
| `<model>_pplus.py` | Likelihood (`my_likelihood`) of that geometry |
| `<model>_pplus.yaml` | Cobaya + PolyChord configuration |
| `<model>_pplus_zcmb.py` | Copy of the likelihood that reads `zCMB` instead of `zHD` |
| `null_tests.py` | Maximum-likelihood fits to the real data and to isotropic mocks |
| `res_<model>_pplus_real.txt` | Maximum-likelihood fit to the real data (z_HD) |
| `res_<model>_pplus_mock.txt` | Fits to the 64 isotropic mocks |
| `res_<model>_pplus_zcmb_real.txt` | Maximum-likelihood fit to the real data (z_CMB) |
| `log_mock.txt` | Log of the mock run |
| `<model>.png`, `<model>_out.png` | Posterior contour plots |

`rh2s2/` also holds `check1.py`, `check2.py` and `check3.py`, which are the validation scripts described in "Development checks" below.

---

## Installation

Python 3 with:

```
numpy scipy pandas astropy cobaya getdist matplotlib emcee tqdm corner
```

plus PolyChord (`pypolychord`), installed as described in the Cobaya documentation. The `<model>_pplus.py` files import `emcee`, `tqdm`, `getdist`, `matplotlib` and `corner` at the top, so **all of these must be installed, including for the mock tests**, which import those files.

```bash
python -m venv venv && source venv/bin/activate
pip install numpy scipy pandas astropy cobaya getdist matplotlib emcee tqdm corner
# then install PolyChord as described in the Cobaya docs
```

The YAML files contain a hard-coded PolyChord location (`path: /home/tanay/PolyChordLite`). **Edit it to your own PolyChord install.**

## Data

`data.zip` contains `pplus.csv` and `pplus_cov.txt` from the public Pantheon+&SH0ES data release (Scolnic et al. 2022, ApJ 938, 113). The likelihood uses the columns `zHD`, `zHEL`, `IS_CALIBRATOR`, `m_b_corr`, `CEPH_DIST`, `RA`, `DEC` and the full 1701x1701 covariance matrix.

**Unpack the data once after cloning:**

```bash
cd models
unzip data.zip          # creates pplus.csv and pplus_cov.txt in models/
```

The model files read `./pplus.csv` and `./pplus_cov.txt` from the **current directory**, so every script must be run from inside the relevant model folder (`lcdm/`, `rh2s2/`, `nil/` or `solv/`). Link the two unpacked files into each model folder once:

```bash
for d in lcdm rh2s2 nil solv; do
  ln -sf ../pplus.csv $d/pplus.csv
  ln -sf ../pplus_cov.txt $d/pplus_cov.txt
done
```

(Copying the two files into each folder works equally well.)

---

## 1. Nested sampling (main results, table 3)

All runs are started from inside the model folder, for example:

```bash
cd rh2s2
cobaya-run -f rh2s2_pplus.yaml      # use `mpirun -n <N> cobaya-run -f ...` for MPI runs
```

PolyChord settings (sec. 3.3 of the paper): `nlive: 100d (900 for LCDM)`, `num_repeats: 6d`, `precision_criterion: 0.001`, `boost_posterior: 2.0`, `do_clustering: True`, `write_dead: True`, `feedback: 1`; `d` is the number of sampled parameters. Priors are listed in table 2 of the paper. Output goes to `<model>_chains/chain*`.
> **The chains are not stored in this repository** because of their size. Running the YAML files
> regenerates them. <!-- choose one: the chains used in the paper are available on request from the
> contact below / are archived at Zenodo: DOI ... -->

### 1.1 Thurston geometries

`rh2s2/`, `nil/` and `solv/` sample (h, σ, e², Ω_m, Ω_κ, l_a, b_a, M₀) with `w̄ = 0`.

### 1.2 Flat ΛCDM

Flat ΛCDM is **not** a separate code. It is the likelihood of R × H²/S² with all anisotropic parameters set to zero, so that Ω_Λ = 1 − Ω_m and the luminosity distance reduces to Eq. (52) of the paper. The file `lcdm/lcdm.yaml` is:

```yaml
likelihood:
    import: import_module('rh2s2_pplus').my_likelihood

params:
    h:      {prior: {min: 0.5, max: 1.0}, latex: h}
    om_im:  {prior: {min: 0.2, max: 0.5}, latex: \Omega_m}
    M0:     {prior: {min: -21, max: -18}, latex: M_0}
    sigma:  0
    e2:     0
    om_k:   0
    la:     150
    ba:     30
    wbar:   0

sampler:
    polychord:
        path: /home/tanay/PolyChordLite     # edit to your install
        nlive: 900
        num_repeats: 6d
        precision_criterion: 0.001
        boost_posterior: 2.0
        do_clustering: True
        write_dead: True
        feedback: 1

output: lcdm_chains/chain
```

Only h, Ω_m and M₀ are sampled. The direction (l_a, b_a) has no effect when e² = 0. `lcdm/rh2s2_pplus.py` must be identical to `rh2s2/rh2s2_pplus.py` (check with `diff lcdm/rh2s2_pplus.py rh2s2/rh2s2_pplus.py`).

As a consistency check, a direct maximum-likelihood fit of this limit gives chi2 = 1522.42, the same as the minimum chi2 in `lcdm_chains/chain.1.txt`.

### 1.3 Reproducing the numbers in table 3

These scripts read the chains, so they work only after you have run the samplers (or obtained the paper chains, see above). From `models/`:

```bash
python check_table3.py   # posterior means, std, <chi2>, min chi2 and log Z of every chain
python check3_all.py     # minimum chi2 in every chain
python lr_chains.py      # L0/R0 lower bounds from the chain samples
```

Reference values from the chains used in the paper:

| | flat LCDM | R x H^2/S^2 | Nil | Solv |
|---|---|---|---|---|
| log Z (`chain.stats`) | -771.894 | -776.197 | -776.666 | -776.183 |
| <chi2> (posterior mean) | 1525.41 | 1522.02 | 1522.01 | 1522.21 |
| chi2_min (smallest sample) | 1522.42 | 1513.30 | 1512.62 | 1513.05 |

Nested sampling is stochastic, so a rerun will reproduce these only up to the
sampling noise (about 0.1 in log Z).

---

## 2. Isotropic mock tests (secs. 3.4 and 4.4)

`null_tests.py` fits the real data or isotropic mocks by maximum likelihood, because nested sampling is too slow for repeated runs.

**Method**

- Fiducial model: flat LCDM at h = 0.734, Omega_m = 0.334, M0 = -19.248.
- Mocks use the real redshifts, sky positions and calibrator structure. Hubble-flow magnitudes are `M0 + mu_LCDM(z) + noise`, with noise drawn from the full covariance through its Cholesky factor. Cepheid-calibrated supernovae keep their `CEPH_DIST` moduli.
- Each dataset is fitted with flat LCDM, then with the geometry. The preferred direction is scanned on a 20-point grid on one hemisphere (lambda and -lambda are equivalent). At each grid point the other six parameters are optimised with Nelder-Mead, and the two best directions are refined.
- Test statistic: `dchi2 = chi2_min(LCDM) - chi2_min(geometry)`.
- p-value: `(1 + #{dchi2_mock >= dchi2_real}) / (1 + N_mock)`.
- Seeds 1000, 1001, ... are used, so all three geometries see the same mocks.

**Usage** (from inside the geometry folder):

```bash
cd rh2s2
python null_tests.py rh2s2_pplus real 1          # real data, about 2 min
python null_tests.py rh2s2_pplus mock 64 16      # 64 mocks, 16 worker processes
```

Arguments: `<module name> <real|mock> <number of runs> [number of workers]`. Results are written line by line to `res_<module>_<mode>.txt` with the columns

```
dchi2 sigma e2 Om Ok la ba h M0 chi2_lcdm chi2_geo
```

**Performance.** The likelihood is limited by memory bandwidth. One dataset takes about 2.2 min on one core. On a 32-core / 64-thread machine, 16 workers gave about 2.3 datasets/min, and 32 or 64 workers were slower. **Use at most about 16 workers.**

**Summary** (from `models/`):

```bash
python analyse_mock.py
```

Results of the run in the paper (64 mocks per geometry):

| Geometry | Real dchi2 | Mock median | Mock 95% | p |
|---|---|---|---|---|
| R x H^2/S^2 | 9.16 | 5.69 | 14.06 | 0.169 |
| Nil | 8.80 | 5.03 | 11.93 | 0.138 |
| Solv | 8.90 | 5.00 | 12.71 | 0.169 |

The real improvement over flat LCDM is consistent with isotropic mocks.

---

## 3. Redshift-frame check (z_HD vs z_CMB)

The `*_zcmb.py` files are copies of the model files with the column `zHD` replaced by `zCMB`:

```bash
sed 's/"zHD"/"zCMB"/' rh2s2_pplus.py > rh2s2_pplus_zcmb.py
python null_tests.py rh2s2_pplus_zcmb real 1
```

Then, from `models/`:

```bash
python compare_z.py     # real-data fits with z_HD and z_CMB; p-values against the z_HD mocks
```

With z_CMB, flat LCDM fits much worse (chi2 = 1580.6 against 1522.4). The best-fit direction moves from about (275 deg, 7 deg) to (20 deg, 25 deg), so the best-fit axis is not robust. The mocks were generated with z_HD, so the z_CMB p-values are approximate.

---

## Development checks (`rh2s2/`)

| Script | What it checks |
|---|---|
| `check1.py` | The numpy angle function agrees with astropy (below 1e-15 rad), same likelihood value, timing |
| `check2.py` | chi2 of LCDM at the table 3 means, the optimised minimum, and the smoothness of the likelihood around the best fit |
| `check3.py` | Minimum chi2 in the `rh2s2` chain |

They are not needed to reproduce the paper.

---

## Notes and caveats

- `null_tests.py` replaces the astropy-based `ang_sep` of the model files with a numpy dot-product version. It gives the same angles and avoids the astropy overhead.
- The direction grid has about 30 degree spacing, so best-fit axes are approximate. The same grid is applied to the data and the mocks.
- The mock noise follows the Pantheon+ covariance, which is slightly larger than the scatter of the real residuals (chi2_red about 0.90), so the mock dchi2 distribution may be slightly broad and the quoted p-values are approximate.
- The mocks calibrate dchi2 and the best-fit sigma and Omega_kappa. They do not calibrate the Bayesian evidence.
- The Solv results are leading-order estimates in z/L (appendix A of the paper).

## Citation

```bibtex
@article{Gupta:2026thurston,
  author        = {Gupta, Tanay and Verma, Anshul and Panda, Sukanta and Aluri, Pavan K.},
  title         = {Thurston geometries and parameter constraints from {SNIa} data},
  year          = {2026},
  eprint        = {2602.02936},
  archivePrefix = {arXiv}
}
```

Please also cite the Pantheon+ release (Scolnic et al. 2022), PolyChord (Handley et al. 2015) and Cobaya.

## Contact

Tanay Gupta (tanay23@iiserb.ac.in)
