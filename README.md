# Thurston geometries and SNIa constraints (Pantheon+ & SH0ES)

Code for fitting anisotropic Thurston geometries (R × H²/S², Nil, Solv) to the
Pantheon+ & SH0ES Type Ia supernova sample, and comparing each with its isotropic limit.

**Authors:** Tanay Gupta, Anshul Verma, Sukanta Panda, Pavan K. Aluri
**Status:** manuscript under revision. Production chains are running, and no final results
are reported here yet.

---

## What the model is

Each geometry has two scale factors, `a(t)` and `b(t)` (preferred direction), with
mean scale factor `A = (a²b)^(1/3)`. The source is **matter plus a cosmological
constant** (`p_Λ = -ρ_Λ`), and `w̄` is the equation of state of the matter component only.

Time variable: `τ = ln A`, with a prime meaning `d/dτ`. State vector:
`[h, σ, e², Ω_m, Ω_κ]`. The constraint `Ω_m + Ω_Λ + Ω_κ + σ² = 1` holds at all epochs,
so `Ω_Λ = 1 - Ω_m - Ω_κ - σ²` needs no ODE of its own.

Common to all geometries:

```
h'/h  = -3 [ σ² + Ω_κ/3 + (1 + w̄) Ω_m / 2 ]
Ω_m'  = Ω_m [ 6σ² + 2Ω_κ - 3(1 + w̄)(1 - Ω_m) ]
(e²)' = 6σ(1 - e²)
```

Geometry-specific:

| Geometry | σ′ geometry terms | Ω_κ′ |
|---|---|---|
| R × H²/S² | `(1+σ)(σ²-σ+1) + (2-σ)[Ω_κ-(1+σ)²]/2` | `-2Ω_κ(1+σ+h'/h)` |
| Nil, Solv | `(1+σ)(σ²-σ+1) - (2-σ)[Ω_κ+(1+σ)²]/2 - Ω_κ(1+σ)` | `-2Ω_κ(1-2σ+h'/h)` |

All geometries share the source term `(3σ/2)[(w̄+1)Ω_m + σ² + Ω_κ - 1]` in σ′.

Isotropic limit: `σ = e² = Ω_κ = 0` gives `h'/h = -(3/2)Ω_m(1+w̄)`. At `w̄ = 0` this is
exactly flat ΛCDM. Curvature sources shear, so there is no non-flat isotropic limit.

Luminosity distance (planar-gauge, direction-dependent):

```
d_L = c (1+z_hel) ∫ dA / (A² H) · (1-e²)^(1/6) / sqrt(1 - e² · trig²(α))
1 + z = (1/A) · sqrt(1 - e² · trig²(α)) / (1-e²)^(1/3)
```

`α` is the angle between the SN direction and the preferred axis. The trigonometric
factors are `cos²α` in `A(z)` and `sin²α` in `f` for R × H²/S² and Solv, and the reverse
for Nil. The fitted likelihood is `χ² = Δμᵀ C⁻¹ Δμ` with the full Pantheon+SH0ES
covariance. Calibrator rows use the Cepheid distance modulus.

---

## Repository layout

```
production/
├── rh2s2/   rh2s2_pplus.py   rh2s2_pplus.yaml   pplus.csv  pplus_cov.txt
├── nil/     nil_pplus.py     nil_pplus.yaml     pplus.csv  pplus_cov.txt
├── solv/    solv_pplus.py    solv_pplus.yaml    pplus.csv  pplus_cov.txt
└── lcdm/    (reuses rh2s2_pplus.py)  lcdm.yaml  isow.yaml
```

- `*_pplus.py`: Cobaya external likelihood `my_likelihood(h, sigma, e2, om_im, om_k, la, ba, M0, wbar)`.
- `lcdm.yaml`: flat ΛCDM baseline, the same likelihood with `σ = e² = Ω_κ = w̄ = 0` fixed.
- `isow.yaml`: isotropic baseline with free `w̄`, which is the correct null for the free-`w̄` anisotropic fits.
- `model/` (archive, if included): earlier scripts, check scripts and Mathematica notebooks.

Data: `pplus.csv` and `pplus_cov.txt` are the Pantheon+SH0ES data and covariance
(`TODO`: add source link and note whether the files are included or must be downloaded).

---

## Installation

Python ≥ 3.10 (developed on 3.14) with:

```
numpy scipy pandas astropy cobaya getdist mpi4py
```

PolyChord: `cobaya-install polychord`, or build PolyChordLite manually and set `path:`
in each YAML (currently `/home/tanay/PolyChordLite`, so **edit this for your machine**).

---

## Running

Each model runs from its own folder. One MPI job per model:

```bash
cd production/rh2s2
export OMP_NUM_THREADS=1
nohup mpirun -n 15 cobaya-run rh2s2_pplus.yaml -r > rh2s2.log 2>&1 &
```

Do the same for `nil` and `solv`. For the baselines:

```bash
cd production/lcdm
nohup sh -c "mpirun -n 15 cobaya-run lcdm.yaml -r > lcdm.log 2>&1; \
             mpirun -n 15 cobaya-run isow.yaml -r > isow.log 2>&1" &
```

- `-r` resumes from the last checkpoint, so rerun the same command after a crash.
  **Do not edit a YAML between a run and its resume.** Use a new `output:` prefix for any change.
- Keep the number of MPI ranks across all concurrent jobs at or below the core count.
- Progress: check the timestamps of `*_polychord_raw/chain.resume` and `chain_dead.txt`.
- Use identical PolyChord settings (`nlive`, `num_repeats`, `precision_criterion`) in every
  YAML so that `Δ log Z` is a fair comparison.

### Priors

| Parameter | R × H²/S² | Nil, Solv |
|---|---|---|
| h | [0.5, 1.0] | same |
| σ | [-0.2, 0.2] | same |
| e² | [0, 0.2] | same |
| Ω_m | [0.2, 0.5] | same |
| Ω_κ | [-0.25, 0.25] | [0, 0.25] (these need κ < 0) |
| l_a | [0°, 360°] | same |
| b_a | [0°, 90°] | same |
| M₀ | [-21, -18] | same |
| w̄ | [-0.3, 0.3] | same |

`b_a` is restricted to one hemisphere, because the likelihood depends only on `cos²α` and
`(l, b)` and its antipode give identical fits.

---

## Implementation notes

The original version solved about 1700 ODE systems per likelihood call (one per supernova).
The current version is mathematically equivalent but much faster:

1. The system in `τ = ln A` does not depend on `α`, so **one** `solve_ivp` (DOP853, dense output) serves all supernovae.
2. Each supernova's `τ_i` follows from its algebraic `A(z)` relation by vectorised bisection (`tau_of_z`).
3. The distance integral uses 24-point Gauss–Legendre on the shared solution (`dist_int`).
4. Calibrator rows skip the ODE and integral entirely.
5. Guards return `-1e30` for runaway solutions: a terminal blow-up event, non-finite output, a vanishing `d ln A/dz` denominator checked per supernova over its own range, and a non-positive distance integral. The ODE range is extended when it is too short for the highest-z supernova.

Redshifts: the **primary analysis uses `zHD`** for the integral and `zHEL` in the prefactor.
Using `zCMB` gave χ²(ΛCDM) = 1580.6 against 1522.4 for `zHD`, a ~58 difference that matches the
Pantheon+SH0ES covariance convention.

---

## Validation performed

| Test | Result |
|---|---|
| Isotropic limit vs direct ΛCDM integral (all 3 geometries) | Δχ² ≈ 1e-8 to 3e-7 |
| New vs old per-SN likelihood (accepted points) | Δχ² ≈ 1e-9 to 1e-4 |
| Random prior points (1000 per model) | no NaNs, max ≈ 0.06 s per call |
| Time per call | ≈ 0.01–0.03 s (old: 0.6–1.3 s) |
| ΛCDM best fit, `zHD` (new likelihood) | χ² = 1522.4 (old `lcdm.py`: 1522.6, difference from coarse Romberg integration) |

---

## Known limitations

- **Curvature focusing is not included in d_L.** The luminosity distance uses the planar-gauge (Bianchi-I-like) form, so curvature enters the likelihood only through the expansion rate `h(z)`. The distance relations should be re-derived for each geometry.
- **Planar gauge for Nil.** `P^y = 0` is not preserved exactly by the Nil geodesic equations, so the gauge is an approximation.
- **Solv rays are leading order.** The Solv redshift is valid to leading order in `z√(-κ)/c`, with corrections of relative order `(z√(-κ)/c)²`. For the fitted `L₀/R₀` these are not parametrically negligible compared with the fitted σ and e². Treat Solv constraints as leading-order estimates.
- **`A(z=0) ≠ 1` in the code.** The algebraic `A(z)` relation is α-dependent at `z = 0`, while the manuscript sets `A₀ = 1`. The code reproduces the earlier convention. `TODO`: decide which is intended.
- **Rejected parameter points.** Runaway or sign-changing regions of the ODE are rejected (`-1e30`), which cuts some prior volume. One Nil test point (h=0.74, σ=0.01, e²=0.04, Ω_m=0.35, Ω_κ=0.02, w̄=-0.1) is rejected by the new code but finite in the old one. `TODO`: confirm this is a legitimate rejection.
- **`L₀/R₀` is a reparametrisation of `Ω_κ`.** Quote lower limits, not symmetric errors.
- **Topology does not enter the analysis.** Locally, Nil, Solv and R × H² are the Bianchi II, VI₀ and III geometries.
- **Peculiar-velocity corrections.** `zHD` assumes standard-model flows, which could affect an isotropy test. A `zCMB` robustness run is planned.

---

## Results

`TODO`: to be added after the chains finish. Planned contents:

- Posteriors for h, σ, e², Ω_m, Ω_κ, w̄ and the axis for each geometry.
- χ²_min and `log Z` for ΛCDM, the isotropic free-`w̄` baseline, and each anisotropic model.
- `L₀/R₀` lower limits computed per chain sample.
- Calibration of significance with isotropic mocks (using the real sky positions, redshifts and covariance).
- Checks that no posterior sits at a prior edge.

---

## Citation

`TODO`: add the paper reference or arXiv ID.

If you use the data, please also cite Pantheon+ (Scolnic et al. 2022; Brout et al. 2022) and SH0ES (Riess et al. 2022).

## License

`TODO`
