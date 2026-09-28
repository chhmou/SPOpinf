# SPOpInf — Stochastic Physics-Constrained Operator Inference

MATLAB code for *Stochastic Physics-Constrained Operator Inference for Complex Systems*.

SPOpInf learns a complete stochastic quadratic model, both drift and additive diffusion, from snapshot data. The drift and the noise levels are calibrated jointly with an EM-based iterative algorithm that maximizes an Euler–Maruyama increment likelihood. The energy-preserving structure of the quadratic operator is imposed exactly as linear equality constraints.

The repository covers the two settings of the paper:

| Setting | Cases | Data |
|---|---|---|
| I — identification of SDEs | dyad, triad-based (TBH), Lorenz–84, Lorenz–96, Charney–DeVore (CDV) | generated from scratch by the scripts in `sde/` |
| II — stochastic ROMs of PDEs | stochastic Burgers, 2-D quasi-geostrophic (QG) | **not included**; set `dataPath` (see below) |

## Requirements

- MATLAB R2020a or newer. `exportgraphics` is used when figures are saved.
- Statistics and Machine Learning Toolbox. It provides `ksdensity` (Burgers and QG PDF diagnostics) and `normpdf` (TBH reference density).

No compiled code or other external packages are needed.

## Repository layout

```text
spopinf-github/
├── src/                          shared SPOpInf core (used by every case)
│   ├── constraints/              energy-preserving constraint index sets
│   ├── tensor_indexing/          upper-triangular parametrization of the quadratic tensor
│   └── solvers/                  em_stochastic_solver_sparse.m (constrained EM solver)
├── sde/                          Setting I
│   ├── <system>_stochastic_moi_vs_opinf_sde.m    full experiment per system
│   └── data_generation/generate_<system>_data.m  training data only -> sde/data/
├── pde/                          Setting II
│   ├── burgers/                  stochastic Burgers: driver, settings, FOM/G-ROM/OpInf helpers
│   └── qg/                       2-D QG: three pipeline drivers and their helpers
└── examples/                     one-line entry points that run the drivers in order
```

Every driver adds `src/` (and its own helper folder) to the MATLAB path itself, so no manual `addpath` is needed.

## Setting I: SDE benchmarks (data generated from scratch)

No SDE data are shipped. Each system is simulated by Euler–Maruyama from a fixed random seed (`rng_seed = 10`, time step `dt = 0.005`).

**Generate the training data only**

```matlab
cd sde/data_generation
generate_dyad_data          % or: generate_tbh_data, generate_l84_data, generate_l96_data, generate_cdv_data
```

Each generator writes `sde/data/<system>_training_data.mat`. The file contains:

- the full trajectory `U` (dim × N, transient included) and the time vector `t_save`;
- `dt`, `T`, `Ttrans`, `rng_seed`;
- the true noise amplitudes and the known constant forcing `Fconst` (where the system has one);
- the physical parameters of the system.

To generate all five in one go, run `examples/generate_all_sde_data`.

**Run the full experiments**

Each experiment script regenerates the same trajectory itself, so it needs no files on disk. It then:

1. forms the forward-difference drift target and removes the known forcing;
2. fits plain OpInf and SPOpInf;
3. compares the fitted operators and noise amplitudes with the truth;
4. runs same-noise pathwise tests and independent-seed ensemble statistics.

```matlab
cd sde
dyad_stochastic_moi_vs_opinf_sde
```

`examples/run_all_sde` runs all five scripts. Outputs go to `results_<system>_moi_vs_opinf_sde/run_<timestamp>/`, which holds `log.txt`, `results.mat`, and the figures.

Training lengths used in the paper: dyad T = 2000, TBH T = 1000, L84 T = 100, L96 T = 1000, CDV T = 5000.

**Reproducibility note.** Lorenz–96 and CDV draw their initial condition with `randn` *before* `rng(rng_seed)` is called. Their trajectories therefore depend on the generator state when the script starts. A fresh MATLAB session starts in that state, and so does calling `rng('default')` first; the two SDE runners in `examples/` do this before every script.

## Setting II: PDE cases (bring your own data)

No PDE data are shipped. Each driver starts with a single line to edit:

```matlab
dataPath = '<path to data>';
```

Until `dataPath` is set, the driver stops with an error that names the missing folder.

### Stochastic Burgers — `pde/burgers/`

Driver: `run_sbe_pathwise_comparison.m`. Set `dataPath` to the folder that **contains** `test3p0en03/`.

```text
<dataPath>/
└── test3p0en03/
    ├── sbe_fom_data_sig3p0en03_nu5p0en03_lam1p1en02.mat    variable PAR   (FOM parameters)
    ├── sbe_proj_data_sig3p0en03_nu5p0en03_lam1p1en02.mat   variable fom_proj (modes × snapshots)
    └── sbe_noise_dw_sig3p0en03.mat                         variable dW    (noise increments)
```

The file names encode σ = 3e-3, ν = 5e-3 and λ = 9νπ²/L² (see `sbe_additive_settings.m`).

`PAR` must provide the fields `L, N, dt, t_skip, nu, lam, gam, sigmas, eigens, SeignLap, Nmodes_dW, U, Grid_x, dis_x, dx`.

The reduced dimension defaults to r = 6. Set another value with `setenv('SBE_R','4')` before running. `examples/run_burgers_paper_cases` runs r = 2, 4, 6, the cases in the paper.

The driver trains on the snapshots up to t = 500 and evaluates to t = 1000. It compares G-ROM, G-ROM (σ), OpInf, OpInf (σ) and SPOpInf (labelled CPOpInf in the code). The stochastic models are driven by the saved FOM noise path; the deterministic ones use zero diffusion. Outputs go to `pde/burgers/results/pathwise/run_rXX_<timestamp>/`.

### 2-D quasi-geostrophic — `pde/qg/`

Set `dataPath` to the folder that contains `FOM/` and `test1/`:

```text
<dataPath>/
├── FOM/
│   └── mean_data.mat                                variable psi_mean (FOM time-averaged streamfunction)
└── test1/
    ├── pod_fd_m3500.mat                             variables POD_p, POD_w (POD bases)
    └── qge_proj_data_npod3500_re450_ro3p6en03.mat   variable w_proj (POD coefficients, ≥ 30 modes × ≥ 7001 snapshots on 10 ≤ t ≤ 80)
```

`dataPath` is set independently in each of the three drivers. Run them in this order (the first two are also chained by `examples/run_qg_pipeline`):

1. `qg_moi_vs_galerkin_r_sweep.m`
   - SPOpInf vs the intrusive G-ROM for r = 5, 10, 15, 20, 25, 30, with 400 training pairs.
   - Builds and caches the Galerkin operators in `pde/qg/data_cache/`.
   - Checkpoints each r in `results_qg_moi_vs_galerkin_r_sweep/r_XX/`.
2. `qg_stochastic_closure_vs_moi_r_sweep.m`
   - Stochastic Galerkin-closure variant.
   - Reuses the step-1 checkpoints and the cached operators.
   - Writes `results_qg_stochastic_closure_vs_moi_r_sweep/`, including `mean_streamfunction_l2.csv`.
3. `qg_galerkin_rom.m` (optional): a standalone deterministic G-ROM baseline at r = 30.

The r = 25 and r = 30 fits form large dense normal matrices and need substantial memory and run time. Completed dimensions are reused when a sweep is restarted.

## Outputs and version control

Everything the code writes is covered by `.gitignore`, so nothing generated is committed by accident:

- run folders (`results/`, `results_*/`), generated data (`data/`) and operator caches (`data_cache/`);
- data and figure files (`*.mat`, `*.fig`, `*.png`, …).
