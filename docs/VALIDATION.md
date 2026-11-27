# OP2 Validation Guide

## Overview

Two kinds of checks:

- regression scripts (sections 1-5) that run in minutes and check residual levels against bands;
- verification cases (sections 6-10) that run the solver against reference solutions and
  experiments, mostly on a GPU, and are judged by their physical results.

They assume the shared repository layout:

- OP2 app build in `./` (or `build/` with CMake, `BIN=...` in the case scripts)
- preprocessed meshes in `meshes-op2`
- OP2 outputs in `outputs-op2`
- helper executables in `.helpers/bin` (`scripts/build_helpers.sh`)

The OP2 app does not generate meshes directly. Meshes are produced by the OP2-local wrapper:

```bash
./scripts/preprocess_mesh.sh
```

## Regression scripts

### 1. Smoke test

Runs the OP2 path on basic inviscid cases and checks output residuals:

```bash
bash tests/test_smoke.sh
```

This covers:

- mesh preprocessing for `box` and `bump`
- `make seq`
- OP2 `box`
- OP2 `bump`

The test checks:

- `box` preserves the uniform state to near roundoff
- `bump` reaches the stabilized OP2 final `L2(rho)` band:
  `6.9e-04 <= L2(rho) <= 7.1e-04`

### 2. OP2 regression check

```bash
bash tests/test_consistency.sh
```

This runs:

- OP2 `box`
- OP2 `bump`

and checks the final `L2(rho)` residual levels.

Expected behavior:

- `box`: final residual remains near machine zero
- `bump`: OP2 final `L2(rho)` remains in
  `6.9e-04 <= L2(rho) <= 7.1e-04`

### 3. Flat-plate viscous benchmark

```bash
bash scripts/run_flatplate_validation.sh
```

This runs OP2 on `flatplate_develop`, checks the final residual band, and postprocesses the OP2
result into benchmark CSV files:

- wall skin friction `Cf`
- wall pressure coefficient `Cp`
- normalized velocity profiles at 20%, 50%, and 80% of plate length

Current expected status:

- final OP2 `L2(rho)`:
  `4.0e-08 <= L2(rho) <= 4.15e-08`
- wall/profile CSV files are generated and non-empty

The script should pass when the OP2 implementation remains within the validated residual band
and produces usable benchmark outputs.

### 4. Implicit scheme

```bash
BUILD_DIR=build bash tests/test_implicit.sh
```

This runs 12 implicit iterations on the bump with the CMake build (`README.md`,
"Implicit Time Stepping") for PETSc and hypre, each OP2 variant (seq, genseq,
OpenMP, CUDA and their MPI builds) and the linear-solve options of
`OP2_SOLVER.md` section 17 (`pc`, `pmat`, `op`, `jacobian_interval`,
`partition`, `reorder`): 35 runs. Each must reduce `L2(rho)` below a tenth of
its first iteration's, and the runs of one algorithm must agree to 1%.

### 5. Build variants and formula meshes

```bash
COMPILE_TESTS=TRUE RUN_TESTS=TRUE bash tests/nssolver_op2_test.sh
bash tests/test_formula_meshes.sh
```

The first builds every Makefile variant of the app (seq, genseq, OpenMP, CUDA, and their MPI
builds) and runs each on `box`, `bump` and `flatplate_develop`, checking that the run completes;
the second only checks that the formula-driven 3D meshes (`bump3d`, `axisymmetric_body`,
`naca_wing`) preprocess.

## Verification cases

| Section | Case | Checks | Result | Cost |
|---|---|---|---|---|
| 6 | TMR flat plate | steady SA, viscous wall, TMR boundaries | $C_f$ = FUN3D's to 0.07% on each grid from 69x49 | three coarsest 1.5 min on a laptop GPU, two finest 9 min on an A100 |
| 7 | Cylinder, Re 100 | laminar unsteady NS, dual time | St 0.163, $\bar C_D$ 1.36, $C_L'$ 0.32 | 15 min on a V100 |
| 8 | 18% circular arc | transonic shock oscillation, SA URANS | St 0.159 (experiment 0.156) | 2.7 h on a V100 |
| 9 | OAT15A | transonic buffet onset, amplitude, frequency, wall pressure | onset 3.2° (experiment 3.1-3.25°), St 0.070 (0.066) | 4 h per run on a V100 |
| 10 | ONERA M6 | Euler on an SU2 tetrahedral mesh, 3D transonic wing | $C_L$ 0.288; upper $C_p$ RMS 0.07-0.12 against the experiment, lower 0.03-0.05 | 2 min on a laptop GPU |

Cases 6-10 each have a script that preprocesses the mesh, runs the solver and prints the
case's figures of merit; `BIN` names the executable (`build/nssolver_op2_seq` by
default) and `LAUNCH` an optional launcher such as `mpirun -np 4`. On the `cl`
cluster the `scripts/cl/*.sbatch` jobs run the same scripts on one GPU
(`ARCH=70` with `-w erdos|wald` for the V100 build of `scripts/cl/build.sbatch`)
and leave their logs in `~/op2sol/logs`. Options after a script's name go to every
run; the jobs pass `--solve-config configs/solve.json` (GMRES with block Jacobi or ILU
to a relative residual of 1e-3, `OP2_SOLVER.md` section 17), as the commands below
do.

### 6. TMR flat plate

The NASA Turbulence Modeling Resource's 2D zero-pressure-gradient flat plate, SA
verification case: M 0.2, Re 5e6 per unit length, T 300 K, free-stream
$\tilde\nu = 3\nu$ (`configs/tmr_flatplate.cfg`: second order unlimited as the
TMR reference runs, implicit, CFL 10 growing by 1.3 to 1e4, 300 iterations).
TMR's boundaries: total pressure and temperature inflow at x = -1/3, symmetry
ahead of the plate, the plate from x = 0 to 2, pressure outflow at x = 2 and a
Riemann far field at the top.

```bash
BIN=build/nssolver_op2_cuda GRIDS="35x25 69x49 137x97" scripts/tmr/flatplate.sh \
  --solve-config configs/solve.json
sbatch scripts/cl/flatplate.sbatch          # 273x193 and 545x385, 250 iterations
```

The script fetches TMR's PLOT3D grid family (one archive from nasa.gov),
preprocesses each grid (`plot3d2d`), runs it and prints the skin friction at
x = 0.97 and the drag (`scripts/tmr/flatplate_forces.py`) against CFL3D's on the
545x385 grid. Outputs: `outputs-op2/tmr/flatplate_<grid>.{h5,log}` and
`flatplate_<grid>_wall.csv`.

| Grid | $C_f(0.97)$ | FUN3D | $C_D$ | FUN3D |
|---|---|---|---|---|
| 35x25 | 0.0026785 | 0.0026674 | 0.0028003 | 0.0027739 |
| 69x49 | 0.0026968 | 0.0026950 | 0.0028252 | 0.0028226 |
| 137x97 | 0.0027020 | 0.0027022 | 0.0028367 | 0.0028400 |
| 273x193 | 0.0027039 | 0.0027045 | 0.0028435 | 0.0028479 |
| 545x385 | 0.0027047 | 0.0027054 | 0.0028482 | 0.0028525 |

The skin friction matches FUN3D's on the same grid to 0.07% on 69x49 and to
0.03% from 137x97 on, the drag to 0.2%. The density residual falls to round-off, an L2 norm of about
$10^{-16}$, in 100 (35x25) to 220 (137x97) iterations and 450 on 273x193. On
545x385 the linear solves stop at GMRES's 100 iterations above their tolerance
and the residual falls a decade per 100 iterations, to $4 \cdot 10^{-14}$ in 500.
The cl script's 250 iterations leave these two at $10^{-13}$ and
$3 \cdot 10^{-12}$, with the coefficients converged to the digits shown. A fully
prescribed inflow instead of TMR's raises $C_f$ by 0.45%: the boundaries matter
at this level.

### 7. Cylinder at Re 100

Laminar vortex shedding behind a circular cylinder: Re 100 on the diameter,
M 0.2, T 300 K (`configs/cylinder_re100.cfg`: second order unlimited, dual time
stepping, BDF2). The mesh is a periodic O-grid (`cylinder` preprocessing):
`ntheta` by `nr` points out to 30 diameters, the first off the wall at `FIRST`,
one cell of 0.1 diameters across the span.

```bash
BIN=build/nssolver_op2_cuda GRID="192 128" FIRST=0.0025 DT=1.75e-3 TRIGGER=300 STEPS=1200 \
  scripts/cylinder/run.sh --solve-config configs/solve.json --inner-iterations 30
GRID="192 128" FIRST=0.0025 DT=1.75e-3 TRIGGER=300 STEPS=1200 sbatch scripts/cl/cylinder.sbatch
```

The run starts with a 1% cross flow for `TRIGGER` steps to break the symmetry,
then runs `STEPS` steps in the plain free stream from it, and prints the
Strouhal number, mean drag and lift amplitude over the last five periods
(`scripts/cylinder/forces.py`). The period is about 0.0873, so `DT` 1.75e-3 is
50 steps a period.

| Grid | Steps a period | St | $\bar C_D$ | $C_L'$ |
|---|---|---|---|---|
| 96x64 (`FIRST` 0.005) | 50 | 0.1547 | 1.335 | 0.311 |
| 192x128 | 50 | 0.1626 | 1.358 | 0.324 |
| 192x128 | 100 | 0.1633 | 1.359 | 0.326 |
| 384x256 (`FIRST` 0.00125) | 50 | 0.1638 | 1.362 | 0.326 |
| Williamson (1989), Park et al. (1998) | | 0.164-0.166 | 1.33-1.37 | 0.32-0.34 |

On 192x128 a step takes about 7 inner iterations to `inner_tolerance` 1e-3, none
more than 20. Before the boundary gradients were made exact for linear fields
(`OP2_SOLVER.md` section 5), a spurious spanwise velocity at the outflow kicked
the inner iterations: some steps stopped at the 30-iteration cap, and one run in
five blew up. `forces.py` reports a diverged history instead of its periods.

### 8. 18% circular arc

The biconvex circular-arc section of McDevitt et al. (1976): self-excited shock
oscillations at zero incidence, M 0.76, Re 11e6, T 300 K, SA
(`configs/circular_arc.cfg`: van Albada limiter, `update_limit` 0.2, dual time
stepping). The O-grid has 201 points around the section and 129 out from it
(`nssolver_preprocess_op2_helper circular_arc meshes-op2/arc/arc18.h5
surface_points=201 normal_points=129`; `scripts/cl/airfoil.sbatch` makes it).

```bash
CASE=circular_arc MACH=0.76 ALPHA=0 DT=1.9e-4 STEPS=2000 BIN=build/nssolver_op2_cuda \
  scripts/airfoil/run.sh --solve-config configs/solve.json
CASE=circular_arc MACH=0.76 ALPHA=0 DT=1.9e-4 STEPS=2000 sbatch scripts/cl/airfoil.sbatch
```

`scripts/airfoil/run.sh` sets the free stream from Mach, Reynolds number,
temperature and incidence (`scripts/airfoil/freestream.py`), starts from it
uniformly, and prints the Strouhal number and the lift and drag (mean and
amplitude) over the last four periods (`scripts/airfoil/forces.py`). 2000 steps
of 1.9e-4 s are 15.8 periods of 127 steps; the oscillation sets in by itself.

| | St | $C_L$ | $C_D$ |
|---|---|---|---|
| Current numerics (RTX 4050 laptop GPU) | 0.1588 | $\pm 0.3281$ | $0.0670 \pm 0.0085$ |
| Before exact boundary gradients (V100) | 0.1574 | $\pm 0.3283$ | $0.0667 \pm 0.0089$ |
| McDevitt et al., reduced frequency 0.49 | 0.156 | | |

A step takes 12.5 inner iterations to `inner_tolerance` 1e-3 on average, none
more than 18. Before the boundary gradients were made exact for linear fields
(`OP2_SOLVER.md` section 5), 143 of the 2000 steps stopped at the 30-iteration
cap, and the frequency was 0.9% lower.

### 9. OAT15A transonic buffet

The ONERA OAT15A supercritical section (Jacquin et al. 2009), test case 1 of the
AIAA DPW-8/AePW-4 buffet working group: M 0.73, Re 3e6 on the chord, T 271 K,
buffet from about 3.1-3.25 degrees at 68.75 Hz on the 0.23 m model (Strouhal
number 0.066). `configs/oat15a.cfg`: SA with the compressibility correction
(SA-comp, `OP2_SOLVER.md` section 12), van Albada limiter, `update_limit` 0.2.

**Data.** The grid (S. Deck's committee grid) and the experiment's mean and RMS
pressure come from the working group's site,
`dpw.larc.nasa.gov/DPW8/ONERA_OAT15A/` (`ONERA-Deck_Grids.REV00/Deck-ONERA-OAT15A.zip`,
`Experimental_Data/onera-oat15a.zip` with `Cp.dat` and `Cp_rms.dat`). The
geometry is ONERA's intellectual property and stays outside this repository (on
`cl` in `~/op2sol/data/oat15a`). `scripts/cl/airfoil.sbatch` preprocesses it to
unit chord, one cell across a span of 0.01 (799104 nodes, 399552 a plane):

```bash
nssolver_preprocess_op2_helper plot3d_blocks meshes-op2/oat15a/deck.h5 grid_OAT15A_Deck.x \
  scale=4.3478260869565215 wall_x0=-0.01 wall_x1=1.01 wall_y=0.2
```

**Runs.** One unit of time, c/U, is 4.15 ms; a buffet period is about 14.2 c/U.
All through `scripts/cl/airfoil.sbatch` on a V100 (`ARCH=70 sbatch -w wald`),
with `EXTRA` holding the options after the common ones,
`--implicit-cfl 10 --implicit-cfl-growth 2 --implicit-cfl-max 1e5 --solve-config configs/solve_dual.json`:

1. A first-order steady start, `ALPHA=3.9 ITERATIONS=600 TAG=oat15a_a3.9_first`,
   `EXTRA="--second-order 0 --implicit-cfl 10 --implicit-cfl-growth 1.05 --implicit-cfl-max 200"`.
   A second-order steady run diverges at buffet incidences: there is no steady state.
2. Dual time from it at `DT=1e-3` (0.24 c/U, 59 steps a period), `--inner-iterations 20`,
   until the cycle has settled: 3 to 6 periods (300-400 steps a run, chained by `RESTART`).
   A new incidence restarts from a settled cycle at a nearby one.
3. Statistics at `DT=2.5e-4` (236 steps a period), `STEPS=800` (3.4 periods),
   `--inner-iterations 30 --snapshot-interval 10`.

The runs reported here began with standard SA at 3.9 degrees (items 1 and 2 with
400 steps, then 800 steps at 2.5e-4), switched to SA-comp at 3.9 and 3.5 degrees for 300 steps at
1e-3, continued 3.5 degrees for 300 more, and started 3.25 and 3.0 degrees (400
steps) from that. A step at 1e-3 takes about 27 s; at 2.5e-4 a run makes about
210 steps an hour.

**Post-processing.**

```bash
python3 scripts/airfoil/forces.py outputs-op2/oat15a/TAG.history.csv 0.73 3e6 271 ALPHA 0.01
CASE=oat15a TAG=TAG sbatch scripts/cl/wall_series.sbatch     # each snapshot's wall state
python3 scripts/airfoil/cpstats.py outputs-op2/oat15a/TAG.history.csv outputs-op2/oat15a/wall TAG \
  0.73 3e6 271 ALPHA 2.5e-4 Cp.dat Cp_rms.dat cp_TAG.png
```

`cpstats.py` averages over whole periods, between the first and last lift
maxima the snapshots cover, and prints the lift's mean and swing, the Strouhal
number, the shock's travel (the steepest pressure rise on the upper surface)
and the RMS peak, with the experiment's.

| Model, α | $\Delta C_L$ | St | $\bar C_L$ | Shock travel, x/c | Peak $C_p'$ (x/c) | Experiment's peak |
|---|---|---|---|---|---|---|
| SA, 3.9-4.5° | decays, 0.86 a period | 0.074-0.075 | | | | |
| SA, 5.5° | steady | | 0.967 | | | |
| SA-comp, 3.0° | decays to steady | | | | | |
| SA-comp, 3.25° | 0.105 | | | | | 0.22 |
| SA-comp, 3.5° | 0.220 | 0.0705 | 0.937 | 0.385-0.546 | 0.359 (0.50) | 0.31 (0.45) |
| SA-comp, 3.9° | 0.317 | 0.0708 | 0.925 | 0.31-0.55 | 0.391 (0.49) | 0.336 (0.45) |

Standard SA damps the shock's oscillation at every incidence tried, as most
URANS-SA studies of this case report; SA-comp sustains it. $\Delta C_L^2$ grows
linearly with incidence: through 3.25 and 3.5 degrees it puts the onset at 3.18
degrees (experiment 3.1-3.25) and predicts 0.329 at 3.9 (0.317 run). The cycle
at 3.9 degrees is the same at 1e-3 and 2.5e-4 ($\Delta C_L$ 0.316 and 0.317).
The lower surface and the leading-edge plateau match the experiment's mean
pressure; the shock's pressure fluctuation peaks about 15% above the
experiment's and 0.04-0.05 c further aft, and the mean pressure rise lies 0.03-0.05 c aft of
the experiment's.

These runs predate exact boundary gradients (`OP2_SOLVER.md` section 5). With them, 300
more steps at 1e-3 from the 3.5-degree cycle give the same cycle, $\Delta C_L$ 0.219 against 0.216
the 300 steps before, St 0.0704 and $\bar C_L$ 0.937 against 0.939, with 15% fewer linear
iterations; 152 rather than 239 of the steps end at the 20 inner iterations.

**Video.** `scripts/airfoil/video.py` renders a dual-time run's snapshots as a
video of the lower symmetry plane (its boundary quads are the section's cells),
interpolated linearly onto the frame's pixels: the Mach number with the sonic
line, numerical schlieren or $C_p$, above the lift's history.

```bash
CASE=oat15a TAG=TAG ALPHA=3.5 sbatch scripts/cl/video.sbatch        # FIELD=schlieren|cp, EXTRA="--stills 220,344"
```

A smooth video needs about 100 snapshots a period (`--snapshot-interval 2` at
2.5e-4): 480 steps from the 3.5-degree statistics run took 2 h 15 min on a V100
and wrote 240 snapshots of 77 MB, which render in 2 minutes on 16 CPU cores. The job
first runs the script's `check`, which on 15 snapshots of a 60-step run gives:

- an exact linear field interpolates to $10^{-15}$;
- the far field, beyond 63 chords, holds M 0.730 and $C_p$ within $\pm 0.004$;
- the two symmetry planes agree to $2.5\cdot10^{-4}$ of the free stream's pressure, $2.4\cdot10^{-3}$
  of its speed and $3.7\cdot10^{-3}$ of its density, and their nodes carry up to $1.2\cdot10^{-4}$ of
  its speed through them (`OP2_SOLVER.md` section 8): one plane is the section;
- the history's lift less the snapshots' pressure lift varies by $2\cdot10^{-5}$ at their own steps
  (its mean, $1.5\cdot10^{-4}$, is the viscous lift), against $10^{-3}$ a step off: each frame
  shows its step.

Before the boundary gradients were made exact for linear fields (`OP2_SOLVER.md` section 5),
the planes carried up to 0.6% of the speed through them and the shock's pressure differed by
0.8% between them.

### 10. ONERA M6

The ONERA M6 wing in inviscid transonic flow, SU2's tutorial case: M 0.8395,
3.06 degrees, SU2's 108k-node tetrahedral mesh (`configs/onera_m6.cfg`: second
order with the van Albada limiter, implicit, CFL 10 growing by 1.3 to 1e4, 150
iterations). Its upper surface carries the lambda shock: two shocks inboard that
merge outboard. The experiment is Schmitt and Charpin's (AGARD AR-138, test
2308, Re 11.7e6), surface pressure at seven span stations.

```bash
BIN=build/nssolver_op2_cuda scripts/run_onera_m6_validation.sh --solve-config configs/solve.json
```

The script downloads SU2's mesh and configuration and the experiment's sections
(NASA's NPARC archive), preprocesses the mesh, runs it, writes the wall forces
and $C_p$ (`nssolver_onera_m6_validation_helper`, which also cuts the wall at
the experiment's stations) and compares the sections with the experiment
(`scripts/onera_m6_sections.py`, with a plot where matplotlib is installed):
`docs/ONERA_M6_VALIDATION.md`.

The density residual falls to round-off ($3 \cdot 10^{-14}$, eleven orders) in 150
iterations, 2 minutes on a laptop GPU. $C_L$ 0.288 and $C_D$ 0.0132 on the
wing's planform area (0.759); the $C_p$ RMS difference from the experiment on
each side of each section:

| y/b | 0.20 | 0.44 | 0.65 | 0.80 | 0.90 | 0.96 | 0.99 |
|---|---|---|---|---|---|---|---|
| upper | 0.104 | 0.102 | 0.076 | 0.076 | 0.115 | 0.070 | 0.114 |
| lower | 0.055 | 0.041 | 0.026 | 0.028 | 0.033 | 0.025 | 0.034 |

The leading-edge suction peaks, the lower surface and the shocks' pattern match;
the upper-surface differences are the shocks' positions: the computed shocks sit
a few percent of the chord aft of the experiment's and are sharper: without a
boundary layer, nothing thickens their foot. At y/b 0.99 the experiment's aft
upper surface carries more suction than the computation's: the tip vortex's.

## Manual workflow

### Build helpers

```bash
./scripts/build_helpers.sh
```

This compiles the local helper executables under `./.helpers/bin` without using CMake.

### Preprocess a mesh

```bash
./scripts/preprocess_mesh.sh box meshes-op2/box.h5
./scripts/preprocess_mesh.sh bump meshes-op2/bump.h5
./scripts/preprocess_mesh.sh flatplate meshes-op2/flatplate.h5
./scripts/preprocess_mesh.sh hydra meshes-op2/hydra.h5 meshes/hydra.jm70.grid.1.hdf
```

### Build and run OP2

```bash
make config
make seq
./nssolver_op2_seq --config configs/box.cfg
./nssolver_op2_seq --config configs/bump.cfg
./nssolver_op2_seq --config configs/flatplate_develop.cfg
./nssolver_op2_seq --config configs/hydra_benchmark.cfg
```

For alternate backends, build `make openmp` or `make cuda` and run the matching
binary. The validation scripts also support backend selection through
`OP2_TARGET`, for example:

```bash
OP2_TARGET=openmp bash tests/test_smoke.sh
OP2_TARGET=cuda bash tests/test_consistency.sh
OP2_TARGET=seq bash scripts/run_flatplate_validation.sh
```

### Convert solution to VTK

```bash
bash scripts/hdf5_to_vtk.sh meshes-op2/box.h5 outputs-op2/box_solution.h5 outputs-op2/box_solution.vtk
```

## Comparison thresholds

The helper scripts currently use pragmatic engineering thresholds:

- `box` final `L2(rho)` must remain near zero
- OP2 bump final `L2(rho)` target is about `6.97872e-04`
- flat-plate final `L2(rho)` target is about `4.07298e-08`

These thresholds should only be tightened after solver changes are validated over multiple runs.

The targets are residual levels after the configs' fixed iteration counts,
not converged states. They moved when the control volumes were closed
(median-dual edges, per-node boundary sub-faces and fluxes) and the
components the viscous scheme enforces stopped carrying a residual: the
flat plate's `L2(rho)` is now that of its free nodes only. Both cases now
reach a steady state: RK4 keeps reducing the residual (bump: `5.6e-06` after
30000 iterations, where the earlier numerics diverged), and the implicit
scheme (`--time-scheme implicit`, `OP2_SOLVER.md` section 17) converges both
to round-off in 20-35 iterations.
