# OP2 Solver

This directory contains the OP2-based version of the solver workflow.

Detailed mathematics and implementation notes live in:

- `docs/OP2_SOLVER.md`
- `docs/VALIDATION.md`

## Scope

The OP2 application ports the main non-periodic solver path to OP2 sets, maps, dats, and C-style kernels.

What is included (the numerics in `docs/OP2_SOLVER.md`):

- the OP2 solver driver in `nssolver_op2.cpp`, for every OP2 backend (seq, genseq, OpenMP, CUDA,
  MPI)
- HLLC fluxes, second-order MUSCL reconstruction with an edge-local or van Albada limiter
- full Navier-Stokes viscous fluxes, Spalart-Allmaras (optionally SA-comp)
- explicit RK4 and implicit (Newton-Krylov) pseudo-time stepping, and dual time stepping for
  unsteady flow
- HDF5 mesh preprocessing using the existing C++20 mesh code (Hydra, procedural, PLOT3D and
  multi-block planar grids)
- HDF5 solution output, snapshots and force histories; wall extraction, VTK conversion and
  videos of planar cases
- configs for `box`, `bump`, `hydra`, `hydra_benchmark`, `flatplate_develop`, the TMR flat
  plate, the cylinder, the circular arc and the OAT15A

What is not yet ported to OP2:

- true periodic coupling
- direct procedural mesh generation inside the OP2 app
- a full rotating-frame/turbomachinery model

The limiter path is edge-local in both the original and OP2 solvers. This avoids indirect write/reduction patterns that are a poor fit for OP2.

Periodic Hydra sector boundaries are currently expected to be preprocessed into `SlipWall`, matching the non-OP2 baseline fallback.

## Workflow

1. Build the helper tools:

```bash
./scripts/build_helpers.sh
```

2. Generate OP2-ready meshes:

```bash
./scripts/preprocess_mesh.sh box meshes-op2/box.h5
./scripts/preprocess_mesh.sh bump meshes-op2/bump.h5
./scripts/preprocess_mesh.sh hydra meshes-op2/hydra.h5 meshes/hydra.jm70.grid.1.hdf
```

3. Build the OP2 app:

```bash
make config
make seq
```

Use `make openmp` or `make cuda` to build those backends instead.

4. Build the helper tools and preprocess a mesh:

```bash
./scripts/build_helpers.sh
./scripts/preprocess_mesh.sh box meshes-op2/box.h5
```

5. Run a case:

```bash
./nssolver_op2_seq --config configs/box.cfg
./nssolver_op2_seq --config configs/hydra_benchmark.cfg
```

The runner scripts and tests can target any built backend through `OP2_TARGET`:

```bash
OP2_TARGET=seq bash tests/test_smoke.sh
OP2_TARGET=openmp bash tests/test_smoke.sh
OP2_TARGET=cuda bash tests/test_smoke.sh
```

Supported values are `seq`, `genseq`, `openmp`, and `cuda`. The default is `seq`.

6. Convert HDF5 solution output to VTK:

```bash
scripts/hdf5_to_vtk.sh meshes-op2/box.h5 outputs-op2/box_solution.h5 outputs-op2/box_solution.vtk
```

You can also drive the common flows from `make`:

```bash
make helpers-build
make preprocess-box
make smoke
make consistency
make flatplate
```

Helper binaries are built locally under `.helpers/bin`. This repository does not depend on CMake for its own workflow.

## Implicit Time Stepping

`time_scheme = implicit` (or `--time-scheme implicit`) solves backward-Euler
pseudo-time steps with a first-order Jacobian assembled into a solver library
(PETSc, hypre, Ginkgo, AmgX, rocALUTION) through OP2's solver layer, which is
on OP2's `feature/solvers-prototype` branch and built by its CMake. Build the
app with CMake against that OP2, installed:

```bash
cmake -S . -B build -DOP2_DIR=<op2-prefix>/lib/cmake/op2 && cmake --build build
OP_SOLVE_CONFIG='{"library": "petsc"}' build/nssolver_op2_seq --config configs/bump.cfg \
  --time-scheme implicit --implicit-cfl 10 --solve-config configs/solve.json --pc bjacobi
BUILD_DIR=build tests/test_implicit.sh
```

Any config key can be given on the command line as `--key value`. The scheme
and the linear solve's options (`pc`, `pmat`, `op`, `partition`, `reorder`,
...) are in `docs/OP2_SOLVER.md`, section 17; dual time stepping for unsteady
flow (`--time-step`, `--physical-steps`) in section 18, and the verification
runs' results (TMR flat plate, cylinder, circular arc, OAT15A buffet) in section
19; `docs/VALIDATION.md` describes how to run them.

A planar dual-time run's snapshots (`--snapshot-interval N`) render as a
video of a symmetry plane, after a sanity check of the snapshots:

```bash
python3 scripts/airfoil/video.py check MESH.h5 OUTDIR TAG OUTDIR/TAG.history.csv MACH RE T ALPHA
python3 scripts/airfoil/video.py render MESH.h5 OUTDIR TAG OUTDIR/TAG.history.csv MACH RE T ALPHA out.mp4
```

## Formula-Driven Preprocessing Meshes

The preprocessing helper supports procedural mesh generation with explicit size
and shape overrides via `key=value` arguments. `bump3d` and
`axisymmetric_body` default to parameter sets with clear compression and
turning features, so the plain case names are the recommended starting points.

```bash
bash scripts/preprocess_mesh.sh bump3d meshes-op2/bump3d.h5
bash scripts/preprocess_mesh.sh axisymmetric_body meshes-op2/axisymmetric_body.h5
bash scripts/preprocess_mesh.sh naca_wing meshes-op2/naca_wing.h5
```

Available procedural cases:

- `bump3d`: 3D localized hill in a structured channel. Primary controls:
  `nx`, `ny`, `nz`, `lx`, `ly`, `lz`, `bump_center_x`, `bump_half_width_x`,
  `bump_center_z`, `bump_half_width_z`, `bump_height`. Default preset:
  `61x29x31`, `lx=2.4`, `ly=0.55`, `lz=0.8`, `bump_center_x=1.1`,
  `bump_half_width_x=0.35`, `bump_center_z=0.4`, `bump_half_width_z=0.16`,
  `bump_height=0.14`.
- `axisymmetric_body`: wedge-sector body-of-revolution mesh. Primary controls:
  `nx`, `nr`, `ntheta`, `lx`, `body_start_x`, `body_length`,
  `body_radius_max`, `body_tail_radius`, `farfield_radius`,
  `wedge_angle_degrees`, `radial_growth`. Default preset: `101x29x15`,
  `lx=2.6`, `body_start_x=0.35`, `body_length=1.7`, `body_radius_max=0.24`,
  `body_tail_radius=0.02`, `farfield_radius=0.5`,
  `wedge_angle_degrees=18`, `radial_growth=3.5`.
- `naca_wing`: structured farfield-over-wing mesh using a spanwise-tapered,
  swept NACA 4-digit thickness/camber law on the lower boundary. Primary
  controls: `nx`, `ny`, `nz`, `lx`, `ly`, `lz`, `wing_origin_x`,
  `wing_center_z`, `span`, `root_chord`, `tip_chord`, `sweep_degrees`,
  `thickness_ratio`, `camber_ratio`, `camber_position`, `vertical_offset`,
  `wall_normal_growth`. Default preset: `121x41x81`, `lx=2.8`, `ly=0.9`,
  `lz=1.6`, `wing_origin_x=0.45`, `wing_center_z=0.8`, `span=1.0`,
  `root_chord=0.8`, `tip_chord=0.4`, `sweep_degrees=20`,
  `thickness_ratio=0.12`, `camber_ratio=0.0`, `camber_position=0.4`,
  `vertical_offset=0.0`, `wall_normal_growth=4.0`.

OP2 solver setups for the documented defaults:

- [configs/bump3d.cfg](configs/bump3d.cfg): inviscid bump-channel run using `meshes-op2/bump3d.h5`
- [configs/axisymmetric_body.cfg](configs/axisymmetric_body.cfg): inviscid body-of-revolution run using `meshes-op2/axisymmetric_body.h5`
- [configs/naca_wing.cfg](configs/naca_wing.cfg): baseline finite-wing run using `meshes-op2/naca_wing.h5`

## Validation

OP2-local scripts are provided for smoke, consistency, and flat-plate benchmark checks:

```bash
bash tests/test_smoke.sh
bash tests/test_consistency.sh
bash scripts/run_flatplate_validation.sh
```

The flat-plate script checks the OP2 residual target and verifies that wall/profile benchmark CSV outputs are produced.

## Numerical Stability Notes

Recent stability work focused on making the OP2 path insensitive to ordinary
debug versus optimized compiler choices. The `bump` case was the motivating
regression: `DEBUG=1 make genseq` and an optimized `make genseq` previously
produced visibly different HDF5 solutions.

The OP2 kernels now use a guarded HLLC flux path:

- zero-area faces return zero flux
- pure supersonic left/right states return the corresponding physical flux
- near-singular HLLC denominators fall back to HLL
- non-finite or out-of-wavefan contact speeds fall back to HLL
- near-zero contact speeds blend the star-left and star-right fluxes
  linearly, which keeps the branch insensitive to round-off around symmetric
  states without the jump an HLL switch put into the residual
- invalid HLL denominators fall back to Rusanov flux

The RK and implicit updates also normalize tiny conserved quantities:

- momentum components below `1.0e-12` of the node's largest component are set to zero
- transported turbulence variable values below `1.0e-14 * rho` are set to zero

This cleanup removes roundoff-scale transverse momentum and turbulence traces
that can otherwise be amplified differently by optimized and debug builds. The
momentum floor is relative to the node's own momentum: an absolute one
(`1.0e-7 * rho` before) also zeroed the real wall-normal velocity of nodes next
to a wall or symmetry plane, which held the TMR flat plate's residual at
`1e-12` instead of round-off. The current regression expectation is that the
optimized and debug `genseq` bump solutions compare cleanly with:

```bash
h5diff -p 1e-6 -v debug_bump_solution.h5 opt_bump_solution.h5
```

The validation targets in `docs/VALIDATION.md` were updated after these changes.

## HDF5 Mesh Datasets and Output

The mesh datasets the OP2 app reads (zero-based connectivity) are listed in
`docs/OP2_SOLVER.md`, section 14; its outputs (solution, residual and force
histories, snapshots) and the post-processing tools in section 15.
