# OP2 Solver

This directory contains the OP2-based version of the solver workflow.

Detailed mathematics and implementation notes live in:

- `docs/OP2_SOLVER.md`
- `docs/VALIDATION.md`

## Scope

The OP2 application ports the main non-periodic solver path to OP2 sets, maps, dats, and C-style kernels.

What is included:

- OP2 sequential solver driver in `nssolver_op2.cpp`
- Edge-local second-order MUSCL reconstruction
- Thin-layer viscous fluxes
- Spalart-Allmaras source term support
- HDF5 mesh preprocessing using the existing C++20 mesh code
- HDF5 solution output from the OP2 app
- Separate HDF5-to-VTK conversion step
- Configs for `box`, `bump`, `hydra`, `hydra_benchmark`, and `flatplate_develop`

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

## Formula-Driven Preprocessing Meshes

The preprocessing helper supports procedural mesh generation with explicit size
and shape overrides via `key=value` arguments. `bump3d` and
`axisymmetric_body` now default to the stronger parameter sets that gave the
clearest compression and turning features in the initial solver runs, so the
plain case names are the recommended starting points.

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

Reference solver setups for the documented defaults:

- [configs/bump3d.cfg](/home/ireguly/nssolver-op2/configs/bump3d.cfg): stronger inviscid bump-channel run using `meshes-op2/bump3d.h5`
- [configs/axisymmetric_body.cfg](/home/ireguly/nssolver-op2/configs/axisymmetric_body.cfg): stronger inviscid body-of-revolution run using `meshes-op2/axisymmetric_body.h5`
- [configs/naca_wing.cfg](/home/ireguly/nssolver-op2/configs/naca_wing.cfg): baseline finite-wing run using `meshes-op2/naca_wing.h5`

## Validation

OP2-local scripts are provided for smoke, consistency, and flat-plate benchmark checks:

```bash
bash tests/test_smoke.sh
bash tests/test_consistency.sh
bash scripts/run_flatplate_validation.sh
```

The flat-plate script is expected to pass; the OP2 and reference implementations now match in `Cp`, `Cf`, and sampled velocity profiles.

## HDF5 Mesh Datasets

The OP2 app expects these datasets:

- `node_coordinates` `[N,3]`
- `node_volume` `[N,1]`
- `node_wall_distance` `[N,1]`
- `edge-->node` `[E,2]`
- `edge_weights` `[E,3]`
- `bface-->node` `[F,4]`
- `bface_normal` `[F,3]`
- `bface_area` `[F,1]`
- `bface_group` `[F,1]`
- `bface_type` `[F,1]`

Connectivity is written zero-based.

## Output

The OP2 app writes HDF5 solution datasets:

- `q`
- `primitive`
- `dt`

These are written into the configured solution HDF5 file and can be visualized through the converter step.
