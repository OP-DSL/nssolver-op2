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
