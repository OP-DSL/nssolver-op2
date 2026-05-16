# OP2 Validation Guide

## Overview

This directory contains OP2-local smoke and consistency workflows. They assume the shared repository layout:

- OP2 app build in `./`
- preprocessed meshes in `meshes-op2`
- OP2 outputs in `outputs-op2`

The OP2 app does not generate meshes directly. Meshes are produced by the OP2-local wrapper:

```bash
./scripts/preprocess_mesh.sh
```

## Scripts

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
  `4.9e-03 <= L2(rho) <= 5.1e-03`

### 2. Consistency check against the original solver

```bash
bash tests/test_consistency.sh
```

This runs:

- original `box`
- original `bump`
- OP2 `box`
- OP2 `bump`

and compares the final `L2(rho)` residual levels.

Expected behavior:

- `box`: both solvers near machine zero
- `bump`: original solver final `L2(rho)` remains in
  `5.8e-03 <= L2(rho) <= 6.1e-03`
- `bump`: OP2 final `L2(rho)` remains in
  `4.9e-03 <= L2(rho) <= 5.1e-03`
- `bump`: OP2 and original remain within the configured relative tolerance

### 3. Flat-plate viscous benchmark comparison

```bash
bash scripts/run_flatplate_validation.sh
```

This is a stricter comparison. It runs the original solver and OP2 on `flatplate_develop`, postprocesses the OP2 result into the same CSV benchmark format, and compares:

- wall skin friction `Cf`
- wall pressure coefficient `Cp`
- normalized velocity profiles at 20%, 50%, and 80% of plate length

Current expected status:

- velocity profiles: matched
- `Cf`: matched
- `Cp`: matched
- final `L2(rho)` for both local reference and OP2:
  `2.3e-03 <= L2(rho) <= 2.4e-03`

The script should pass when the OP2 and reference implementations remain aligned.

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
- local bump final `L2(rho)` target is about `5.95101e-03`
- OP2 bump final `L2(rho)` target is about `4.99452e-03`
- flat-plate final `L2(rho)` target is about `2.34059e-03`
- flat-plate profile comparisons are strict
- flat-plate `Cp` should remain matched to the reference solver

These thresholds should only be tightened after solver changes are validated over multiple runs.
