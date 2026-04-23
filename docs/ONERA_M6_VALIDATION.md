# ONERA M6 Validation Case

This case uses the public SU2 ONERA M6 inviscid tutorial mesh and freestream setup:

- Tutorial: https://su2code.github.io/tutorials/Inviscid_ONERAM6/
- Source mesh: `public-data/onera_m6/mesh_ONERAM6_inv_ffd.su2`
- Source config: `public-data/onera_m6/inv_ONERAM6.cfg`

## Download and conversion

Bootstrap the public tutorial assets with:

```bash
./scripts/download_onera_m6_su2_case.sh
```

This script downloads:

- `inv_ONERAM6.cfg` from the public `su2code/Tutorials` repository
- the `su2code/Tutorials` source archive from GitHub
- `mesh_ONERAM6_inv_ffd.su2` extracted from that archive

The files are stored under:

- `public-data/onera_m6/inv_ONERAM6.cfg`
- `public-data/onera_m6/mesh_ONERAM6_inv_ffd.su2`

Convert the SU2 mesh to the OP2 HDF5 mesh format with:

```bash
./scripts/preprocess_mesh.sh onera_m6 meshes-op2/onera_m6.h5 public-data/onera_m6/mesh_ONERAM6_inv_ffd.su2
```

Both `run_onera_m6_validation.sh` and `compare_onera_m6_orders.sh` call the download script automatically, so a fresh checkout does not need a manual data-prep step.

## What this case validates

This is a regression and setup validation case for the current solver and mesh importer. It validates:

- SU2 tetra mesh import into the OP2 HDF5 mesh format
- triangular boundary-face support in preprocessing and OP2 kernels
- correct mapping of the ONERA M6 wall, farfield, and symmetry markers
- stable inviscid advancement on the imported mesh
- reproducible wing force and surface `Cp` postprocessing

It is not yet a claim of agreement with SU2's final converged solution or with the Schmitt-Carpin experimental dataset. The current solver path differs materially from the SU2 tutorial:

- explicit local-time stepping instead of SU2 implicit multigrid
- HLLC edge fluxes instead of SU2 JST-centered dissipation
- first-order baseline run (`second_order = 0`)

Use this case as a numerical baseline for importer and solver changes. Do not treat the 100-iteration coefficients below as published aerodynamic truth.

## Imported mesh expectations

The imported OP2 mesh generated from the official SU2 file should have:

- nodes: `108396`
- tetra elements in source SU2 mesh: `582752`
- OP2 dual edges: `710525`
- boundary faces: `38756`

The SU2 marker order in the source mesh is:

1. `LOWER_SIDE`
2. `TIP`
3. `UPPER_SIDE`
4. `XNORMAL_FACES`
5. `ZNORMAL_FACES`
6. `YNORMAL_FACE`
7. `SYMMETRY_FACE`

The validation helper uses wall groups `1,2,3`.

## Freestream state

The dimensional freestream is derived from the SU2 tutorial inputs:

- `Mach = 0.8395`
- `AOA = 3.06 deg`
- `p_inf = 101325 Pa`
- `T_inf = 288.15 K`
- `gamma = 1.4`
- `R = 287.05 J/(kg K)`

Derived values used in the config and validation helper:

- `rho_inf = 1.2250122659906946 kg/m^3`
- `u_inf = 285.26805348028574 m/s`
- `v_inf = 0.0 m/s`
- `w_inf = 15.249834230540356 m/s`

## Running the case

```bash
./scripts/run_onera_m6_validation.sh
```

This will:

1. preprocess the public SU2 mesh into `meshes-op2/onera_m6.h5` if needed
2. run `nssolver_op2_seq` with `configs/onera_m6.cfg`
3. write a coefficient summary JSON
4. write a surface `Cp` CSV
5. export a VTK file for visualization

Outputs:

- `outputs-op2/onera_m6_solution.h5`
- `outputs-op2/onera_m6_solution.residual.csv`
- `outputs-op2/onera_m6_summary.json`
- `outputs-op2/onera_m6_surface_cp.csv`
- `outputs-op2/onera_m6_solution.vtk`

## Current baseline

With `configs/onera_m6.cfg`:

- iterations: `100`
- CFL: `0.01`
- inviscid
- first-order

Observed baseline after 100 iterations:

- `L2(rho) / L2(rho)_initial = 0.664117`
- `reference_area = 0.758799`
- `CD = 0.0144481`
- `CL = 0.0149413`
- `CX = 0.01363`
- `CZ = 0.0156912`

These numbers should be used as a regression check for this exact solver configuration. If they move substantially after a code change, inspect:

- SU2 import topology
- boundary face node counts
- wall/farfield/symmetry marker mapping
- boundary flux assembly on triangular faces

## Second-order comparison

A stable second-order configuration is provided in:

- `configs/onera_m6_second_order.cfg`

This uses:

- `second_order = 1`
- `iterations = 100`
- `cfl = 0.005`

Run the side-by-side comparison with:

```bash
./scripts/compare_onera_m6_orders.sh
```

This writes:

- `outputs-op2/onera_m6_second_order_solution.h5`
- `outputs-op2/onera_m6_second_order_summary.json`
- `outputs-op2/onera_m6_second_order_surface_cp.csv`
- `outputs-op2/onera_m6_order_comparison.json`

Current comparison at equal iteration count (`100`) shows:

- first-order:
  - `CD = 0.0144481`
  - `CL = 0.0149413`
  - global `Cp min/max = -0.456643 / 0.846524`
- second-order:
  - `CD = 0.0108547`
  - `CL = 0.0118057`
  - global `Cp min/max = -0.244485 / 0.58331`

Interpretation:

- both runs are stable and generate positive lift and drag
- both show upper-surface suction, so the imported ONERA geometry and marker mapping are behaving physically
- the second-order run is presently more conservative than the first-order baseline at the same iteration count
- relative to the SU2 tutorial, the current solver still under-resolves the stronger transonic pressure signature expected from the ONERA M6 lambda-shock pattern

That last point is expected. The SU2 tutorial uses implicit multigrid and JST-centered numerics, while this solver path is still explicit edge-based HLLC. The comparison script is therefore a numerical trend check, not a code-to-code equivalence claim.
