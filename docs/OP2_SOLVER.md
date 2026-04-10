# OP2 Compressible Navier-Stokes Solver

## 1. Scope

This document describes the mathematics, data layout, execution model, and validation workflow of the OP2 implementation in this directory.

The OP2 solver is a node-centered, edge-based finite-volume solver for the steady compressible Navier-Stokes equations with optional Spalart-Allmaras turbulence transport. The high-level driver remains C++, while kernels are written in a constrained C-style suitable for OP2 backends.

The current OP2 implementation supports:

- 3D node-centered edge-based control volumes
- HLLC inviscid fluxes
- Green-Gauss primitive gradients
- edge-local second-order MUSCL reconstruction
- thin-layer viscous fluxes
- explicit four-stage Runge-Kutta marching with local time stepping
- Spalart-Allmaras source-term update
- Hydra-derived HDF5 meshes after preprocessing

It does not support:

- periodic coupling across mesh sector cuts
- rotating-frame source terms
- multigrid
- direct procedural mesh generation inside the OP2 executable

Hydra periodic cuts are intentionally preprocessed as slip walls in the current benchmark path.

## 2. Governing Equations

The solver advances the steady compressible RANS equations in pseudo-time:

\[
\frac{\partial}{\partial t}\int_{V_i} \mathbf{Q}\,dV
+
\int_{\partial V_i}\left(\mathbf{F}_{inv}-\mathbf{F}_{visc}\right)\cdot d\mathbf{S}
=
\int_{V_i}\mathbf{D}_{SA}\,dV
\]

The conservative state is

\[
\mathbf{Q}=
\begin{bmatrix}
\rho \\
\rho u \\
\rho v \\
\rho w \\
\rho E \\
\rho \tilde{\nu}
\end{bmatrix}
\]

and the primitive state is

\[
\mathbf{W}=
\begin{bmatrix}
\rho \\
u \\
v \\
w \\
p \\
\tilde{\nu}
\end{bmatrix}.
\]

For a perfect gas,

\[
p=(\gamma-1)\rho\left(E-\frac{1}{2}(u^2+v^2+w^2)\right),
\qquad
a=\sqrt{\gamma p/\rho}.
\]

The OP2 implementation stores and updates the conservative state, then reconstructs the primitive state before gradient, flux, and source-term evaluation.

## 3. Edge-Based Finite-Volume Discretization

For node \(i\), the semi-discrete residual is

\[
\mathbf{R}_i
=
\sum_{j\in N(i)}
\left(\mathbf{F}^{inv}_{ij}-\mathbf{F}^{visc}_{ij}\right)
\cdot \Delta\mathbf{S}_{ij}

+
\sum_{f\in \partial \Omega_i}
\left(\mathbf{F}^{inv}_{f}-\mathbf{F}^{visc}_{f}\right)\cdot \Delta\mathbf{S}_{f}

-
V_i\mathbf{D}_{SA,i}.
\]

Here:

- \(V_i\) is the nodal dual volume
- \(\Delta \mathbf{S}_{ij}\) is the directed dual-face area vector associated with edge \(i\to j\)
- \(\Delta \mathbf{S}_f\) is the boundary-face area vector

The solver uses OP2 edge loops for interior fluxes and OP2 boundary-face loops for boundary fluxes.

## 4. Primitive/Conservative Conversion

Given primitive state \(\mathbf{W}\), the conservative state is

\[
\rho u = \rho u,\quad
\rho v = \rho v,\quad
\rho w = \rho w,
\]

\[
\rho E = \rho \left(\frac{p}{(\gamma-1)\rho} + \frac{1}{2}(u^2+v^2+w^2)\right),
\qquad
\rho \tilde{\nu} = \rho \tilde{\nu}.
\]

The inverse conversion is performed at every Runge-Kutta stage before flux assembly.

Implementation:

- `initialize_q_kernel`
- `q_to_primitive_kernel`
- `primitive_to_conservative_op2`
- `conservative_to_primitive_op2`

## 5. Gradient Reconstruction

Primitive gradients are computed with a Green-Gauss approximation over the nodal dual control volume:

\[
\nabla \mathbf{W}_i
=
\frac{1}{V_i}
\left[
\sum_{j\in N(i)} \frac{1}{2}(\mathbf{W}_i+\mathbf{W}_j)\otimes \Delta \mathbf{S}_{ij}
+
\sum_{f\in \partial \Omega_i} \mathbf{W}_f\otimes \Delta \mathbf{S}_f
\right].
\]

In the OP2 implementation:

- `edge_grad_kernel` contributes the interior edge term through indirect `OP_INC` updates to both endpoints
- `bface_grad_kernel` contributes the boundary-face term through indirect `OP_INC` updates to the four boundary nodes
- `normalize_grad_kernel` divides by nodal volume

The gradient storage is packed as 18 doubles per node:

- 3 components for each of the 6 primitive variables

The layout is:

\[
[\partial_x \rho,\partial_y \rho,\partial_z \rho,\partial_x u,\dots,\partial_z \tilde{\nu}]
\]

## 6. Second-Order MUSCL Reconstruction

For second-order runs, left and right interface states are reconstructed from node-centered primitives and gradients.

For edge \(i\leftrightarrow j\), let

\[
\Delta \mathbf{r}_{ij} = \frac{1}{2}(\mathbf{x}_j-\mathbf{x}_i),
\qquad
\Delta \mathbf{r}_{ji} = -\Delta \mathbf{r}_{ij}.
\]

The reconstructed states are

\[
\mathbf{W}_L = \mathbf{W}_i + \phi_i \left(\nabla \mathbf{W}_i \cdot \Delta \mathbf{r}_{ij}\right),
\]

\[
\mathbf{W}_R = \mathbf{W}_j + \phi_j \left(\nabla \mathbf{W}_j \cdot \Delta \mathbf{r}_{ji}\right).
\]

### 6.1 Edge-Local Limiter

The original nodal-extrema limiter was replaced by an edge-local projected-slope limiter so that the OP2 implementation does not need indirect `OP_RW` or indirect reduction access patterns for limiter assembly.

For each primitive component \(m\),

\[
\delta^{proj}_{i,m} = \nabla W_{i,m}\cdot \Delta \mathbf{r}_{ij},
\qquad
\delta^{edge}_{i,m} = W_{j,m} - W_{i,m}.
\]

The scalar limiter contribution is

\[
\phi_{i,m}=
\begin{cases}
1, & |\delta^{proj}_{i,m}| < \varepsilon \\
0, & \delta^{proj}_{i,m}\,\delta^{edge}_{i,m}\le 0 \\
\min\left(1,\frac{\delta^{edge}_{i,m}}{\delta^{proj}_{i,m}}\right), & \text{otherwise}
\end{cases}
\]

and the node-edge limiter is

\[
\phi_i = \min_m \phi_{i,m}.
\]

This limiter is directional and edge-local. It is less multidimensional than a Venkatakrishnan-style nodal extrema limiter, but it maps cleanly onto OP2 and the original solver now uses the same algorithm for consistency.

Implementation:

- `scalar_edge_limiter_op2`
- `edge_limiter_scale_op2`
- `reconstruct_primitive_op2`

## 7. Inviscid Flux: HLLC

Given reconstructed left/right states and the edge area vector \(\Delta \mathbf{S}_{ij}\), the solver computes the HLLC flux.

Let

\[
\hat{\mathbf{n}} = \frac{\Delta \mathbf{S}_{ij}}{\|\Delta \mathbf{S}_{ij}\|},
\qquad
u_n = \mathbf{u}\cdot \hat{\mathbf{n}}.
\]

Wave-speed estimates are

\[
S_L = \min(u_{n,L}-a_L, u_{n,R}-a_R),
\qquad
S_R = \max(u_{n,L}+a_L, u_{n,R}+a_R).
\]

The contact speed \(S_M\) follows the usual HLLC estimate:

\[
S_M =
\frac{
p_R-p_L + \rho_L u_{n,L}(S_L-u_{n,L}) - \rho_R u_{n,R}(S_R-u_{n,R})
}{
\rho_L(S_L-u_{n,L}) - \rho_R(S_R-u_{n,R})
}.
\]

The flux is then assembled piecewise from left, star-left, star-right, or right states.

Implementation:

- `physical_flux_op2`
- `hllc_flux_op2`
- called from `edge_flux_kernel` and `boundary_flux_kernel`

## 8. Thin-Layer Viscous Flux

The viscous path uses an edge-based thin-layer approximation. For an edge-aligned direction,

\[
\frac{\partial u}{\partial n}\approx \frac{u_R-u_L}{|\Delta \mathbf{r}|},
\qquad
\frac{\partial T}{\partial n}\approx \frac{T_R-T_L}{|\Delta \mathbf{r}|},
\qquad
\frac{\partial \tilde{\nu}}{\partial n}\approx \frac{\tilde{\nu}_R-\tilde{\nu}_L}{|\Delta \mathbf{r}|}.
\]

The laminar viscosity uses Sutherland's law:

\[
\mu(T)=\mu_{ref}\left(\frac{T}{T_{ref}}\right)^{3/2}
\frac{T_{ref}+S}{T+S}.
\]

Thermal conductivity is

\[
k = \mu \frac{\gamma c_v}{Pr}.
\]

The OP2 implementation adds eddy viscosity from the SA working variable through the standard \(f_{v1}\) relation.

The thin-layer viscous flux is then approximated from the normal derivatives of velocity, temperature, and SA variable.

Implementation:

- `dynamic_viscosity_op2`
- `thermal_conductivity_op2`
- `eddy_viscosity_op2`
- `thin_layer_viscous_flux_op2`

## 9. Boundary Conditions

Boundary faces are stored as quads and processed in a dedicated OP2 boundary-face loop.

Supported boundary types:

- `Farfield`
- `Inlet`
- `Outlet`
- `SlipWall`
- `NoSlipWall`

### 9.1 Inviscid Boundary State Construction

The solver forms a face-averaged interior primitive state \(\mathbf{W}_{int}\), then constructs a ghost state \(\mathbf{W}_{ghost}\):

- `Farfield`, `Inlet`: replace with freestream primitive
- `Outlet`: copy interior state, replace only pressure with freestream pressure
- `SlipWall`: reflect normal velocity
- `NoSlipWall`: reflect normal velocity for inviscid flux, and enforce zero wall velocity in the viscous wall state

Implementation:

- `make_boundary_ghost_state_op2`
- `boundary_flux_kernel`

### 9.2 Boundary-Node State Enforcement

For viscous runs, the OP2 implementation also enforces boundary node states before each Runge-Kutta stage:

- inlet/farfield nodes are reset to freestream
- no-slip wall nodes are forced to zero velocity
- slip-wall nodes have normal velocity removed
- outlet nodes use the configured static pressure

This is done through:

- `enforce_boundary_node_kernel`

This kernel is intentionally limited to boundary nodes and uses indirect `OP_RW` only on `q` through a boundary-node to node map.

## 10. Local Time Stepping

The pseudo-time step at node \(i\) is

\[
\Delta t_i = \text{CFL}\frac{V_i}{\lambda_i + \varepsilon}
\]

where the inviscid spectral contribution is accumulated from connected edges:

\[
\lambda_i^{inv}
=
\sum_{j\in N(i)} |\Delta S_{ij}|\left(|u_n|+a\right).
\]

For viscous runs an additional penalty is added:

\[
\lambda_i^{visc} \sim 2\mu \frac{|\Delta S_{ij}|^2}{\rho_i V_i}.
\]

Implementation:

- `edge_spectral_kernel`
- `compute_dt_scaled_kernel`

## 11. Runge-Kutta Marching

The code uses a four-stage explicit RK update. Each outer iteration performs:

1. store the current conservative state into `q0`
2. convert `q` to primitive
3. build gradients if needed
4. compute local pseudo-time step
5. for each RK stage:
   - enforce boundary-node state if needed
   - convert `q` to primitive
   - zero residual
   - accumulate interior edge fluxes
   - accumulate boundary-face fluxes
   - add SA source term
   - update `q`
6. assemble a final residual for monitoring

The update is

\[
\mathbf{Q}^{(k)}_i = \mathbf{Q}^{(0)}_i - \alpha_k \frac{\Delta t_i}{V_i}\mathbf{R}^{(k-1)}_i
\]

with stage coefficients

\[
\alpha = \left[\frac{1}{4},\frac{1}{3},\frac{1}{2},1\right].
\]

Implementation:

- `rk_update_kernel`

## 12. Spalart-Allmaras Source Term

The OP2 implementation currently includes the SA source contribution in nodal source form. The exact transport flux remains embedded in the conservative/viscous update, while the explicit source term is added in a separate node loop.

The source structure is

\[
D_{SA} = P_{SA} - D_{SA}^{wall}
\]

with wall-distance dependence and a simple vorticity magnitude surrogate built from velocity gradients.

Implementation:

- `sa_source_kernel`

This is intentionally a compact engineering implementation rather than a full production-grade SA closure with all optional compressibility or transition corrections.

## 13. OP2 Data Model

The OP2 mesh uses four sets:

- `nodes`
- `edges`
- `bfaces`
- `bnodes`

and three maps:

- `edge-->node` with arity 2
- `bface-->node` with arity 4
- `bnode-->node` with arity 1

The principal dats are:

- node geometry: `node_coordinates`, `node_volume`, `node_wall_distance`
- edge geometry: `edge_weights`
- boundary geometry: `bface_normal`, `bface_area`, `bface_type`, `bface_group`
- boundary-node metadata: `bnode_dirichlet`, `bnode_wall`, `bnode_slip`, `bnode_normal`
- solution/state: `q`, `q0`, `primitive`, `residual`, `spectral`, `dt`, `grad`

### 13.1 Access Pattern Design

The implementation deliberately uses:

- indirect `OP_INC` for edge-to-node residual and gradient accumulation
- direct writes for node-owned temporaries
- indirect `OP_RW` only where unavoidable on boundary-node state enforcement

The old nodal-limiter assembly path was removed because it required indirect `OP_RW` and indirect reductions, which are a poor fit for OP2 portability and performance.

## 14. Mesh Preprocessing Contract

The OP2 executable does not read the original internal mesh objects or raw Hydra layout directly. Instead, `nssolver_preprocess_op2` writes an OP2-oriented HDF5 mesh with:

- `node_coordinates [N,3]`
- `node_volume [N,1]`
- `node_wall_distance [N,1]`
- `edge-->node [E,2]`
- `edge_weights [E,3]`
- `bface-->node [F,4]`
- `bface_normal [F,3]`
- `bface_area [F,1]`
- `bface_group [F,1]`
- `bface_type [F,1]`
- `bnode-->node [Nb,1]`
- `bnode_dirichlet [Nb,1]`
- `bnode_wall [Nb,1]`
- `bnode_slip [Nb,1]`
- `bnode_normal [Nb,3]`

All connectivity is zero-based in the preprocessed OP2 mesh.

The local preprocessing helper also supports procedural structured meshes with
runtime parameter overrides. `bump3d` and `axisymmetric_body` now default to
the stronger geometry presets that produced the clearest visible flow features
in the initial exploratory runs.

```bash
bash scripts/preprocess_mesh.sh bump3d meshes-op2/bump3d.h5
bash scripts/preprocess_mesh.sh axisymmetric_body meshes-op2/axisymmetric_body.h5
bash scripts/preprocess_mesh.sh naca_wing meshes-op2/naca_wing.h5
```

Implemented procedural 3D generators:

- `bump3d`: lower-wall hill defined by an analytic bump law in `x` and `z`
  with default preset `61x29x31`, `lx=2.4`, `ly=0.55`, `lz=0.8`,
  `bump_center_x=1.1`, `bump_half_width_x=0.35`, `bump_center_z=0.4`,
  `bump_half_width_z=0.16`, `bump_height=0.14`
- `axisymmetric_body`: analytic body-of-revolution embedded in a wedge-sector
  farfield with default preset `101x29x15`, `lx=2.6`, `body_start_x=0.35`,
  `body_length=1.7`, `body_radius_max=0.24`, `body_tail_radius=0.02`,
  `farfield_radius=0.5`, `wedge_angle_degrees=18`, `radial_growth=3.5`
- `naca_wing`: finite swept tapered wing from a NACA 4-digit section law
  embedded in a structured farfield block with default preset `121x41x81`,
  `lx=2.8`, `ly=0.9`, `lz=1.6`, `wing_origin_x=0.45`, `wing_center_z=0.8`,
  `span=1.0`, `root_chord=0.8`, `tip_chord=0.4`, `sweep_degrees=20`,
  `thickness_ratio=0.12`, `camber_ratio=0.0`, `camber_position=0.4`,
  `vertical_offset=0.0`, `wall_normal_growth=4.0`

Documented config files for the default procedural cases:

- [configs/bump3d.cfg](/home/ireguly/nssolver-op2/configs/bump3d.cfg)
- [configs/axisymmetric_body.cfg](/home/ireguly/nssolver-op2/configs/axisymmetric_body.cfg)
- [configs/naca_wing.cfg](/home/ireguly/nssolver-op2/configs/naca_wing.cfg)

## 15. Output Contract

The OP2 executable writes:

- `q`
- `primitive`
- `dt`

to the configured HDF5 output file. It also writes a residual-history CSV alongside the HDF5 solution:

- `<solution>.residual.csv`

The local helper `nssolver_hdf5_to_vtk_helper` converts OP2 HDF5 output plus the OP2 mesh file into a VTK file for visualization.

## 16. Validation Strategy

The OP2 path is validated against the non-OP2 reference solver on non-periodic cases:

- `box`: exact uniform-state preservation
- `bump`: inviscid nonlinear steady-state consistency
- `flatplate_develop`: viscous boundary-layer consistency
- `hydra_benchmark`: non-rotating Hydra benchmark consistency

The shipped validation scripts live under `scripts/` and `tests/`.

Known status at the time of writing:

- `box` matches to roundoff
- `bump` is close in residual level and qualitative convergence
- `flatplate_develop` matches well in velocity profiles and skin friction
- `flatplate_develop` still shows a pressure-coefficient mismatch between original and OP2 paths

That `Cp` discrepancy is a known open issue and should not be hidden by the validation scripts.
