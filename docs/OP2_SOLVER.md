# OP2 Compressible Navier-Stokes Solver

## 1. Scope

This document describes the mathematics, data layout, execution model, and validation workflow of the OP2 implementation in this directory.

The OP2 solver is a node-centered, edge-based finite-volume solver for the steady compressible Navier-Stokes equations with optional Spalart-Allmaras turbulence transport. The high-level driver remains C++, while kernels are written in a constrained C-style suitable for OP2 backends.

The current OP2 implementation supports:

- 3D node-centered edge-based control volumes
- HLLC inviscid fluxes
- Green-Gauss primitive gradients
- edge-local second-order MUSCL reconstruction
- full Navier-Stokes viscous fluxes (edge-corrected averaged gradients)
- explicit four-stage Runge-Kutta marching with local time stepping
- implicit pseudo-time stepping, and dual time stepping for unsteady flow
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
\left(\mathbf{F}^{inv}_{f,i}-\mathbf{F}^{visc}_{f,i}\right)\cdot \Delta\mathbf{S}_{f,i}

-
V_i\mathbf{D}_{SA,i}.
\]

Here:

- \(V_i\) is the nodal dual volume
- \(\Delta \mathbf{S}_{ij}\) is the directed dual-face area vector associated with edge \(i\to j\)
- \(\Delta \mathbf{S}_{f,i}\) is node \(i\)'s median sub-face of boundary face \(f\) (node, edge midpoints, face centroid)

The dual mesh is the median dual: within each cell, edge \(i\to j\)'s dual face is the quad (edge midpoint, centroid of one adjacent cell face, cell centroid, centroid of the other), summed over the cells around the edge. Together with the boundary sub-faces, every control volume closes,

\[
\sum_{j\in N(i)} \Delta\mathbf{S}_{ij} + \sum_{f\in \partial \Omega_i} \Delta\mathbf{S}_{f,i} = 0,
\]

to round-off, so a uniform state has no residual anywhere, boundaries included.

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
\sum_{f\in \partial \Omega_i} \mathbf{W}_{f,i}\otimes \Delta \mathbf{S}_{f,i}
\right],
\]

where \(\mathbf{W}_{f,i}\) is the face's (bi)linear interpolant averaged over node \(i\)'s sub-face: \((9\mathbf{W}_i + 3\mathbf{W}_{next} + 3\mathbf{W}_{prev} + \mathbf{W}_{opp})/16\) on quads, \((22\mathbf{W}_i + 7\mathbf{W}_{next} + 7\mathbf{W}_{prev})/36\) on triangles, which is exact for linear fields on flat faces.

In the OP2 implementation:

- `edge_grad_kernel` contributes the interior edge term through indirect `OP_INC` updates to both endpoints
- `bface_grad_kernel` contributes each node's sub-face term through indirect `OP_INC` updates to the face's nodes
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
\phi_i = \min_m \phi_{i,m}
\]

over the mean-flow components: \(\tilde\nu\) is advected first-order (as the SA reference
solutions are) and does not limit them. `limiter = 0` reconstructs unlimited. For smooth subsonic
flows that is the right choice: the limiter switches in boundary layers and holds the iteration in a
limit cycle a few orders above round-off.

`limiter = 2` is van Albada's, per component and differentiable: the gradient's projection
\(u=\nabla W_i\cdot\Delta\mathbf{r}_{ij}/2\) and the edge's half difference \(e=(W_j-W_i)/2\) give the
increment \((u(e^2+\varepsilon^2)+e(u^2+\varepsilon^2))/(u^2+e^2+2\varepsilon^2)\), \(\varepsilon\) 1% of the
free stream's density, speed of sound and pressure. For flows with shocks, where unlimited
reconstruction fails and the edge limiter's switches stall the implicit scheme's inner iterations
(dual time stepping), it converges them as the unlimited scheme does smooth flows.

This limiter is directional and edge-local. It is less multidimensional than a Venkatakrishnan-style nodal extrema limiter, but it maps cleanly onto OP2 loops.

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

The flux is then assembled piecewise from left, star-left, star-right, or right states. Across a band \(|S_M| \le 10^{-8}(|S_L|+|S_R|+|S_M|)\) the star-left and star-right fluxes are blended linearly: they agree to \(O(S_M)\), so the flux stays continuous where a sign test would flip with round-off (every edge whose dual face is parallel to the flow sits at \(S_M \approx 0\)). Falling back to HLL there instead, as the solver once did, puts an \(O(1)\) jump into the residual, which stalls convergence and breaks Newton's method. HLL remains the fallback for degenerate wave-speed estimates only.

Implementation:

- `physical_flux_op2`
- `hllc_flux_op2`
- called from `edge_flux_kernel` and `boundary_flux_kernel`

## 8. Viscous Flux

Each edge's viscous flux is the full Navier-Stokes one, from the nodal states (not the reconstructed
ones). The gradients of \(u, v, w, T\) are the edge's average of the two nodes' Green-Gauss
gradients, with the component along the edge \(\mathbf{e}=\Delta\mathbf{r}/|\Delta\mathbf{r}|\)
replaced by the edge difference:

\[
\nabla\phi_{ij} = \overline{\nabla\phi} + \left(\frac{\phi_j-\phi_i}{|\Delta\mathbf{r}|}
- \overline{\nabla\phi}\cdot\mathbf{e}\right)\mathbf{e},
\]

which keeps the stencil compact along the edge and adds the cross terms. The flux through
\(\Delta\mathbf{S}_{ij}\) is \(\boldsymbol{\tau}\cdot\Delta\mathbf{S}\) for momentum and
\(\bar{\mathbf{u}}\cdot\boldsymbol{\tau}\cdot\Delta\mathbf{S} + k\nabla T\cdot\Delta\mathbf{S}\) for
energy, with

\[
\boldsymbol{\tau} = (\mu+\mu_t)\left(\nabla\mathbf{u}+\nabla\mathbf{u}^T-\tfrac{2}{3}(\nabla\cdot\mathbf{u})\mathbf{I}\right),
\qquad
k = c_p\left(\frac{\mu}{Pr}+\frac{\mu_t}{Pr_t}\right),\ Pr_t = 0.9,
\]

Sutherland's law for \(\mu\) and \(\mu_t = \rho\tilde\nu f_{v1}\). With zero gradients the same
function gives the compact part alone, which the first-order Jacobian and the no-slip wall's flux
take. A slip wall (a symmetry plane) takes the normal stress \(\mathbf{n}\cdot\boldsymbol{\tau}\cdot\mathbf{n}\)
of the node's gradient through its sub-face (`symmetry_stress_kernel`): it balances the interior
edges' \(-\tfrac{2}{3}\mu\nabla\cdot\mathbf{u}\), which otherwise drives a spurious velocity
normal to the plane.

SA's diffusion, \(\frac{1}{\sigma}[\nabla\cdot((\mu+\rho\tilde\nu)\nabla\tilde\nu) + c_{b2}\rho|\nabla\tilde\nu|^2]\),
is taken as node-based codes take it,

\[
\frac{1+c_{b2}}{\sigma}\nabla\cdot\left((\mu+\rho\tilde\nu)\nabla\tilde\nu\right)
- \frac{c_{b2}}{\sigma}(\mu+\rho\tilde\nu)_i\nabla^2\tilde\nu,
\]

both through compact edge differences, so every edge's coefficient stays positive. The
\(c_{b2}|\nabla\tilde\nu|^2\) source from nodal gradients feeds on their noise on stretched cells
(it drove \(\tilde\nu/\nu\) to \(10^4\) ahead of a flat plate's leading edge).

Implementation:

- `viscous_flux_op2`, `viscous_gradients_op2`, `stress_op2`
- `sa_cb2_coefficient_op2` (the second part, in `edge_flux_kernel` and `jac_edge_kernel`)
- `symmetry_stress_kernel`

## 9. Boundary Conditions

Boundary faces are stored as quads and processed in a dedicated OP2 boundary-face loop.

Supported boundary types:

- `Farfield`, `Inlet`: the free stream, imposed strongly at their nodes in viscous runs
- `Characteristic`: a Riemann far field, weak
- `TotalInlet`: subsonic inflow at the free stream's total pressure and temperature, weak
- `Outlet`
- `SlipWall`
- `NoSlipWall`

### 9.1 Inviscid Boundary State Construction

Each node of a boundary face takes the flux through its own sub-face \(\Delta\mathbf{S}_{f,i}\), from its own state \(\mathbf{W}_i\) and a ghost state built from it. (A single flux from the face-averaged state, shared equally, leaves the boundary blind to node-to-node oscillations along it.) The ghost state \(\mathbf{W}_{ghost}\) is:

- `Farfield`, `Inlet`: replace with freestream primitive
- `Characteristic`: the Riemann invariants \(u_n \pm 2a/(\gamma-1)\) along the outward normal, the
  outgoing one from the node and the incoming one from the free stream; the tangential velocity,
  entropy and \(\tilde\nu\) from the side the flow comes from (`characteristic_state_op2`)
- `TotalInlet`: the static pressure from the node, the free stream's total pressure and temperature
  and its direction (`total_inlet_state_op2`)
- `Outlet`: copy interior state, replace only pressure with freestream pressure
- `SlipWall`: reflect normal velocity
- `NoSlipWall`: reflect normal velocity for inviscid flux, and enforce zero wall velocity in the viscous wall state

Implementation:

- `make_boundary_ghost_state_op2`
- `boundary_flux_kernel`

### 9.2 Boundary-Node State Enforcement

For viscous runs, the OP2 implementation also enforces boundary node states before each Runge-Kutta stage and after each implicit update:

- inlet/farfield nodes are reset to freestream
- no-slip wall nodes are forced to zero velocity, and to \(\tilde\nu=0\) with SA

These components have no residual (`zero_constrained_kernel`): the enforcement sets them. Slip walls and outlets are weak, through their ghost states' fluxes only. Projecting slip nodes' velocity onto the wall would undo part of every update the residual asks for, and a node on two slip planes has no single normal to project on.

This is done through:

- `enforce_boundary_node_kernel`
- `zero_constrained_kernel`, from the node's Dirichlet and no-slip entries (`node_bc_kernel`)

This kernel is intentionally limited to boundary nodes and uses indirect `OP_RW` only on `q` through a boundary-node to node map.

## 10. Local Time Stepping

The pseudo-time step at node \(i\) is

\[
\Delta t_i = \text{CFL}\frac{V_i}{\lambda_i + \varepsilon}
\]

where the inviscid spectral contribution is accumulated over the node's whole surface, its edges and its boundary sub-faces:

\[
\lambda_i^{inv}
=
\sum_{j\in N(i)} |\Delta S_{ij}|\left(|u_n|+a\right)
+
\sum_{f\in \partial \Omega_i} |\Delta S_{f,i}|\left(|u_n|+a\right).
\]

For viscous runs an additional penalty is added:

\[
\lambda_i^{visc} \sim 2\mu \frac{|\Delta S_{ij}|^2}{\rho_i V_i}.
\]

Implementation:

- `edge_spectral_kernel`
- `bface_spectral_kernel`
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

The production and destruction of the standard model (no \(f_{t2}\)), as nodal sources:

\[
\rho\,(c_{b1}\tilde S\tilde\nu - c_{w1}f_w(\tilde\nu/d)^2),
\]

with the wall distance \(d\) of the preprocessed mesh. \(\tilde S = \Omega + \bar S\) is kept positive
smoothly, as the NASA Turbulence Modeling Resource's SA does (\(\bar S \ge -c_2\Omega\), else
\(\Omega + \Omega(c_2^2\Omega + c_3\bar S)/((c_3-2c_2)\Omega-\bar S)\), \(c_2=0.7\), \(c_3=0.9\)): a hard clip of it
switches in the buffer layer. The diffusion is in section 8.

`sa_comp = 1` selects SA-comp, the Turbulence Modeling Resource's compressibility correction: the
destruction gains \(C_5\tilde\nu^2/a^2\,\partial_j u_i\partial_j u_i\), \(C_5 = 3.5\), \(a\) the local
speed of sound. It lowers the eddy viscosity of separated shear layers; on the TMR flat plate
(M 0.2) it moves \(C_f\) by 0.02%. On OAT15A (M 0.73, Re 3e6, the full grid) standard SA damps the
shock's oscillation at every angle tried (3.9 to 5.5 degrees); with SA-comp buffet sets in at about
3.2 degrees (the experiment's onset lies between 3.1 and 3.25): \(\Delta C_L\) 0.105 at 3.25,
0.220 at 3.5, 0.317 at 3.9, Strouhal number 0.070-0.071 (experiment 0.066), the same with time steps
of 0.24 and 0.06 c/U. The shock's pressure fluctuation peaks about 15% above the experiment's and
0.04-0.05 c further aft (Cp RMS 0.36 at x/c 0.50 against 0.31 at 0.45 at 3.5 degrees).

In the implicit scheme's Jacobian the source's derivative by \(\tilde\nu\) is a difference of the
source terms themselves: the destruction's through \(r\) and \(f_w\) as well (near walls about
twice the \(2D/\tilde\nu\) of \(D\propto\tilde\nu^2\), whose Newton steps overshoot into a
two-step oscillation), and the production's only where it damps.

Implementation:

- `sa_terms_op2`, `sa_source_kernel`
- `jac_node_kernel`

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
- boundary geometry: `bface_normal`, `bface_node_normals`, `bface_area`, `bface_num_nodes`, `bface_type`, `bface_group`
- boundary-node metadata: `bnode_dirichlet`, `bnode_wall`, and per node `node_bc` from them
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
- `bface_node_normals [F,12]`: each node's median sub-face area vector (zero for a triangle's fourth)
- `bface_num_nodes [F,1]`
- `bface_area [F,1]`
- `bface_group [F,1]`
- `bface_type [F,1]`
- `bnode-->node [Nb,1]`
- `bnode_dirichlet [Nb,1]`
- `bnode_wall [Nb,1]`
- `bnode_slip [Nb,1]`
- `bnode_normal [Nb,3]`

`node_wall_distance` is the distance to the nearest no-slip wall face, which the
SA source terms read (a node far from every wall has a large value, and no wall
at all leaves 1e30). The procedural structured meshes (`bump3d`,
`axisymmetric_body`, `naca_wing`) used to leave it at the largest `double`
(a loop that ran before the boundary faces were counted), which switched the
SA production and destruction off: meshes written before this fix have to be
generated again. For them the distance is exact to the ymin boundary's no-slip
faces within three columns of the node's own, and to a sample of them beyond;
with at most 4096 such faces (a 2D grid's), exact to all of them.

The structured generators also read 2D PLOT3D grids (`plot3d2d`, one block,
extruded one cell between slip planes, its jmin boundary a no-slip wall over
`wall_x0` to `wall_x1`) and build circular-cylinder O-grids (`cylinder`,
periodic around). Their boundary faces point away from their own cell.

Planar multi-block grids go through a general hexahedral builder: the blocks'
points merged where they coincide (block interfaces, C-grid wake cuts),
extruded one layer, median-dual edges from each hexahedron's twelve edges,
boundary faces the faces only one cell has, typed by where they lie, and the
wall distance exact to every no-slip face. A block that folds over itself is an
error. `plot3d_blocks` reads Fortran unformatted multi-block PLOT3D planes (the
AIAA buffet workshop's OAT15A grids); `circular_arc` builds an O-grid about a
biconvex circular-arc section (a C-grid's wake cut would make its sharp
trailing edge a concave corner, where the wall's normals cross), its lines
normal to the wall within `blend_length` and straight to equal angles on the
far-field circle beyond.

All connectivity is zero-based in the preprocessed OP2 mesh. Meshes written before `bface_node_normals` existed must be preprocessed again. (The solver no longer reads `bnode_slip` and `bnode_normal`.)

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

The OP2 path is validated directly on non-periodic cases:

- `box`: exact uniform-state preservation
- `bump`: inviscid nonlinear steady-state regression
- `flatplate_develop`: viscous boundary-layer regression and postprocessing
- `hydra_benchmark`: non-rotating Hydra benchmark run

The shipped validation scripts live under `scripts/` and `tests/`.

Known status at the time of writing:

- `box` remains near roundoff
- `bump` remains inside the validated OP2 residual band
- `flatplate_develop` remains inside the validated OP2 residual band and produces wall/profile benchmark CSVs

## 17. Implicit Pseudo-Time Stepping

`time_scheme = implicit` replaces the four-stage RK update by backward-Euler
pseudo-time steps through OP2's solver layer (`op_solve.h`, OP2's
`feature/solvers-prototype`):

$$
\left(\frac{V_i}{\Delta t_i} I + \frac{\partial R}{\partial q}\right)\Delta q = -R(q),
\qquad q \leftarrow q + \omega\,\Delta q
$$

with $V_i/\Delta t_i = \lambda_i / \text{CFL}$ from the same spectral radius
as the explicit step, the CFL starting at `implicit_cfl`, multiplied by
`implicit_cfl_growth` each iteration up to `implicit_cfl_max`, and
$\omega$ = `implicit_relax`. An update is kept where it is admissible, as the
RK update's, and counted as rejected otherwise.

The matrix is $\partial R/\partial q$ of the first-order residual, with
$6\times 6$ blocks on the edge graph, assembled by par loops straight into the
solver library's storage (`op_arg_mat`):

- each edge: the derivatives of its first-order flux (HLLC less the
  viscous flux's compact part) by both end states, by one-sided differences
  of the same flux functions (step $10^{-7}\max(|q_c|, s_c)$ in component
  $c$), as four blocks, and those of SA's non-conservative diffusion part;
- each boundary sub-face: the derivative of the node's boundary flux (ghost
  state included) by the node's state, the same way, on the node's diagonal
  block;
- each node: the pseudo-time term (and in dual time stepping the physical
  one) and the SA source's derivative (section 12).

For first-order runs this is the residual's Jacobian (Newton's method at
large CFL); second-order runs keep it as an approximation.

Components the viscous scheme enforces (all at Dirichlet nodes, momentum and
SA's at no-slip nodes) are rows of the identity with a zero right-hand side. The
system is solved in scaled variables, $S^{-1} A S\,x = S^{-1} b$ with
$\delta q = S x$ and $S$ the reference state
$(\rho_\infty, \rho_\infty a_\infty, \rho_\infty a_\infty, \rho_\infty a_\infty, \rho_\infty a_\infty^2, \rho_\infty \nu_{ref})$:
the flow's is FUN3D's nondimensionalization, and
$\nu_{ref} = \max(\tilde\nu_\infty, \kappa_{SA}\nu_\infty)$ with $\kappa_{SA}$ the
config key `sa_scale` (1000), after Ceze and Fidkowski (AIAA 2013-2686): $\tilde\nu$
reaches $10^3$-$10^4\,\nu_\infty$ in boundary layers, and a free-stream reference
lets the SA rows dominate the Krylov norms (in the residual and, as GMRES's
default left preconditioning measures it, in the update), so that a relative
tolerance no longer controls the flow equations. The scaling is a similarity
transform per node: ILU and block-Jacobi preconditioners are invariant to it,
and it changes only what the Krylov method minimises and when it stops.
`sa_scale = 1` gives the earlier free-stream reference. The residual history
(`.residual.csv`) carries the SA residual's L2 norm, `l2_nu`.

The linear solve takes the OP2 implicit proxy's options (`apps/c/implicit`),
as config keys or `--key value` on the command line:

| option | values | meaning |
|---|---|---|
| `solve_config` | file | the solver layer's JSON (`configs/solve.json`); `OP_SOLVE_CONFIG` over it picks the library |
| `pc` | `lib`, `bjacobi`, `neumann` | the library's preconditioner, or par loops reading the assembled blocks: block Jacobi, or a Neumann series of `terms` steps |
| `pmat` | `A`, `decoupled` | the library's preconditioner from the matrix, or from each block's diagonal only |
| `op` | `assembled`, `matfree` | the operator: the matrix, or differences of the full (second-order) residual with the matrix as preconditioner (JFNK) |
| `jacobian_interval` | N | reassemble every N iterations |
| `partition` | `kway`, `random`, `inertial` | ParMETIS, OP2's random, or its inertial partitioner |
| `reorder` | 0, 1 | renumber each rank's mesh after partitioning (`op_renumber`) |
| `dump_interval`, `dump_prefix` | N, path | every N-th iteration writes the system just solved to `<prefix>_<iteration>.h5` (see below) |
| `diags`, `timing`, `check` | | OP2's diagnostics, timing output, and a check that fails the run unless L2(rho) falls by `check` |

The solvers are named `nssolver`, `nssolver_bjacobi` and `nssolver_neumann`
in the JSON, each in every library's own terms. `tests/test_implicit.sh`
runs the scheme across libraries, variants and these options.

**Replaying the linear systems.** With `--dump-interval N --dump-prefix P`
each N-th iteration writes, after its solve, `P_<iteration>.h5`: the matrix
(`A`, OP2's `op_mat_write_hdf5` layout), the right-hand side `A/b` and the
solution `A/x` the solve returned (from a zero initial guess), and a line to
`P_index.csv` (iteration, CFL, linear iterations, residual). OP2's
`solve_hdf5_6_<variant>` driver reads a file on any rank count and solves it
with any library and settings; `scripts/replay_systems.sh -v VARIANTS
FILE.h5...` tabulates iterations and true residuals for several settings
(`OP_SOLVE_CONFIG` overrides over `configs/solve.json`). The first setting,
on the run's rank count, reproduces the run's solve exactly; block-Jacobi
preconditioners follow the partition, so other rank counts differ. The
dumped `A/x` is itself a solve to the linear tolerance, so the error against
it is of that order: judge a setting by its true residual. A par-loop
preconditioner (`pc = bjacobi`, `neumann`) or `op = matfree` is not part of
the file; the replay solves the assembled system with library
preconditioners, and `pmat = decoupled`'s matrix P is not dumped.

`outlet_backflow = freestream` gives an outlet face whose interior flow
points inward the free-stream ghost state, as an inlet would. The default,
`extrapolate`, keeps the interior state with the far-field pressure, under
which a reversed outflow can feed itself. With closed control volumes no case
here develops backflow (the two give identical histories), so this is a
safeguard only.

## 18. Dual Time Stepping

With `time_step` \(\Delta t > 0\) (implicit scheme) each of `physical_steps` steps solves

\[
\mathbf{R}(\mathbf{Q}) + V\,\frac{c_0\mathbf{Q} - c_1\mathbf{Q}^n + c_2\mathbf{Q}^{n-1}}{\Delta t} = 0,
\]

second-order backward differences (\(c = 3/2, 2, 1/2\); `time_order = 2`) after a first-order first
step (\(c = 1, 1, 0\)), by the pseudo-time iterations of section 17: the physical term adds
\(c_0V/\Delta t\) to the diagonal blocks (`jac_node_kernel`), the matrix-free operator differences
the whole residual, and the CFL ramp restarts every step. A step ends when the residual has
fallen by `inner_tolerance` or after `inner_iterations`. `restart_file` starts from a solution's
`q`; `snapshot_interval` writes `q` and the primitives every that many steps.

The inner iterations are inexact Newton steps: `configs/solve_dual.json` solves each to a relative
residual of 0.1 rather than 1e-3, which keeps their count (11 to 14 a step on the transonic
circular arc) and takes a sixth of the linear iterations, 2.4 times less time. For transonic
starts `update_limit` (at most that relative change of density and pressure per node and
iteration) scales back the update of the few nodes that would overshoot; skipping it, as the
admissibility check otherwise does, leaves them out of step with their neighbours and blows up.

The output file's `.history.csv` has each step's force on the no-slip walls,
\(\sum(p\,\Delta\mathbf{S} - \mathbf{R}_{1..3})\) over their nodes' sub-faces: a wall node's momentum
residual, which enforcing no slip discards, is its flux \(p\Delta\mathbf{S}\) through the wall less
the traction that would balance it.

Implementation:

- `unsteady_residual_kernel`, `implicit_residual`
- `wall_pressure_force_kernel`, `wall_force_kernel`

## 19. Verification Runs

- `scripts/tmr/flatplate.sh`: the NASA Turbulence Modeling Resource's 2D zero-pressure-gradient
  flat plate (SA, M 0.2, Re 5e6 per unit length) on its PLOT3D grid family (`plot3d2d`
  preprocessing), with TMR's boundaries (total-pressure inflow, Riemann top, pressure outflow);
  skin friction at x = 0.97 and drag against the CFL3D and FUN3D reference values
  (`scripts/tmr/flatplate_forces.py`).
- `scripts/cylinder/run.sh`: laminar vortex shedding behind a circular cylinder at Re 100
  (`cylinder` preprocessing, a periodic O-grid), by dual time stepping; Strouhal number, mean drag
  and lift amplitude (`scripts/cylinder/forces.py`).
- `scripts/airfoil/run.sh`: transonic airfoils by SA, steady or by dual time stepping, the free
  stream from Mach, Reynolds number, temperature and incidence (`freestream.py`): the 18% circular
  arc (McDevitt et al.; self-excited shock oscillation at zero incidence) and the ONERA OAT15A on
  the buffet workshop's grid (shock buffet; its geometry stays ONERA's and outside this
  repository). Lift, drag and the oscillation's Strouhal number from the history (`forces.py`),
  the wall's pressure from `nssolver_op2_benchmark_postprocess_helper wall`.
