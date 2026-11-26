# OP2 Compressible Navier-Stokes Solver

## 1. Scope

This document describes the mathematics, data layout, execution model, and validation workflow of the OP2 implementation in this directory.

The OP2 solver is a node-centered, edge-based finite-volume solver for the compressible Navier-Stokes and RANS equations (Spalart-Allmaras), steady or unsteady. The high-level driver remains C++, while kernels are written in a constrained C-style suitable for OP2 backends.

The current OP2 implementation supports:

- 3D node-centered edge-based median-dual control volumes, closed at boundaries (section 3)
- HLLC inviscid fluxes (section 7)
- Green-Gauss primitive gradients (section 5)
- second-order MUSCL reconstruction with an edge-local or van Albada limiter, or none (section 6)
- full Navier-Stokes viscous fluxes from edge-corrected averaged gradients (section 8)
- Spalart-Allmaras without $f_{t2}$, optionally with the compressibility correction SA-comp (section 12)
- far-field, characteristic, total-pressure inflow, outlet, slip and adiabatic no-slip boundaries (section 9)
- explicit four-stage Runge-Kutta marching with local time stepping (sections 10, 11)
- implicit pseudo-time stepping by Newton-Krylov through OP2's solver layer (section 17)
- dual time stepping with second-order backward differences for unsteady flow (section 18)
- Hydra-derived HDF5 meshes after preprocessing, and procedural, PLOT3D and
  multi-block planar meshes (section 14)

It does not support:

- periodic coupling across mesh sector cuts
- rotating-frame source terms
- multigrid
- direct procedural mesh generation inside the OP2 executable

Hydra periodic cuts are intentionally preprocessed as slip walls in the current benchmark path.

## 2. Governing Equations

The solver drives the residual of the compressible RANS equations to zero by marching in pseudo-time (steady states), or within each physical time step of dual time stepping (section 18):

$$
\frac{\partial}{\partial t}\int_{V_i} \mathbf{Q}\,dV
+
\int_{\partial V_i}\left(\mathbf{F}_{inv}-\mathbf{F}_{visc}\right)\cdot d\mathbf{S}
=
\int_{V_i}\mathbf{D}_{SA}\,dV
$$

The conservative state is

$$
\mathbf{Q}=
\begin{bmatrix}
\rho \\
\rho u \\
\rho v \\
\rho w \\
\rho E \\
\rho \tilde{\nu}
\end{bmatrix}
$$

and the primitive state is

$$
\mathbf{W}=
\begin{bmatrix}
\rho \\
u \\
v \\
w \\
p \\
\tilde{\nu}
\end{bmatrix}.
$$

For a perfect gas,

$$
p=(\gamma-1)\rho\left(E-\frac{1}{2}(u^2+v^2+w^2)\right),
\qquad
a=\sqrt{\gamma p/\rho},
$$

with $\gamma$ and the gas constant $R$ from the config (`gamma`, `gas_constant`; 1.4 and
287.05 J/(kg K)), the temperature $T = p/(\rho R)$, Sutherland's law for the viscosity
($\mu = 1.7894\cdot10^{-5}$ Pa s at 288.15 K, $S = 110.4$ K) and a Prandtl number of 0.72.

The OP2 implementation stores and updates the conservative state, then reconstructs the primitive state before gradient, flux, and source-term evaluation.

## 3. Edge-Based Finite-Volume Discretization

For node $i$, the semi-discrete residual is

$$
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
$$

Here:

- $V_i$ is the nodal dual volume
- $\Delta \mathbf{S}_{ij}$ is the directed dual-face area vector associated with edge $i\to j$
- $\Delta \mathbf{S}_{f,i}$ is node $i$'s median sub-face of boundary face $f$ (node, edge midpoints, face centroid)

The dual mesh is the median dual: within each cell, edge $i\to j$'s dual face is the quad (edge midpoint, centroid of one adjacent cell face, cell centroid, centroid of the other), summed over the cells around the edge. Together with the boundary sub-faces, every control volume closes,

$$
\sum_{j\in N(i)} \Delta\mathbf{S}_{ij} + \sum_{f\in \partial \Omega_i} \Delta\mathbf{S}_{f,i} = 0,
$$

to round-off, so a uniform state has no residual anywhere, boundaries included (the
preprocessor checks it, section 14).

The solver uses OP2 edge loops for interior fluxes and OP2 boundary-face loops for boundary fluxes.

## 4. Primitive/Conservative Conversion

Given primitive state $\mathbf{W}$, the conservative state is

$$
\rho u = \rho u,\quad
\rho v = \rho v,\quad
\rho w = \rho w,
$$

$$
\rho E = \rho \left(\frac{p}{(\gamma-1)\rho} + \frac{1}{2}(u^2+v^2+w^2)\right),
\qquad
\rho \tilde{\nu} = \rho \tilde{\nu}.
$$

The inverse conversion precedes every residual evaluation: each Runge-Kutta stage, each implicit
iteration. It keeps density and pressure above $10^{-12}$; an update that would take them below
`rho_floor` or `p_floor` is not kept (sections 11 and 17). Each update zeroes momentum
components below $10^{-12}$ of the node's largest one and $\rho\tilde\nu$ below $10^{-14}\rho$:
round-off traces, such as a planar run's $w$, that optimized and debug builds would otherwise
carry differently. A floor on the velocity itself would also zero the real wall-normal velocity
of the nodes next to a wall, which grows as $y^2$, and hold the residual above round-off there.

Implementation:

- `initialize_q_kernel`
- `q_to_primitive_kernel`
- `primitive_to_conservative_op2`
- `conservative_to_primitive_op2`

## 5. Gradient Reconstruction

Primitive gradients are computed with a Green-Gauss approximation over the nodal dual control volume:

$$
\nabla \mathbf{W}_i
=
\frac{1}{V_i}
\left[
\sum_{j\in N(i)} \frac{1}{2}(\mathbf{W}_i+\mathbf{W}_j)\otimes \Delta \mathbf{S}_{ij}
+
\sum_{f\in \partial \Omega_i} \mathbf{W}_{f,i}\otimes \Delta \mathbf{S}_{f,i}
\right],
$$

where $\mathbf{W}_{f,i}$ is the face's (bi)linear interpolant averaged over node $i$'s sub-face: $(9\mathbf{W}_i + 3\mathbf{W}_{next} + 3\mathbf{W}_{prev} + \mathbf{W}_{opp})/16$ on quads, $(22\mathbf{W}_i + 7\mathbf{W}_{next} + 7\mathbf{W}_{prev})/36$ on triangles, which is exact for linear fields on flat faces.

In the OP2 implementation:

- `edge_grad_kernel` contributes the interior edge term through indirect `OP_INC` updates to both endpoints
- `bface_grad_kernel` contributes each node's sub-face term through indirect `OP_INC` updates to the face's nodes
- `normalize_grad_kernel` divides by nodal volume

The gradient storage is packed as 18 doubles per node:

- 3 components for each of the 6 primitive variables

The layout is:

$$
[\partial_x \rho,\partial_y \rho,\partial_z \rho,\partial_x u,\dots,\partial_z \tilde{\nu}]
$$

## 6. Second-Order MUSCL Reconstruction

For second-order runs, left and right interface states are reconstructed from node-centered primitives and gradients.

For edge $i\leftrightarrow j$, let

$$
\Delta \mathbf{r}_{ij} = \frac{1}{2}(\mathbf{x}_j-\mathbf{x}_i),
\qquad
\Delta \mathbf{r}_{ji} = -\Delta \mathbf{r}_{ij}.
$$

The reconstructed states are

$$
\mathbf{W}_L = \mathbf{W}_i + \phi_i \left(\nabla \mathbf{W}_i \cdot \Delta \mathbf{r}_{ij}\right),
$$

$$
\mathbf{W}_R = \mathbf{W}_j + \phi_j \left(\nabla \mathbf{W}_j \cdot \Delta \mathbf{r}_{ji}\right),
$$

with the limiting of section 6.1. Only the mean flow is reconstructed: $\tilde\nu$ is advected
first-order (as the SA reference solutions are), $\tilde\nu_L = \tilde\nu_i$, $\tilde\nu_R = \tilde\nu_j$. The
viscous flux takes the nodal states (section 8).

### 6.1 Limiters

`limiter` selects one of three; each needs the two nodes of its edge only, so the OP2 loops keep
to `OP_READ` and `OP_INC` access (a nodal-extrema limiter would need indirect `OP_RW` or
reductions).

`limiter = 1` (the default) is edge-local and scalar. For each mean-flow component $m$,

$$
\delta^{proj}_{i,m} = \nabla W_{i,m}\cdot \Delta \mathbf{r}_{ij},
\qquad
\delta^{edge}_{i,m} = W_{j,m} - W_{i,m}.
$$

The scalar limiter contribution is

$$
\phi_{i,m}=
\begin{cases}
1, & |\delta^{proj}_{i,m}| < \varepsilon \\
0, & \delta^{proj}_{i,m}\,\delta^{edge}_{i,m}\le 0 \\
\min\left(1,\frac{\delta^{edge}_{i,m}}{\delta^{proj}_{i,m}}\right), & \text{otherwise}
\end{cases}
$$

and the node-edge limiter is

$$
\phi_i = \min_m \phi_{i,m}
$$

over $\rho, u, v, w, p$ ($\varepsilon = 10^{-14}$). It is directional, less multidimensional than a
Venkatakrishnan-style nodal limiter, and it switches: in boundary layers its switching holds the
iteration in a limit cycle a few orders above round-off, and with shocks it stalls the implicit
scheme's inner iterations.

`limiter = 2` is van Albada's, per component and differentiable: the gradient's projection
$u=\nabla W_{i,m}\cdot\Delta\mathbf{r}_{ij}$ and the edge's half difference $e=(W_{j,m}-W_{i,m})/2$
give the increment

$$
W_{L,m} - W_{i,m} = \frac{u\,(e^2+\varepsilon_m^2)+e\,(u^2+\varepsilon_m^2)}{u^2+e^2+2\varepsilon_m^2},
$$

$u$ where the two agree, toward 0 where they differ in sign, $\varepsilon_m$ 1% of the free stream's
density, speed of sound (for the velocity components) and pressure.

`limiter = 0` reconstructs unlimited. Smooth flows converge best unlimited; flows with shocks,
where unlimited reconstruction fails, take van Albada's, with which the implicit scheme
converges as the unlimited one does smooth flows (`configs/oat15a.cfg`).

Implementation:

- `edge_limiter_scale_op2`, `scalar_edge_limiter_op2` (`limiter = 1`)
- `van_albada_reconstruct_op2` (`limiter = 2`)
- `reconstruct_primitive_op2`, from `edge_flux_kernel`

## 7. Inviscid Flux: HLLC

Given reconstructed left/right states and the edge area vector $\Delta \mathbf{S}_{ij}$, the solver computes the HLLC flux.

Let

$$
\hat{\mathbf{n}} = \frac{\Delta \mathbf{S}_{ij}}{\|\Delta \mathbf{S}_{ij}\|},
\qquad
u_n = \mathbf{u}\cdot \hat{\mathbf{n}}.
$$

Wave-speed estimates are

$$
S_L = \min(u_{n,L}-a_L, u_{n,R}-a_R),
\qquad
S_R = \max(u_{n,L}+a_L, u_{n,R}+a_R).
$$

The contact speed $S_M$ follows the usual HLLC estimate:

$$
S_M =
\frac{
p_R-p_L + \rho_L u_{n,L}(S_L-u_{n,L}) - \rho_R u_{n,R}(S_R-u_{n,R})
}{
\rho_L(S_L-u_{n,L}) - \rho_R(S_R-u_{n,R})
}.
$$

The flux is then assembled piecewise from left, star-left, star-right, or right states. Across a band $|S_M| \le 10^{-8}(|S_L|+|S_R|+|S_M|)$ the star-left and star-right fluxes are blended linearly: they agree to $O(S_M)$, so the flux stays continuous where a sign test would flip with round-off (every edge whose dual face is parallel to the flow sits at $S_M \approx 0$). Falling back to HLL there instead puts an $O(1)$ jump into the residual, which stalls convergence and breaks Newton's method. HLL is the fallback for degenerate wave-speed estimates only (a vanishing denominator of $S_M$, $S_M$ outside $(S_L, S_R)$, or a non-finite star state), and Rusanov's flux for a degenerate HLL.

Implementation:

- `physical_flux_op2`
- `hllc_flux_op2`
- called from `edge_flux_kernel` and `boundary_flux_kernel`

## 8. Viscous Flux

Each edge's viscous flux is the full Navier-Stokes one, from the nodal states (not the reconstructed
ones). The gradients of $u, v, w, T$ are the edge's average of the two nodes' Green-Gauss
gradients, with the component along the edge $\mathbf{e}=\Delta\mathbf{r}/|\Delta\mathbf{r}|$
replaced by the edge difference:

$$
\nabla\phi_{ij} = \overline{\nabla\phi} + \left(\frac{\phi_j-\phi_i}{|\Delta\mathbf{r}|}
- \overline{\nabla\phi}\cdot\mathbf{e}\right)\mathbf{e},
$$

which keeps the stencil compact along the edge and adds the cross terms. The flux through
$\Delta\mathbf{S}_{ij}$ is $\boldsymbol{\tau}\cdot\Delta\mathbf{S}$ for momentum and
$\bar{\mathbf{u}}\cdot\boldsymbol{\tau}\cdot\Delta\mathbf{S} + k\nabla T\cdot\Delta\mathbf{S}$ for
energy, with

$$
\boldsymbol{\tau} = (\mu+\mu_t)\left(\nabla\mathbf{u}+\nabla\mathbf{u}^T-\tfrac{2}{3}(\nabla\cdot\mathbf{u})\mathbf{I}\right),
\qquad
k = c_p\left(\frac{\mu}{Pr}+\frac{\mu_t}{Pr_t}\right),\ Pr_t = 0.9,
$$

Sutherland's law for $\mu$ (section 2), $\mu_t = \rho\tilde\nu f_{v1}$, and $\mu$, $\mu_t$ and
$\bar{\mathbf{u}}$ the averages of the two nodes'. With zero gradients the same function gives the
compact part alone, which the first-order Jacobian takes.

A no-slip wall node's flux through its sub-face is that compact part between the node and a wall
state of zero velocity at the node's temperature, at the distance $V_i/(n_f|\Delta\mathbf{S}_{f,i}|)$
($n_f$ the face's node count): the wall is adiabatic.

A slip wall (a symmetry plane) takes the normal stress $-\tfrac{2}{3}(\mu+\mu_t)\nabla_t\cdot\mathbf{u}$ of the
node's gradient through its sub-face (`symmetry_stress_kernel`), $\nabla_t\cdot\mathbf{u}$ the divergence
less the normal velocity's normal derivative: it balances the interior edges'
$-\tfrac{2}{3}(\mu+\mu_t)\nabla\cdot\mathbf{u}$ of the flow along the plane, which otherwise drives a velocity
through it. All of $\mathbf{n}\cdot\boldsymbol{\tau}\cdot\mathbf{n}$ would cancel the viscous damping of
that velocity, which then grows on a one-cell extrusion. The plane's normal velocity is imposed
weakly, by the ghost state (section 9.1), and stays small but not zero: on the OAT15A's one-cell
extrusion the planes' nodes carry up to 0.6% of the free stream's speed through them, equal and
opposite on the two planes, with no mass crossing either.

SA's diffusion, $\frac{1}{\sigma}[\nabla\cdot((\mu+\rho\tilde\nu)\nabla\tilde\nu) + c_{b2}\rho|\nabla\tilde\nu|^2]$,
is taken as node-based codes take it,

$$
\frac{1+c_{b2}}{\sigma}\nabla\cdot\left((\mu+\rho\tilde\nu)\nabla\tilde\nu\right)
- \frac{c_{b2}}{\sigma}(\mu+\rho\tilde\nu)_i\nabla^2\tilde\nu,
$$

both through compact edge differences, so every edge's coefficient stays positive. The
$c_{b2}|\nabla\tilde\nu|^2$ source from nodal gradients feeds on their noise on stretched cells
(it drove $\tilde\nu/\nu$ to $10^4$ ahead of a flat plate's leading edge).

Implementation:

- `viscous_flux_op2`, `viscous_gradients_op2`, `stress_op2`
- `sa_cb2_coefficient_op2` (the second part, in `edge_flux_kernel` and `jac_edge_kernel`)
- `boundary_node_flux_op2` (the no-slip wall's flux)
- `symmetry_stress_kernel`

## 9. Boundary Conditions

Boundary faces are quads or triangles (`bface_num_nodes`; a triangle's fourth node is unused),
processed in dedicated OP2 boundary-face loops.

Supported boundary types:

- `Farfield`, `Inlet`: the free stream, imposed strongly at their nodes in viscous runs
- `Characteristic`: a Riemann far field, weak
- `TotalInlet`: subsonic inflow at the free stream's total pressure and temperature, weak
- `Outlet`: the free stream's pressure, weak
- `SlipWall`: a symmetry plane or inviscid wall, weak
- `NoSlipWall`: an adiabatic wall, its velocity (and $\tilde\nu$) imposed strongly

### 9.1 Inviscid Boundary State Construction

Each node of a boundary face takes the flux through its own sub-face $\Delta\mathbf{S}_{f,i}$, from its own state $\mathbf{W}_i$ and a ghost state built from it. (A single flux from the face-averaged state, shared equally, leaves the boundary blind to node-to-node oscillations along it.) The ghost state $\mathbf{W}_{ghost}$ is:

- `Farfield`, `Inlet`: replace with freestream primitive
- `Characteristic`: the Riemann invariants $u_n \pm 2a/(\gamma-1)$ along the outward normal, the
  outgoing one from the node and the incoming one from the free stream; the tangential velocity,
  entropy and $\tilde\nu$ from the side the flow comes from, blended linearly across
  $|u_n| < 0.05a$ where the flow runs along the boundary (a switch there would flip with round-off
  and stall Newton's method); supersonic outflow takes the node's state, supersonic inflow the
  free stream's (`characteristic_state_op2`)
- `TotalInlet`: the static pressure from the node (at most the total pressure), the free stream's
  total pressure and temperature and its direction (`total_inlet_state_op2`)
- `Outlet`: the node's state with the free stream's pressure. Under backflow (normal velocity into
  the domain) `outlet_backflow = freestream` takes the free stream instead, as an inlet would; the
  default, `extrapolate`, keeps the node's state, under which a reversed outflow can feed itself.
  No case here has developed backflow (the two give identical histories), so this is a safeguard.
- `SlipWall`: the node's state with its normal velocity reflected. HLLC between the two has its
  contact on the face: no mass crosses it
- `NoSlipWall`: the same reflection, and $\tilde\nu$ negated, for the inviscid flux; the viscous
  flux is section 8's

Implementation:

- `make_boundary_ghost_state_op2`
- `boundary_flux_kernel`

### 9.2 Boundary-Node State Enforcement

For viscous runs, the OP2 implementation also enforces boundary node states before each Runge-Kutta stage and after each implicit update:

- inlet/farfield nodes are reset to freestream
- no-slip wall nodes are forced to zero velocity, and to $\tilde\nu=0$ with SA

These components have no residual (`zero_constrained_kernel`): the enforcement sets them. Slip walls and outlets are weak, through their ghost states' fluxes only. Projecting slip nodes' velocity onto the wall would undo part of every update the residual asks for, and a node on two slip planes has no single normal to project on.

This is done through:

- `enforce_boundary_node_kernel`
- `zero_constrained_kernel`, from the node's Dirichlet and no-slip entries (`node_bc_kernel`)

This kernel is intentionally limited to boundary nodes and uses indirect `OP_RW` only on `q` through a boundary-node to node map.

## 10. Local Time Stepping

The pseudo-time step at node $i$ is

$$
\Delta t_i = \text{CFL}\frac{V_i}{\lambda_i + \varepsilon}
$$

where the inviscid spectral contribution is accumulated over the node's whole surface, its edges and its boundary sub-faces:

$$
\lambda_i^{inv}
=
\sum_{j\in N(i)} |\Delta S_{ij}|\left(|u_n|+a\right)
+
\sum_{f\in \partial \Omega_i} |\Delta S_{f,i}|\left(|u_n|+a\right).
$$

For viscous runs each edge and boundary sub-face adds, from the node's own state,

$$
\lambda_i^{visc} = \sum 2(\mu+\mu_t)_i \frac{|\Delta S|^2}{\rho_i V_i}.
$$

The implicit scheme takes the same $\lambda_i$ for its pseudo-time term (section 17).

Implementation:

- `edge_spectral_kernel`
- `bface_spectral_kernel`
- `compute_dt_scaled_kernel`

## 11. Runge-Kutta Marching

The code uses a four-stage explicit RK update. Each outer iteration performs:

1. enforce the boundary-node state and store the conservative state into `q0`
2. convert `q` to primitive
3. build gradients if needed (second order, viscous or SA): the stages reuse them
4. compute local pseudo-time step
5. for each RK stage:
   - enforce boundary-node state if needed
   - convert `q` to primitive
   - zero residual
   - accumulate interior edge fluxes
   - accumulate boundary-face fluxes, and in viscous runs the symmetry planes' stress
   - add SA source term
   - zero the enforced components' residual
   - update `q`, keeping `q0` at a node where the result is not finite or falls below
     `rho_floor` or `p_floor`
6. assemble a final residual for monitoring

The update is

$$
\mathbf{Q}^{(k)}_i = \mathbf{Q}^{(0)}_i - \alpha_k \frac{\Delta t_i}{V_i}\mathbf{R}^{(k-1)}_i
$$

with stage coefficients

$$
\alpha = \left[\frac{1}{4},\frac{1}{3},\frac{1}{2},1\right].
$$

Implementation:

- `rk_update_kernel`

## 12. Spalart-Allmaras Source Term

The production and destruction of the standard model (no $f_{t2}$), as nodal sources:

$$
\rho\,(c_{b1}\tilde S\tilde\nu - c_{w1}f_w(\tilde\nu/d)^2),
$$

with the wall distance $d$ of the preprocessed mesh (at least $10^{-8}$), the vorticity magnitude
$\Omega$, $\bar S = \tilde\nu f_{v2}/(\kappa^2 d^2)$, $r = \min(\tilde\nu/(\tilde S\kappa^2d^2), 10)$ and the
standard constants ($c_{b1} = 0.1355$, $c_{b2} = 0.622$, $\sigma = 2/3$, $\kappa = 0.41$, $c_{w2} = 0.3$,
$c_{w3} = 2$, $c_{v1} = 7.1$). $\tilde S = \Omega + \bar S$ is kept positive smoothly, as the NASA
Turbulence Modeling Resource's SA does ($\bar S \ge -c_2\Omega$, else
$\Omega + \Omega(c_2^2\Omega + c_3\bar S)/((c_3-2c_2)\Omega-\bar S)$, $c_2=0.7$, $c_3=0.9$): a hard clip of it
switches in the buffer layer. Nodes with $\tilde\nu \le 0$ have no source. The diffusion is in section 8.

`sa_comp = 1` selects SA-comp, the Turbulence Modeling Resource's compressibility correction (Shur
et al.'s, for mixing layers): the destruction gains

$$
\rho\,C_5\frac{\tilde\nu^2}{a^2}\frac{\partial u_i}{\partial x_j}\frac{\partial u_i}{\partial x_j},
\qquad C_5 = 3.5,
$$

$a$ the local speed of sound. It lowers the eddy viscosity of separated shear layers, and barely
touches attached boundary layers (0.02% in the TMR flat plate's skin friction at M 0.2). The
OAT15A buffet needs it: standard SA damps the shock's oscillation (section 19).

In the implicit scheme's Jacobian the source's derivative by $\tilde\nu$ is a difference of the
source terms themselves: the destruction's through $r$ and $f_w$ as well (near walls about
twice the $2D/\tilde\nu$ of $D\propto\tilde\nu^2$, whose Newton steps overshoot into a
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
- implicit scheme: the Jacobian `A` (an `op_mat` on the edge graph), `bdiag`, `rhs`, `dq`, and in
  dual time stepping the earlier time levels `qn`, `qnm1`

### 13.1 Access Pattern Design

The implementation deliberately uses:

- indirect `OP_INC` for edge-to-node residual and gradient accumulation
- direct writes for node-owned temporaries
- indirect `OP_RW` only where unavoidable on boundary-node state enforcement

Nodal extrema (for a limiter, say) would need indirect `OP_RW` and indirect reductions, a poor fit
for OP2's portability and performance: hence the edge-local limiters of section 6.1.

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
at all leaves 1e30). For the procedural structured meshes (`bump3d`,
`axisymmetric_body`, `naca_wing`) it is exact to the ymin boundary's no-slip
faces within three columns of the node's own, and to a sample of them beyond;
with at most 4096 such faces (a 2D grid's), exact to all of them. Such meshes
written before the distance was computed for them hold the largest `double`,
which switches the SA sources off: generate them again.

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

SU2 meshes (`su2`, `onera_m6`) are tetrahedral: median-dual edges from each
tetrahedron's six, the markers typed by name (`UPPER_SIDE`, `LOWER_SIDE` and
`TIP` walls, `SYMMETRY_FACE` a slip plane, the rest far field). The importer
summed half the tetrahedra's dual faces with the wrong sign before November
2026, so its earlier meshes' control volumes do not close: preprocess them again.

Every node's control volume must close: its edges' dual faces (outward) and its
boundary sub-faces sum to zero. The preprocessor rejects a mesh whose largest
imbalance exceeds `1e-8` of the node's total face area (the builders above
close to round-off, about `1e-14`), as it does non-positive volumes and
zero-area faces.

All connectivity is zero-based in the preprocessed OP2 mesh. The solver stops on a mesh
that lacks a dataset it reads (one written before `bface_node_normals` existed): preprocess
it again. (The solver no longer reads `bnode_slip` and `bnode_normal`.)

The local preprocessing helper also supports procedural structured meshes with
runtime parameter overrides; `bump3d` and `axisymmetric_body` default to
geometries with clearly visible flow features.

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

- [configs/bump3d.cfg](../configs/bump3d.cfg)
- [configs/axisymmetric_body.cfg](../configs/axisymmetric_body.cfg)
- [configs/naca_wing.cfg](../configs/naca_wing.cfg)

## 15. Output Contract

The OP2 executable writes `q`, `primitive`, `dt` and `residual` per node to the configured HDF5
output file (`output_file`, `<stem>.h5`), and beside it:

- `<stem>.residual.csv`: each iteration's L2 norms of the density, energy and SA residuals and the
  density residual's maximum
- in dual time stepping, `<stem>.history.csv`: each physical step's time, wall force (section 18),
  inner iterations, their residual reduction and linear iterations
- with `snapshot_interval` N, `<stem>_NNNNNN.h5` every N steps: `q` and `primitive`, numbered by
  the run's own steps from 1 (a restarted run counts again from 1)

A run restarts from any of these files' `q` (`restart_file`).

Post-processing:

- `nssolver_op2_benchmark_postprocess_helper wall MESH SOLUTION PREFIX 0` writes the no-slip
  wall's nodes with their pressure, density and temperature (`PREFIX_wall.csv`), which
  `scripts/airfoil/cp.py` and `cpstats.py` read
- `nssolver_hdf5_to_vtk_helper` (`scripts/hdf5_to_vtk.sh`) writes a solution's nodes with density,
  pressure and velocity as a legacy VTK point cloud
- `scripts/airfoil/video.py` renders a planar case's dual-time snapshots as a video of one
  symmetry plane (`docs/VALIDATION.md` section 9)

## 16. Validation Strategy

The OP2 path is validated directly on non-periodic cases:

- `box`: exact uniform-state preservation
- `bump`: inviscid nonlinear steady-state regression
- `flatplate_develop`: viscous boundary-layer regression and postprocessing
- `hydra_benchmark`: non-rotating Hydra benchmark run

The shipped validation scripts live under `scripts/` and `tests/`; `docs/VALIDATION.md` gives
their residual bands. `tests/test_implicit.sh` runs the implicit scheme on the bump across OP2's
backends, both solver libraries and the options of section 17. The physics is verified against
reference solutions and experiments by the cases of `docs/VALIDATION.md` sections 6-10
(section 19).

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
$\omega$ = `implicit_relax`, for `iterations` iterations (steady runs). An update is kept
where it is admissible, as the RK update's, and counted as rejected otherwise.

`update_limit` > 0 scales each node's update so that its density and pressure change by at most
that fraction. In transonic starts a few nodes would otherwise overshoot; skipping their
updates, as the admissibility check does, leaves them out of step with their neighbours and
blows up (`configs/oat15a.cfg` takes 0.2).

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
`sa_scale = 1` gives the free stream's own reference. The residual history
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

## 18. Dual Time Stepping

With `time_step` $\Delta t > 0$ (implicit scheme) each of `physical_steps` steps solves

$$
\mathbf{R}(\mathbf{Q}) + V\,\frac{c_0\mathbf{Q} - c_1\mathbf{Q}^n + c_2\mathbf{Q}^{n-1}}{\Delta t} = 0,
$$

second-order backward differences ($c = 3/2, 2, 1/2$; `time_order = 2`) after a first-order first
step ($c = 1, 1, 0$), by the pseudo-time iterations of section 17: the physical term adds
$c_0V/\Delta t$ to the diagonal blocks (`jac_node_kernel`), the matrix-free operator differences
the whole residual, and the CFL ramp restarts every step. A step ends when the density
residual's L2 norm has fallen by `inner_tolerance` or after `inner_iterations`.

`restart_file` starts from a solution's or snapshot's `q`, which holds one time level: a restarted
run's first step is first-order again. `snapshot_interval` writes `q` and the primitives every
that many steps (section 15).

The inner iterations are inexact Newton steps: `configs/solve_dual.json` solves each to a relative
residual of 0.1 rather than 1e-3, which keeps their count (11 to 14 a step on the transonic
circular arc) and takes a sixth of the linear iterations, 2.4 times less time.

The output file's `.history.csv` has each step's force on the no-slip walls,
$\sum(p\,\Delta\mathbf{S} - \mathbf{R}_{1..3})$ over their nodes' sub-faces: a wall node's momentum
residual, which enforcing no slip discards, is its flux $p\Delta\mathbf{S}$ through the wall less
the traction that would balance it.

Implementation:

- `unsteady_residual_kernel`, `implicit_residual`
- `wall_pressure_force_kernel`, `wall_force_kernel`

## 19. Verification Runs

`docs/VALIDATION.md` (sections 6-10) gives each verification case's set-up, commands,
post-processing and results:

- The NASA TMR zero-pressure-gradient flat plate (SA, M 0.2, Re 5e6): the skin friction at
  x = 0.97 matches FUN3D's on the same grid to 0.03% from the 69x49 grid to the 545x385 one
  (0.0026957 against 0.0026950 on 69x49).
- Laminar vortex shedding behind a cylinder at Re 100, by dual time stepping: Strouhal number
  0.163, mean drag 1.36 and lift amplitude 0.32 on the 192x128 grid at 50 steps a period.
- The 18% circular arc (SA, M 0.76, Re 11e6): self-excited shock oscillation at Strouhal number
  0.157 (the experiment's reduced frequency 0.49 is 0.156), lift $\pm 0.33$.
- The ONERA OAT15A buffet (SA-comp, M 0.73, Re 3e6): onset at about 3.2 degrees (experiment
  3.1-3.25), $\Delta C_L$ 0.105, 0.220 and 0.317 at 3.25, 3.5 and 3.9 degrees, Strouhal number
  0.070-0.071 (experiment 0.066), the same with time steps of 0.24 and 0.06 c/U; the shock's
  pressure fluctuation peaks about 15% above the experiment's and 0.04-0.05 c further aft.
  Standard SA damps the shock's oscillation at every incidence tried (3.9 to 5.5 degrees).
- The ONERA M6 wing (Euler, M 0.8395, 3.06 degrees, SU2's tutorial mesh): converged to round-off
  in 150 implicit iterations, $C_L$ 0.288 and the lambda shock against Schmitt and Charpin's
  sections (upper-surface $C_p$ within 0.07-0.12 RMS, the shocks a little aft of the
  experiment's, as inviscid flow puts them).
