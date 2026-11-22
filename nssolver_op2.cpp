#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "op_seq.h"
#include "op_hdf5.h"
#include "op_solve.h"

#include "op2_config.h"
#include "op2_kernels.h"
#include <op_profile.h>

double op2_gamma = 1.4;
double op2_gas_constant = 287.05;
double op2_prandtl = 0.72;
double op2_mu_ref = 1.7894e-5;
double op2_t_ref = 288.15;
double op2_sutherland = 110.4;
double op2_freestream[6] = {1.225, 120.0, 0.0, 0.0, 101325.0, 0.0};
double op2_rho_floor = 1.0e-4;
double op2_p_floor = 100.0;
int op2_second_order = 0;
int op2_limiter = 1;
int op2_include_viscous = 0;
int op2_include_sa = 0;
int op2_outlet_backflow = 0;
double op2_scale[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

struct ResidualNorms {
  double l2_rho = 0.0;
  double l2_rhoE = 0.0;
  double linf_rho = 0.0;
};

struct Mesh {
  op_set nodes, edges, bfaces, bnodes;
  op_map edge_to_nodes, bface_to_nodes, bnode_to_node;
  op_dat node_coords, node_volume, node_wall_distance, edge_weights;
  op_dat bface_normal, bface_area, bface_num_nodes, bface_type, bface_shares;
  op_dat bnode_dirichlet, bnode_wall;
  op_dat node_bc;  // the node's Dirichlet and no-slip entries (constrained_op2)
};

static void usage(const char *argv0) {
  std::cerr << "usage: " << argv0 << " --config <config.cfg> [--<key> <value> ...]\n"
            << "  any config key, '-' or '_': e.g. --time-scheme implicit --implicit-cfl 50\n"
            << "  --solve-config FILE --pc lib|bjacobi|neumann --terms M --pmat A|decoupled\n"
            << "  --op assembled|matfree --partition kway|random|inertial --reorder 0|1\n"
            << "  --jacobian-interval N\n"
            << "  --dump-interval N --dump-prefix PATH --diags N --timing 0|1 --check FACTOR\n";
}

static void write_residual_history_csv_op2(const std::string &path, const std::vector<double> &l2_rho,
                                           const std::vector<double> &l2_rhoE, const std::vector<double> &linf_rho) {
  std::filesystem::path out_path(path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open residual history file: " + path);
  }
  out << "iteration,l2_rho,l2_rhoE,linf_rho\n";
  for (std::size_t i = 0; i < l2_rho.size(); ++i) {
    out << i << ',' << l2_rho[i] << ',' << l2_rhoE[i] << ',' << linf_rho[i] << '\n';
  }
}

// --- The residual R(q), which both schemes drive to zero ----------------------

// For viscous runs, impose the free stream at Dirichlet nodes and no velocity
// at no-slip ones on the nodal conservative state.
static void enforce_boundary(const Mesh &m, op_dat q) {
  op_par_loop(enforce_boundary_node_kernel, "enforce_boundary_node_kernel", m.bnodes,
              op_arg_dat(m.bnode_dirichlet, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(m.bnode_wall, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(q, 0, m.bnode_to_node, NVAR_OP2, "double", OP_RW));
}

static void to_primitive(const Mesh &m, op_dat q, op_dat prim) {
  op_par_loop(q_to_primitive_kernel, "q_to_primitive_kernel", m.nodes,
              op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(prim, -1, OP_ID, NPRIM_OP2, "double", OP_WRITE));
}

// Green-Gauss gradients from interior edges and boundary faces, divided by the
// nodal control volume, for second-order reconstruction and viscous fluxes;
// inviscid first-order runs still zero them so the flux kernels read no stale
// values.
static void gradients(const Mesh &m, bool needed, op_dat prim, op_dat grad) {
  op_par_loop(zero_grad_kernel, "zero_grad_kernel", m.nodes,
              op_arg_dat(grad, -1, OP_ID, NGRAD_OP2, "double", OP_WRITE));
  if (!needed) return;

  op_par_loop(edge_grad_kernel, "edge_grad_kernel", m.edges,
              op_arg_dat(prim, 0, m.edge_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 1, m.edge_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(m.edge_weights, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(grad, 0, m.edge_to_nodes, NGRAD_OP2, "double", OP_INC),
              op_arg_dat(grad, 1, m.edge_to_nodes, NGRAD_OP2, "double", OP_INC));

  op_par_loop(bface_grad_kernel, "bface_grad_kernel", m.bfaces,
              op_arg_dat(m.bface_num_nodes, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(m.bface_shares, -1, OP_ID, 12, "double", OP_READ),
              op_arg_dat(prim, 0, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 1, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 2, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 3, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(grad, 0, m.bface_to_nodes, NGRAD_OP2, "double", OP_INC),
              op_arg_dat(grad, 1, m.bface_to_nodes, NGRAD_OP2, "double", OP_INC),
              op_arg_dat(grad, 2, m.bface_to_nodes, NGRAD_OP2, "double", OP_INC),
              op_arg_dat(grad, 3, m.bface_to_nodes, NGRAD_OP2, "double", OP_INC));

  op_par_loop(normalize_grad_kernel, "normalize_grad_kernel", m.nodes,
              op_arg_dat(m.node_volume, -1, OP_ID, 1, "double", OP_READ),
              op_arg_dat(grad, -1, OP_ID, NGRAD_OP2, "double", OP_RW));
}

// Interior and boundary fluxes, then the SA source, into a cleared residual;
// none in the components the viscous scheme enforces. force, if given, takes
// the flow's force on the no-slip walls first.
static void flux_residual(const Mesh &m, op_dat prim, op_dat grad, op_dat res, double *force = nullptr) {
  op_par_loop(zero_var_kernel, "zero_var_kernel", m.nodes,
              op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));

  // In second-order mode the kernel performs edge-local limited reconstruction
  // directly from the endpoint gradients.
  op_par_loop(edge_flux_kernel, "edge_flux_kernel", m.edges,
              op_arg_dat(m.node_coords, 0, m.edge_to_nodes, 3, "double", OP_READ),
              op_arg_dat(m.node_coords, 1, m.edge_to_nodes, 3, "double", OP_READ),
              op_arg_dat(prim, 0, m.edge_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 1, m.edge_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(grad, 0, m.edge_to_nodes, NGRAD_OP2, "double", OP_READ),
              op_arg_dat(grad, 1, m.edge_to_nodes, NGRAD_OP2, "double", OP_READ),
              op_arg_dat(m.edge_weights, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(res, 0, m.edge_to_nodes, NVAR_OP2, "double", OP_INC),
              op_arg_dat(res, 1, m.edge_to_nodes, NVAR_OP2, "double", OP_INC));

  // Physical boundary fluxes, each node's from its own state and its
  // boundary-type-specific ghost through its sub-face.
  op_par_loop(boundary_flux_kernel, "boundary_flux_kernel", m.bfaces,
              op_arg_dat(m.bface_type, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(m.bface_normal, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(m.bface_area, -1, OP_ID, 1, "double", OP_READ),
              op_arg_dat(m.bface_num_nodes, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(m.bface_shares, -1, OP_ID, 12, "double", OP_READ),
              op_arg_dat(m.node_volume, 0, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 1, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 2, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 3, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(prim, 0, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 1, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 2, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 3, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(res, 0, m.bface_to_nodes, NVAR_OP2, "double", OP_INC),
              op_arg_dat(res, 1, m.bface_to_nodes, NVAR_OP2, "double", OP_INC),
              op_arg_dat(res, 2, m.bface_to_nodes, NVAR_OP2, "double", OP_INC),
              op_arg_dat(res, 3, m.bface_to_nodes, NVAR_OP2, "double", OP_INC));
  if (op2_include_viscous)
    op_par_loop(symmetry_stress_kernel, "symmetry_stress_kernel", m.bfaces,
                op_arg_dat(m.bface_type, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(m.bface_normal, -1, OP_ID, 3, "double", OP_READ),
                op_arg_dat(m.bface_area, -1, OP_ID, 1, "double", OP_READ),
                op_arg_dat(m.bface_num_nodes, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(m.bface_shares, -1, OP_ID, 12, "double", OP_READ),
                op_arg_dat(prim, 0, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                op_arg_dat(prim, 1, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                op_arg_dat(prim, 2, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                op_arg_dat(prim, 3, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                op_arg_dat(grad, 0, m.bface_to_nodes, NGRAD_OP2, "double", OP_READ),
                op_arg_dat(grad, 1, m.bface_to_nodes, NGRAD_OP2, "double", OP_READ),
                op_arg_dat(grad, 2, m.bface_to_nodes, NGRAD_OP2, "double", OP_READ),
                op_arg_dat(grad, 3, m.bface_to_nodes, NGRAD_OP2, "double", OP_READ),
                op_arg_dat(res, 0, m.bface_to_nodes, NVAR_OP2, "double", OP_INC),
                op_arg_dat(res, 1, m.bface_to_nodes, NVAR_OP2, "double", OP_INC),
                op_arg_dat(res, 2, m.bface_to_nodes, NVAR_OP2, "double", OP_INC),
                op_arg_dat(res, 3, m.bface_to_nodes, NVAR_OP2, "double", OP_INC));

  op_par_loop(sa_source_kernel, "sa_source_kernel", m.nodes,
              op_arg_dat(prim, -1, OP_ID, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(grad, -1, OP_ID, NGRAD_OP2, "double", OP_READ),
              op_arg_dat(m.node_wall_distance, -1, OP_ID, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, -1, OP_ID, 1, "double", OP_READ),
              op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_RW));

  if (force != nullptr) {
    force[0] = force[1] = force[2] = 0.0;
    op_par_loop(wall_pressure_force_kernel, "wall_pressure_force_kernel", m.bfaces,
                op_arg_dat(m.bface_type, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(m.bface_num_nodes, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(m.bface_shares, -1, OP_ID, 12, "double", OP_READ),
                op_arg_dat(prim, 0, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                op_arg_dat(prim, 1, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                op_arg_dat(prim, 2, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                op_arg_dat(prim, 3, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                op_arg_gbl(force, 3, "double", OP_INC));
    op_par_loop(wall_force_kernel, "wall_force_kernel", m.bnodes,
                op_arg_dat(m.bnode_wall, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(res, 0, m.bnode_to_node, NVAR_OP2, "double", OP_READ),
                op_arg_gbl(force, 3, "double", OP_INC));
  }

  op_par_loop(zero_constrained_kernel, "zero_constrained_kernel", m.nodes,
              op_arg_dat(m.node_bc, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_RW));
}

// The local spectral radius over each node's whole surface, edges' and
// boundary sub-faces', and from it each node's explicit time step.
static void local_time_step(const Mesh &m, op_dat prim, double cfl, op_dat spectral, op_dat dt) {
  op_par_loop(zero_scalar_kernel, "zero_scalar_kernel", m.nodes,
              op_arg_dat(spectral, -1, OP_ID, 1, "double", OP_WRITE));

  op_par_loop(edge_spectral_kernel, "edge_spectral_kernel", m.edges,
              op_arg_dat(prim, 0, m.edge_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 1, m.edge_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(m.edge_weights, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(m.node_volume, 0, m.edge_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 1, m.edge_to_nodes, 1, "double", OP_READ),
              op_arg_dat(spectral, 0, m.edge_to_nodes, 1, "double", OP_INC),
              op_arg_dat(spectral, 1, m.edge_to_nodes, 1, "double", OP_INC));
  op_par_loop(bface_spectral_kernel, "bface_spectral_kernel", m.bfaces,
              op_arg_dat(m.bface_num_nodes, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(m.bface_shares, -1, OP_ID, 12, "double", OP_READ),
              op_arg_dat(prim, 0, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 1, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 2, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 3, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(m.node_volume, 0, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 1, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 2, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 3, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(spectral, 0, m.bface_to_nodes, 1, "double", OP_INC),
              op_arg_dat(spectral, 1, m.bface_to_nodes, 1, "double", OP_INC),
              op_arg_dat(spectral, 2, m.bface_to_nodes, 1, "double", OP_INC),
              op_arg_dat(spectral, 3, m.bface_to_nodes, 1, "double", OP_INC));
  op_par_loop(compute_dt_scaled_kernel, "compute_dt_scaled_kernel", m.nodes,
              op_arg_dat(m.node_volume, -1, OP_ID, 1, "double", OP_READ),
              op_arg_dat(spectral, -1, OP_ID, 1, "double", OP_READ),
              op_arg_gbl(&cfl, 1, "double", OP_READ),
              op_arg_dat(dt, -1, OP_ID, 1, "double", OP_WRITE));
}

static ResidualNorms residual_norms(const Mesh &m, op_dat res) {
  double l2_rho_sq = 0.0;
  double l2_rhoE_sq = 0.0;
  double linf_rho = 0.0;
  op_par_loop(residual_norm_kernel, "residual_norm_kernel", m.nodes,
              op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_gbl(&l2_rho_sq, 1, "double", OP_INC),
              op_arg_gbl(&l2_rhoE_sq, 1, "double", OP_INC),
              op_arg_gbl(&linf_rho, 1, "double", OP_MAX));
  ResidualNorms norms;
  norms.l2_rho = std::sqrt(l2_rho_sq / static_cast<double>(op_get_size(m.nodes)));
  norms.l2_rhoE = std::sqrt(l2_rhoE_sq / static_cast<double>(op_get_size(m.nodes)));
  norms.linf_rho = linf_rho;
  return norms;
}

// --- The implicit scheme ----------------------------------------------------

// What the linear solve's callbacks reach: the state the operator linearises
// about, the matrix the par-loop preconditioners read, and work dats.
struct Implicit {
  const Mesh *m;
  bool full_gradients;
  op_dat q, res, spectral, bdiag;
  op_dat dinv, y;
  op_dat q_pert, prim_pert, grad_pert, res_pert;
  op_mat A;
  int terms;
  double cfl;
  // Dual time stepping: the two earlier time levels and c0, c1, c2 over the
  // time step (all 0 when steady).
  op_dat qn, qnm1;
  double physical[3];
};

// The residual of the implicit scheme at state q: flux_residual's, and in dual
// time stepping the physical term (unsteady_residual_kernel).
static void implicit_residual(Implicit &c, op_dat q, op_dat prim, op_dat grad, op_dat res,
                              double *force = nullptr) {
  const Mesh &m = *c.m;
  flux_residual(m, prim, grad, res, force);
  if (c.qn == NULL) return;
  op_par_loop(unsteady_residual_kernel, "unsteady_residual_kernel", m.nodes,
              op_arg_dat(m.node_bc, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(m.node_volume, -1, OP_ID, 1, "double", OP_READ),
              op_arg_gbl(c.physical, 3, "double", OP_READ),
              op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(c.qn, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(c.qnm1, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_RW));
}

// The first-order Jacobian, or its components' blocks alone (decoupled),
// into M's storage in the solver library.
static void assemble_jacobian(Implicit &c, op_mat M, op_dat prim, op_dat grad, int decoupled) {
  const Mesh &m = *c.m;
  op_mat_zero(M);
  op_par_loop(zero_block_kernel, "zero_block_kernel", m.nodes,
              op_arg_dat(c.bdiag, -1, OP_ID, NBLOCK_OP2, "double", OP_WRITE));
  op_par_loop(jac_bface_kernel, "jac_bface_kernel", m.bfaces,
              op_arg_dat(m.bface_type, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(m.bface_normal, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(m.bface_area, -1, OP_ID, 1, "double", OP_READ),
              op_arg_dat(m.bface_num_nodes, -1, OP_ID, 1, "int", OP_READ),
              op_arg_dat(m.bface_shares, -1, OP_ID, 12, "double", OP_READ),
              op_arg_dat(m.node_volume, 0, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 1, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 2, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, 3, m.bface_to_nodes, 1, "double", OP_READ),
              op_arg_dat(prim, 0, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 1, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 2, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 3, m.bface_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(c.bdiag, 0, m.bface_to_nodes, NBLOCK_OP2, "double", OP_INC),
              op_arg_dat(c.bdiag, 1, m.bface_to_nodes, NBLOCK_OP2, "double", OP_INC),
              op_arg_dat(c.bdiag, 2, m.bface_to_nodes, NBLOCK_OP2, "double", OP_INC),
              op_arg_dat(c.bdiag, 3, m.bface_to_nodes, NBLOCK_OP2, "double", OP_INC));
  op_par_loop(jac_edge_kernel, "jac_edge_kernel", m.edges,
              op_arg_dat(m.node_coords, 0, m.edge_to_nodes, 3, "double", OP_READ),
              op_arg_dat(m.node_coords, 1, m.edge_to_nodes, 3, "double", OP_READ),
              op_arg_dat(prim, 0, m.edge_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(prim, 1, m.edge_to_nodes, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(m.edge_weights, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(m.node_bc, 0, m.edge_to_nodes, 2, "int", OP_READ),
              op_arg_dat(m.node_bc, 1, m.edge_to_nodes, 2, "int", OP_READ),
              op_arg_gbl(&decoupled, 1, "int", OP_READ),
              op_arg_mat(M, 0, m.edge_to_nodes, 0, m.edge_to_nodes, NVAR_OP2, "double", OP_INC),
              op_arg_mat(M, 0, m.edge_to_nodes, 1, m.edge_to_nodes, NVAR_OP2, "double", OP_INC),
              op_arg_mat(M, 1, m.edge_to_nodes, 0, m.edge_to_nodes, NVAR_OP2, "double", OP_INC),
              op_arg_mat(M, 1, m.edge_to_nodes, 1, m.edge_to_nodes, NVAR_OP2, "double", OP_INC));
  op_par_loop(jac_node_kernel, "jac_node_kernel", m.nodes,
              op_arg_dat(m.node_bc, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(c.spectral, -1, OP_ID, 1, "double", OP_READ),
              op_arg_dat(c.bdiag, -1, OP_ID, NBLOCK_OP2, "double", OP_READ),
              op_arg_dat(prim, -1, OP_ID, NPRIM_OP2, "double", OP_READ),
              op_arg_dat(grad, -1, OP_ID, NGRAD_OP2, "double", OP_READ),
              op_arg_dat(m.node_wall_distance, -1, OP_ID, 1, "double", OP_READ),
              op_arg_dat(m.node_volume, -1, OP_ID, 1, "double", OP_READ),
              op_arg_gbl(&c.cfl, 1, "double", OP_READ),
              op_arg_gbl(c.physical, 1, "double", OP_READ),
              op_arg_gbl(&decoupled, 1, "int", OP_READ),
              op_arg_mat(M, -1, OP_ID, -1, OP_ID, NVAR_OP2, "double", OP_INC));
  op_mat_assemble(M);
}

static void bjacobi(op_dat r, op_dat z, void *ctx) {
  Implicit *c = (Implicit *)ctx;
  op_par_loop(apply_block_kernel, "apply_block_kernel", c->m->nodes,
              op_arg_dat(c->dinv, -1, OP_ID, NBLOCK_OP2, "double", OP_READ),
              op_arg_dat(r, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(z, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));
}

// y = A x through the assembled matrix, read in solver memory.
static void matvec(Implicit *c, op_dat x, op_dat y) {
  const Mesh &m = *c->m;
  op_par_loop(mat_diag_kernel, "mat_diag_kernel", m.nodes,
              op_arg_mat(c->A, -1, OP_ID, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(y, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));
  op_par_loop(mat_edge_kernel, "mat_edge_kernel", m.edges,
              op_arg_mat(c->A, 0, m.edge_to_nodes, 1, m.edge_to_nodes, NVAR_OP2, "double", OP_READ),
              op_arg_mat(c->A, 1, m.edge_to_nodes, 0, m.edge_to_nodes, NVAR_OP2, "double", OP_READ),
              op_arg_dat(x, 0, m.edge_to_nodes, NVAR_OP2, "double", OP_READ),
              op_arg_dat(x, 1, m.edge_to_nodes, NVAR_OP2, "double", OP_READ),
              op_arg_dat(y, 0, m.edge_to_nodes, NVAR_OP2, "double", OP_INC),
              op_arg_dat(y, 1, m.edge_to_nodes, NVAR_OP2, "double", OP_INC));
}

// z = sum_{i<=m} (I - Dinv A)^i Dinv r: block Jacobi, then m steps that each
// apply A through its blocks in solver memory.
static void neumann(op_dat r, op_dat z, void *ctx) {
  Implicit *c = (Implicit *)ctx;
  bjacobi(r, z, ctx);
  for (int i = 0; i < c->terms; i++) {
    matvec(c, z, c->y);
    op_par_loop(neumann_step_kernel, "neumann_step_kernel", c->m->nodes,
                op_arg_dat(c->dinv, -1, OP_ID, NBLOCK_OP2, "double", OP_READ),
                op_arg_dat(r, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                op_arg_dat(c->y, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                op_arg_dat(z, -1, OP_ID, NVAR_OP2, "double", OP_RW));
  }
}

// y = (V / dt + dR/dq) x by a difference of the full (second-order) residual:
// Jacobian-free Newton-Krylov, the assembled first-order matrix its
// preconditioner.
static void matfree(op_dat x, op_dat y, void *ctx) {
  Implicit *c = (Implicit *)ctx;
  const Mesh &m = *c->m;
  // A step of 1e-7 relative to the scaled state's RMS entry.
  double x2 = 0.0;
  double q2 = 0.0;
  op_par_loop(sum_squares_kernel, "sum_squares_kernel", m.nodes,
              op_arg_dat(x, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(c->q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_gbl(&x2, 1, "double", OP_INC),
              op_arg_gbl(&q2, 1, "double", OP_INC));
  const double entries = (double)NVAR_OP2 * op_get_size(m.nodes);
  double eps = x2 > 0.0 ? 1.0e-7 * (1.0 + std::sqrt(q2 / entries)) / std::sqrt(x2 / entries) : 1.0;
  op_par_loop(perturb_kernel, "perturb_kernel", m.nodes,
              op_arg_dat(c->q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_gbl(&eps, 1, "double", OP_READ),
              op_arg_dat(c->q_pert, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));
  to_primitive(m, c->q_pert, c->prim_pert);
  gradients(m, c->full_gradients, c->prim_pert, c->grad_pert);
  implicit_residual(*c, c->q_pert, c->prim_pert, c->grad_pert, c->res_pert);
  op_par_loop(matfree_combine_kernel, "matfree_combine_kernel", m.nodes,
              op_arg_dat(m.node_bc, -1, OP_ID, 2, "int", OP_READ),
              op_arg_dat(c->spectral, -1, OP_ID, 1, "double", OP_READ),
              op_arg_gbl(&c->cfl, 1, "double", OP_READ),
              op_arg_gbl(&eps, 1, "double", OP_READ),
              op_arg_dat(x, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(c->res_pert, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(c->res, -1, OP_ID, NVAR_OP2, "double", OP_READ),
              op_arg_dat(y, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));
}

static double wall_time() {
  deviceSync();
  double cpu, now;
  op_timers(&cpu, &now);
  return now;
}

int main(int argc, char **argv) {
  try {
    SolverConfig cfg;
    std::string config_path;
    std::vector<std::pair<std::string, std::string>> overrides;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
        config_path = argv[++i];
      } else if (std::strncmp(argv[i], "--", 2) == 0 && i + 1 < argc) {
        std::string key = argv[i] + 2;
        std::replace(key.begin(), key.end(), '-', '_');
        overrides.emplace_back(key, argv[++i]);
      }
    }
    if (config_path.empty()) {
      usage(argv[0]);
      return 1;
    }
    load_config_file(config_path, cfg);
    for (const auto &[key, value] : overrides) {
      if (!apply_config_value(cfg, key, value)) {
        usage(argv[0]);
        throw std::runtime_error("unknown option --" + key);
      }
    }
    validate_config(cfg);
    if (cfg.init_mode == "hydra_benchmark") {
      derive_hydra_benchmark_primitive(cfg);
    }
    const bool implicit = cfg.time_scheme == "implicit";

    op2_gamma = cfg.gamma;
    op2_gas_constant = cfg.gas_constant;
    op2_rho_floor = cfg.rho_floor;
    op2_p_floor = cfg.p_floor;
    op2_second_order = cfg.second_order;
    op2_limiter = cfg.limiter;
    op2_include_viscous = cfg.include_viscous;
    op2_include_sa = cfg.include_sa;
    op2_outlet_backflow = cfg.outlet_backflow == "freestream";
    for (int i = 0; i < 6; ++i) op2_freestream[i] = cfg.primitive[i];

    // The implicit scheme's reference state: rho, rho a, rho a^2 and rho nu
    // of the free stream, nu at least its laminar viscosity.
    {
      const double *p = cfg.primitive;
      const double a = std::sqrt(cfg.gamma * p[4] / p[0]);
      const double nu = dynamic_viscosity_op2(p[4] / (p[0] * cfg.gas_constant)) / p[0];
      const double scale[6] = {p[0], p[0] * a, p[0] * a, p[0] * a, p[0] * a * a, p[0] * std::max(p[5], nu)};
      for (int i = 0; i < 6; ++i) op2_scale[i] = scale[i];
    }

    op_init(argc, argv, cfg.diags);
    if (op_is_root()) std::cout << "[phase] op_init\n";

    // Expose solver configuration constants to OP2 code generation so CUDA
    // backends materialize matching device-side symbols.
    op_decl_const(1, "double", &op2_gamma);
    op_decl_const(1, "double", &op2_gas_constant);
    op_decl_const(1, "double", &op2_prandtl);
    op_decl_const(1, "double", &op2_mu_ref);
    op_decl_const(1, "double", &op2_t_ref);
    op_decl_const(1, "double", &op2_sutherland);
    op_decl_const(NPRIM_OP2, "double", op2_freestream);
    op_decl_const(1, "double", &op2_rho_floor);
    op_decl_const(1, "double", &op2_p_floor);
    op_decl_const(1, "int", &op2_second_order);
    op_decl_const(1, "int", &op2_limiter);
    op_decl_const(1, "int", &op2_include_viscous);
    op_decl_const(1, "int", &op2_include_sa);
    op_decl_const(1, "int", &op2_outlet_backflow);
    op_decl_const(NVAR_OP2, "double", op2_scale);

    const char *mesh_file = cfg.mesh_file.c_str();
    Mesh m;
    if (op_is_root()) std::cout << "[phase] decl sets\n";
    m.nodes = op_decl_set_hdf5_infer_size(mesh_file, "nodes", "node_coordinates");
    m.edges = op_decl_set_hdf5_infer_size(mesh_file, "edges", "edge-->node");
    m.bfaces = op_decl_set_hdf5_infer_size(mesh_file, "bfaces", "bface-->node");
    m.bnodes = op_decl_set_hdf5_infer_size(mesh_file, "bnodes", "bnode-->node");

    if (op_is_root()) std::cout << "[phase] decl maps\n";
    m.edge_to_nodes = op_decl_map_hdf5(m.edges, m.nodes, 2, mesh_file, "edge-->node");
    m.bface_to_nodes = op_decl_map_hdf5(m.bfaces, m.nodes, 4, mesh_file, "bface-->node");
    m.bnode_to_node = op_decl_map_hdf5(m.bnodes, m.nodes, 1, mesh_file, "bnode-->node");

    if (op_is_root()) std::cout << "[phase] decl dats\n";
    m.node_coords = op_decl_dat_hdf5(m.nodes, 3, "double", mesh_file, "node_coordinates");
    m.node_volume = op_decl_dat_hdf5(m.nodes, 1, "double", mesh_file, "node_volume");
    m.node_wall_distance = op_decl_dat_hdf5(m.nodes, 1, "double", mesh_file, "node_wall_distance");
    m.edge_weights = op_decl_dat_hdf5(m.edges, 3, "double", mesh_file, "edge_weights");
    m.bface_normal = op_decl_dat_hdf5(m.bfaces, 3, "double", mesh_file, "bface_normal");
    m.bface_area = op_decl_dat_hdf5(m.bfaces, 1, "double", mesh_file, "bface_area");
    m.bface_num_nodes = op_decl_dat_hdf5(m.bfaces, 1, "int", mesh_file, "bface_num_nodes");
    m.bface_type = op_decl_dat_hdf5(m.bfaces, 1, "int", mesh_file, "bface_type");
    m.bface_shares = op_decl_dat_hdf5(m.bfaces, 12, "double", mesh_file, "bface_node_normals");
    op_dat bface_group = op_decl_dat_hdf5(m.bfaces, 1, "int", mesh_file, "bface_group");
    (void)bface_group;
    m.bnode_dirichlet = op_decl_dat_hdf5(m.bnodes, 1, "int", mesh_file, "bnode_dirichlet");
    m.bnode_wall = op_decl_dat_hdf5(m.bnodes, 1, "int", mesh_file, "bnode_wall");

    if (cfg.partition == "random")
      op_partition("RANDOM", NULL, m.nodes, NULL, NULL);
    else if (cfg.partition == "inertial")
      op_partition("INERTIAL", NULL, m.nodes, NULL, m.node_coords);
    else
      op_partition("PARMETIS", "KWAY", m.edges, m.edge_to_nodes, NULL);
    if (cfg.reorder) {
      op_set_coords(m.nodes, m.node_coords);
      op_renumber(m.edge_to_nodes);
    }

    op_dat q = cfg.restart_file.empty()
                   ? op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "q")
                   : op_decl_dat_hdf5(m.nodes, NVAR_OP2, "double", cfg.restart_file.c_str(), "q");
    op_dat q0 = op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "q0");
    op_dat prim = op_decl_dat_temp_char(m.nodes, NPRIM_OP2, "double", sizeof(double), "primitive");
    op_dat res = op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "residual");
    op_dat spectral = op_decl_dat_temp_char(m.nodes, 1, "double", sizeof(double), "spectral");
    op_dat dt = op_decl_dat_temp_char(m.nodes, 1, "double", sizeof(double), "dt");
    op_dat grad = op_decl_dat_temp_char(m.nodes, NGRAD_OP2, "double", sizeof(double), "grad");
    const bool full_gradients = cfg.second_order || cfg.include_viscous || cfg.include_sa;

    // Which components each node's boundary entries fix.
    m.node_bc = op_decl_dat_temp_char(m.nodes, 2, "int", sizeof(int), "node_bc");
    op_par_loop(zero_node_bc_kernel, "zero_node_bc_kernel", m.nodes,
                op_arg_dat(m.node_bc, -1, OP_ID, 2, "int", OP_WRITE));
    op_par_loop(node_bc_kernel, "node_bc_kernel", m.bnodes,
                op_arg_dat(m.bnode_dirichlet, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(m.bnode_wall, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(m.node_bc, 0, m.bnode_to_node, 2, "int", OP_INC));

    // The implicit scheme's matrices, solver and work dats.
    Implicit imp{};
    op_mat P = NULL;
    op_solver solver = NULL;
    op_dat rhs = NULL, dq = NULL;
    if (implicit) {
      op_solve_init();
      if (!cfg.solve_config.empty()) op_solve_set_config(cfg.solve_config.c_str());
      imp.m = &m;
      imp.full_gradients = full_gradients;
      imp.q = q;
      imp.res = res;
      imp.spectral = spectral;
      imp.terms = cfg.terms;
      imp.cfl = cfg.implicit_cfl;
      imp.bdiag = op_decl_dat_temp_char(m.nodes, NBLOCK_OP2, "double", sizeof(double), "bdiag");
      rhs = op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "rhs");
      dq = op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "dq");
      if (cfg.pc != "lib")
        imp.dinv = op_decl_dat_temp_char(m.nodes, NBLOCK_OP2, "double", sizeof(double), "dinv");
      if (cfg.pc == "neumann")
        imp.y = op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "neumann_y");
      if (cfg.op == "matfree") {
        imp.q_pert = op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "q_pert");
        imp.prim_pert = op_decl_dat_temp_char(m.nodes, NPRIM_OP2, "double", sizeof(double), "prim_pert");
        imp.grad_pert = op_decl_dat_temp_char(m.nodes, NGRAD_OP2, "double", sizeof(double), "grad_pert");
        imp.res_pert = op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "res_pert");
      }

      op_sparsity sp = op_decl_sparsity(m.nodes, m.nodes, 1, &m.edge_to_nodes, &m.edge_to_nodes, "jacobian");
      imp.A = op_decl_mat(sp, NVAR_OP2, "double", "A");
      if (cfg.pmat == "decoupled") P = op_decl_mat(sp, NVAR_OP2, "double", "P");
      solver = op_decl_solver(imp.A, (cfg.pc == "lib" ? std::string("nssolver") : "nssolver_" + cfg.pc).c_str());
      if (cfg.pc == "bjacobi") op_solver_set_pc(solver, bjacobi, &imp);
      if (cfg.pc == "neumann") op_solver_set_pc(solver, neumann, &imp);
      if (P != NULL) op_solver_set_pc_mat(solver, P);
      if (cfg.op == "matfree") op_solver_set_operator(solver, matfree, &imp);
    }

    if (op_is_root()) std::cout << "[phase] initialize\n";
    if (cfg.restart_file.empty())
      op_par_loop(initialize_q_kernel, "initialize_q_kernel", m.nodes,
                  op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));

    double initial_l2_rho = -1.0;
    std::vector<double> history_l2_rho;
    std::vector<double> history_l2_rhoE;
    std::vector<double> history_linf_rho;
    history_l2_rho.reserve(static_cast<std::size_t>(cfg.iterations));
    history_l2_rhoE.reserve(static_cast<std::size_t>(cfg.iterations));
    history_linf_rho.reserve(static_cast<std::size_t>(cfg.iterations));

    // The implicit scheme's phases, and its linear solves.
    double assembly_time = 0.0, commit_time = 0.0, solve_time = 0.0;
    long long linear_iterations = 0;
    int unconverged_solves = 0, rejected_total = 0;
    double cfl_max = cfg.implicit_cfl_max > 0.0 ? cfg.implicit_cfl_max : cfg.implicit_cfl;

    double wall_t1, wall_t2, cpu;
    op_timers(&cpu, &wall_t1);

    // The implicit scheme starts from the residual at the initial state.
    const bool unsteady = implicit && cfg.time_step > 0.0;
    if (unsteady) {
      imp.qn = op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "qn");
      imp.qnm1 = op_decl_dat_temp_char(m.nodes, NVAR_OP2, "double", sizeof(double), "qnm1");
    }
    if (implicit) {
      enforce_boundary(m, q);
      to_primitive(m, q, prim);
      gradients(m, full_gradients, prim, grad);
      flux_residual(m, prim, grad, res);
    }

    // Dual time stepping's record: each step's wall force and inner iterations.
    std::ofstream history;
    std::filesystem::path stem(cfg.output_file);
    stem.replace_extension();
    if (unsteady && op_is_root()) {
      if (stem.has_parent_path()) std::filesystem::create_directories(stem.parent_path());
      history.open(stem.string() + ".history.csv");
      history << "step,time,fx,fy,fz,inner_iterations,inner_reduction,linear_iterations\n";
      history.precision(12);
    }

    op_profile_start("nssolver_op2");
    const int steps = unsteady ? cfg.physical_steps : 1;
    const int inner_max = unsteady ? cfg.inner_iterations : cfg.iterations;
    long long count = 0;
    for (int step = 0; step < steps; ++step) {
      double step_l2_0 = -1.0;
      long long step_its = 0;
      if (unsteady) {
        // q^n-1 <- q^n, q^n <- q, and the step's coefficients: first order on
        // the first step, which has no q^n-1.
        if (step > 0)
          op_par_loop(copy_q_kernel, "copy_q_kernel", m.nodes,
                      op_arg_dat(imp.qn, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                      op_arg_dat(imp.qnm1, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));
        op_par_loop(copy_q_kernel, "copy_q_kernel", m.nodes,
                    op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                    op_arg_dat(imp.qn, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));
        const bool second = cfg.time_order == 2 && step > 0;
        imp.physical[0] = (second ? 1.5 : 1.0) / cfg.time_step;
        imp.physical[1] = (second ? 2.0 : 1.0) / cfg.time_step;
        imp.physical[2] = (second ? 0.5 : 0.0) / cfg.time_step;
        if (step == 0)
          op_par_loop(copy_q_kernel, "copy_q_kernel", m.nodes,
                      op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                      op_arg_dat(imp.qnm1, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));
        implicit_residual(imp, q, prim, grad, res);
      }
      int iter = 0;
      for (; iter < inner_max; ++iter) {
        int its = 0;
        double linear_residual = 0.0;
        int rejected = 0;
        if (!implicit) {
          enforce_boundary(m, q);

          // Save the RK baseline state used by all four stages.
          op_par_loop(copy_q_kernel, "copy_q_kernel", m.nodes,
                      op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                      op_arg_dat(q0, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));

          // Primitives once per nonlinear iteration feed gradients, timestep
          // estimates, and all edge and boundary flux kernels.
          to_primitive(m, q, prim);
          gradients(m, full_gradients, prim, grad);
          local_time_step(m, prim, cfg.cfl, spectral, dt);

          for (int stage = 0; stage < 4; ++stage) {
            // Reapply boundary constraints at the start of every RK stage so the
            // stage residual is assembled from a consistent boundary state.
            enforce_boundary(m, q);
            to_primitive(m, q, prim);
            flux_residual(m, prim, grad, res);

            // Advance one stage of the classical four-stage explicit RK scheme.
            op_par_loop(rk_update_kernel, "rk_update_kernel", m.nodes,
                        op_arg_gbl(&stage, 1, "int", OP_READ),
                        op_arg_dat(dt, -1, OP_ID, 1, "double", OP_READ),
                        op_arg_dat(m.node_volume, -1, OP_ID, 1, "double", OP_READ),
                        op_arg_dat(q0, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                        op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                        op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));

            // Leave the stage in an enforced state so the next stage starts from
            // physically admissible wall and slip values.
            enforce_boundary(m, q);
          }

          // Reassemble the final residual for convergence monitoring from the
          // converged state at the end of the full RK iteration.
          enforce_boundary(m, q);
          to_primitive(m, q, prim);
          flux_residual(m, prim, grad, res);
        } else {
          // One pseudo-time step: (V / dt + dR/dq) dq = -R about the current
          // state, whose primitives, gradients and residual are current.
          imp.cfl = std::min(cfl_max, cfg.implicit_cfl * std::pow(cfg.implicit_cfl_growth, iter));
          local_time_step(m, prim, imp.cfl, spectral, dt);
          if (iter % cfg.jacobian_interval == 0) {
            double t0 = wall_time();
            assemble_jacobian(imp, imp.A, prim, grad, 0);
            if (P != NULL) assemble_jacobian(imp, P, prim, grad, 1);
            if (cfg.pc != "lib")
              op_par_loop(invert_block_kernel, "invert_block_kernel", m.nodes,
                          op_arg_mat(imp.A, -1, OP_ID, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                          op_arg_dat(imp.dinv, -1, OP_ID, NBLOCK_OP2, "double", OP_WRITE));
            double t1 = wall_time();
            assembly_time += t1 - t0;
          }

          op_par_loop(implicit_rhs_kernel, "implicit_rhs_kernel", m.nodes,
                      op_arg_dat(m.node_bc, -1, OP_ID, 2, "int", OP_READ),
                      op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                      op_arg_dat(rhs, -1, OP_ID, NVAR_OP2, "double", OP_WRITE),
                      op_arg_dat(dq, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));
          double t0 = wall_time();
          if (!op_solve(solver, rhs, dq)) unconverged_solves++;
          solve_time += wall_time() - t0;
          op_solver_stats(solver, &its, &linear_residual);
          if (cfg.dump_interval > 0 && (count + 1) % cfg.dump_interval == 0) {
            char suffix[32];
            snprintf(suffix, sizeof suffix, "_%06lld.h5", count + 1);
            const std::string file = cfg.dump_prefix + suffix;
            const std::filesystem::path parent = std::filesystem::path(file).parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
            op_mat_write_hdf5(imp.A, file.c_str());
            op_fetch_data_hdf5_file_path(rhs, file.c_str(), "A/b");
            op_fetch_data_hdf5_file_path(dq, file.c_str(), "A/x");
            if (op_is_root()) {
              const std::string index = cfg.dump_prefix + "_index.csv";
              const bool fresh = !std::filesystem::exists(index) || count + 1 == cfg.dump_interval;
              std::ofstream out(index, fresh ? std::ios::trunc : std::ios::app);
              if (fresh) out << "iteration,cfl,linear_iterations,linear_residual,file\n";
              out << count + 1 << "," << imp.cfl << "," << its << "," << linear_residual << "," << file << "\n";
              std::cout << "[dump] " << file << "\n";
            }
          }
          linear_iterations += its;

          double relax = cfg.implicit_relax;
          op_par_loop(implicit_update_kernel, "implicit_update_kernel", m.nodes,
                      op_arg_dat(dq, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                      op_arg_gbl(&relax, 1, "double", OP_READ),
                      op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_RW),
                      op_arg_gbl(&rejected, 1, "int", OP_INC));
          rejected_total += rejected;

          // The new state's residual: monitored, and the next step's.
          enforce_boundary(m, q);
          to_primitive(m, q, prim);
          gradients(m, full_gradients, prim, grad);
          implicit_residual(imp, q, prim, grad, res);
        }

        // Reduce the assembled residual field to scalar convergence metrics.
        ResidualNorms norms = residual_norms(m, res);
        history_l2_rho.push_back(norms.l2_rho);
        history_l2_rhoE.push_back(norms.l2_rhoE);
        history_linf_rho.push_back(norms.linf_rho);
        if (initial_l2_rho < 0.0) initial_l2_rho = norms.l2_rho;

        step_its += its;
        if (step_l2_0 < 0.0) step_l2_0 = norms.l2_rho;
        const bool inner_done = unsteady && norms.l2_rho <= cfg.inner_tolerance * step_l2_0;
        const bool should_print = !unsteady && (iter == 0 || iter + 1 == cfg.iterations ||
                                                ((iter + 1) % std::max(cfg.progress_interval, 1) == 0));
        if (should_print && op_is_root()) {
          std::cout << "[progress] iter " << (iter + 1) << "/" << cfg.iterations
                    << " | L2(rho)=" << norms.l2_rho
                    << " | L2/L2_0=" << (initial_l2_rho > 0.0 ? norms.l2_rho / initial_l2_rho : 1.0)
                    << " | Linf(rho)=" << norms.linf_rho;
          if (implicit)
            std::cout << " | CFL=" << imp.cfl << " | linear its=" << its << " res=" << linear_residual
                      << (rejected ? " | rejected=" + std::to_string(rejected) : "");
          std::cout << "\n";
        }
        ++count;
        if (inner_done) {
          ++iter;
          break;
        }
      }

      if (unsteady) {
        // The converged step's wall force, from its spatial residual.
        double force[3];
        implicit_residual(imp, q, prim, grad, res, force);
        const double reduction = step_l2_0 > 0.0 ? history_l2_rho.back() / step_l2_0 : 1.0;
        if (op_is_root()) {
          history << step + 1 << "," << (step + 1) * cfg.time_step << "," << force[0] << "," << force[1] << ","
                  << force[2] << "," << iter << "," << reduction << "," << step_its << "\n";
          history.flush();
          if (step == 0 || step + 1 == steps || (step + 1) % std::max(cfg.progress_interval, 1) == 0)
            std::cout << "[step] " << step + 1 << "/" << steps << " | t=" << (step + 1) * cfg.time_step
                      << " | inner its=" << iter << " | reduction=" << reduction << " | linear its=" << step_its
                      << " | force=(" << force[0] << ", " << force[1] << ", " << force[2] << ")\n";
        }
        if (cfg.snapshot_interval > 0 && (step + 1) % cfg.snapshot_interval == 0) {
          char suffix[32];
          snprintf(suffix, sizeof suffix, "_%06d.h5", step + 1);
          const std::string file = stem.string() + suffix;
          if (op_is_root()) std::filesystem::remove(file);
          op_fetch_data_hdf5_file(q, file.c_str());
          op_fetch_data_hdf5_file(prim, file.c_str());
        }
      }
    }

    to_primitive(m, q, prim);

    op_profile_end();
    op_timers(&cpu, &wall_t2);
    double walltime = wall_t2 - wall_t1;
    op_printf("Max total runtime = %f\n", walltime);
    if (implicit) {
      op_printf("implicit: %lld linear iterations, %d unconverged solves, %d rejected node updates\n",
                linear_iterations, unconverged_solves, rejected_total);
      op_printf("time: assembly %.4f s (loops and commit), solve %.4f s\n", assembly_time, solve_time);
      (void)commit_time;
    }

    q->name = strdup("q");
    prim->name = strdup("primitive");
    dt->name = strdup("dt");
    res->name = strdup("residual");
    std::filesystem::path output_path(cfg.output_file);
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    std::filesystem::remove(output_path);
    if (op_is_root()) std::cout << "[phase] write hdf5\n";
    op_fetch_data_hdf5_file(q, cfg.output_file.c_str());
    op_fetch_data_hdf5_file(prim, cfg.output_file.c_str());
    op_fetch_data_hdf5_file(dt, cfg.output_file.c_str());
    op_fetch_data_hdf5_file(res, cfg.output_file.c_str());
    std::filesystem::path residual_path(cfg.output_file);
    residual_path.replace_extension(".residual.csv");
    write_residual_history_csv_op2(residual_path.string(), history_l2_rho, history_l2_rhoE, history_linf_rho);

    if (op_is_root()) std::cout << "Wrote solution HDF5: " << cfg.output_file << "\n";
    if (op_is_root()) std::cout << "Wrote residual CSV: " << residual_path.string() << "\n";

    // check FACTOR fails the run unless L2(rho) fell by FACTOR, or on a
    // non-finite residual.
    const double reduction = history_l2_rho.empty() || initial_l2_rho <= 0.0
                                 ? 1.0
                                 : history_l2_rho.back() / initial_l2_rho;
    bool failed = !std::isfinite(reduction);
    if (cfg.check > 0.0 && !(reduction <= cfg.check)) {
      op_printf("check failed: L2(rho) reduction %.3e, required %.1e\n", reduction, cfg.check);
      failed = true;
    }

    if (cfg.timing) op_timing_output();
    op_profile_output();
    if (implicit) op_solve_exit();
    op_exit();
    return failed ? 1 : 0;
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
