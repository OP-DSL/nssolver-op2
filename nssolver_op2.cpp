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

#include "op2_config.h"
#include "op2_kernels.h"

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
int op2_include_viscous = 0;
int op2_include_sa = 0;

struct ResidualNorms {
  double l2_rho = 0.0;
  double l2_rhoE = 0.0;
  double linf_rho = 0.0;
};

static void usage() {
  std::cerr << "usage: ./nssolver_op2_seq --config <config.cfg>\n";
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

int main(int argc, char **argv) {
  try {
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
        config_path = argv[++i];
      }
    }
    if (config_path.empty()) {
      usage();
      return 1;
    }

    SolverConfig cfg;
    load_config_file(config_path, cfg);
    if (cfg.init_mode == "hydra_benchmark") {
      derive_hydra_benchmark_primitive(cfg);
    }

    op2_gamma = cfg.gamma;
    op2_gas_constant = cfg.gas_constant;
    op2_rho_floor = cfg.rho_floor;
    op2_p_floor = cfg.p_floor;
    op2_second_order = cfg.second_order;
    op2_include_viscous = cfg.include_viscous;
    op2_include_sa = cfg.include_sa;
    for (int i = 0; i < 6; ++i) op2_freestream[i] = cfg.primitive[i];

    std::cout << "[phase] op_init\n";
    op_init(argc, argv, 0);

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
    op_decl_const(1, "int", &op2_include_viscous);
    op_decl_const(1, "int", &op2_include_sa);

    std::cout << "[phase] decl sets\n";
    op_set nodes = op_decl_set_hdf5_infer_size(cfg.mesh_file.c_str(), "nodes", "node_coordinates");
    op_set edges = op_decl_set_hdf5_infer_size(cfg.mesh_file.c_str(), "edges", "edge-->node");
    op_set bfaces = op_decl_set_hdf5_infer_size(cfg.mesh_file.c_str(), "bfaces", "bface-->node");
    op_set bnodes = op_decl_set_hdf5_infer_size(cfg.mesh_file.c_str(), "bnodes", "bnode-->node");

    std::cout << "[phase] decl maps\n";
    op_map edge_to_nodes = op_decl_map_hdf5(edges, nodes, 2, cfg.mesh_file.c_str(), "edge-->node");
    op_map bface_to_nodes = op_decl_map_hdf5(bfaces, nodes, 4, cfg.mesh_file.c_str(), "bface-->node");
    op_map bnode_to_node = op_decl_map_hdf5(bnodes, nodes, 1, cfg.mesh_file.c_str(), "bnode-->node");

    std::cout << "[phase] decl dats\n";
    op_dat node_coords = op_decl_dat_hdf5(nodes, 3, "double", cfg.mesh_file.c_str(), "node_coordinates");
    op_dat node_volume = op_decl_dat_hdf5(nodes, 1, "double", cfg.mesh_file.c_str(), "node_volume");
    op_dat node_wall_distance = op_decl_dat_hdf5(nodes, 1, "double", cfg.mesh_file.c_str(), "node_wall_distance");
    op_dat edge_weights = op_decl_dat_hdf5(edges, 3, "double", cfg.mesh_file.c_str(), "edge_weights");
    op_dat bface_normal = op_decl_dat_hdf5(bfaces, 3, "double", cfg.mesh_file.c_str(), "bface_normal");
    op_dat bface_area = op_decl_dat_hdf5(bfaces, 1, "double", cfg.mesh_file.c_str(), "bface_area");
    op_dat bface_type = op_decl_dat_hdf5(bfaces, 1, "int", cfg.mesh_file.c_str(), "bface_type");
    op_dat bface_group = op_decl_dat_hdf5(bfaces, 1, "int", cfg.mesh_file.c_str(), "bface_group");
    (void)bface_group;
    op_dat bnode_dirichlet = op_decl_dat_hdf5(bnodes, 1, "int", cfg.mesh_file.c_str(), "bnode_dirichlet");
    op_dat bnode_wall = op_decl_dat_hdf5(bnodes, 1, "int", cfg.mesh_file.c_str(), "bnode_wall");
    op_dat bnode_slip = op_decl_dat_hdf5(bnodes, 1, "int", cfg.mesh_file.c_str(), "bnode_slip");
    op_dat bnode_normal = op_decl_dat_hdf5(bnodes, 3, "double", cfg.mesh_file.c_str(), "bnode_normal");
    (void)node_coords;
    (void)node_wall_distance;
    (void)bnode_to_node;
    (void)bnode_normal;

    op_dat q = op_decl_dat_temp_char(nodes, NVAR_OP2, "double", sizeof(double), "q");
    op_dat q0 = op_decl_dat_temp_char(nodes, NVAR_OP2, "double", sizeof(double), "q0");
    op_dat prim = op_decl_dat_temp_char(nodes, NPRIM_OP2, "double", sizeof(double), "primitive");
    op_dat res = op_decl_dat_temp_char(nodes, NVAR_OP2, "double", sizeof(double), "residual");
    op_dat spectral = op_decl_dat_temp_char(nodes, 1, "double", sizeof(double), "spectral");
    op_dat dt = op_decl_dat_temp_char(nodes, 1, "double", sizeof(double), "dt");
    op_dat grad = op_decl_dat_temp_char(nodes, NGRAD_OP2, "double", sizeof(double), "grad");

    std::cout << "[phase] initialize\n";
    op_par_loop(initialize_q_kernel, "initialize_q_kernel", nodes,
                op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));

    double initial_l2_rho = -1.0;
    std::vector<double> history_l2_rho;
    std::vector<double> history_l2_rhoE;
    std::vector<double> history_linf_rho;
    history_l2_rho.reserve(static_cast<std::size_t>(cfg.iterations));
    history_l2_rhoE.reserve(static_cast<std::size_t>(cfg.iterations));
    history_linf_rho.reserve(static_cast<std::size_t>(cfg.iterations));

    for (int iter = 0; iter < cfg.iterations; ++iter) {
      // Keep boundary nodes in a physically admissible state before building any
      // stage data. For viscous runs this imposes no-slip or slip constraints on
      // the nodal conservative state stored on the boundary map.
      op_par_loop(enforce_boundary_node_kernel, "enforce_boundary_node_kernel", bnodes,
                  op_arg_dat(bnode_dirichlet, -1, OP_ID, 1, "int", OP_READ),
                  op_arg_dat(bnode_wall, -1, OP_ID, 1, "int", OP_READ),
                  op_arg_dat(bnode_slip, -1, OP_ID, 1, "int", OP_READ),
                  op_arg_dat(bnode_normal, -1, OP_ID, 3, "double", OP_READ),
                  op_arg_dat(q, 0, bnode_to_node, NVAR_OP2, "double", OP_RW));

      // Save the RK baseline state used by all four stages.
      op_par_loop(copy_q_kernel, "copy_q_kernel", nodes,
                  op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                  op_arg_dat(q0, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));

      // Convert conservative variables to primitive form once per nonlinear
      // iteration. The primitive field feeds gradients, timestep estimates, and
      // all edge and boundary flux kernels.
      op_par_loop(q_to_primitive_kernel, "q_to_primitive_kernel", nodes,
                  op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                  op_arg_dat(prim, -1, OP_ID, NPRIM_OP2, "double", OP_WRITE));

      if (cfg.second_order || cfg.include_sa) {
        // Assemble Green-Gauss gradients from interior edges and boundary faces,
        // then divide by the nodal control volume to obtain nodal primitive
        // gradients used by second-order reconstruction and SA diffusion terms.
        op_par_loop(zero_grad_kernel, "zero_grad_kernel", nodes,
                    op_arg_dat(grad, -1, OP_ID, NGRAD_OP2, "double", OP_WRITE));

        op_par_loop(edge_grad_kernel, "edge_grad_kernel", edges,
                    op_arg_dat(prim, 0, edge_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(prim, 1, edge_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(edge_weights, -1, OP_ID, 3, "double", OP_READ),
                    op_arg_dat(grad, 0, edge_to_nodes, NGRAD_OP2, "double", OP_INC),
                    op_arg_dat(grad, 1, edge_to_nodes, NGRAD_OP2, "double", OP_INC));

        op_par_loop(bface_grad_kernel, "bface_grad_kernel", bfaces,
                    op_arg_dat(bface_normal, -1, OP_ID, 3, "double", OP_READ),
                    op_arg_dat(prim, 0, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(prim, 1, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(prim, 2, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(prim, 3, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(grad, 0, bface_to_nodes, NGRAD_OP2, "double", OP_INC),
                    op_arg_dat(grad, 1, bface_to_nodes, NGRAD_OP2, "double", OP_INC),
                    op_arg_dat(grad, 2, bface_to_nodes, NGRAD_OP2, "double", OP_INC),
                    op_arg_dat(grad, 3, bface_to_nodes, NGRAD_OP2, "double", OP_INC));

        op_par_loop(normalize_grad_kernel, "normalize_grad_kernel", nodes,
                    op_arg_dat(node_volume, -1, OP_ID, 1, "double", OP_READ),
                    op_arg_dat(grad, -1, OP_ID, NGRAD_OP2, "double", OP_RW));
      } else {
        // First-order runs still zero the gradient storage so the flux kernels do
        // not consume stale values.
        op_par_loop(zero_grad_kernel, "zero_grad_kernel", nodes,
                    op_arg_dat(grad, -1, OP_ID, NGRAD_OP2, "double", OP_WRITE));
      }

      // Estimate the local spectral radius and convert it into an explicit
      // timestep for each node.
      op_par_loop(zero_scalar_kernel, "zero_scalar_kernel", nodes,
                  op_arg_dat(spectral, -1, OP_ID, 1, "double", OP_WRITE));

      op_par_loop(edge_spectral_kernel, "edge_spectral_kernel", edges,
                  op_arg_dat(prim, 0, edge_to_nodes, NPRIM_OP2, "double", OP_READ),
                  op_arg_dat(prim, 1, edge_to_nodes, NPRIM_OP2, "double", OP_READ),
                  op_arg_dat(edge_weights, -1, OP_ID, 3, "double", OP_READ),
                  op_arg_dat(node_volume, 0, edge_to_nodes, 1, "double", OP_READ),
                  op_arg_dat(node_volume, 1, edge_to_nodes, 1, "double", OP_READ),
                  op_arg_dat(spectral, 0, edge_to_nodes, 1, "double", OP_INC),
                  op_arg_dat(spectral, 1, edge_to_nodes, 1, "double", OP_INC));
      double cfl = cfg.cfl;
      op_par_loop(compute_dt_scaled_kernel, "compute_dt_scaled_kernel", nodes,
                  op_arg_dat(node_volume, -1, OP_ID, 1, "double", OP_READ),
                  op_arg_dat(spectral, -1, OP_ID, 1, "double", OP_READ),
                  op_arg_gbl(&cfl, 1, "double", OP_READ),
                  op_arg_dat(dt, -1, OP_ID, 1, "double", OP_WRITE));

      for (int stage = 0; stage < 4; ++stage) {
        // Reapply boundary constraints at the start of every RK stage so the
        // stage residual is assembled from a consistent boundary state.
        op_par_loop(enforce_boundary_node_kernel, "enforce_boundary_node_kernel", bnodes,
                    op_arg_dat(bnode_dirichlet, -1, OP_ID, 1, "int", OP_READ),
                    op_arg_dat(bnode_wall, -1, OP_ID, 1, "int", OP_READ),
                    op_arg_dat(bnode_slip, -1, OP_ID, 1, "int", OP_READ),
                    op_arg_dat(bnode_normal, -1, OP_ID, 3, "double", OP_READ),
                    op_arg_dat(q, 0, bnode_to_node, NVAR_OP2, "double", OP_RW));

        // Refresh primitives from the stage state.
        op_par_loop(q_to_primitive_kernel, "q_to_primitive_kernel", nodes,
                    op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                    op_arg_dat(prim, -1, OP_ID, NPRIM_OP2, "double", OP_WRITE));

        // Start from a clean residual vector before edge, boundary, and source
        // contributions are accumulated.
        op_par_loop(zero_var_kernel, "zero_var_kernel", nodes,
                    op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));

        // Assemble interior fluxes. In second-order mode the kernel performs
        // edge-local limited reconstruction directly from the endpoint gradients.
        op_par_loop(edge_flux_kernel, "edge_flux_kernel", edges,
                    op_arg_dat(node_coords, 0, edge_to_nodes, 3, "double", OP_READ),
                    op_arg_dat(node_coords, 1, edge_to_nodes, 3, "double", OP_READ),
                    op_arg_dat(prim, 0, edge_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(prim, 1, edge_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(grad, 0, edge_to_nodes, NGRAD_OP2, "double", OP_READ),
                    op_arg_dat(grad, 1, edge_to_nodes, NGRAD_OP2, "double", OP_READ),
                    op_arg_dat(edge_weights, -1, OP_ID, 3, "double", OP_READ),
                    op_arg_dat(res, 0, edge_to_nodes, NVAR_OP2, "double", OP_INC),
                    op_arg_dat(res, 1, edge_to_nodes, NVAR_OP2, "double", OP_INC));

        // Add physical boundary fluxes using face-averaged interior states and
        // the boundary-type-specific ghost construction.
        op_par_loop(boundary_flux_kernel, "boundary_flux_kernel", bfaces,
                    op_arg_dat(bface_type, -1, OP_ID, 1, "int", OP_READ),
                    op_arg_dat(bface_normal, -1, OP_ID, 3, "double", OP_READ),
                    op_arg_dat(bface_area, -1, OP_ID, 1, "double", OP_READ),
                    op_arg_dat(node_volume, 0, bface_to_nodes, 1, "double", OP_READ),
                    op_arg_dat(node_volume, 1, bface_to_nodes, 1, "double", OP_READ),
                    op_arg_dat(node_volume, 2, bface_to_nodes, 1, "double", OP_READ),
                    op_arg_dat(node_volume, 3, bface_to_nodes, 1, "double", OP_READ),
                    op_arg_dat(prim, 0, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(prim, 1, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(prim, 2, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(prim, 3, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(res, 0, bface_to_nodes, NVAR_OP2, "double", OP_INC),
                    op_arg_dat(res, 1, bface_to_nodes, NVAR_OP2, "double", OP_INC),
                    op_arg_dat(res, 2, bface_to_nodes, NVAR_OP2, "double", OP_INC),
                    op_arg_dat(res, 3, bface_to_nodes, NVAR_OP2, "double", OP_INC));

        // Add the Spalart-Allmaras source contribution as a nodal update to the
        // transported turbulence variable residual.
        op_par_loop(sa_source_kernel, "sa_source_kernel", nodes,
                    op_arg_dat(prim, -1, OP_ID, NPRIM_OP2, "double", OP_READ),
                    op_arg_dat(grad, -1, OP_ID, NGRAD_OP2, "double", OP_READ),
                    op_arg_dat(node_wall_distance, -1, OP_ID, 1, "double", OP_READ),
                    op_arg_dat(node_volume, -1, OP_ID, 1, "double", OP_READ),
                    op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_RW));

        // Advance one stage of the classical four-stage explicit RK scheme.
        op_par_loop(rk_update_kernel, "rk_update_kernel", nodes,
                    op_arg_gbl(&stage, 1, "int", OP_READ),
                    op_arg_dat(dt, -1, OP_ID, 1, "double", OP_READ),
                    op_arg_dat(node_volume, -1, OP_ID, 1, "double", OP_READ),
                    op_arg_dat(q0, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                    op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                    op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));

        // Leave the stage in an enforced state so the next stage starts from
        // physically admissible wall and slip values.
        op_par_loop(enforce_boundary_node_kernel, "enforce_boundary_node_kernel", bnodes,
                    op_arg_dat(bnode_dirichlet, -1, OP_ID, 1, "int", OP_READ),
                    op_arg_dat(bnode_wall, -1, OP_ID, 1, "int", OP_READ),
                    op_arg_dat(bnode_slip, -1, OP_ID, 1, "int", OP_READ),
                    op_arg_dat(bnode_normal, -1, OP_ID, 3, "double", OP_READ),
                    op_arg_dat(q, 0, bnode_to_node, NVAR_OP2, "double", OP_RW));
      }

      // Reassemble the final residual for convergence monitoring from the
      // converged state at the end of the full RK iteration.
      op_par_loop(enforce_boundary_node_kernel, "enforce_boundary_node_kernel", bnodes,
                  op_arg_dat(bnode_dirichlet, -1, OP_ID, 1, "int", OP_READ),
                  op_arg_dat(bnode_wall, -1, OP_ID, 1, "int", OP_READ),
                  op_arg_dat(bnode_slip, -1, OP_ID, 1, "int", OP_READ),
                  op_arg_dat(bnode_normal, -1, OP_ID, 3, "double", OP_READ),
                  op_arg_dat(q, 0, bnode_to_node, NVAR_OP2, "double", OP_RW));

      op_par_loop(q_to_primitive_kernel, "q_to_primitive_kernel", nodes,
                  op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                  op_arg_dat(prim, -1, OP_ID, NPRIM_OP2, "double", OP_WRITE));

      op_par_loop(zero_var_kernel, "zero_var_kernel", nodes,
                  op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_WRITE));

      op_par_loop(edge_flux_kernel, "edge_flux_kernel", edges,
                  op_arg_dat(node_coords, 0, edge_to_nodes, 3, "double", OP_READ),
                  op_arg_dat(node_coords, 1, edge_to_nodes, 3, "double", OP_READ),
                  op_arg_dat(prim, 0, edge_to_nodes, NPRIM_OP2, "double", OP_READ),
                  op_arg_dat(prim, 1, edge_to_nodes, NPRIM_OP2, "double", OP_READ),
                  op_arg_dat(grad, 0, edge_to_nodes, NGRAD_OP2, "double", OP_READ),
                  op_arg_dat(grad, 1, edge_to_nodes, NGRAD_OP2, "double", OP_READ),
                  op_arg_dat(edge_weights, -1, OP_ID, 3, "double", OP_READ),
                  op_arg_dat(res, 0, edge_to_nodes, NVAR_OP2, "double", OP_INC),
                  op_arg_dat(res, 1, edge_to_nodes, NVAR_OP2, "double", OP_INC));

      op_par_loop(boundary_flux_kernel, "boundary_flux_kernel", bfaces,
                  op_arg_dat(bface_type, -1, OP_ID, 1, "int", OP_READ),
                  op_arg_dat(bface_normal, -1, OP_ID, 3, "double", OP_READ),
                  op_arg_dat(bface_area, -1, OP_ID, 1, "double", OP_READ),
                  op_arg_dat(node_volume, 0, bface_to_nodes, 1, "double", OP_READ),
                  op_arg_dat(node_volume, 1, bface_to_nodes, 1, "double", OP_READ),
                  op_arg_dat(node_volume, 2, bface_to_nodes, 1, "double", OP_READ),
                  op_arg_dat(node_volume, 3, bface_to_nodes, 1, "double", OP_READ),
                  op_arg_dat(prim, 0, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                  op_arg_dat(prim, 1, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                  op_arg_dat(prim, 2, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                  op_arg_dat(prim, 3, bface_to_nodes, NPRIM_OP2, "double", OP_READ),
                  op_arg_dat(res, 0, bface_to_nodes, NVAR_OP2, "double", OP_INC),
                  op_arg_dat(res, 1, bface_to_nodes, NVAR_OP2, "double", OP_INC),
                  op_arg_dat(res, 2, bface_to_nodes, NVAR_OP2, "double", OP_INC),
                  op_arg_dat(res, 3, bface_to_nodes, NVAR_OP2, "double", OP_INC));

      op_par_loop(sa_source_kernel, "sa_source_kernel", nodes,
                  op_arg_dat(prim, -1, OP_ID, NPRIM_OP2, "double", OP_READ),
                  op_arg_dat(grad, -1, OP_ID, NGRAD_OP2, "double", OP_READ),
                  op_arg_dat(node_wall_distance, -1, OP_ID, 1, "double", OP_READ),
                  op_arg_dat(node_volume, -1, OP_ID, 1, "double", OP_READ),
                  op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_RW));

      // Reduce the assembled residual field to scalar convergence metrics.
      double l2_rho_sq = 0.0;
      double l2_rhoE_sq = 0.0;
      double linf_rho = 0.0;
      op_par_loop(residual_norm_kernel, "residual_norm_kernel", nodes,
                  op_arg_dat(res, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                  op_arg_gbl(&l2_rho_sq, 1, "double", OP_INC),
                  op_arg_gbl(&l2_rhoE_sq, 1, "double", OP_INC),
                  op_arg_gbl(&linf_rho, 1, "double", OP_MAX));
      ResidualNorms norms;
      norms.l2_rho = std::sqrt(l2_rho_sq / static_cast<double>(op_get_size(nodes)));
      norms.l2_rhoE = std::sqrt(l2_rhoE_sq / static_cast<double>(op_get_size(nodes)));
      norms.linf_rho = linf_rho;
      history_l2_rho.push_back(norms.l2_rho);
      history_l2_rhoE.push_back(norms.l2_rhoE);
      history_linf_rho.push_back(norms.linf_rho);
      if (initial_l2_rho < 0.0) initial_l2_rho = norms.l2_rho;

      const bool should_print = (iter == 0 || iter + 1 == cfg.iterations ||
                                 ((iter + 1) % std::max(cfg.progress_interval, 1) == 0));
      if (should_print) {
        std::cout << "[progress] iter " << (iter + 1) << "/" << cfg.iterations
                  << " | L2(rho)=" << norms.l2_rho
                  << " | L2/L2_0=" << (initial_l2_rho > 0.0 ? norms.l2_rho / initial_l2_rho : 1.0)
                  << " | Linf(rho)=" << norms.linf_rho << "\n";
      }
    }

    op_par_loop(q_to_primitive_kernel, "q_to_primitive_kernel", nodes,
                op_arg_dat(q, -1, OP_ID, NVAR_OP2, "double", OP_READ),
                op_arg_dat(prim, -1, OP_ID, NPRIM_OP2, "double", OP_WRITE));

    q->name = strdup("q");
    prim->name = strdup("primitive");
    dt->name = strdup("dt");
    std::filesystem::path output_path(cfg.output_file);
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    std::filesystem::remove(output_path);
    std::cout << "[phase] write hdf5\n";
    op_fetch_data_hdf5_file(q, cfg.output_file.c_str());
    op_fetch_data_hdf5_file(prim, cfg.output_file.c_str());
    op_fetch_data_hdf5_file(dt, cfg.output_file.c_str());
    std::filesystem::path residual_path(cfg.output_file);
    residual_path.replace_extension(".residual.csv");
    write_residual_history_csv_op2(residual_path.string(), history_l2_rho, history_l2_rhoE, history_linf_rho);

    std::cout << "Wrote solution HDF5: " << cfg.output_file << "\n";
    std::cout << "Wrote residual CSV: " << residual_path.string() << "\n";
    op_exit();
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
