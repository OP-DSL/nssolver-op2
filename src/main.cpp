#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nssolver/hydra_benchmark.hpp"
#include "nssolver/hydra_reader.hpp"
#include "nssolver/mesh.hpp"
#include "nssolver/solver.hpp"
#include "nssolver/validation.hpp"
#include "nssolver/vtk_writer.hpp"

int main(int argc, char** argv) {
    using namespace nssolver;

    std::string case_name = "box";
    std::string hydra_path;
    if (argc >= 2) {
        case_name = argv[1];
    }
    if (argc >= 3) {
        hydra_path = argv[2];
    }

    Mesh mesh;
    Real flat_plate_leading_edge_x = 0.2;
    GasModel gas {};
    std::cout << "[phase] selecting case '" << case_name << "'\n";
    const bool develop_flat_plate = case_name == "flatplate_develop";
    const bool hydra_benchmark = case_name == "hydra_benchmark";
    if (case_name == "box") {
        const StructuredBoxSpec spec {
            .nx = 21,
            .ny = 11,
            .nz = 2,
            .lx = 2.0,
            .ly = 1.0,
            .lz = 0.05,
            .xmin = BoundaryType::Farfield,
            .xmax = BoundaryType::Farfield,
            .ymin = BoundaryType::SlipWall,
            .ymax = BoundaryType::SlipWall,
            .zmin = BoundaryType::SlipWall,
            .zmax = BoundaryType::SlipWall,
        };
        mesh = make_structured_box_mesh(spec);
    } else if (case_name == "bump") {
        mesh = make_bump_channel_mesh(BumpChannelSpec {
            .nx = 81,
            .ny = 33,
            .nz = 2,
            .lx = 3.0,
            .ly = 1.0,
            .lz = 0.05,
            .bump_center = 1.5,
            .bump_half_width = 0.5,
            .bump_height = 0.08,
            .inlet = BoundaryType::Farfield,
            .outlet = BoundaryType::Farfield,
            .lower_wall = BoundaryType::SlipWall,
            .upper_wall = BoundaryType::SlipWall,
        });
    } else if (case_name == "flatplate" || develop_flat_plate) {
        const FlatPlateSpec spec {
            .nx = 81,
            .ny = 41,
            .nz = 2,
            .lx = 1.5,
            .ly = 0.4,
            .lz = 0.02,
            .leading_edge_x = 0.3,
            .wall_normal_growth = 2000.0,
            .inlet = BoundaryType::Inlet,
            .outlet = BoundaryType::Outlet,
            .plate_type = BoundaryType::NoSlipWall,
            .lower_upstream_type = BoundaryType::SlipWall,
            .upper_type = BoundaryType::Farfield,
        };
        flat_plate_leading_edge_x = spec.leading_edge_x;
        mesh = make_flat_plate_mesh(spec);
    } else if (case_name == "hydra" || hydra_benchmark) {
        if (hydra_path.empty()) {
            throw std::runtime_error(case_name + " case requires a mesh path argument");
        }
        mesh = read_hydra_hdf5(hydra_path);
        const MeshValidationReport report = validate_mesh(mesh);
        if (!report.ok) {
            throw std::runtime_error("Hydra mesh failed validation");
        }
        if (hydra_benchmark) {
            gas.gamma = 1.39976;
        }
    } else {
        throw std::runtime_error("unknown case name");
    }
    std::cout << "[phase] mesh ready: nodes=" << mesh.nodes.count
              << ", edges=" << mesh.edges.count
              << ", boundary_faces=" << mesh.boundary_faces.count << '\n';

    SolverOptions options {};
    options.iterations = 160;
    options.cfl = 0.15;
    options.second_order = case_name == "box";
    options.verbose = true;
    options.progress_interval = 10;
    options.freestream.primitive = Primitive {.rho = 1.225, .u = 120.0, .v = 0.0, .w = 0.0, .p = 101325.0, .nu_tilde = 0.0};
    if (case_name == "flatplate" || develop_flat_plate) {
        options.include_viscous = true;
        options.cfl = 0.005;
        options.iterations = develop_flat_plate ? 400 : 0;
        options.progress_interval = develop_flat_plate ? 50 : 500;
        options.freestream.primitive = Primitive {.rho = 1.225, .u = 20.0, .v = 0.0, .w = 0.0, .p = 101325.0, .nu_tilde = 0.0};
    } else if (case_name == "bump") {
        options.cfl = 0.08;
        options.freestream.primitive = Primitive {.rho = 1.225, .u = 80.0, .v = 0.0, .w = 0.0, .p = 101325.0, .nu_tilde = 0.0};
    } else if (hydra_benchmark) {
        const std::filesystem::path grid_path {hydra_path};
        const std::filesystem::path mesh_dir = grid_path.has_parent_path() ? grid_path.parent_path() : std::filesystem::path(".");
        const auto inflow = read_hydra_inflow_profile((mesh_dir / "bc.sub_inflow.H01R-C_INLET.xml").string());
        const auto outflow = read_hydra_outflow_profile((mesh_dir / "bc.sub_outflow.H01R-C_EXIT.xml").string());
        const HydraBenchmarkConditions conditions = make_hydra_benchmark_conditions(inflow, outflow, gas, 0.20);

        options.iterations = 240;
        options.cfl = 0.08;
        options.second_order = false;
        options.include_viscous = false;
        options.include_sa = false;
        options.progress_interval = 10;
        options.freestream.primitive = conditions.primitive;

        std::cout << "[benchmark] gamma=" << gas.gamma
                  << ", inlet_rows=" << inflow.rows
                  << ", outlet_rows=" << outflow.rows
                  << ", mach=" << conditions.mach
                  << ", Tt_in=" << conditions.inlet_total_temperature
                  << ", Pt_in=" << conditions.inlet_total_pressure
                  << ", p_static_benchmark=" << conditions.primitive.p
                  << ", p_exit_original_mean=" << conditions.original_exit_static_pressure << '\n';
        std::cout << "[benchmark] original exit static pressure is not imposed because rotor work/rotating-frame physics "
                     "is not implemented; outlet uses matched subsonic static pressure for a non-rotating steady benchmark\n";
    }

    FlowState state;
    std::cout << "[phase] initializing state\n";
    if (case_name == "flatplate") {
        initialize_blasius_flat_plate_state(mesh, state, gas, options, flat_plate_leading_edge_x);
    } else {
        initialize_uniform_state(mesh, options.freestream, gas, state);
    }
    if (case_name == "flatplate" || develop_flat_plate) {
        enforce_boundary_state(mesh, gas, options, state);
    }
    std::cout << "[phase] solving\n";
    const SolverHistory history = run_solver(mesh, gas, options, state);
    std::cout << "[phase] assembling final residual and writing outputs\n";
    assemble_residual(mesh, gas, options, state);
    const ResidualNorms norms = compute_residual_norms(state);

    write_vtk_legacy(case_name + "_solution.vtk", mesh, state);
    write_residual_history_csv(case_name + "_residual.csv", history);
    if (case_name == "flatplate" || develop_flat_plate) {
        write_flat_plate_benchmark_outputs(case_name, mesh, state, gas, options, flat_plate_leading_edge_x);
    } else if (case_name == "bump") {
        write_bump_benchmark_outputs(case_name, mesh, state, options);
    }
    std::cout << "Wrote " << case_name << "_solution.vtk\n";
    std::cout << residual_summary(norms) << '\n';
    return 0;
}
