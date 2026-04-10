#pragma once

#include <iosfwd>
#include <string>

#include "nssolver/flux.hpp"
#include "nssolver/mesh.hpp"
#include "nssolver/physics.hpp"
#include "nssolver/state.hpp"

namespace nssolver {

struct SolverOptions {
    Real cfl {0.5};
    Real viscous_penalty {0.0};
    int iterations {200};
    Real residual_tolerance {1.0e-10};
    bool second_order {false};
    bool include_viscous {false};
    bool include_sa {false};
    bool verbose {false};
    int progress_interval {25};
    Real rho_floor {1.0e-4};
    Real p_floor {100.0};
    Freestream freestream {};
};

struct ResidualNorms {
    Real l2_rho {};
    Real l2_rhoE {};
    Real linf_rho {};
};

struct SolverHistory {
    std::vector<Real> l2_rho;
    std::vector<Real> l2_rhoE;
    std::vector<Real> linf_rho;
    bool stopped_on_nonfinite {false};
    bool converged {false};
    int final_iteration {0};
};

struct StateDiagnostics {
    Real min_rho {};
    Real min_p {};
    Real min_rhoE {};
    Real max_mach {};
    bool finite {true};
};

void initialize_uniform_state(const Mesh& mesh, const Freestream& freestream, const GasModel& gas, FlowState& state);
void enforce_boundary_state(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state);
void compute_gradients(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state);
Primitive reconstruct_primitive(const FlowState& state, Index i, const Vec3& delta_r, Real limiter_scale);
void compute_local_time_step(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state);
void assemble_residual(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state);
void rk4_step(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state);
ResidualNorms compute_residual_norms(const FlowState& state);
StateDiagnostics compute_state_diagnostics(const FlowState& state, const GasModel& gas);
SolverHistory run_solver(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state);
std::string residual_summary(const ResidualNorms& norms);
void write_progress_line(std::ostream& out, int iteration, int iterations, const ResidualNorms& norms, Real initial_l2_rho);

}  // namespace nssolver
