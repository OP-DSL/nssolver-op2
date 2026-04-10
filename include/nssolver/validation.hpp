#pragma once

#include <string>

#include "nssolver/mesh.hpp"
#include "nssolver/physics.hpp"
#include "nssolver/solver.hpp"
#include "nssolver/state.hpp"

namespace nssolver {

void write_residual_history_csv(const std::string& path, const SolverHistory& history);
void initialize_blasius_flat_plate_state(const Mesh& mesh,
                                         FlowState& state,
                                         const GasModel& gas,
                                         const SolverOptions& options,
                                         Real leading_edge_x);
void write_flat_plate_benchmark_outputs(const std::string& prefix,
                                        const Mesh& mesh,
                                        const FlowState& state,
                                        const GasModel& gas,
                                        const SolverOptions& options,
                                        Real leading_edge_x);
void write_flat_plate_benchmark_outputs_from_wall_type(const std::string& prefix,
                                                       const Mesh& mesh,
                                                       const FlowState& state,
                                                       const GasModel& gas,
                                                       const SolverOptions& options,
                                                       Real leading_edge_x,
                                                       BoundaryType wall_type);
void write_bump_benchmark_outputs(const std::string& prefix,
                                  const Mesh& mesh,
                                  const FlowState& state,
                                  const SolverOptions& options);

}  // namespace nssolver
