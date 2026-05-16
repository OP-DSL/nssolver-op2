#pragma once

#include <string>

#include "nssolver/mesh.hpp"
#include "nssolver/physics.hpp"
#include "nssolver/state.hpp"

namespace nssolver {

struct BenchmarkOptions {
    Primitive freestream {};
};

void write_flat_plate_benchmark_outputs(const std::string& prefix,
                                        const Mesh& mesh,
                                        const FlowState& state,
                                        const GasModel& gas,
                                        const BenchmarkOptions& options,
                                        Real leading_edge_x);
void write_flat_plate_benchmark_outputs_from_wall_type(const std::string& prefix,
                                                       const Mesh& mesh,
                                                       const FlowState& state,
                                                       const GasModel& gas,
                                                       const BenchmarkOptions& options,
                                                       Real leading_edge_x,
                                                       BoundaryType wall_type);

}  // namespace nssolver
