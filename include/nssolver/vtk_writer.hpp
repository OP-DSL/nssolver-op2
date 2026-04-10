#pragma once

#include <string>

#include "nssolver/mesh.hpp"
#include "nssolver/state.hpp"

namespace nssolver {

void write_vtk_legacy(const std::string& path, const Mesh& mesh, const FlowState& state);

}  // namespace nssolver
