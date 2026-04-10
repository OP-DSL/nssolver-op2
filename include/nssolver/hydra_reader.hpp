#pragma once

#include <string>

#include "nssolver/mesh.hpp"

namespace nssolver {

struct MeshValidationReport {
    bool ok {true};
    std::size_t zero_or_negative_volumes {};
    std::size_t zero_area_edges {};
    std::size_t zero_area_boundary_faces {};
};

MeshValidationReport validate_mesh(const Mesh& mesh);
Mesh read_hydra_hdf5(const std::string& path);

}  // namespace nssolver
