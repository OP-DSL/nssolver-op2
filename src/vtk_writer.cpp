#include "nssolver/vtk_writer.hpp"

#include <fstream>
#include <stdexcept>

namespace nssolver {

void write_vtk_legacy(const std::string& path, const Mesh& mesh, const FlowState& state) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open VTK output file");
    }

    out << "# vtk DataFile Version 3.0\n";
    out << "nssolver output\n";
    out << "ASCII\n";
    out << "DATASET POLYDATA\n";
    out << "POINTS " << mesh.nodes.count << " double\n";
    for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
        out << mesh.nodes.x[i] << ' ' << mesh.nodes.y[i] << ' ' << mesh.nodes.z[i] << '\n';
    }

    out << "VERTICES " << mesh.nodes.count << ' ' << 2 * mesh.nodes.count << '\n';
    for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
        out << "1 " << i << '\n';
    }

    out << "POINT_DATA " << mesh.nodes.count << '\n';
    out << "SCALARS density double 1\nLOOKUP_TABLE default\n";
    for (Real value : state.rho) {
        out << value << '\n';
    }
    out << "SCALARS pressure double 1\nLOOKUP_TABLE default\n";
    for (Real value : state.p) {
        out << value << '\n';
    }
    out << "VECTORS velocity double\n";
    for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
        out << state.u[i] << ' ' << state.v[i] << ' ' << state.w[i] << '\n';
    }
}

}  // namespace nssolver
