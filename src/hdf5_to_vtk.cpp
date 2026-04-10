#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nssolver/hdf5_utils.hpp"
#include "nssolver/mesh.hpp"
#include "nssolver/physics.hpp"
#include "nssolver/state.hpp"
#include "nssolver/vtk_writer.hpp"

#ifdef NSSOLVER_HAVE_HDF5
#include <hdf5.h>
#endif

namespace nssolver {

namespace {

#ifdef NSSOLVER_HAVE_HDF5
Mesh read_op2_mesh_hdf5(const std::string& path) {
    const hdf5::Handle file = hdf5::open_file_readonly(path);
    const auto coords = hdf5::read_dataset<Real>(file, "node_coordinates").second;
    const auto node_volume = hdf5::read_dataset<Real>(file, "node_volume").second;
    const auto wall_distance = hdf5::read_dataset<Real>(file, "node_wall_distance").second;
    const auto edge_nodes = hdf5::read_dataset<Index>(file, "edge-->node").second;
    const auto edge_weights = hdf5::read_dataset<Real>(file, "edge_weights").second;
    const auto bface_nodes = hdf5::read_dataset<Index>(file, "bface-->node").second;
    const auto bface_normals = hdf5::read_dataset<Real>(file, "bface_normal").second;
    const auto bface_area = hdf5::read_dataset<Real>(file, "bface_area").second;
    const auto bface_group = hdf5::read_dataset<Index>(file, "bface_group").second;
    const auto bface_type = hdf5::read_dataset<int>(file, "bface_type").second;

    Mesh mesh;
    mesh.nodes.count = coords.size() / 3;
    mesh.nodes.x.resize(mesh.nodes.count);
    mesh.nodes.y.resize(mesh.nodes.count);
    mesh.nodes.z.resize(mesh.nodes.count);
    mesh.nodes.vol = node_volume;
    mesh.nodes.wall_dist = wall_distance;
    for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
        mesh.nodes.x[i] = coords[3 * i + 0];
        mesh.nodes.y[i] = coords[3 * i + 1];
        mesh.nodes.z[i] = coords[3 * i + 2];
    }

    mesh.edges.count = edge_nodes.size() / 2;
    mesh.edges.node_L.resize(mesh.edges.count);
    mesh.edges.node_R.resize(mesh.edges.count);
    mesh.edges.nx.resize(mesh.edges.count);
    mesh.edges.ny.resize(mesh.edges.count);
    mesh.edges.nz.resize(mesh.edges.count);
    mesh.edges.area.resize(mesh.edges.count);
    for (std::size_t e = 0; e < mesh.edges.count; ++e) {
        mesh.edges.node_L[e] = edge_nodes[2 * e + 0];
        mesh.edges.node_R[e] = edge_nodes[2 * e + 1];
        mesh.edges.nx[e] = edge_weights[3 * e + 0];
        mesh.edges.ny[e] = edge_weights[3 * e + 1];
        mesh.edges.nz[e] = edge_weights[3 * e + 2];
        mesh.edges.area[e] = norm(Vec3 {mesh.edges.nx[e], mesh.edges.ny[e], mesh.edges.nz[e]});
    }

    mesh.boundary_faces.count = bface_nodes.size() / 4;
    mesh.boundary_faces.n1.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.n2.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.n3.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.n4.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.nx.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.ny.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.nz.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.area = bface_area;
    mesh.boundary_faces.group_id = bface_group;
    mesh.boundary_faces.type.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.name.resize(mesh.boundary_faces.count);
    for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
        mesh.boundary_faces.n1[f] = bface_nodes[4 * f + 0];
        mesh.boundary_faces.n2[f] = bface_nodes[4 * f + 1];
        mesh.boundary_faces.n3[f] = bface_nodes[4 * f + 2];
        mesh.boundary_faces.n4[f] = bface_nodes[4 * f + 3];
        mesh.boundary_faces.nx[f] = bface_normals[3 * f + 0];
        mesh.boundary_faces.ny[f] = bface_normals[3 * f + 1];
        mesh.boundary_faces.nz[f] = bface_normals[3 * f + 2];
        mesh.boundary_faces.type[f] = static_cast<BoundaryType>(bface_type[f]);
        mesh.boundary_faces.name[f] = "group_" + std::to_string(mesh.boundary_faces.group_id[f]);
    }
    return mesh;
}

FlowState read_solution_hdf5(const std::string& path, const GasModel& gas, std::size_t node_count) {
    const hdf5::Handle file = hdf5::open_file_readonly(path);
    const auto q = hdf5::read_dataset<Real>(file, "q").second;
    if (q.size() != 6 * node_count) {
        throw std::runtime_error("solution dataset 'q' has unexpected shape");
    }

    FlowState state;
    state.resize(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
        state.rho[i] = q[6 * i + 0];
        state.rhou[i] = q[6 * i + 1];
        state.rhov[i] = q[6 * i + 2];
        state.rhow[i] = q[6 * i + 3];
        state.rhoE[i] = q[6 * i + 4];
        state.rhoNu[i] = q[6 * i + 5];
    }
    update_primitives(state, gas);
    return state;
}
#endif

}  // namespace

}  // namespace nssolver

int main(int argc, char** argv) {
    using namespace nssolver;

#ifndef NSSOLVER_HAVE_HDF5
    (void)argc;
    (void)argv;
    throw std::runtime_error("nssolver_hdf5_to_vtk requires HDF5-enabled build");
#else
    if (argc != 4) {
        std::cerr << "usage: nssolver_hdf5_to_vtk <mesh.h5> <solution.h5> <output.vtk>\n";
        return 1;
    }

    const GasModel gas {};
    const Mesh mesh = read_op2_mesh_hdf5(argv[1]);
    const FlowState state = read_solution_hdf5(argv[2], gas, mesh.nodes.count);
    const std::filesystem::path out_path(argv[3]);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    write_vtk_legacy(argv[3], mesh, state);
    std::cout << "Wrote VTK: " << argv[3] << "\n";
    return 0;
#endif
}
