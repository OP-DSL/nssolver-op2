#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <array>
#include <map>
#include <vector>

#include "nssolver/hydra_reader.hpp"
#include "nssolver/mesh.hpp"

#ifdef NSSOLVER_HAVE_HDF5
#include <H5Cpp.h>
#endif

namespace nssolver {

namespace {

using ParameterMap = std::map<std::string, std::string>;

#ifdef NSSOLVER_HAVE_HDF5
template <typename T>
void write_dataset_2d(H5::H5File& file,
                      const std::string& name,
                      hsize_t dim0,
                      hsize_t dim1,
                      const T* data,
                      const H5::PredType& type) {
    const hsize_t dims[2] = {dim0, dim1};
    H5::DataSpace space(2, dims);
    H5::DataSet dataset = file.createDataSet(name, type, space);
    dataset.write(data, type);
}

template <typename T>
void write_dataset_1d(H5::H5File& file,
                      const std::string& name,
                      hsize_t dim0,
                      const T* data,
                      const H5::PredType& type) {
    write_dataset_2d(file, name, dim0, 1, data, type);
}

std::vector<Real> interleave3(const std::vector<Real>& x, const std::vector<Real>& y, const std::vector<Real>& z) {
    std::vector<Real> values(3 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        values[3 * i + 0] = x[i];
        values[3 * i + 1] = y[i];
        values[3 * i + 2] = z[i];
    }
    return values;
}

std::vector<Index> interleave4(const std::vector<Index>& a,
                               const std::vector<Index>& b,
                               const std::vector<Index>& c,
                               const std::vector<Index>& d) {
    std::vector<Index> values(4 * a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        values[4 * i + 0] = a[i];
        values[4 * i + 1] = b[i];
        values[4 * i + 2] = c[i];
        values[4 * i + 3] = d[i];
    }
    return values;
}

void write_op2_mesh_hdf5(const Mesh& mesh, const std::string& output_path) {
    const std::filesystem::path out_path(output_path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    H5::H5File file(output_path, H5F_ACC_TRUNC);
    const auto coords = interleave3(mesh.nodes.x, mesh.nodes.y, mesh.nodes.z);
    const auto edge_weights = interleave3(mesh.edges.nx, mesh.edges.ny, mesh.edges.nz);
    const auto bface_normals = interleave3(mesh.boundary_faces.nx, mesh.boundary_faces.ny, mesh.boundary_faces.nz);
    const auto bface_nodes =
        interleave4(mesh.boundary_faces.n1, mesh.boundary_faces.n2, mesh.boundary_faces.n3, mesh.boundary_faces.n4);
    std::vector<Index> bnode_to_node;
    std::vector<int> bnode_dirichlet;
    std::vector<int> bnode_wall;
    std::vector<int> bnode_slip;
    std::vector<Real> bnode_normal(3 * mesh.nodes.count, 0.0);
    std::vector<char> is_boundary(mesh.nodes.count, 0);
    std::vector<char> is_dirichlet(mesh.nodes.count, 0);
    std::vector<char> is_wall(mesh.nodes.count, 0);
    std::vector<char> is_slip(mesh.nodes.count, 0);
    for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
        const std::array<Index, 4> nodes = {
            mesh.boundary_faces.n1[f], mesh.boundary_faces.n2[f], mesh.boundary_faces.n3[f], mesh.boundary_faces.n4[f]};
        const Vec3 normal {mesh.boundary_faces.nx[f], mesh.boundary_faces.ny[f], mesh.boundary_faces.nz[f]};
        for (Index node : nodes) {
            is_boundary[node] = 1;
            if (mesh.boundary_faces.type[f] == BoundaryType::Farfield || mesh.boundary_faces.type[f] == BoundaryType::Inlet) {
                is_dirichlet[node] = 1;
            } else if (mesh.boundary_faces.type[f] == BoundaryType::NoSlipWall) {
                is_wall[node] = 1;
            } else if (mesh.boundary_faces.type[f] == BoundaryType::SlipWall) {
                is_slip[node] = 1;
                bnode_normal[3 * static_cast<std::size_t>(node) + 0] += normal.x;
                bnode_normal[3 * static_cast<std::size_t>(node) + 1] += normal.y;
                bnode_normal[3 * static_cast<std::size_t>(node) + 2] += normal.z;
            }
        }
    }

    for (std::size_t node = 0; node < mesh.nodes.count; ++node) {
        if (!is_boundary[node]) {
            continue;
        }
        bnode_to_node.push_back(static_cast<Index>(node));
        bnode_dirichlet.push_back(is_dirichlet[node] ? 1 : 0);
        bnode_wall.push_back(is_wall[node] ? 1 : 0);
        bnode_slip.push_back(is_slip[node] ? 1 : 0);
    }
    std::vector<Real> bnode_normal_compact(3 * bnode_to_node.size());
    for (std::size_t i = 0; i < bnode_to_node.size(); ++i) {
        const std::size_t node = static_cast<std::size_t>(bnode_to_node[i]);
        bnode_normal_compact[3 * i + 0] = bnode_normal[3 * node + 0];
        bnode_normal_compact[3 * i + 1] = bnode_normal[3 * node + 1];
        bnode_normal_compact[3 * i + 2] = bnode_normal[3 * node + 2];
    }

    write_dataset_2d(file, "node_coordinates", mesh.nodes.count, 3, coords.data(), H5::PredType::NATIVE_DOUBLE);
    write_dataset_1d(file, "node_volume", mesh.nodes.count, mesh.nodes.vol.data(), H5::PredType::NATIVE_DOUBLE);
    write_dataset_1d(
        file, "node_wall_distance", mesh.nodes.count, mesh.nodes.wall_dist.data(), H5::PredType::NATIVE_DOUBLE);

    {
        std::vector<Index> edge_nodes(2 * mesh.edges.count);
        for (std::size_t e = 0; e < mesh.edges.count; ++e) {
            edge_nodes[2 * e + 0] = mesh.edges.node_L[e];
            edge_nodes[2 * e + 1] = mesh.edges.node_R[e];
        }
        write_dataset_2d(file, "edge-->node", mesh.edges.count, 2, edge_nodes.data(), H5::PredType::NATIVE_INT32);
    }
    write_dataset_2d(file, "edge_weights", mesh.edges.count, 3, edge_weights.data(), H5::PredType::NATIVE_DOUBLE);

    write_dataset_2d(file, "bface-->node", mesh.boundary_faces.count, 4, bface_nodes.data(), H5::PredType::NATIVE_INT32);
    write_dataset_2d(
        file, "bface_normal", mesh.boundary_faces.count, 3, bface_normals.data(), H5::PredType::NATIVE_DOUBLE);
    write_dataset_1d(
        file, "bface_area", mesh.boundary_faces.count, mesh.boundary_faces.area.data(), H5::PredType::NATIVE_DOUBLE);
    write_dataset_1d(
        file, "bface_group", mesh.boundary_faces.count, mesh.boundary_faces.group_id.data(), H5::PredType::NATIVE_INT32);
    {
        std::vector<int> types(mesh.boundary_faces.count);
        for (std::size_t i = 0; i < mesh.boundary_faces.count; ++i) {
            types[i] = static_cast<int>(mesh.boundary_faces.type[i]);
        }
        write_dataset_1d(file, "bface_type", mesh.boundary_faces.count, types.data(), H5::PredType::NATIVE_INT32);
    }
    write_dataset_1d(file, "bnode-->node", bnode_to_node.size(), bnode_to_node.data(), H5::PredType::NATIVE_INT32);
    write_dataset_1d(file, "bnode_dirichlet", bnode_dirichlet.size(), bnode_dirichlet.data(), H5::PredType::NATIVE_INT32);
    write_dataset_1d(file, "bnode_wall", bnode_wall.size(), bnode_wall.data(), H5::PredType::NATIVE_INT32);
    write_dataset_1d(file, "bnode_slip", bnode_slip.size(), bnode_slip.data(), H5::PredType::NATIVE_INT32);
    write_dataset_2d(
        file, "bnode_normal", bnode_to_node.size(), 3, bnode_normal_compact.data(), H5::PredType::NATIVE_DOUBLE);
}
#endif

int parse_int_param(const ParameterMap& params, const std::string& key, int default_value) {
    const auto it = params.find(key);
    return it == params.end() ? default_value : std::stoi(it->second);
}

Real parse_real_param(const ParameterMap& params, const std::string& key, Real default_value) {
    const auto it = params.find(key);
    return it == params.end() ? default_value : std::stod(it->second);
}

ParameterMap parse_parameters(int argc, char** argv, int first_index) {
    ParameterMap params;
    for (int i = first_index; i < argc; ++i) {
        const std::string arg = argv[i];
        const std::size_t eq = arg.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 >= arg.size()) {
            throw std::runtime_error("expected parameter in key=value form, got: " + arg);
        }
        params[arg.substr(0, eq)] = arg.substr(eq + 1);
    }
    return params;
}

Mesh build_mesh(const std::string& case_name, const std::string& input_path, const ParameterMap& params) {
    if (case_name == "box") {
        return make_structured_box_mesh(StructuredBoxSpec {
            .nx = 21, .ny = 11, .nz = 2, .lx = 2.0, .ly = 1.0, .lz = 0.05, .xmin = BoundaryType::Farfield,
            .xmax = BoundaryType::Farfield, .ymin = BoundaryType::SlipWall, .ymax = BoundaryType::SlipWall,
            .zmin = BoundaryType::SlipWall, .zmax = BoundaryType::SlipWall});
    }
    if (case_name == "bump") {
        return make_bump_channel_mesh(BumpChannelSpec {
            .nx = 81, .ny = 33, .nz = 2, .lx = 3.0, .ly = 1.0, .lz = 0.05, .bump_center = 1.5, .bump_half_width = 0.5,
            .bump_height = 0.08, .inlet = BoundaryType::Farfield, .outlet = BoundaryType::Farfield,
            .lower_wall = BoundaryType::SlipWall, .upper_wall = BoundaryType::SlipWall});
    }
    if (case_name == "flatplate" || case_name == "flatplate_develop") {
        return make_flat_plate_mesh(FlatPlateSpec {
            .nx = 81, .ny = 41, .nz = 2, .lx = 1.5, .ly = 0.4, .lz = 0.02, .leading_edge_x = 0.3,
            .wall_normal_growth = 2000.0, .inlet = BoundaryType::Inlet, .outlet = BoundaryType::Outlet,
            .plate_type = BoundaryType::NoSlipWall, .lower_upstream_type = BoundaryType::SlipWall,
            .upper_type = BoundaryType::Farfield});
    }
    if (case_name == "bump3d") {
        return make_bump_3d_mesh(Bump3DSpec {
            .nx = parse_int_param(params, "nx", 61),
            .ny = parse_int_param(params, "ny", 29),
            .nz = parse_int_param(params, "nz", 31),
            .lx = parse_real_param(params, "lx", 2.4),
            .ly = parse_real_param(params, "ly", 0.55),
            .lz = parse_real_param(params, "lz", 0.8),
            .bump_center_x = parse_real_param(params, "bump_center_x", 1.1),
            .bump_half_width_x = parse_real_param(params, "bump_half_width_x", 0.35),
            .bump_center_z = parse_real_param(params, "bump_center_z", 0.4),
            .bump_half_width_z = parse_real_param(params, "bump_half_width_z", 0.16),
            .bump_height = parse_real_param(params, "bump_height", 0.14),
            .inlet = BoundaryType::Farfield, .outlet = BoundaryType::Farfield,
            .lower_wall = BoundaryType::SlipWall, .upper_wall = BoundaryType::SlipWall,
            .side_wall = BoundaryType::SlipWall});
    }
    if (case_name == "axisymmetric_body") {
        return make_axisymmetric_body_mesh(AxisymmetricBodySpec {
            .nx = parse_int_param(params, "nx", 101),
            .nr = parse_int_param(params, "nr", 29),
            .ntheta = parse_int_param(params, "ntheta", 15),
            .lx = parse_real_param(params, "lx", 2.6),
            .body_start_x = parse_real_param(params, "body_start_x", 0.35),
            .body_length = parse_real_param(params, "body_length", 1.7),
            .body_radius_max = parse_real_param(params, "body_radius_max", 0.24),
            .body_tail_radius = parse_real_param(params, "body_tail_radius", 0.02),
            .farfield_radius = parse_real_param(params, "farfield_radius", 0.5),
            .wedge_angle_degrees = parse_real_param(params, "wedge_angle_degrees", 18.0),
            .radial_growth = parse_real_param(params, "radial_growth", 3.5),
            .inlet = BoundaryType::Farfield, .outlet = BoundaryType::Farfield,
            .body_wall = BoundaryType::SlipWall, .farfield = BoundaryType::Farfield,
            .wedge_wall = BoundaryType::SlipWall});
    }
    if (case_name == "naca_wing") {
        return make_naca_finite_wing_mesh(NacaWingSpec {
            .nx = parse_int_param(params, "nx", 121),
            .ny = parse_int_param(params, "ny", 41),
            .nz = parse_int_param(params, "nz", 81),
            .lx = parse_real_param(params, "lx", 2.8),
            .ly = parse_real_param(params, "ly", 0.9),
            .lz = parse_real_param(params, "lz", 1.6),
            .wing_origin_x = parse_real_param(params, "wing_origin_x", 0.45),
            .wing_center_z = parse_real_param(params, "wing_center_z", 0.8),
            .span = parse_real_param(params, "span", 1.0),
            .root_chord = parse_real_param(params, "root_chord", 0.8),
            .tip_chord = parse_real_param(params, "tip_chord", 0.4),
            .sweep_degrees = parse_real_param(params, "sweep_degrees", 20.0),
            .thickness_ratio = parse_real_param(params, "thickness_ratio", 0.12),
            .camber_ratio = parse_real_param(params, "camber_ratio", 0.0),
            .camber_position = parse_real_param(params, "camber_position", 0.4),
            .vertical_offset = parse_real_param(params, "vertical_offset", 0.0),
            .wall_normal_growth = parse_real_param(params, "wall_normal_growth", 4.0),
            .inlet = BoundaryType::Farfield, .outlet = BoundaryType::Farfield,
            .wing_wall = BoundaryType::NoSlipWall,
            .lower_farfield = BoundaryType::SlipWall,
            .upper_farfield = BoundaryType::Farfield,
            .side_farfield = BoundaryType::Farfield});
    }
    if (case_name == "hydra" || case_name == "hydra_benchmark") {
        if (input_path.empty()) {
            throw std::runtime_error(case_name + " preprocessing requires an input Hydra HDF5 mesh path");
        }
        return read_hydra_hdf5(input_path);
    }
    throw std::runtime_error("unknown preprocessing case: " + case_name);
}

}  // namespace

}  // namespace nssolver

int main(int argc, char** argv) {
    using namespace nssolver;

#ifndef NSSOLVER_HAVE_HDF5
    (void)argc;
    (void)argv;
    throw std::runtime_error("nssolver_preprocess_op2 requires HDF5-enabled build");
#else
    if (argc < 3) {
        std::cerr << "usage: nssolver_preprocess_op2 <case> <output_mesh.h5> [input_mesh.hdf] [key=value ...]\n";
        return 1;
    }

    const std::string case_name = argv[1];
    const std::string output_path = argv[2];
    int param_index = 3;
    std::string input_path;
    if ((case_name == "hydra" || case_name == "hydra_benchmark") && argc >= 4) {
        input_path = argv[3];
        param_index = 4;
    }
    const ParameterMap params = parse_parameters(argc, argv, param_index);

    const Mesh mesh = build_mesh(case_name, input_path, params);
    const MeshValidationReport report = validate_mesh(mesh);
    if (!report.ok) {
        throw std::runtime_error("preprocessed mesh failed validation");
    }
    write_op2_mesh_hdf5(mesh, output_path);
    std::cout << "Wrote OP2 mesh: " << output_path << "\n";
    std::cout << "nodes=" << mesh.nodes.count << ", edges=" << mesh.edges.count
              << ", boundary_faces=" << mesh.boundary_faces.count << "\n";
    return 0;
#endif
}
