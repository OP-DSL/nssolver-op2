#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nssolver/hdf5_utils.hpp"
#include "nssolver/mesh.hpp"
#include "nssolver/physics.hpp"
#include "nssolver/state.hpp"

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
    std::vector<Index> bface_num_nodes;
    if (hdf5::dataset_exists(file, "bface_num_nodes")) {
        bface_num_nodes = hdf5::read_dataset<Index>(file, "bface_num_nodes").second;
    }
    const auto bface_group = hdf5::read_dataset<Index>(file, "bface_group").second;
    const auto bface_type = hdf5::read_dataset<int>(file, "bface_type").second;

    Mesh mesh;
    mesh.nodes.count = coords.size() / 3;
    mesh.nodes.x.resize(mesh.nodes.count);
    mesh.nodes.y.resize(mesh.nodes.count);
    mesh.nodes.z.resize(mesh.nodes.count);
    mesh.nodes.vol = node_volume;
    mesh.nodes.wall_dist = wall_distance;
    mesh.node_to_edges.assign(mesh.nodes.count, {});
    mesh.node_to_boundary_faces.assign(mesh.nodes.count, {});
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
        mesh.node_to_edges[mesh.edges.node_L[e]].push_back(static_cast<Index>(e));
        mesh.node_to_edges[mesh.edges.node_R[e]].push_back(static_cast<Index>(e));
    }

    mesh.boundary_faces.count = bface_nodes.size() / 4;
    mesh.boundary_faces.n1.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.n2.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.n3.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.n4.resize(mesh.boundary_faces.count);
    mesh.boundary_faces.num_nodes.resize(mesh.boundary_faces.count, 4);
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
        if (!bface_num_nodes.empty()) {
            mesh.boundary_faces.num_nodes[f] = bface_num_nodes[f];
        }
        mesh.boundary_faces.nx[f] = bface_normals[3 * f + 0];
        mesh.boundary_faces.ny[f] = bface_normals[3 * f + 1];
        mesh.boundary_faces.nz[f] = bface_normals[3 * f + 2];
        mesh.boundary_faces.type[f] = static_cast<BoundaryType>(bface_type[f]);
        mesh.boundary_faces.name[f] = "group_" + std::to_string(mesh.boundary_faces.group_id[f]);
        const auto face_nodes = boundary_face_nodes(mesh.boundary_faces, f);
        const std::size_t face_node_count = boundary_face_num_nodes(mesh.boundary_faces, f);
        for (std::size_t local = 0; local < face_node_count; ++local) {
            mesh.node_to_boundary_faces[face_nodes[local]].push_back(static_cast<Index>(f));
        }
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

struct CoefficientSummary {
    Real reference_area {};
    Real dynamic_pressure {};
    Vec3 total_force {};
    Real cd {};
    Real cl {};
    Real cx {};
    Real cy {};
    Real cz {};
};

std::vector<Index> parse_group_list(const std::string& text) {
    std::vector<Index> groups;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            groups.push_back(static_cast<Index>(std::stoi(token)));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return groups;
}

bool contains_group(const std::vector<Index>& groups, Index group_id) {
    return std::find(groups.begin(), groups.end(), group_id) != groups.end();
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Vec3 normalize(const Vec3& v) {
    return v / std::max(norm(v), 1.0e-14);
}

Vec3 face_centroid(const Mesh& mesh, std::size_t f) {
    const auto face_nodes = boundary_face_nodes(mesh.boundary_faces, f);
    const std::size_t face_node_count = boundary_face_num_nodes(mesh.boundary_faces, f);
    Vec3 centroid {};
    for (std::size_t local = 0; local < face_node_count; ++local) {
        const Index node = face_nodes[local];
        centroid.x += mesh.nodes.x[node];
        centroid.y += mesh.nodes.y[node];
        centroid.z += mesh.nodes.z[node];
    }
    return centroid / static_cast<Real>(face_node_count);
}

Real face_pressure(const Mesh& mesh, const FlowState& state, std::size_t f) {
    const auto face_nodes = boundary_face_nodes(mesh.boundary_faces, f);
    const std::size_t face_node_count = boundary_face_num_nodes(mesh.boundary_faces, f);
    Real pressure = 0.0;
    for (std::size_t local = 0; local < face_node_count; ++local) {
        pressure += state.p[face_nodes[local]];
    }
    return pressure / static_cast<Real>(face_node_count);
}

CoefficientSummary compute_coefficients(const Mesh& mesh,
                                        const FlowState& state,
                                        const Primitive& freestream,
                                        const std::vector<Index>& wall_groups) {
    const Vec3 velocity {freestream.u, freestream.v, freestream.w};
    const Real vinf = norm(velocity);
    const Real qinf = 0.5 * freestream.rho * vinf * vinf;
    const Vec3 drag_dir = velocity / vinf;
    const Vec3 span_dir {0.0, 1.0, 0.0};
    const Vec3 lift_dir = normalize(cross(drag_dir, span_dir));

    CoefficientSummary summary {};
    summary.dynamic_pressure = qinf;
    for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
        if (!contains_group(wall_groups, mesh.boundary_faces.group_id[f])) {
            continue;
        }
        const Vec3 area_vector {
            mesh.boundary_faces.nx[f],
            mesh.boundary_faces.ny[f],
            mesh.boundary_faces.nz[f],
        };
        const Real pressure = face_pressure(mesh, state, f);
        summary.total_force = summary.total_force + pressure * area_vector;
        if (area_vector.z > 0.0) {
            summary.reference_area += area_vector.z;
        }
    }

    const Real denom = std::max(qinf * summary.reference_area, 1.0e-14);
    summary.cd = dot(summary.total_force, drag_dir) / denom;
    summary.cl = dot(summary.total_force, lift_dir) / denom;
    summary.cx = summary.total_force.x / denom;
    summary.cy = summary.total_force.y / denom;
    summary.cz = summary.total_force.z / denom;
    return summary;
}

void write_surface_cp_csv(const std::string& path,
                          const Mesh& mesh,
                          const FlowState& state,
                          const Primitive& freestream,
                          const std::vector<Index>& wall_groups) {
    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    const Vec3 velocity {freestream.u, freestream.v, freestream.w};
    const Real qinf = 0.5 * freestream.rho * dot(velocity, velocity);

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open surface Cp output: " + path);
    }
    out << "group_id,x,y,z,cp\n";
    for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
        if (!contains_group(wall_groups, mesh.boundary_faces.group_id[f])) {
            continue;
        }
        const Vec3 centroid = face_centroid(mesh, f);
        const Real pressure = face_pressure(mesh, state, f);
        const Real cp = (pressure - freestream.p) / qinf;
        out << mesh.boundary_faces.group_id[f] << ','
            << centroid.x << ','
            << centroid.y << ','
            << centroid.z << ','
            << cp << '\n';
    }
}

void write_summary_json(const std::string& path, const CoefficientSummary& summary) {
    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open summary output: " + path);
    }
    out << "{\n";
    out << "  \"reference_area\": " << summary.reference_area << ",\n";
    out << "  \"dynamic_pressure\": " << summary.dynamic_pressure << ",\n";
    out << "  \"force\": {\n";
    out << "    \"x\": " << summary.total_force.x << ",\n";
    out << "    \"y\": " << summary.total_force.y << ",\n";
    out << "    \"z\": " << summary.total_force.z << "\n";
    out << "  },\n";
    out << "  \"coefficients\": {\n";
    out << "    \"cd\": " << summary.cd << ",\n";
    out << "    \"cl\": " << summary.cl << ",\n";
    out << "    \"cx\": " << summary.cx << ",\n";
    out << "    \"cy\": " << summary.cy << ",\n";
    out << "    \"cz\": " << summary.cz << "\n";
    out << "  }\n";
    out << "}\n";
}

}  // namespace

}  // namespace nssolver

int main(int argc, char** argv) {
    using namespace nssolver;

#ifndef NSSOLVER_HAVE_HDF5
    (void)argc;
    (void)argv;
    throw std::runtime_error("nssolver_onera_m6_validation requires HDF5-enabled build");
#else
    if (argc != 11) {
        std::cerr << "usage: nssolver_onera_m6_validation <mesh.h5> <solution.h5> <summary.json> <surface_cp.csv> "
                     "<rho_inf> <u_inf> <v_inf> <w_inf> <p_inf> <wall_group_ids>\n";
        return 1;
    }

    const Mesh mesh = read_op2_mesh_hdf5(argv[1]);
    GasModel gas {};
    const FlowState state = read_solution_hdf5(argv[2], gas, mesh.nodes.count);
    const Primitive freestream {
        .rho = std::stod(argv[5]),
        .u = std::stod(argv[6]),
        .v = std::stod(argv[7]),
        .w = std::stod(argv[8]),
        .p = std::stod(argv[9]),
        .nu_tilde = 0.0,
    };
    const std::vector<Index> wall_groups = parse_group_list(argv[10]);
    const CoefficientSummary summary = compute_coefficients(mesh, state, freestream, wall_groups);
    write_summary_json(argv[3], summary);
    write_surface_cp_csv(argv[4], mesh, state, freestream, wall_groups);
    std::cout << "reference_area=" << summary.reference_area
              << " cd=" << summary.cd
              << " cl=" << summary.cl
              << " cx=" << summary.cx
              << " cz=" << summary.cz
              << "\n";
    return 0;
#endif
}
