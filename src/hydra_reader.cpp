#include "nssolver/hydra_reader.hpp"
#include "nssolver/hdf5_utils.hpp"

#ifdef NSSOLVER_HAVE_HDF5
#include <hdf5.h>
#endif

#include <array>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace nssolver {

MeshValidationReport validate_mesh(const Mesh& mesh) {
    MeshValidationReport report;
    for (Real volume : mesh.nodes.vol) {
        if (volume <= 0.0) {
            ++report.zero_or_negative_volumes;
        }
    }
    for (Real area : mesh.edges.area) {
        if (area <= 0.0) {
            ++report.zero_area_edges;
        }
    }
    for (Real area : mesh.boundary_faces.area) {
        if (area <= 0.0) {
            ++report.zero_area_boundary_faces;
        }
    }
    report.ok = report.zero_or_negative_volumes == 0 && report.zero_area_edges == 0 && report.zero_area_boundary_faces == 0;
    return report;
}

Mesh read_hydra_hdf5(const std::string& path) {
#ifdef NSSOLVER_HAVE_HDF5
    auto choose_dataset_name = [&](const hdf5::Handle& file, const std::initializer_list<const char*> names) {
        for (const char* name : names) {
            if (hdf5::dataset_exists(file, name)) {
                return std::string(name);
            }
        }
        std::ostringstream message;
        message << "missing required dataset; tried";
        for (const char* name : names) {
            message << ' ' << name;
        }
        throw std::runtime_error(message.str());
    };

    auto column_length = [](const std::vector<hsize_t>& dims) -> std::size_t {
        if (dims.size() == 1) {
            return static_cast<std::size_t>(dims[0]);
        }
        if (dims.size() == 2 && dims[1] == 1) {
            return static_cast<std::size_t>(dims[0]);
        }
        return 0;
    };

    auto to_zero_based_node = [](Index raw, std::size_t node_count, const std::string& dataset_name) -> Index {
        if (raw <= 0 || static_cast<std::size_t>(raw) > node_count) {
            throw std::runtime_error(dataset_name + " contains node index outside [1,N]");
        }
        return raw - 1;
    };

    auto decode_fixed_string = [](const std::vector<unsigned char>& bytes, std::size_t offset, std::size_t width) {
        std::string value;
        const std::size_t end = std::min(bytes.size(), offset + width);
        for (std::size_t i = offset; i < end; ++i) {
            if (bytes[i] == 0) {
                break;
            }
            if (std::isprint(bytes[i])) {
                value.push_back(static_cast<char>(bytes[i]));
            }
        }
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
            value.pop_back();
        }
        return value;
    };

    auto hydra_boundary_type = [](Index hydra_type) {
        switch (hydra_type) {
            case 2:
                return BoundaryType::NoSlipWall;
            case 4:
                return BoundaryType::Inlet;
            case 5:
                return BoundaryType::Outlet;
            case 9:
            case 10:
                // TODO: implement rotational periodic coupling. Until then these sector cuts
                // are imported explicitly but treated as slip walls for smoke/stability runs.
                return BoundaryType::SlipWall;
            default:
                return BoundaryType::Farfield;
        }
    };

    auto quad_normal = [](const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4) {
        const Vec3 d1 = p3 - p1;
        const Vec3 d2 = p4 - p2;
        return 0.5 * Vec3 {
            d1.y * d2.z - d1.z * d2.y,
            d1.z * d2.x - d1.x * d2.z,
            d1.x * d2.y - d1.y * d2.x,
        };
    };

    auto tetra_volume = [](const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
        const Vec3 ad = a - d;
        const Vec3 bd = b - d;
        const Vec3 cd = c - d;
        const Vec3 cross_cd {
            bd.y * cd.z - bd.z * cd.y,
            bd.z * cd.x - bd.x * cd.z,
            bd.x * cd.y - bd.y * cd.x,
        };
        return std::abs(ad.x * cross_cd.x + ad.y * cross_cd.y + ad.z * cross_cd.z) / 6.0;
    };

    auto hex_volume = [&](const std::array<Vec3, 8>& p) {
        return tetra_volume(p[0], p[1], p[3], p[4]) +
               tetra_volume(p[1], p[2], p[3], p[6]) +
               tetra_volume(p[1], p[3], p[4], p[6]) +
               tetra_volume(p[1], p[4], p[5], p[6]) +
               tetra_volume(p[3], p[4], p[6], p[7]);
    };

    try {
        const hdf5::Handle file = hdf5::open_file_readonly(path);

        Mesh mesh;
        const auto [coord_dims, coords] = hdf5::read_dataset<Real>(file, "node_coordinates");
        if (coord_dims.size() != 2 || coord_dims[1] != 3) {
            throw std::runtime_error("node_coordinates must be [N,3]");
        }
        mesh.nodes.count = static_cast<std::size_t>(coord_dims[0]);
        mesh.nodes.x.resize(mesh.nodes.count);
        mesh.nodes.y.resize(mesh.nodes.count);
        mesh.nodes.z.resize(mesh.nodes.count);
        mesh.nodes.vol.assign(mesh.nodes.count, 0.0);
        mesh.nodes.wall_dist.assign(mesh.nodes.count, 0.0);
        for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
            mesh.nodes.x[i] = coords[3 * i + 0];
            mesh.nodes.y[i] = coords[3 * i + 1];
            mesh.nodes.z[i] = coords[3 * i + 2];
        }

        const std::string edge_node_name = choose_dataset_name(file, {"edge-->node", "edge->node"});
        const auto [edge_dims, edge_nodes] = hdf5::read_dataset<Index>(file, edge_node_name);
        if (edge_dims.size() != 2 || edge_dims[1] != 2) {
            throw std::runtime_error(edge_node_name + " must be [E,2]");
        }
        const auto [weight_dims, edge_weights] = hdf5::read_dataset<Real>(file, "edge_weights");
        if (weight_dims.size() != 2 || weight_dims[1] != 3 || weight_dims[0] != edge_dims[0]) {
            throw std::runtime_error("edge_weights must be [E,3]");
        }
        mesh.edges.count = static_cast<std::size_t>(edge_dims[0]);
        mesh.edges.node_L.resize(mesh.edges.count);
        mesh.edges.node_R.resize(mesh.edges.count);
        mesh.edges.nx.resize(mesh.edges.count);
        mesh.edges.ny.resize(mesh.edges.count);
        mesh.edges.nz.resize(mesh.edges.count);
        mesh.edges.area.resize(mesh.edges.count);
        for (std::size_t e = 0; e < mesh.edges.count; ++e) {
            mesh.edges.node_L[e] = to_zero_based_node(edge_nodes[2 * e + 0], mesh.nodes.count, edge_node_name);
            mesh.edges.node_R[e] = to_zero_based_node(edge_nodes[2 * e + 1], mesh.nodes.count, edge_node_name);
            mesh.edges.nx[e] = edge_weights[3 * e + 0];
            mesh.edges.ny[e] = edge_weights[3 * e + 1];
            mesh.edges.nz[e] = edge_weights[3 * e + 2];
            mesh.edges.area[e] = norm(Vec3 {mesh.edges.nx[e], mesh.edges.ny[e], mesh.edges.nz[e]});
        }

        const std::string hex_node_name = choose_dataset_name(file, {"hex-->node", "hex->node"});
        const auto [hex_dims, hex_nodes] = hdf5::read_dataset<Index>(file, hex_node_name);
        if (hex_dims.size() != 2 || hex_dims[1] != 8) {
            throw std::runtime_error(hex_node_name + " must be [H,8]");
        }
        for (std::size_t h = 0; h < static_cast<std::size_t>(hex_dims[0]); ++h) {
            std::array<Index, 8> nodes {};
            std::array<Vec3, 8> points {};
            for (int n = 0; n < 8; ++n) {
                nodes[n] = to_zero_based_node(hex_nodes[8 * h + n], mesh.nodes.count, hex_node_name);
                points[n] = {mesh.nodes.x[nodes[n]], mesh.nodes.y[nodes[n]], mesh.nodes.z[nodes[n]]};
            }
            const Real volume = hex_volume(points);
            for (Index node : nodes) {
                mesh.nodes.vol[node] += volume / 8.0;
            }
        }

        std::vector<BoundaryType> group_types;
        std::vector<std::string> group_names;
        if (hdf5::dataset_exists(file, "surface_group_type")) {
            const auto [type_dims, surface_types] = hdf5::read_dataset<Index>(file, "surface_group_type");
            const std::size_t group_count = column_length(type_dims);
            if (group_count == 0 || group_count != surface_types.size()) {
                throw std::runtime_error("surface_group_type must be [G] or [G,1]");
            }
            group_types.resize(group_count);
            for (std::size_t group = 0; group < group_count; ++group) {
                group_types[group] = hydra_boundary_type(surface_types[group]);
            }
        }
        if (hdf5::dataset_exists(file, "surface_groups")) {
            const auto [name_dims, surface_names] = hdf5::read_dataset<unsigned char>(file, "surface_groups");
            if (name_dims.size() == 2) {
                group_names.resize(static_cast<std::size_t>(name_dims[0]));
                const std::size_t width = static_cast<std::size_t>(name_dims[1]);
                for (std::size_t group = 0; group < group_names.size(); ++group) {
                    group_names[group] = decode_fixed_string(surface_names, group * width, width);
                }
            }
        }

        const std::string quad_node_name = choose_dataset_name(file, {"quad-->node", "quad->node"});
        const std::string quad_group_name = choose_dataset_name(file, {"quad-->group", "quad->group"});
        const auto [quad_dims, quad_nodes] = hdf5::read_dataset<Index>(file, quad_node_name);
        const auto [group_dims, quad_groups] = hdf5::read_dataset<Index>(file, quad_group_name);
        if (quad_dims.size() != 2 || quad_dims[1] != 4 || column_length(group_dims) != static_cast<std::size_t>(quad_dims[0])) {
            throw std::runtime_error("quad boundary datasets have inconsistent shapes");
        }
        mesh.boundary_faces.count = static_cast<std::size_t>(quad_dims[0]);
        mesh.boundary_faces.n1.resize(mesh.boundary_faces.count);
        mesh.boundary_faces.n2.resize(mesh.boundary_faces.count);
        mesh.boundary_faces.n3.resize(mesh.boundary_faces.count);
        mesh.boundary_faces.n4.resize(mesh.boundary_faces.count);
        mesh.boundary_faces.nx.resize(mesh.boundary_faces.count);
        mesh.boundary_faces.ny.resize(mesh.boundary_faces.count);
        mesh.boundary_faces.nz.resize(mesh.boundary_faces.count);
        mesh.boundary_faces.area.resize(mesh.boundary_faces.count);
        mesh.boundary_faces.group_id.resize(mesh.boundary_faces.count);
        mesh.boundary_faces.type.resize(mesh.boundary_faces.count, BoundaryType::Farfield);
        mesh.boundary_faces.name.resize(mesh.boundary_faces.count);
        for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
            const Index n1 = to_zero_based_node(quad_nodes[4 * f + 0], mesh.nodes.count, quad_node_name);
            const Index n2 = to_zero_based_node(quad_nodes[4 * f + 1], mesh.nodes.count, quad_node_name);
            const Index n3 = to_zero_based_node(quad_nodes[4 * f + 2], mesh.nodes.count, quad_node_name);
            const Index n4 = to_zero_based_node(quad_nodes[4 * f + 3], mesh.nodes.count, quad_node_name);
            const Vec3 p1 {mesh.nodes.x[n1], mesh.nodes.y[n1], mesh.nodes.z[n1]};
            const Vec3 p2 {mesh.nodes.x[n2], mesh.nodes.y[n2], mesh.nodes.z[n2]};
            const Vec3 p3 {mesh.nodes.x[n3], mesh.nodes.y[n3], mesh.nodes.z[n3]};
            const Vec3 p4 {mesh.nodes.x[n4], mesh.nodes.y[n4], mesh.nodes.z[n4]};
            const Vec3 normal = quad_normal(p1, p2, p3, p4);

            mesh.boundary_faces.n1[f] = n1;
            mesh.boundary_faces.n2[f] = n2;
            mesh.boundary_faces.n3[f] = n3;
            mesh.boundary_faces.n4[f] = n4;
            mesh.boundary_faces.nx[f] = normal.x;
            mesh.boundary_faces.ny[f] = normal.y;
            mesh.boundary_faces.nz[f] = normal.z;
            mesh.boundary_faces.area[f] = norm(normal);
            mesh.boundary_faces.group_id[f] = quad_groups[f];
            const Index group_id = quad_groups[f];
            if (group_id <= 0) {
                throw std::runtime_error(quad_group_name + " contains non-positive group id");
            }
            if (!group_types.empty() && static_cast<std::size_t>(group_id) > group_types.size()) {
                throw std::runtime_error(quad_group_name + " references a missing surface_group_type row");
            }
            if (!group_names.empty() && static_cast<std::size_t>(group_id) > group_names.size()) {
                throw std::runtime_error(quad_group_name + " references a missing surface_groups row");
            }
            if (group_id > 0 && static_cast<std::size_t>(group_id) <= group_types.size()) {
                mesh.boundary_faces.type[f] = group_types[static_cast<std::size_t>(group_id - 1)];
            }
            if (group_id > 0 && static_cast<std::size_t>(group_id) <= group_names.size() &&
                !group_names[static_cast<std::size_t>(group_id - 1)].empty()) {
                mesh.boundary_faces.name[f] = group_names[static_cast<std::size_t>(group_id - 1)];
            } else {
                mesh.boundary_faces.name[f] = "group_" + std::to_string(group_id);
            }
        }

        if (hdf5::dataset_exists(file, "node_wall_distance")) {
            const auto [wd_dims, wall_distance] = hdf5::read_dataset<Real>(file, "node_wall_distance");
            if (column_length(wd_dims) == mesh.nodes.count) {
                mesh.nodes.wall_dist = wall_distance;
            } else {
                throw std::runtime_error("node_wall_distance must be [N] or [N,1]");
            }
        }

        mesh.node_to_edges.assign(mesh.nodes.count, {});
        mesh.node_to_boundary_faces.assign(mesh.nodes.count, {});
        for (Index e = 0; e < static_cast<Index>(mesh.edges.count); ++e) {
            mesh.node_to_edges[mesh.edges.node_L[e]].push_back(e);
            mesh.node_to_edges[mesh.edges.node_R[e]].push_back(e);
        }
        for (Index f = 0; f < static_cast<Index>(mesh.boundary_faces.count); ++f) {
            const std::array<Index, 4> nodes = {
                mesh.boundary_faces.n1[f], mesh.boundary_faces.n2[f], mesh.boundary_faces.n3[f], mesh.boundary_faces.n4[f]};
            for (Index n : nodes) {
                mesh.node_to_boundary_faces[n].push_back(f);
            }
        }

        return mesh;
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("Failed to read Hydra HDF5: ") + ex.what());
    }
#else
    (void) path;
    throw std::runtime_error("Hydra HDF5 support is not enabled in this build");
#endif
}

}  // namespace nssolver
