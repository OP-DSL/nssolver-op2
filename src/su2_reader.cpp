#include "nssolver/hydra_reader.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nssolver {

namespace {

struct TetCell {
    std::array<Index, 4> nodes {};
};

struct TriFace {
    std::array<Index, 3> nodes {};
    std::string marker;
    BoundaryType type {BoundaryType::Farfield};
    Index group_id {};
};

std::string trim_copy(const std::string& text) {
    std::size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }
    std::size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(start, end - start);
}

std::string read_trimmed_line(std::istream& in) {
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("unexpected end of SU2 mesh file");
    }
    return trim_copy(line);
}

std::pair<std::string, std::string> split_key_value(const std::string& line) {
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) {
        throw std::runtime_error("expected key=value line in SU2 mesh file: " + line);
    }
    return {trim_copy(line.substr(0, eq)), trim_copy(line.substr(eq + 1))};
}

BoundaryType su2_marker_type(const std::string& marker) {
    if (marker == "UPPER_SIDE" || marker == "LOWER_SIDE" || marker == "TIP") {
        return BoundaryType::NoSlipWall;
    }
    if (marker == "SYMMETRY_FACE") {
        return BoundaryType::SlipWall;
    }
    return BoundaryType::Farfield;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Real tetra_volume(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    return std::abs(dot(a - d, cross(b - d, c - d))) / 6.0;
}

Vec3 triangle_area_vector(const Vec3& a, const Vec3& b, const Vec3& c) {
    return 0.5 * cross(b - a, c - a);
}

Vec3 face_centroid(const Vec3& a, const Vec3& b, const Vec3& c) {
    return (a + b + c) / 3.0;
}

Vec3 tet_centroid(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    return 0.25 * (a + b + c + d);
}

std::pair<Index, Index> edge_key(Index a, Index b) {
    return a < b ? std::pair {a, b} : std::pair {b, a};
}

Vec3 quad_area_vector(const std::array<Vec3, 4>& poly) {
    return 0.5 * cross(poly[2] - poly[0], poly[3] - poly[1]);
}

Mesh build_su2_tet_mesh(const std::vector<Vec3>& points, const std::vector<TetCell>& tets, const std::vector<TriFace>& boundary) {
    Mesh mesh;
    mesh.nodes.count = points.size();
    mesh.nodes.x.resize(mesh.nodes.count);
    mesh.nodes.y.resize(mesh.nodes.count);
    mesh.nodes.z.resize(mesh.nodes.count);
    mesh.nodes.vol.assign(mesh.nodes.count, 0.0);
    mesh.nodes.wall_dist.assign(mesh.nodes.count, 0.0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        mesh.nodes.x[i] = points[i].x;
        mesh.nodes.y[i] = points[i].y;
        mesh.nodes.z[i] = points[i].z;
    }

    std::map<std::pair<Index, Index>, Vec3> edge_area;
    std::map<std::array<Index, 3>, Index> face_to_opposite;
    std::map<Index, std::string> group_names;
    std::map<Index, BoundaryType> group_types;

    for (const TetCell& tet : tets) {
        const Vec3 p0 = points[static_cast<std::size_t>(tet.nodes[0])];
        const Vec3 p1 = points[static_cast<std::size_t>(tet.nodes[1])];
        const Vec3 p2 = points[static_cast<std::size_t>(tet.nodes[2])];
        const Vec3 p3 = points[static_cast<std::size_t>(tet.nodes[3])];
        const Real volume = tetra_volume(p0, p1, p2, p3);
        for (Index node : tet.nodes) {
            mesh.nodes.vol[static_cast<std::size_t>(node)] += volume / 4.0;
        }

        const Vec3 centroid = tet_centroid(p0, p1, p2, p3);
        const std::array<Vec3, 4> coords = {p0, p1, p2, p3};
        const std::array<std::tuple<int, int, int, int>, 6> dual_specs = {{
            {0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2},
            {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 3, 0, 1},
        }};
        for (const auto& spec : dual_specs) {
            const int ia = std::get<0>(spec);
            const int ib = std::get<1>(spec);
            const int ic = std::get<2>(spec);
            const int id = std::get<3>(spec);
            const Vec3 midpoint = 0.5 * (coords[ia] + coords[ib]);
            const Vec3 face1 = face_centroid(coords[ia], coords[ib], coords[ic]);
            const Vec3 face2 = face_centroid(coords[ia], coords[ib], coords[id]);
            std::array<Vec3, 4> dual_poly = {midpoint, face1, centroid, face2};
            Vec3 contribution = quad_area_vector(dual_poly);
            const Vec3 edge_vec = coords[ib] - coords[ia];
            if (dot(contribution, edge_vec) < 0.0) {
                contribution = -1.0 * contribution;
            }
            edge_area[edge_key(tet.nodes[ia], tet.nodes[ib])] = edge_area[edge_key(tet.nodes[ia], tet.nodes[ib])] + contribution;
        }

        const std::array<std::tuple<int, int, int, int>, 4> faces = {{
            {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 3, 1}, {1, 2, 3, 0},
        }};
        for (const auto& face : faces) {
            std::array<Index, 3> key = {
                tet.nodes[std::get<0>(face)],
                tet.nodes[std::get<1>(face)],
                tet.nodes[std::get<2>(face)],
            };
            std::sort(key.begin(), key.end());
            face_to_opposite[key] = tet.nodes[std::get<3>(face)];
        }
    }

    mesh.edges.count = edge_area.size();
    mesh.edges.node_L.reserve(mesh.edges.count);
    mesh.edges.node_R.reserve(mesh.edges.count);
    mesh.edges.nx.reserve(mesh.edges.count);
    mesh.edges.ny.reserve(mesh.edges.count);
    mesh.edges.nz.reserve(mesh.edges.count);
    mesh.edges.area.reserve(mesh.edges.count);
    for (const auto& [key, area_vector] : edge_area) {
        mesh.edges.node_L.push_back(key.first);
        mesh.edges.node_R.push_back(key.second);
        mesh.edges.nx.push_back(area_vector.x);
        mesh.edges.ny.push_back(area_vector.y);
        mesh.edges.nz.push_back(area_vector.z);
        mesh.edges.area.push_back(norm(area_vector));
    }

    mesh.boundary_faces.count = boundary.size();
    mesh.boundary_faces.n1.reserve(boundary.size());
    mesh.boundary_faces.n2.reserve(boundary.size());
    mesh.boundary_faces.n3.reserve(boundary.size());
    mesh.boundary_faces.n4.reserve(boundary.size());
    mesh.boundary_faces.num_nodes.reserve(boundary.size());
    mesh.boundary_faces.nx.reserve(boundary.size());
    mesh.boundary_faces.ny.reserve(boundary.size());
    mesh.boundary_faces.nz.reserve(boundary.size());
    mesh.boundary_faces.area.reserve(boundary.size());
    mesh.boundary_faces.group_id.reserve(boundary.size());
    mesh.boundary_faces.type.reserve(boundary.size());
    mesh.boundary_faces.name.reserve(boundary.size());
    for (const TriFace& face : boundary) {
        const Vec3 a = points[static_cast<std::size_t>(face.nodes[0])];
        const Vec3 b = points[static_cast<std::size_t>(face.nodes[1])];
        const Vec3 c = points[static_cast<std::size_t>(face.nodes[2])];
        Vec3 normal = triangle_area_vector(a, b, c);
        std::array<Index, 3> key = face.nodes;
        std::sort(key.begin(), key.end());
        const auto it = face_to_opposite.find(key);
        if (it != face_to_opposite.end()) {
            const Vec3 interior = points[static_cast<std::size_t>(it->second)];
            if (dot(normal, interior - face_centroid(a, b, c)) > 0.0) {
                normal = -1.0 * normal;
            }
        }
        mesh.boundary_faces.n1.push_back(face.nodes[0]);
        mesh.boundary_faces.n2.push_back(face.nodes[1]);
        mesh.boundary_faces.n3.push_back(face.nodes[2]);
        mesh.boundary_faces.n4.push_back(face.nodes[2]);
        mesh.boundary_faces.num_nodes.push_back(3);
        mesh.boundary_faces.nx.push_back(normal.x);
        mesh.boundary_faces.ny.push_back(normal.y);
        mesh.boundary_faces.nz.push_back(normal.z);
        mesh.boundary_faces.area.push_back(norm(normal));
        mesh.boundary_faces.group_id.push_back(face.group_id);
        mesh.boundary_faces.type.push_back(face.type);
        mesh.boundary_faces.name.push_back(face.marker);
        group_names[face.group_id] = face.marker;
        group_types[face.group_id] = face.type;
    }

    std::vector<Index> wall_nodes;
    for (const TriFace& face : boundary) {
        if (face.type != BoundaryType::NoSlipWall && face.type != BoundaryType::SlipWall) {
            continue;
        }
        wall_nodes.insert(wall_nodes.end(), face.nodes.begin(), face.nodes.end());
    }
    std::sort(wall_nodes.begin(), wall_nodes.end());
    wall_nodes.erase(std::unique(wall_nodes.begin(), wall_nodes.end()), wall_nodes.end());
    for (Index node : wall_nodes) {
        mesh.nodes.wall_dist[static_cast<std::size_t>(node)] = 0.0;
    }

    mesh.node_to_edges.assign(mesh.nodes.count, {});
    mesh.node_to_boundary_faces.assign(mesh.nodes.count, {});
    for (Index e = 0; e < static_cast<Index>(mesh.edges.count); ++e) {
        mesh.node_to_edges[static_cast<std::size_t>(mesh.edges.node_L[e])].push_back(e);
        mesh.node_to_edges[static_cast<std::size_t>(mesh.edges.node_R[e])].push_back(e);
    }
    for (Index f = 0; f < static_cast<Index>(mesh.boundary_faces.count); ++f) {
        const std::array<Index, 3> nodes = {
            mesh.boundary_faces.n1[static_cast<std::size_t>(f)],
            mesh.boundary_faces.n2[static_cast<std::size_t>(f)],
            mesh.boundary_faces.n3[static_cast<std::size_t>(f)],
        };
        for (Index node : nodes) {
            mesh.node_to_boundary_faces[static_cast<std::size_t>(node)].push_back(f);
        }
    }
    return mesh;
}

}  // namespace

Mesh read_su2_mesh(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open SU2 mesh: " + path);
    }

    std::string line = read_trimmed_line(in);
    auto [ndime_key, ndime_value] = split_key_value(line);
    if (ndime_key != "NDIME" || std::stoi(ndime_value) != 3) {
        throw std::runtime_error("only 3D SU2 meshes are supported");
    }

    line = read_trimmed_line(in);
    auto [nelem_key, nelem_value] = split_key_value(line);
    if (nelem_key != "NELEM") {
        throw std::runtime_error("expected NELEM after NDIME in SU2 mesh");
    }
    const Index nelem = static_cast<Index>(std::stoi(nelem_value));
    std::vector<TetCell> tets;
    tets.reserve(static_cast<std::size_t>(nelem));
    for (Index i = 0; i < nelem; ++i) {
        std::stringstream row(read_trimmed_line(in));
        int type = 0;
        row >> type;
        if (type != 10) {
            throw std::runtime_error("only tetrahedral SU2 elements (type 10) are supported");
        }
        TetCell cell;
        row >> cell.nodes[0] >> cell.nodes[1] >> cell.nodes[2] >> cell.nodes[3];
        tets.push_back(cell);
    }

    line = read_trimmed_line(in);
    auto [npoin_key, npoin_value] = split_key_value(line);
    if (npoin_key != "NPOIN") {
        throw std::runtime_error("expected NPOIN after NELEM block in SU2 mesh");
    }
    const Index npoin = static_cast<Index>(std::stoi(npoin_value));
    std::vector<Vec3> points;
    points.reserve(static_cast<std::size_t>(npoin));
    for (Index i = 0; i < npoin; ++i) {
        std::stringstream row(read_trimmed_line(in));
        Vec3 point {};
        row >> point.x >> point.y >> point.z;
        points.push_back(point);
    }

    line = read_trimmed_line(in);
    auto [nmark_key, nmark_value] = split_key_value(line);
    if (nmark_key != "NMARK") {
        throw std::runtime_error("expected NMARK after NPOIN block in SU2 mesh");
    }
    const Index nmark = static_cast<Index>(std::stoi(nmark_value));
    std::vector<TriFace> boundary;
    Index next_group_id = 1;
    for (Index marker = 0; marker < nmark; ++marker) {
        auto [tag_key, tag_value] = split_key_value(read_trimmed_line(in));
        auto [count_key, count_value] = split_key_value(read_trimmed_line(in));
        if (tag_key != "MARKER_TAG" || count_key != "MARKER_ELEMS") {
            throw std::runtime_error("malformed SU2 marker block");
        }
        const Index marker_elems = static_cast<Index>(std::stoi(count_value));
        const BoundaryType type = su2_marker_type(tag_value);
        for (Index i = 0; i < marker_elems; ++i) {
            std::stringstream row(read_trimmed_line(in));
            int elem_type = 0;
            row >> elem_type;
            if (elem_type != 5) {
                throw std::runtime_error("only triangular SU2 boundary elements (type 5) are supported");
            }
            TriFace face;
            row >> face.nodes[0] >> face.nodes[1] >> face.nodes[2];
            face.marker = tag_value;
            face.type = type;
            face.group_id = next_group_id;
            boundary.push_back(face);
        }
        ++next_group_id;
    }

    return build_su2_tet_mesh(points, tets, boundary);
}

}  // namespace nssolver
