#include "nssolver/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <unordered_map>
#include <functional>
#include <limits>
#include <stdexcept>

namespace nssolver {

namespace {

using MappingFunction = std::function<Vec3(Index, Index, Index)>;
using BoundaryTypeFunction = std::function<BoundaryType(Index, Index)>;
using BoundaryNameFunction = std::function<std::string(Index, Index)>;

Index node_id(Index i, Index j, Index k, Index nx, Index ny) {
    return k * nx * ny + j * nx + i;
}

Index adjacent_cell_count_for_x_edge(Index j, Index k, Index ny, Index nz) {
    Index count = 0;
    for (Index dj = -1; dj <= 0; ++dj) {
        const Index cj = j + dj;
        if (0 <= cj && cj < ny - 1) {
            for (Index dk = -1; dk <= 0; ++dk) {
                const Index ck = k + dk;
                if (0 <= ck && ck < nz - 1) {
                    ++count;
                }
            }
        }
    }
    return count;
}

Index adjacent_cell_count_for_y_edge(Index i, Index k, Index nx, Index nz) {
    Index count = 0;
    for (Index di = -1; di <= 0; ++di) {
        const Index ci = i + di;
        if (0 <= ci && ci < nx - 1) {
            for (Index dk = -1; dk <= 0; ++dk) {
                const Index ck = k + dk;
                if (0 <= ck && ck < nz - 1) {
                    ++count;
                }
            }
        }
    }
    return count;
}

Index adjacent_cell_count_for_z_edge(Index i, Index j, Index nx, Index ny) {
    Index count = 0;
    for (Index di = -1; di <= 0; ++di) {
        const Index ci = i + di;
        if (0 <= ci && ci < nx - 1) {
            for (Index dj = -1; dj <= 0; ++dj) {
                const Index cj = j + dj;
                if (0 <= cj && cj < ny - 1) {
                    ++count;
                }
            }
        }
    }
    return count;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

Vec3 quad_area_vector(const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4) {
    const Vec3 d1 = p3 - p1;
    const Vec3 d2 = p4 - p2;
    return 0.5 * cross(d1, d2);
}

Real tetra_volume(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    return std::abs(dot(a - d, cross(b - d, c - d))) / 6.0;
}

Real hex_volume(const std::array<Vec3, 8>& p) {
    return tetra_volume(p[0], p[1], p[3], p[4]) +
           tetra_volume(p[1], p[2], p[3], p[6]) +
           tetra_volume(p[1], p[3], p[4], p[6]) +
           tetra_volume(p[1], p[4], p[5], p[6]) +
           tetra_volume(p[3], p[4], p[6], p[7]);
}

void add_edge(Mesh& mesh, Index left, Index right, const Vec3& area_vector) {
    mesh.edges.node_L.push_back(left);
    mesh.edges.node_R.push_back(right);
    mesh.edges.nx.push_back(area_vector.x);
    mesh.edges.ny.push_back(area_vector.y);
    mesh.edges.nz.push_back(area_vector.z);
    mesh.edges.area.push_back(norm(area_vector));
}

void add_boundary_face(Mesh& mesh,
                       Index n1,
                       Index n2,
                       Index n3,
                       Index n4,
                       const Vec3& normal,
                       BoundaryType type,
                       const std::string& name,
                       Index group_id,
                       Index num_nodes = 4) {
    mesh.boundary_faces.n1.push_back(n1);
    mesh.boundary_faces.n2.push_back(n2);
    mesh.boundary_faces.n3.push_back(n3);
    mesh.boundary_faces.n4.push_back(n4);
    mesh.boundary_faces.num_nodes.push_back(num_nodes);
    mesh.boundary_faces.nx.push_back(normal.x);
    mesh.boundary_faces.ny.push_back(normal.y);
    mesh.boundary_faces.nz.push_back(normal.z);
    mesh.boundary_faces.area.push_back(norm(normal));
    mesh.boundary_faces.group_id.push_back(group_id);
    mesh.boundary_faces.type.push_back(type);
    mesh.boundary_faces.name.push_back(name);
}

void finalize_mesh(Mesh& mesh) {
    mesh.edges.count = mesh.edges.node_L.size();
    mesh.boundary_faces.count = mesh.boundary_faces.n1.size();
    mesh.node_to_edges.assign(mesh.nodes.count, {});
    mesh.node_to_boundary_faces.assign(mesh.nodes.count, {});

    for (Index e = 0; e < static_cast<Index>(mesh.edges.count); ++e) {
        mesh.node_to_edges[mesh.edges.node_L[e]].push_back(e);
        mesh.node_to_edges[mesh.edges.node_R[e]].push_back(e);
    }

    for (Index f = 0; f < static_cast<Index>(mesh.boundary_faces.count); ++f) {
        const auto nodes = boundary_face_nodes(mesh.boundary_faces, static_cast<std::size_t>(f));
        const std::size_t node_count = boundary_face_num_nodes(mesh.boundary_faces, static_cast<std::size_t>(f));
        for (std::size_t local = 0; local < node_count; ++local) {
            mesh.node_to_boundary_faces[nodes[local]].push_back(f);
        }
    }
}

// The squared distance from p to triangle abc (Ericson, Real-Time Collision
// Detection: the closest point by the region of p).
Real triangle_distance2(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const Real d1 = dot(ab, ap), d2 = dot(ac, ap);
    const auto sq = [&](const Vec3& q) { const Vec3 d = p - q; return dot(d, d); };
    if (d1 <= 0.0 && d2 <= 0.0) return sq(a);
    const Vec3 bp = p - b;
    const Real d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return sq(b);
    const Real vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return sq(a + (d1 / (d1 - d3)) * ab);
    const Vec3 cp = p - c;
    const Real d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return sq(c);
    const Real vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return sq(a + (d2 / (d2 - d6)) * ac);
    const Real va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
        return sq(b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b));
    const Real denom = 1.0 / (va + vb + vc);
    return sq(a + (vb * denom) * ab + (vc * denom) * ac);
}

Real stretched_coordinate(Index idx, Index count, Real length, Real growth) {
    const Real xi = static_cast<Real>(idx) / static_cast<Real>(count - 1);
    if (std::abs(growth - 1.0) < 1.0e-12) {
        return length * xi;
    }
    return length * (std::exp(std::log(growth) * xi) - 1.0) / (growth - 1.0);
}

Mesh build_structured_mesh(Index nx,
                           Index ny,
                           Index nz,
                           const MappingFunction& mapping,
                           const BoundaryTypeFunction& xmin_type,
                           const BoundaryTypeFunction& xmax_type,
                           const BoundaryTypeFunction& ymin_type,
                           const BoundaryTypeFunction& ymax_type,
                           const BoundaryTypeFunction& zmin_type,
                           const BoundaryTypeFunction& zmax_type,
                           const BoundaryNameFunction& xmin_name,
                           const BoundaryNameFunction& xmax_name,
                           const BoundaryNameFunction& ymin_name,
                           const BoundaryNameFunction& ymax_name,
                           const BoundaryNameFunction& zmin_name,
                           const BoundaryNameFunction& zmax_name,
                           bool periodic_i = false) {
    if (nx < 2 || ny < 2 || nz < 2) {
        throw std::invalid_argument("Structured mesh requires at least 2 nodes in each direction");
    }

    // Periodic in i (an O-grid): i = nx - 1 is i = 0 again, and there are no
    // xmin and xmax boundaries.
    const Index stride = periodic_i ? nx - 1 : nx;
    const auto nid = [=](Index i, Index j, Index k) {
        return node_id(periodic_i && i == nx - 1 ? 0 : i, j, k, stride, ny);
    };

    Mesh mesh;
    const auto node_count = static_cast<std::size_t>(stride) * ny * nz;
    mesh.nodes.count = node_count;
    mesh.nodes.x.resize(node_count);
    mesh.nodes.y.resize(node_count);
    mesh.nodes.z.resize(node_count);
    mesh.nodes.vol.assign(node_count, 0.0);
    mesh.nodes.wall_dist.assign(node_count, 0.0);

    for (Index k = 0; k < nz; ++k) {
        for (Index j = 0; j < ny; ++j) {
            for (Index i = 0; i < stride; ++i) {
                const Index id = nid(i, j, k);
                const Vec3 p = mapping(i, j, k);
                mesh.nodes.x[id] = p.x;
                mesh.nodes.y[id] = p.y;
                mesh.nodes.z[id] = p.z;
            }
        }
    }

    // Boundary faces point away from their own cell's centroid.
    const auto cell_centroid = [&](Index i, Index j, Index k) {
        Vec3 c {};
        for (Index dk = 0; dk < 2; ++dk)
            for (Index dj = 0; dj < 2; ++dj)
                for (Index di = 0; di < 2; ++di) c = c + 0.125 * mapping(i + di, j + dj, k + dk);
        return c;
    };

    for (Index k = 0; k < nz - 1; ++k) {
        for (Index j = 0; j < ny - 1; ++j) {
            for (Index i = 0; i < nx - 1; ++i) {
                const std::array<Index, 8> nodes = {
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                    nid(i, j, k + 1),
                    nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i, j + 1, k + 1),
                };
                std::array<Vec3, 8> p {};
                for (int n = 0; n < 8; ++n) {
                    p[n] = {mesh.nodes.x[nodes[n]], mesh.nodes.y[nodes[n]], mesh.nodes.z[nodes[n]]};
                }
                const Real cell_volume = hex_volume(p);
                for (Index n : nodes) {
                    mesh.nodes.vol[n] += cell_volume / 8.0;
                }
            }
        }
    }

    // The edges, each node's to its +i, +j and +k neighbours, with the area
    // vectors of the median dual: in each cell, each edge's dual face is the
    // quad (edge midpoint, centroid of one adjacent face, cell centroid,
    // centroid of the other), so every control volume closes.
    std::vector<Index> edge_from[3];
    for (auto& e : edge_from) e.assign(node_count, -1);
    for (Index k = 0; k < nz; ++k) {
        for (Index j = 0; j < ny; ++j) {
            for (Index i = 0; i < stride; ++i) {
                const Index left = nid(i, j, k);
                if (i + 1 < nx) {
                    edge_from[0][left] = static_cast<Index>(mesh.edges.node_L.size());
                    add_edge(mesh, left, nid(i + 1, j, k), {});
                }
                if (j + 1 < ny) {
                    edge_from[1][left] = static_cast<Index>(mesh.edges.node_L.size());
                    add_edge(mesh, left, nid(i, j + 1, k), {});
                }
                if (k + 1 < nz) {
                    edge_from[2][left] = static_cast<Index>(mesh.edges.node_L.size());
                    add_edge(mesh, left, nid(i, j, k + 1), {});
                }
            }
        }
    }

    // A hex's corners as (i, j, k) offsets, its faces, and its edges: the
    // corner they start from, their direction and their two faces.
    constexpr int corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    constexpr int face[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
                                {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 2, 6, 5}};
    constexpr int cell_edge[12][4] = {{0, 0, 0, 2}, {3, 0, 0, 3}, {4, 0, 1, 2}, {7, 0, 1, 3},
                                      {0, 1, 0, 4}, {1, 1, 0, 5}, {4, 1, 1, 4}, {5, 1, 1, 5},
                                      {0, 2, 2, 4}, {1, 2, 2, 5}, {3, 2, 3, 4}, {2, 2, 3, 5}};
    for (Index k = 0; k < nz - 1; ++k) {
        for (Index j = 0; j < ny - 1; ++j) {
            for (Index i = 0; i < nx - 1; ++i) {
                std::array<Vec3, 8> p {};
                Vec3 centroid_cell {};
                for (int n = 0; n < 8; ++n) {
                    p[n] = mapping(i + corner[n][0], j + corner[n][1], k + corner[n][2]);
                    centroid_cell = centroid_cell + 0.125 * p[n];
                }
                std::array<Vec3, 6> centroid_face {};
                for (int f = 0; f < 6; ++f) {
                    for (int n = 0; n < 4; ++n) centroid_face[f] = centroid_face[f] + 0.25 * p[face[f][n]];
                }
                for (const auto& e : cell_edge) {
                    const int a = e[0];
                    const int dir = e[1];
                    const int b_offset[3] = {corner[a][0] + (dir == 0), corner[a][1] + (dir == 1), corner[a][2] + (dir == 2)};
                    const Vec3 pa = p[a];
                    const Vec3 pb = mapping(i + b_offset[0], j + b_offset[1], k + b_offset[2]);
                    const Vec3 midpoint = 0.5 * (pa + pb);
                    Vec3 dual = quad_area_vector(midpoint, centroid_face[e[2]], centroid_cell, centroid_face[e[3]]);
                    if (dot(dual, pb - pa) < 0.0) dual = -1.0 * dual;
                    const Index edge = edge_from[dir][nid(i + corner[a][0], j + corner[a][1], k + corner[a][2])];
                    mesh.edges.nx[edge] += dual.x;
                    mesh.edges.ny[edge] += dual.y;
                    mesh.edges.nz[edge] += dual.z;
                }
            }
        }
    }
    for (std::size_t e = 0; e < mesh.edges.node_L.size(); ++e) {
        mesh.edges.area[e] = norm(Vec3 {mesh.edges.nx[e], mesh.edges.ny[e], mesh.edges.nz[e]});
    }

    for (Index j = 0; j < ny - 1 && !periodic_i; ++j) {
        for (Index k = 0; k < nz - 1; ++k) {
            const std::array<Index, 4> xmin_nodes = {
                nid(0, j, k), nid(0, j + 1, k), nid(0, j + 1, k + 1), nid(0, j, k + 1)};
            const std::array<Index, 4> xmax_nodes = {
                nid(nx - 1, j, k), nid(nx - 1, j, k + 1), nid(nx - 1, j + 1, k + 1), nid(nx - 1, j + 1, k)};
            const Vec3 xmin_normal = quad_area_vector(
                mapping(0, j, k), mapping(0, j + 1, k), mapping(0, j + 1, k + 1), mapping(0, j, k + 1));
            const Vec3 xmax_normal = quad_area_vector(
                mapping(nx - 1, j, k), mapping(nx - 1, j, k + 1), mapping(nx - 1, j + 1, k + 1), mapping(nx - 1, j + 1, k));
            const Vec3 xmin_center = 0.25 * (mapping(0, j, k) + mapping(0, j + 1, k) + mapping(0, j + 1, k + 1) + mapping(0, j, k + 1));
            const Vec3 xmax_center =
                0.25 * (mapping(nx - 1, j, k) + mapping(nx - 1, j, k + 1) + mapping(nx - 1, j + 1, k + 1) + mapping(nx - 1, j + 1, k));
            const Vec3 xmin_oriented = dot(xmin_normal, xmin_center - cell_centroid(0, j, k)) >= 0.0 ? xmin_normal : -1.0 * xmin_normal;
            const Vec3 xmax_oriented = dot(xmax_normal, xmax_center - cell_centroid(nx - 2, j, k)) >= 0.0 ? xmax_normal : -1.0 * xmax_normal;
            add_boundary_face(mesh, xmin_nodes[0], xmin_nodes[1], xmin_nodes[2], xmin_nodes[3], xmin_oriented,
                              xmin_type(j, k), xmin_name(j, k), 0);
            add_boundary_face(mesh, xmax_nodes[0], xmax_nodes[1], xmax_nodes[2], xmax_nodes[3], xmax_oriented,
                              xmax_type(j, k), xmax_name(j, k), 1);
        }
    }

    for (Index i = 0; i < nx - 1; ++i) {
        for (Index k = 0; k < nz - 1; ++k) {
            const std::array<Index, 4> ymin_nodes = {
                nid(i, 0, k), nid(i, 0, k + 1), nid(i + 1, 0, k + 1), nid(i + 1, 0, k)};
            const std::array<Index, 4> ymax_nodes = {
                nid(i, ny - 1, k), nid(i + 1, ny - 1, k), nid(i + 1, ny - 1, k + 1), nid(i, ny - 1, k + 1)};
            const Vec3 ymin_normal = quad_area_vector(
                mapping(i, 0, k), mapping(i, 0, k + 1), mapping(i + 1, 0, k + 1), mapping(i + 1, 0, k));
            const Vec3 ymax_normal = quad_area_vector(
                mapping(i, ny - 1, k), mapping(i + 1, ny - 1, k), mapping(i + 1, ny - 1, k + 1), mapping(i, ny - 1, k + 1));
            const Vec3 ymin_center = 0.25 * (mapping(i, 0, k) + mapping(i, 0, k + 1) + mapping(i + 1, 0, k + 1) + mapping(i + 1, 0, k));
            const Vec3 ymax_center =
                0.25 * (mapping(i, ny - 1, k) + mapping(i + 1, ny - 1, k) + mapping(i + 1, ny - 1, k + 1) + mapping(i, ny - 1, k + 1));
            const Vec3 ymin_oriented = dot(ymin_normal, ymin_center - cell_centroid(i, 0, k)) >= 0.0 ? ymin_normal : -1.0 * ymin_normal;
            const Vec3 ymax_oriented = dot(ymax_normal, ymax_center - cell_centroid(i, ny - 2, k)) >= 0.0 ? ymax_normal : -1.0 * ymax_normal;
            add_boundary_face(mesh, ymin_nodes[0], ymin_nodes[1], ymin_nodes[2], ymin_nodes[3], ymin_oriented,
                              ymin_type(i, k), ymin_name(i, k), 2);
            add_boundary_face(mesh, ymax_nodes[0], ymax_nodes[1], ymax_nodes[2], ymax_nodes[3], ymax_oriented,
                              ymax_type(i, k), ymax_name(i, k), 3);
        }
    }

    for (Index i = 0; i < nx - 1; ++i) {
        for (Index j = 0; j < ny - 1; ++j) {
            const std::array<Index, 4> zmin_nodes = {
                nid(i, j, 0), nid(i + 1, j, 0), nid(i + 1, j + 1, 0), nid(i, j + 1, 0)};
            const std::array<Index, 4> zmax_nodes = {
                nid(i, j, nz - 1), nid(i, j + 1, nz - 1), nid(i + 1, j + 1, nz - 1), nid(i + 1, j, nz - 1)};
            const Vec3 zmin_normal = quad_area_vector(
                mapping(i, j, 0), mapping(i + 1, j, 0), mapping(i + 1, j + 1, 0), mapping(i, j + 1, 0));
            const Vec3 zmax_normal = quad_area_vector(
                mapping(i, j, nz - 1), mapping(i, j + 1, nz - 1), mapping(i + 1, j + 1, nz - 1), mapping(i + 1, j, nz - 1));
            const Vec3 zmin_center = 0.25 * (mapping(i, j, 0) + mapping(i + 1, j, 0) + mapping(i + 1, j + 1, 0) + mapping(i, j + 1, 0));
            const Vec3 zmax_center =
                0.25 * (mapping(i, j, nz - 1) + mapping(i, j + 1, nz - 1) + mapping(i + 1, j + 1, nz - 1) + mapping(i + 1, j, nz - 1));
            const Vec3 zmin_oriented = dot(zmin_normal, zmin_center - cell_centroid(i, j, 0)) >= 0.0 ? zmin_normal : -1.0 * zmin_normal;
            const Vec3 zmax_oriented = dot(zmax_normal, zmax_center - cell_centroid(i, j, nz - 2)) >= 0.0 ? zmax_normal : -1.0 * zmax_normal;
            add_boundary_face(mesh, zmin_nodes[0], zmin_nodes[1], zmin_nodes[2], zmin_nodes[3], zmin_oriented,
                              zmin_type(i, j), zmin_name(i, j), 4);
            add_boundary_face(mesh, zmax_nodes[0], zmax_nodes[1], zmax_nodes[2], zmax_nodes[3], zmax_oriented,
                              zmax_type(i, j), zmax_name(i, j), 5);
        }
    }

    finalize_mesh(mesh);

    // Wall distance for the SA model: the distance to the no-slip faces of
    // the ymin boundary (the wing is carved into it), exact to the faces in a
    // window of columns around the node's own, which holds the nearest wall
    // point wherever the distance is small enough to matter; a node with none
    // in its window takes a sample of the wall's faces. (The loop this
    // replaces ran before the faces were counted and left every node at the
    // largest Real, which switched the SA source off.)
    {
        constexpr Index window = 3, coarse_step = 8;
        const Index fx = nx - 1, fz = nz - 1;
        std::vector<char> wall(static_cast<std::size_t>(fx) * fz, 0);
        for (Index k = 0; k < fz; ++k)
            for (Index i = 0; i < fx; ++i)
                wall[static_cast<std::size_t>(k) * fx + i] = ymin_type(i, k) == BoundaryType::NoSlipWall ? 1 : 0;

        const auto node_at = [&](Index i, Index j, Index k) {
            const std::size_t id = nid(i, j, k);
            return Vec3 {mesh.nodes.x[id], mesh.nodes.y[id], mesh.nodes.z[id]};
        };
        const auto face_distance2 = [&](const Vec3& p, Index i, Index k) {
            const Vec3 a = node_at(i, 0, k), b = node_at(i + 1, 0, k), c = node_at(i + 1, 0, k + 1),
                       d = node_at(i, 0, k + 1);
            return std::min(triangle_distance2(p, a, b, c), triangle_distance2(p, a, c, d));
        };

        // Few wall faces (a 2D grid's): every node to all of them.
        std::vector<std::pair<Index, Index>> wall_faces;
        for (Index k = 0; k < fz; ++k)
            for (Index i = 0; i < fx; ++i)
                if (wall[static_cast<std::size_t>(k) * fx + i]) wall_faces.emplace_back(i, k);
        const bool exhaustive = wall_faces.size() <= 4096;

        for (Index k = 0; k < nz; ++k) {
            for (Index j = 0; j < ny; ++j) {
                for (Index i = 0; i < nx; ++i) {
                    const Vec3 p = node_at(i, j, k);
                    Real best = std::numeric_limits<Real>::max();
                    if (exhaustive) {
                        for (const auto& [ii, kk] : wall_faces) best = std::min(best, face_distance2(p, ii, kk));
                        mesh.nodes.wall_dist[nid(i, j, k)] =
                            best == std::numeric_limits<Real>::max() ? 1.0e30 : std::sqrt(best);
                        continue;
                    }
                    const Index k0 = k > window ? k - window : 0, k1 = std::min(fz - 1, k + window);
                    const Index i0 = i > window ? i - window : 0, i1 = std::min(fx - 1, i + window);
                    for (Index kk = k0; kk <= k1; ++kk)
                        for (Index ii = i0; ii <= i1; ++ii)
                            if (wall[static_cast<std::size_t>(kk) * fx + ii])
                                best = std::min(best, face_distance2(p, ii, kk));
                    if (best == std::numeric_limits<Real>::max())
                        for (Index kk = 0; kk < fz; kk += coarse_step)
                            for (Index ii = 0; ii < fx; ii += coarse_step)
                                if (wall[static_cast<std::size_t>(kk) * fx + ii])
                                    best = std::min(best, face_distance2(p, ii, kk));
                    mesh.nodes.wall_dist[nid(i, j, k)] =
                        best == std::numeric_limits<Real>::max() ? 1.0e30 : std::sqrt(best);
                }
            }
        }
    }
    return mesh;
}


// A hexahedral cell's corners as (i, j, k) offsets, its faces, and its edges:
// the corner they start from, their direction and their two faces.
constexpr int hex_corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
constexpr int hex_face[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
                                {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 2, 6, 5}};
constexpr int hex_edge[12][4] = {{0, 0, 0, 2}, {3, 0, 0, 3}, {4, 0, 1, 2}, {7, 0, 1, 3},
                                 {0, 1, 0, 4}, {1, 1, 0, 5}, {4, 1, 1, 4}, {5, 1, 1, 5},
                                 {0, 2, 2, 4}, {1, 2, 2, 5}, {3, 2, 3, 4}, {2, 2, 3, 5}};

using FaceClassifier = std::function<std::pair<BoundaryType, std::string>(const Vec3&, const Vec3&)>;

// A mesh of hexahedra given by their corners (in hex_corner's order): nodal
// volumes, median-dual edges, and as boundary faces the faces only one cell
// has, pointing away from it, typed by classify(centroid, area vector). The
// wall distance is exact to every no-slip face.
Mesh build_hex_mesh(const std::vector<Vec3>& points, const std::vector<std::array<Index, 8>>& hexes,
                    const FaceClassifier& classify) {
    Mesh mesh;
    const std::size_t n = points.size();
    mesh.nodes.count = n;
    mesh.nodes.x.resize(n);
    mesh.nodes.y.resize(n);
    mesh.nodes.z.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        mesh.nodes.x[i] = points[i].x;
        mesh.nodes.y[i] = points[i].y;
        mesh.nodes.z[i] = points[i].z;
    }
    mesh.nodes.vol.assign(n, 0.0);
    mesh.nodes.wall_dist.assign(n, 1.0e30);

    int edge_end[12];
    for (int e = 0; e < 12; ++e) {
        int c[3] = {hex_corner[hex_edge[e][0]][0], hex_corner[hex_edge[e][0]][1], hex_corner[hex_edge[e][0]][2]};
        c[hex_edge[e][1]] += 1;
        for (int m = 0; m < 8; ++m)
            if (hex_corner[m][0] == c[0] && hex_corner[m][1] == c[1] && hex_corner[m][2] == c[2]) edge_end[e] = m;
    }

    std::unordered_map<std::uint64_t, Index> edge_of;
    edge_of.reserve(3 * hexes.size() + 16);
    struct FaceKeyHash {
        std::size_t operator()(const std::array<Index, 4>& k) const {
            std::uint64_t h = 1469598103934665603ULL;
            for (Index v : k) h = (h ^ static_cast<std::uint64_t>(v)) * 1099511628211ULL;
            return static_cast<std::size_t>(h);
        }
    };
    // Each face by its sorted nodes: its cell and face, or -1 once a second cell has it.
    std::unordered_map<std::array<Index, 4>, std::pair<long long, int>, FaceKeyHash> faces;
    faces.reserve(4 * hexes.size() + 16);

    for (std::size_t h = 0; h < hexes.size(); ++h) {
        std::array<Vec3, 8> p {};
        Vec3 cell {};
        for (int c = 0; c < 8; ++c) {
            p[c] = points[hexes[h][c]];
            cell = cell + 0.125 * p[c];
        }
        const Real volume = hex_volume(p);
        for (int c = 0; c < 8; ++c) mesh.nodes.vol[hexes[h][c]] += volume / 8.0;
        std::array<Vec3, 6> centroid_face {};
        for (int f = 0; f < 6; ++f)
            for (int c = 0; c < 4; ++c) centroid_face[f] = centroid_face[f] + 0.25 * p[hex_face[f][c]];
        for (int e = 0; e < 12; ++e) {
            const Index a = hexes[h][hex_edge[e][0]], b = hexes[h][edge_end[e]];
            const std::uint64_t key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) |
                                      static_cast<std::uint64_t>(std::max(a, b));
            auto [it, fresh] = edge_of.try_emplace(key, static_cast<Index>(mesh.edges.node_L.size()));
            if (fresh) add_edge(mesh, a, b, {});
            const Index edge = it->second;
            const Vec3 pl = points[mesh.edges.node_L[edge]], pr = points[mesh.edges.node_R[edge]];
            Vec3 dual = quad_area_vector(0.5 * (pl + pr), centroid_face[hex_edge[e][2]], cell,
                                         centroid_face[hex_edge[e][3]]);
            if (dot(dual, pr - pl) < 0.0) dual = -1.0 * dual;
            mesh.edges.nx[edge] += dual.x;
            mesh.edges.ny[edge] += dual.y;
            mesh.edges.nz[edge] += dual.z;
        }
        for (int f = 0; f < 6; ++f) {
            std::array<Index, 4> key {};
            for (int c = 0; c < 4; ++c) key[c] = hexes[h][hex_face[f][c]];
            std::sort(key.begin(), key.end());
            auto [it, fresh] = faces.try_emplace(key, static_cast<long long>(h), f);
            if (!fresh) it->second.first = -1;
        }
    }
    for (std::size_t e = 0; e < mesh.edges.node_L.size(); ++e)
        mesh.edges.area[e] = norm(Vec3 {mesh.edges.nx[e], mesh.edges.ny[e], mesh.edges.nz[e]});

    // Boundary faces in a fixed order (the map's is not).
    std::vector<std::pair<long long, int>> boundary;
    for (const auto& [key, owner] : faces)
        if (owner.first >= 0) boundary.push_back(owner);
    std::sort(boundary.begin(), boundary.end());
    std::map<std::string, Index> groups;
    for (const auto& [h, f] : boundary) {
        std::array<Index, 4> ids {};
        Vec3 cell {}, centre {};
        for (int c = 0; c < 8; ++c) cell = cell + 0.125 * points[hexes[h][c]];
        for (int c = 0; c < 4; ++c) {
            ids[c] = hexes[h][hex_face[f][c]];
            centre = centre + 0.25 * points[ids[c]];
        }
        Vec3 normal = quad_area_vector(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]);
        if (dot(normal, centre - cell) < 0.0) normal = -1.0 * normal;
        const auto [type, name] = classify(centre, normal);
        const Index group = groups.try_emplace(name, static_cast<Index>(groups.size())).first->second;
        add_boundary_face(mesh, ids[0], ids[1], ids[2], ids[3], normal, type, name, group);
    }
    finalize_mesh(mesh);

    std::vector<std::array<Vec3, 4>> walls;
    for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
        if (mesh.boundary_faces.type[f] != BoundaryType::NoSlipWall) continue;
        const auto ids = boundary_face_nodes(mesh.boundary_faces, f);
        walls.push_back({points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]});
    }
    for (std::size_t i = 0; i < n; ++i) {
        Real best = std::numeric_limits<Real>::max();
        for (const auto& w : walls)
            best = std::min({best, triangle_distance2(points[i], w[0], w[1], w[2]),
                             triangle_distance2(points[i], w[0], w[2], w[3])});
        if (!walls.empty()) mesh.nodes.wall_dist[i] = std::sqrt(best);
    }
    return mesh;
}

// Planar quad blocks (points (x, y) of block b at i + ni j) as one layer of
// hexahedra between z = 0 and lz: points within tol of each other are one
// (block interfaces, a C-grid's wake cut).
Mesh build_extruded_blocks(const std::vector<std::array<Index, 2>>& dims,
                           const std::vector<std::vector<std::array<Real, 2>>>& blocks, Real lz, Real tol,
                           const FaceClassifier& classify) {
    struct PairHash {
        std::size_t operator()(const std::pair<long long, long long>& k) const {
            return std::hash<long long>()(k.first) * 31 + std::hash<long long>()(k.second);
        }
    };
    std::unordered_map<std::pair<long long, long long>, Index, PairHash> merged;
    std::vector<std::array<Real, 2>> plane;
    std::vector<std::vector<Index>> ids(blocks.size());
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        ids[b].resize(blocks[b].size());
        for (std::size_t i = 0; i < blocks[b].size(); ++i) {
            const auto& q = blocks[b][i];
            const std::pair<long long, long long> key {std::llround(q[0] / tol), std::llround(q[1] / tol)};
            auto [it, fresh] = merged.try_emplace(key, static_cast<Index>(plane.size()));
            if (fresh) plane.push_back(q);
            ids[b][i] = it->second;
        }
    }
    const Index np = static_cast<Index>(plane.size());
    std::vector<Vec3> points(2 * static_cast<std::size_t>(np));
    for (Index i = 0; i < np; ++i) {
        points[i] = {plane[i][0], plane[i][1], 0.0};
        points[np + i] = {plane[i][0], plane[i][1], lz};
    }
    std::vector<std::array<Index, 8>> hexes;
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        const Index ni = dims[b][0], nj = dims[b][1];
        std::size_t positive = 0, negative = 0;
        for (Index j = 0; j + 1 < nj; ++j)
            for (Index i = 0; i + 1 < ni; ++i) {
                const Index q[4] = {ids[b][j * ni + i], ids[b][j * ni + i + 1], ids[b][(j + 1) * ni + i + 1],
                                    ids[b][(j + 1) * ni + i]};
                if (q[0] == q[1] || q[1] == q[2] || q[2] == q[3] || q[3] == q[0])
                    throw std::runtime_error("degenerate quad in block " + std::to_string(b));
                // Counter-clockwise in the plane, so that (i, j, k) is right-handed.
                const Real twice_area = (plane[q[1]][0] - plane[q[0]][0]) * (plane[q[3]][1] - plane[q[0]][1]) -
                                        (plane[q[1]][1] - plane[q[0]][1]) * (plane[q[3]][0] - plane[q[0]][0]);
                if (twice_area >= 0.0) {
                    ++positive;
                    hexes.push_back({q[0], q[1], q[2], q[3], np + q[0], np + q[1], np + q[2], np + q[3]});
                } else {
                    ++negative;
                    hexes.push_back({q[0], q[3], q[2], q[1], np + q[0], np + q[3], np + q[2], np + q[1]});
                }
            }
        // A block's cells all turn one way unless it folds over itself.
        if (positive && negative)
            throw std::runtime_error("block " + std::to_string(b) + " folds: " + std::to_string(std::min(positive, negative)) +
                                     " of its cells are turned over");
    }
    return build_hex_mesh(points, hexes, classify);
}

// An airfoil grid's boundaries: the z planes slip walls, faces whose centre
// lies in the box (x0, x1, |y| <= y) no-slip walls, every other one far field.
FaceClassifier airfoil_classifier(Real x0, Real x1, Real y, BoundaryType farfield) {
    return [=](const Vec3& centre, const Vec3& normal) -> std::pair<BoundaryType, std::string> {
        if (std::abs(normal.z) > 0.5 * norm(normal)) return {BoundaryType::SlipWall, "span"};
        if (centre.x >= x0 && centre.x <= x1 && std::abs(centre.y) <= y) return {BoundaryType::NoSlipWall, "wall"};
        return {farfield, "farfield"};
    };
}

}  // namespace

Mesh make_structured_box_mesh(const StructuredBoxSpec& spec) {
    if (spec.nx < 2 || spec.ny < 2 || spec.nz < 2) {
        throw std::invalid_argument("Structured box mesh requires at least 2 nodes in each direction");
    }

    Mesh mesh;
    const auto node_count = static_cast<std::size_t>(spec.nx) * spec.ny * spec.nz;
    mesh.nodes.count = node_count;
    mesh.nodes.x.resize(node_count);
    mesh.nodes.y.resize(node_count);
    mesh.nodes.z.resize(node_count);
    mesh.nodes.vol.assign(node_count, 0.0);
    mesh.nodes.wall_dist.assign(node_count, 0.0);

    const Real dx = spec.lx / static_cast<Real>(spec.nx - 1);
    const Real dy = spec.ly / static_cast<Real>(spec.ny - 1);
    const Real dz = spec.lz / static_cast<Real>(spec.nz - 1);

    for (Index k = 0; k < spec.nz; ++k) {
        for (Index j = 0; j < spec.ny; ++j) {
            for (Index i = 0; i < spec.nx; ++i) {
                const Index id = node_id(i, j, k, spec.nx, spec.ny);
                mesh.nodes.x[id] = dx * static_cast<Real>(i);
                mesh.nodes.y[id] = dy * static_cast<Real>(j);
                mesh.nodes.z[id] = dz * static_cast<Real>(k);

                const Real wall_x = std::min(mesh.nodes.x[id], spec.lx - mesh.nodes.x[id]);
                const Real wall_y = std::min(mesh.nodes.y[id], spec.ly - mesh.nodes.y[id]);
                const Real wall_z = std::min(mesh.nodes.z[id], spec.lz - mesh.nodes.z[id]);
                mesh.nodes.wall_dist[id] = std::min(wall_x, std::min(wall_y, wall_z));
            }
        }
    }

    const Real cell_volume = dx * dy * dz;
    for (Index k = 0; k < spec.nz - 1; ++k) {
        for (Index j = 0; j < spec.ny - 1; ++j) {
            for (Index i = 0; i < spec.nx - 1; ++i) {
                const std::array<Index, 8> nodes = {
                    node_id(i, j, k, spec.nx, spec.ny),
                    node_id(i + 1, j, k, spec.nx, spec.ny),
                    node_id(i + 1, j + 1, k, spec.nx, spec.ny),
                    node_id(i, j + 1, k, spec.nx, spec.ny),
                    node_id(i, j, k + 1, spec.nx, spec.ny),
                    node_id(i + 1, j, k + 1, spec.nx, spec.ny),
                    node_id(i + 1, j + 1, k + 1, spec.nx, spec.ny),
                    node_id(i, j + 1, k + 1, spec.nx, spec.ny),
                };
                for (Index n : nodes) {
                    mesh.nodes.vol[n] += cell_volume / 8.0;
                }
            }
        }
    }

    for (Index k = 0; k < spec.nz; ++k) {
        for (Index j = 0; j < spec.ny; ++j) {
            for (Index i = 0; i < spec.nx; ++i) {
                const Index left = node_id(i, j, k, spec.nx, spec.ny);
                if (i + 1 < spec.nx) {
                    const Real area = 0.25 * static_cast<Real>(adjacent_cell_count_for_x_edge(j, k, spec.ny, spec.nz)) * dy * dz;
                    add_edge(mesh, left, node_id(i + 1, j, k, spec.nx, spec.ny), {area, 0.0, 0.0});
                }
                if (j + 1 < spec.ny) {
                    const Real area = 0.25 * static_cast<Real>(adjacent_cell_count_for_y_edge(i, k, spec.nx, spec.nz)) * dx * dz;
                    add_edge(mesh, left, node_id(i, j + 1, k, spec.nx, spec.ny), {0.0, area, 0.0});
                }
                if (k + 1 < spec.nz) {
                    const Real area = 0.25 * static_cast<Real>(adjacent_cell_count_for_z_edge(i, j, spec.nx, spec.ny)) * dx * dy;
                    add_edge(mesh, left, node_id(i, j, k + 1, spec.nx, spec.ny), {0.0, 0.0, area});
                }
            }
        }
    }

    for (Index j = 0; j < spec.ny - 1; ++j) {
        for (Index k = 0; k < spec.nz - 1; ++k) {
            add_boundary_face(mesh,
                              node_id(0, j, k, spec.nx, spec.ny),
                              node_id(0, j + 1, k, spec.nx, spec.ny),
                              node_id(0, j + 1, k + 1, spec.nx, spec.ny),
                              node_id(0, j, k + 1, spec.nx, spec.ny),
                              {-dy * dz, 0.0, 0.0},
                              spec.xmin,
                              "xmin",
                              0);
            add_boundary_face(mesh,
                              node_id(spec.nx - 1, j, k, spec.nx, spec.ny),
                              node_id(spec.nx - 1, j, k + 1, spec.nx, spec.ny),
                              node_id(spec.nx - 1, j + 1, k + 1, spec.nx, spec.ny),
                              node_id(spec.nx - 1, j + 1, k, spec.nx, spec.ny),
                              {dy * dz, 0.0, 0.0},
                              spec.xmax,
                              "xmax",
                              1);
        }
    }

    for (Index i = 0; i < spec.nx - 1; ++i) {
        for (Index k = 0; k < spec.nz - 1; ++k) {
            add_boundary_face(mesh,
                              node_id(i, 0, k, spec.nx, spec.ny),
                              node_id(i, 0, k + 1, spec.nx, spec.ny),
                              node_id(i + 1, 0, k + 1, spec.nx, spec.ny),
                              node_id(i + 1, 0, k, spec.nx, spec.ny),
                              {0.0, -dx * dz, 0.0},
                              spec.ymin,
                              "ymin",
                              2);
            add_boundary_face(mesh,
                              node_id(i, spec.ny - 1, k, spec.nx, spec.ny),
                              node_id(i + 1, spec.ny - 1, k, spec.nx, spec.ny),
                              node_id(i + 1, spec.ny - 1, k + 1, spec.nx, spec.ny),
                              node_id(i, spec.ny - 1, k + 1, spec.nx, spec.ny),
                              {0.0, dx * dz, 0.0},
                              spec.ymax,
                              "ymax",
                              3);
        }
    }

    for (Index i = 0; i < spec.nx - 1; ++i) {
        for (Index j = 0; j < spec.ny - 1; ++j) {
            add_boundary_face(mesh,
                              node_id(i, j, 0, spec.nx, spec.ny),
                              node_id(i + 1, j, 0, spec.nx, spec.ny),
                              node_id(i + 1, j + 1, 0, spec.nx, spec.ny),
                              node_id(i, j + 1, 0, spec.nx, spec.ny),
                              {0.0, 0.0, -dx * dy},
                              spec.zmin,
                              "zmin",
                              4);
            add_boundary_face(mesh,
                              node_id(i, j, spec.nz - 1, spec.nx, spec.ny),
                              node_id(i, j + 1, spec.nz - 1, spec.nx, spec.ny),
                              node_id(i + 1, j + 1, spec.nz - 1, spec.nx, spec.ny),
                              node_id(i + 1, j, spec.nz - 1, spec.nx, spec.ny),
                              {0.0, 0.0, dx * dy},
                              spec.zmax,
                              "zmax",
                              5);
        }
    }

    finalize_mesh(mesh);
    return mesh;
}

Mesh make_bump_channel_mesh(const BumpChannelSpec& spec) {
    const Real dz = spec.lz / static_cast<Real>(spec.nz - 1);
    const auto bump = [=](Real x) {
        const Real xi = (x - spec.bump_center) / spec.bump_half_width;
        if (std::abs(xi) >= 1.0) {
            return 0.0;
        }
        const Real shape = 1.0 - xi * xi;
        return spec.bump_height * shape * shape;
    };
    const auto mapping = [=](Index i, Index j, Index k) {
        const Real x = spec.lx * static_cast<Real>(i) / static_cast<Real>(spec.nx - 1);
        const Real y_bottom = bump(x);
        const Real y_top = spec.ly;
        const Real eta = static_cast<Real>(j) / static_cast<Real>(spec.ny - 1);
        const Real y = y_bottom + eta * (y_top - y_bottom);
        return Vec3 {x, y, dz * k};
    };
    const auto constant_name = [](const std::string& value) { return [=](Index, Index) { return value; }; };
    const auto constant_type = [](BoundaryType value) { return [=](Index, Index) { return value; }; };

    return build_structured_mesh(spec.nx, spec.ny, spec.nz, mapping,
                                 constant_type(spec.inlet), constant_type(spec.outlet),
                                 constant_type(spec.lower_wall), constant_type(spec.upper_wall),
                                 constant_type(BoundaryType::SlipWall), constant_type(BoundaryType::SlipWall),
                                 constant_name("inlet"), constant_name("outlet"),
                                 constant_name("lower_wall"), constant_name("upper_wall"),
                                 constant_name("zmin"), constant_name("zmax"));
}

Mesh make_flat_plate_mesh(const FlatPlateSpec& spec) {
    const Real dz = spec.lz / static_cast<Real>(spec.nz - 1);
    const auto mapping = [=](Index i, Index j, Index k) {
        const Real x = spec.lx * static_cast<Real>(i) / static_cast<Real>(spec.nx - 1);
        const Real y = stretched_coordinate(j, spec.ny, spec.ly, spec.wall_normal_growth);
        return Vec3 {x, y, dz * k};
    };
    const auto constant_name = [](const std::string& value) { return [=](Index, Index) { return value; }; };
    const auto constant_type = [](BoundaryType value) { return [=](Index, Index) { return value; }; };
    const auto lower_type = [=](Index i, Index) {
        const Real x_mid = spec.lx * (static_cast<Real>(i) + 0.5) / static_cast<Real>(spec.nx - 1);
        return x_mid >= spec.leading_edge_x ? spec.plate_type : spec.lower_upstream_type;
    };
    const auto lower_name = [=](Index i, Index) {
        const Real x_mid = spec.lx * (static_cast<Real>(i) + 0.5) / static_cast<Real>(spec.nx - 1);
        return x_mid >= spec.leading_edge_x ? std::string("plate") : std::string("lower_upstream");
    };

    return build_structured_mesh(spec.nx, spec.ny, spec.nz, mapping,
                                 constant_type(spec.inlet), constant_type(spec.outlet),
                                 lower_type, constant_type(spec.upper_type),
                                 constant_type(BoundaryType::SlipWall), constant_type(BoundaryType::SlipWall),
                                 constant_name("inlet"), constant_name("outlet"),
                                 lower_name, constant_name("upper"),
                                 constant_name("zmin"), constant_name("zmax"));
}

Mesh make_bump_3d_mesh(const Bump3DSpec& spec) {
    const auto bump = [=](Real x, Real z) {
        const Real xi_x = (x - spec.bump_center_x) / spec.bump_half_width_x;
        const Real xi_z = (z - spec.bump_center_z) / spec.bump_half_width_z;
        if (std::abs(xi_x) >= 1.0 || std::abs(xi_z) >= 1.0) {
            return 0.0;
        }
        const Real shape_x = 1.0 - xi_x * xi_x;
        const Real shape_z = 1.0 - xi_z * xi_z;
        return spec.bump_height * shape_x * shape_x * shape_z * shape_z;
    };
    const auto mapping = [=](Index i, Index j, Index k) {
        const Real x = spec.lx * static_cast<Real>(i) / static_cast<Real>(spec.nx - 1);
        const Real z = spec.lz * static_cast<Real>(k) / static_cast<Real>(spec.nz - 1);
        const Real y_bottom = bump(x, z);
        const Real eta = static_cast<Real>(j) / static_cast<Real>(spec.ny - 1);
        const Real y = y_bottom + eta * (spec.ly - y_bottom);
        return Vec3 {x, y, z};
    };
    const auto constant_name = [](const std::string& value) { return [=](Index, Index) { return value; }; };
    const auto constant_type = [](BoundaryType value) { return [=](Index, Index) { return value; }; };

    return build_structured_mesh(spec.nx, spec.ny, spec.nz, mapping,
                                 constant_type(spec.inlet), constant_type(spec.outlet),
                                 constant_type(spec.lower_wall), constant_type(spec.upper_wall),
                                 constant_type(spec.side_wall), constant_type(spec.side_wall),
                                 constant_name("inlet"), constant_name("outlet"),
                                 constant_name("lower_wall"), constant_name("upper_wall"),
                                 constant_name("side_min"), constant_name("side_max"));
}

Mesh make_axisymmetric_body_mesh(const AxisymmetricBodySpec& spec) {
    const Real theta_extent = spec.wedge_angle_degrees * std::acos(-1.0) / 180.0;
    const auto body_radius = [=](Real x) {
        const Real x0 = spec.body_start_x;
        const Real x1 = spec.body_start_x + spec.body_length;
        if (x <= x0 || x >= x1) {
            return spec.body_tail_radius;
        }
        const Real s = (x - x0) / spec.body_length;
        const Real sears_haack_like = std::pow(std::max(4.0 * s * (1.0 - s), 0.0), 0.75);
        return spec.body_tail_radius + (spec.body_radius_max - spec.body_tail_radius) * sears_haack_like;
    };
    const auto mapping = [=](Index i, Index j, Index k) {
        const Real x = spec.lx * static_cast<Real>(i) / static_cast<Real>(spec.nx - 1);
        const Real r_inner = body_radius(x);
        const Real eta = stretched_coordinate(j, spec.nr, 1.0, spec.radial_growth);
        const Real r = r_inner + eta * (spec.farfield_radius - r_inner);
        const Real theta = -0.5 * theta_extent + theta_extent * static_cast<Real>(k) / static_cast<Real>(spec.ntheta - 1);
        return Vec3 {x, r * std::cos(theta), r * std::sin(theta)};
    };
    const auto constant_name = [](const std::string& value) { return [=](Index, Index) { return value; }; };
    const auto constant_type = [](BoundaryType value) { return [=](Index, Index) { return value; }; };

    return build_structured_mesh(spec.nx, spec.nr, spec.ntheta, mapping,
                                 constant_type(spec.inlet), constant_type(spec.outlet),
                                 constant_type(spec.body_wall), constant_type(spec.farfield),
                                 constant_type(spec.wedge_wall), constant_type(spec.wedge_wall),
                                 constant_name("inlet"), constant_name("outlet"),
                                 constant_name("body"), constant_name("farfield"),
                                 constant_name("wedge_min"), constant_name("wedge_max"));
}

Mesh make_naca_finite_wing_mesh(const NacaWingSpec& spec) {
    const auto thickness = [=](Real x_over_c) {
        if (x_over_c <= 0.0 || x_over_c >= 1.0) {
            return 0.0;
        }
        const Real xc = std::max(x_over_c, 1.0e-8);
        const Real yt = 5.0 * spec.thickness_ratio *
                        (0.2969 * std::sqrt(xc) - 0.1260 * xc - 0.3516 * xc * xc +
                         0.2843 * xc * xc * xc - 0.1015 * xc * xc * xc * xc);
        return std::max(yt, 0.0);
    };
    const auto camber = [=](Real x_over_c) {
        if (spec.camber_ratio <= 0.0) {
            return 0.0;
        }
        const Real p = std::clamp(spec.camber_position, 1.0e-6, 1.0 - 1.0e-6);
        const Real m = spec.camber_ratio;
        if (x_over_c < p) {
            return m * (2.0 * p * x_over_c - x_over_c * x_over_c) / (p * p);
        }
        return m * ((1.0 - 2.0 * p) + 2.0 * p * x_over_c - x_over_c * x_over_c) / ((1.0 - p) * (1.0 - p));
    };
    const auto wing_surface = [=](Real x, Real z) {
        const Real half_span = 0.5 * spec.span;
        const Real z_rel = z - spec.wing_center_z;
        if (std::abs(z_rel) > half_span) {
            return 0.0;
        }
        const Real span_eta = std::abs(z_rel) / half_span;
        const Real chord = spec.root_chord + (spec.tip_chord - spec.root_chord) * span_eta;
        const Real x_le = spec.wing_origin_x + std::abs(z_rel) * std::tan(spec.sweep_degrees * std::acos(-1.0) / 180.0);
        const Real xi = (x - x_le) / chord;
        if (xi <= 0.0 || xi >= 1.0) {
            return 0.0;
        }
        return spec.vertical_offset + chord * (camber(xi) + thickness(xi));
    };
    const auto mapping = [=](Index i, Index j, Index k) {
        const Real x = spec.lx * static_cast<Real>(i) / static_cast<Real>(spec.nx - 1);
        const Real z = spec.lz * static_cast<Real>(k) / static_cast<Real>(spec.nz - 1);
        const Real y_lower = wing_surface(x, z);
        const Real eta = stretched_coordinate(j, spec.ny, 1.0, spec.wall_normal_growth);
        const Real y = y_lower + eta * (spec.ly - y_lower);
        return Vec3 {x, y, z};
    };
    const auto lower_type = [=](Index i, Index k) {
        const Real x_mid = spec.lx * (static_cast<Real>(i) + 0.5) / static_cast<Real>(spec.nx - 1);
        const Real z_mid = spec.lz * (static_cast<Real>(k) + 0.5) / static_cast<Real>(spec.nz - 1);
        return wing_surface(x_mid, z_mid) > spec.vertical_offset + 1.0e-10 ? spec.wing_wall : spec.lower_farfield;
    };
    const auto lower_name = [=](Index i, Index k) {
        const Real x_mid = spec.lx * (static_cast<Real>(i) + 0.5) / static_cast<Real>(spec.nx - 1);
        const Real z_mid = spec.lz * (static_cast<Real>(k) + 0.5) / static_cast<Real>(spec.nz - 1);
        return wing_surface(x_mid, z_mid) > spec.vertical_offset + 1.0e-10 ? std::string("wing") : std::string("lower_farfield");
    };
    const auto constant_name = [](const std::string& value) { return [=](Index, Index) { return value; }; };
    const auto constant_type = [](BoundaryType value) { return [=](Index, Index) { return value; }; };

    return build_structured_mesh(spec.nx, spec.ny, spec.nz, mapping,
                                 constant_type(spec.inlet), constant_type(spec.outlet),
                                 lower_type, constant_type(spec.upper_farfield),
                                 constant_type(spec.side_farfield), constant_type(spec.side_farfield),
                                 constant_name("inlet"), constant_name("outlet"),
                                 lower_name, constant_name("upper_farfield"),
                                 constant_name("side_min"), constant_name("side_max"));
}

Mesh make_plot3d_2d_mesh(const Plot3D2DSpec& spec) {
    std::ifstream in(spec.path);
    if (!in) throw std::runtime_error("cannot open PLOT3D grid: " + spec.path);
    Index blocks = 0, ni = 0, nj = 0;
    in >> blocks >> ni >> nj;
    if (!in || blocks != 1 || ni < 2 || nj < 2)
        throw std::runtime_error("expected a formatted 2D PLOT3D grid of one block: " + spec.path);
    const std::size_t count = static_cast<std::size_t>(ni) * nj;
    std::vector<Real> x(count), y(count);
    for (auto& v : x) in >> v;
    for (auto& v : y) in >> v;
    if (!in) throw std::runtime_error("truncated PLOT3D grid: " + spec.path);

    const auto mapping = [&](Index i, Index j, Index k) {
        const std::size_t id = static_cast<std::size_t>(j) * ni + i;
        return Vec3 {x[id], y[id], spec.lz * static_cast<Real>(k)};
    };
    const auto on_wall = [&](Index i) {
        const Real x_mid = 0.5 * (x[i] + x[i + 1]);
        return x_mid >= spec.wall_x0 && x_mid <= spec.wall_x1;
    };
    const auto lower_type = [=](Index i, Index) { return on_wall(i) ? BoundaryType::NoSlipWall : spec.lower_other; };
    const auto lower_name = [=](Index i, Index) { return on_wall(i) ? std::string("wall") : std::string("lower"); };
    const auto constant_name = [](const std::string& value) { return [=](Index, Index) { return value; }; };
    const auto constant_type = [](BoundaryType value) { return [=](Index, Index) { return value; }; };

    return build_structured_mesh(ni, nj, 2, mapping,
                                 constant_type(spec.inlet), constant_type(spec.outlet),
                                 lower_type, constant_type(spec.upper),
                                 constant_type(BoundaryType::SlipWall), constant_type(BoundaryType::SlipWall),
                                 constant_name("inlet"), constant_name("outlet"),
                                 lower_name, constant_name("upper"),
                                 constant_name("zmin"), constant_name("zmax"));
}

Mesh make_cylinder_mesh(const CylinderSpec& spec) {
    // Radii from the wall's spacing growing geometrically to the far field.
    const Real span = spec.farfield_radius - spec.radius;
    const auto extent = [&](Real g) {
        return std::abs(g - 1.0) < 1.0e-12 ? spec.first_spacing * (spec.nr - 1)
                                           : spec.first_spacing * (std::pow(g, spec.nr - 1) - 1.0) / (g - 1.0);
    };
    if (extent(1.0) > span) throw std::invalid_argument("cylinder: first_spacing too large for nr");
    Real lo = 1.0, hi = 2.0;
    while (extent(hi) < span) hi *= 2.0;
    for (int it = 0; it < 200; ++it) {
        const Real mid = 0.5 * (lo + hi);
        (extent(mid) < span ? lo : hi) = mid;
    }
    const Real growth = 0.5 * (lo + hi);
    const Real two_pi = 2.0 * std::acos(-1.0);
    const Real dz = spec.lz / static_cast<Real>(spec.nz - 1);
    // Clockwise in i, so that (i, j, k) is right-handed.
    const auto mapping = [=](Index i, Index j, Index k) {
        const Real theta = -two_pi * static_cast<Real>(i) / static_cast<Real>(spec.ntheta);
        const Real r = spec.radius + spec.first_spacing * (std::pow(growth, j) - 1.0) / (growth - 1.0);
        return Vec3 {r * std::cos(theta), r * std::sin(theta), dz * k};
    };
    const auto constant_name = [](const std::string& value) { return [=](Index, Index) { return value; }; };
    const auto constant_type = [](BoundaryType value) { return [=](Index, Index) { return value; }; };
    return build_structured_mesh(spec.ntheta + 1, spec.nr, spec.nz, mapping,
                                 constant_type(BoundaryType::SlipWall), constant_type(BoundaryType::SlipWall),
                                 constant_type(spec.wall), constant_type(spec.farfield),
                                 constant_type(BoundaryType::SlipWall), constant_type(BoundaryType::SlipWall),
                                 constant_name("none"), constant_name("none"),
                                 constant_name("wall"), constant_name("farfield"),
                                 constant_name("zmin"), constant_name("zmax"), true);
}

Mesh make_plot3d_blocks_mesh(const Plot3DBlocksSpec& spec) {
    // Fortran unformatted (sequential records, little-endian): the block
    // count, the dimensions (2 or 3 per block), then each block's x, y (and z)
    // in double or single precision. A 3D block must be one plane (nk = 1).
    std::ifstream in(spec.path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open PLOT3D grid: " + spec.path);
    const auto record = [&]() {
        std::int32_t head = 0, tail = 0;
        in.read(reinterpret_cast<char*>(&head), 4);
        std::vector<char> data(static_cast<std::size_t>(head));
        in.read(data.data(), head);
        in.read(reinterpret_cast<char*>(&tail), 4);
        if (!in || head != tail) throw std::runtime_error("bad Fortran record in " + spec.path);
        return data;
    };
    const auto count = record();
    std::int32_t nblocks = 0;
    std::memcpy(&nblocks, count.data(), 4);
    const auto dim_record = record();
    const int ndim = static_cast<int>(dim_record.size() / (4 * static_cast<std::size_t>(nblocks)));
    if (ndim != 2 && ndim != 3) throw std::runtime_error("PLOT3D dimensions record of unexpected size");
    std::vector<std::array<Index, 2>> dims(nblocks);
    std::vector<std::vector<std::array<Real, 2>>> blocks(nblocks);
    for (int b = 0; b < nblocks; ++b) {
        std::int32_t d[3] = {1, 1, 1};
        std::memcpy(d, dim_record.data() + 4 * ndim * b, 4 * ndim);
        if (ndim == 3 && d[2] != 1) throw std::runtime_error("PLOT3D block is not a single plane");
        const std::size_t np = static_cast<std::size_t>(d[0]) * d[1];
        const auto data = record();
        const bool single = data.size() == 4 * ndim * np;
        if (!single && data.size() != 8 * ndim * np) throw std::runtime_error("PLOT3D block record of unexpected size");
        const auto value = [&](std::size_t k) {
            if (single) {
                float f;
                std::memcpy(&f, data.data() + 4 * k, 4);
                return static_cast<Real>(f);
            }
            double v;
            std::memcpy(&v, data.data() + 8 * k, 8);
            return static_cast<Real>(v);
        };
        // Every stride-th point in i and j (and the last): the strides must
        // keep the points of lines blocks share, which the merge checks.
        const Index si = b < static_cast<int>(spec.strides.size()) ? spec.strides[b][0] : 1;
        const Index sj = b < static_cast<int>(spec.strides.size()) ? spec.strides[b][1] : 1;
        const auto picks = [](Index n, Index s) {
            std::vector<Index> k;
            for (Index i = 0; i < n; i += s) k.push_back(i);
            if (k.back() != n - 1) k.push_back(n - 1);
            return k;
        };
        const auto pi = picks(d[0], si), pj = picks(d[1], sj);
        dims[b] = {static_cast<Index>(pi.size()), static_cast<Index>(pj.size())};
        for (Index j : pj)
            for (Index i : pi) {
                const std::size_t k = static_cast<std::size_t>(j) * d[0] + i;
                blocks[b].push_back({spec.scale * value(k), spec.scale * value(np + k)});
            }
    }
    Mesh mesh = build_extruded_blocks(dims, blocks, spec.lz, spec.merge_tolerance,
                                      airfoil_classifier(spec.wall_x0, spec.wall_x1, spec.wall_y, spec.farfield));
    // Lines blocks share that did not merge (strides that miss each other's
    // points) leave far-field faces inside the domain.
    Real outer = 0.0, inner = std::numeric_limits<Real>::max();
    for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
        if (mesh.boundary_faces.type[f] != spec.farfield) continue;
        const auto ids = boundary_face_nodes(mesh.boundary_faces, f);
        const Real r = std::hypot(mesh.nodes.x[ids[0]] - 0.5 * (spec.wall_x0 + spec.wall_x1), mesh.nodes.y[ids[0]]);
        outer = std::max(outer, r);
        inner = std::min(inner, r);
    }
    if (inner < 0.25 * outer) throw std::runtime_error("block interfaces did not merge (far-field faces inside the domain)");
    return mesh;
}

Mesh make_circular_arc_mesh(const CircularArcSpec& spec) {
    // The arcs of a biconvex section of unit chord: y = +-(sqrt(r^2 - (x -
    // 1/2)^2) - (r - h)), h half the thickness.
    const Real h = 0.5 * spec.thickness;
    const Real r = (0.25 + h * h) / (2.0 * h);
    const Real pi = std::acos(-1.0);
    const auto surface = [&](Real x) { return std::sqrt(std::max(r * r - (x - 0.5) * (x - 0.5), 0.0)) - (r - h); };

    // An O-grid (a C-grid's wake cut makes the sharp trailing edge a concave
    // corner, where the wall's normals cross): the section clockwise from the
    // trailing edge along the lower surface and back along the upper, points
    // clustered at both edges; the last point is the first again.
    const Index ns = spec.surface_points;
    std::vector<Real> xs(ns);
    for (Index i = 0; i < ns; ++i) xs[i] = 0.5 * (1.0 + std::cos(pi * static_cast<Real>(i) / (ns - 1)));
    std::vector<std::array<Real, 2>> inner;
    for (Index i = 0; i < ns; ++i) inner.push_back({xs[i], -surface(xs[i])});
    for (Index i = ns - 2; i >= 0; --i) inner.push_back({xs[i], surface(xs[i])});
    const Index ni = static_cast<Index>(inner.size());
    const Index nc = ni - 1;  // distinct points around

    // Off the section: its normal (left of the clockwise line, into the flow)
    // and the far-field circle about mid-chord at equal angles per line from
    // the trailing edge's direction, so that the lines the clustered edges
    // send out cover the flow ahead of and behind them.
    std::vector<std::array<Real, 2>> normal(nc), outer(nc);
    for (Index i = 0; i < nc; ++i) {
        const auto& a = inner[(i + nc - 1) % nc];
        const auto& b = inner[(i + 1) % nc];
        const Real tx = b[0] - a[0], ty = b[1] - a[1];
        const Real len = std::hypot(tx, ty);
        normal[i] = {-ty / len, tx / len};
        const Real theta = -2.0 * pi * static_cast<Real>(i) / static_cast<Real>(nc);
        outer[i] = {0.5 + spec.farfield_radius * std::cos(theta), spec.farfield_radius * std::sin(theta)};
    }
    const Real big_r = spec.farfield_radius;

    // Lines from the section to the circle, the first spacing off the wall's,
    // growing geometrically: normal to the wall within about blend_length of
    // it, straight to the outer point beyond.
    const Index nj = spec.normal_points;
    std::vector<Real> eta(nj);
    {
        const Real total = big_r;
        const auto extent = [&](Real g) { return spec.first_spacing * (std::pow(g, nj - 1) - 1.0) / (g - 1.0); };
        Real lo = 1.0 + 1.0e-9, hi = 2.0;
        while (extent(hi) < total) hi *= 2.0;
        for (int it = 0; it < 200; ++it) {
            const Real mid = 0.5 * (lo + hi);
            (extent(mid) < total ? lo : hi) = mid;
        }
        const Real g = 0.5 * (lo + hi);
        for (Index j = 0; j < nj; ++j) eta[j] = spec.first_spacing * (std::pow(g, j) - 1.0) / (g - 1.0) / total;
        eta[nj - 1] = 1.0;
    }
    std::vector<std::array<Real, 2>> block(static_cast<std::size_t>(ni) * nj);
    for (Index j = 0; j < nj; ++j)
        for (Index i = 0; i < ni; ++i) {
            const Index c = i % nc;
            const Real lx = outer[c][0] - inner[c][0], ly = outer[c][1] - inner[c][1];
            const Real d = eta[j] * std::hypot(lx, ly);
            const Real w = 1.0 - std::exp(-d / spec.blend_length);
            block[static_cast<std::size_t>(j) * ni + i] = {
                inner[c][0] + (1.0 - w) * d * normal[c][0] + w * eta[j] * lx,
                inner[c][1] + (1.0 - w) * d * normal[c][1] + w * eta[j] * ly};
        }
    return build_extruded_blocks({{ni, nj}}, {block}, spec.lz, 1.0e-12,
                                 airfoil_classifier(-0.01, 1.01, 0.5 * spec.thickness + 0.01, spec.farfield));
}

Real total_nodal_volume(const Mesh& mesh) {
    Real total = 0.0;
    for (Real v : mesh.nodes.vol) {
        total += v;
    }
    return total;
}

std::vector<Real> boundary_node_normals(const Mesh& mesh) {
    const auto& faces = mesh.boundary_faces;
    std::vector<Real> shares(12 * faces.count, 0.0);
    for (std::size_t f = 0; f < faces.count; ++f) {
        const auto nodes = boundary_face_nodes(faces, f);
        const std::size_t count = boundary_face_num_nodes(faces, f);
        std::array<Vec3, 4> p {};
        Vec3 centroid {};
        for (std::size_t n = 0; n < count; ++n) {
            p[n] = {mesh.nodes.x[nodes[n]], mesh.nodes.y[nodes[n]], mesh.nodes.z[nodes[n]]};
            centroid = centroid + (1.0 / static_cast<Real>(count)) * p[n];
        }
        const Vec3 normal {faces.nx[f], faces.ny[f], faces.nz[f]};
        for (std::size_t n = 0; n < count; ++n) {
            const Vec3 next = 0.5 * (p[n] + p[(n + 1) % count]);
            const Vec3 prev = 0.5 * (p[n] + p[(n + count - 1) % count]);
            Vec3 share = quad_area_vector(p[n], next, centroid, prev);
            if (dot(share, normal) < 0.0) share = -1.0 * share;
            shares[12 * f + 3 * n + 0] = share.x;
            shares[12 * f + 3 * n + 1] = share.y;
            shares[12 * f + 3 * n + 2] = share.z;
        }
    }
    return shares;
}

}  // namespace nssolver
