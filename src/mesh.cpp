#include "nssolver/mesh.hpp"

#include <algorithm>
#include <cmath>
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
                       Index group_id) {
    mesh.boundary_faces.n1.push_back(n1);
    mesh.boundary_faces.n2.push_back(n2);
    mesh.boundary_faces.n3.push_back(n3);
    mesh.boundary_faces.n4.push_back(n4);
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
        const std::array<Index, 4> nodes = {
            mesh.boundary_faces.n1[f], mesh.boundary_faces.n2[f], mesh.boundary_faces.n3[f], mesh.boundary_faces.n4[f]};
        for (Index n : nodes) {
            mesh.node_to_boundary_faces[n].push_back(f);
        }
    }
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
                           const BoundaryNameFunction& zmax_name) {
    if (nx < 2 || ny < 2 || nz < 2) {
        throw std::invalid_argument("Structured mesh requires at least 2 nodes in each direction");
    }

    Mesh mesh;
    const auto node_count = static_cast<std::size_t>(nx) * ny * nz;
    mesh.nodes.count = node_count;
    mesh.nodes.x.resize(node_count);
    mesh.nodes.y.resize(node_count);
    mesh.nodes.z.resize(node_count);
    mesh.nodes.vol.assign(node_count, 0.0);
    mesh.nodes.wall_dist.assign(node_count, 0.0);

    for (Index k = 0; k < nz; ++k) {
        for (Index j = 0; j < ny; ++j) {
            for (Index i = 0; i < nx; ++i) {
                const Index id = node_id(i, j, k, nx, ny);
                const Vec3 p = mapping(i, j, k);
                mesh.nodes.x[id] = p.x;
                mesh.nodes.y[id] = p.y;
                mesh.nodes.z[id] = p.z;
            }
        }
    }

    Vec3 centroid {};
    for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
        centroid.x += mesh.nodes.x[i];
        centroid.y += mesh.nodes.y[i];
        centroid.z += mesh.nodes.z[i];
    }
    centroid = centroid / static_cast<Real>(mesh.nodes.count);

    for (Index k = 0; k < nz - 1; ++k) {
        for (Index j = 0; j < ny - 1; ++j) {
            for (Index i = 0; i < nx - 1; ++i) {
                const std::array<Index, 8> nodes = {
                    node_id(i, j, k, nx, ny),
                    node_id(i + 1, j, k, nx, ny),
                    node_id(i + 1, j + 1, k, nx, ny),
                    node_id(i, j + 1, k, nx, ny),
                    node_id(i, j, k + 1, nx, ny),
                    node_id(i + 1, j, k + 1, nx, ny),
                    node_id(i + 1, j + 1, k + 1, nx, ny),
                    node_id(i, j + 1, k + 1, nx, ny),
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

    for (Index k = 0; k < nz; ++k) {
        for (Index j = 0; j < ny; ++j) {
            for (Index i = 0; i < nx; ++i) {
                const Index left = node_id(i, j, k, nx, ny);
                if (i + 1 < nx) {
                    const Index right = node_id(i + 1, j, k, nx, ny);
                    const Real weight = 0.25 * static_cast<Real>(adjacent_cell_count_for_x_edge(j, k, ny, nz));
                    const Vec3 span_y = mapping(i, std::min(j + 1, ny - 1), k) - mapping(i, std::max(j - 1, 0), k);
                    const Vec3 span_z = mapping(i, j, std::min(k + 1, nz - 1)) - mapping(i, j, std::max(k - 1, 0));
                    add_edge(mesh, left, right, weight * 0.25 * cross(span_y, span_z));
                }
                if (j + 1 < ny) {
                    const Index right = node_id(i, j + 1, k, nx, ny);
                    const Real weight = 0.25 * static_cast<Real>(adjacent_cell_count_for_y_edge(i, k, nx, nz));
                    const Vec3 span_z = mapping(i, j, std::min(k + 1, nz - 1)) - mapping(i, j, std::max(k - 1, 0));
                    const Vec3 span_x = mapping(std::min(i + 1, nx - 1), j, k) - mapping(std::max(i - 1, 0), j, k);
                    add_edge(mesh, left, right, weight * 0.25 * cross(span_z, span_x));
                }
                if (k + 1 < nz) {
                    const Index right = node_id(i, j, k + 1, nx, ny);
                    const Real weight = 0.25 * static_cast<Real>(adjacent_cell_count_for_z_edge(i, j, nx, ny));
                    const Vec3 span_x = mapping(std::min(i + 1, nx - 1), j, k) - mapping(std::max(i - 1, 0), j, k);
                    const Vec3 span_y = mapping(i, std::min(j + 1, ny - 1), k) - mapping(i, std::max(j - 1, 0), k);
                    add_edge(mesh, left, right, weight * 0.25 * cross(span_x, span_y));
                }
            }
        }
    }

    for (Index j = 0; j < ny - 1; ++j) {
        for (Index k = 0; k < nz - 1; ++k) {
            const std::array<Index, 4> xmin_nodes = {
                node_id(0, j, k, nx, ny), node_id(0, j + 1, k, nx, ny), node_id(0, j + 1, k + 1, nx, ny), node_id(0, j, k + 1, nx, ny)};
            const std::array<Index, 4> xmax_nodes = {
                node_id(nx - 1, j, k, nx, ny), node_id(nx - 1, j, k + 1, nx, ny), node_id(nx - 1, j + 1, k + 1, nx, ny), node_id(nx - 1, j + 1, k, nx, ny)};
            const Vec3 xmin_normal = quad_area_vector(
                mapping(0, j, k), mapping(0, j + 1, k), mapping(0, j + 1, k + 1), mapping(0, j, k + 1));
            const Vec3 xmax_normal = quad_area_vector(
                mapping(nx - 1, j, k), mapping(nx - 1, j, k + 1), mapping(nx - 1, j + 1, k + 1), mapping(nx - 1, j + 1, k));
            const Vec3 xmin_center = 0.25 * (mapping(0, j, k) + mapping(0, j + 1, k) + mapping(0, j + 1, k + 1) + mapping(0, j, k + 1));
            const Vec3 xmax_center =
                0.25 * (mapping(nx - 1, j, k) + mapping(nx - 1, j, k + 1) + mapping(nx - 1, j + 1, k + 1) + mapping(nx - 1, j + 1, k));
            const Vec3 xmin_oriented = dot(xmin_normal, xmin_center - centroid) >= 0.0 ? xmin_normal : -1.0 * xmin_normal;
            const Vec3 xmax_oriented = dot(xmax_normal, xmax_center - centroid) >= 0.0 ? xmax_normal : -1.0 * xmax_normal;
            add_boundary_face(mesh, xmin_nodes[0], xmin_nodes[1], xmin_nodes[2], xmin_nodes[3], xmin_oriented,
                              xmin_type(j, k), xmin_name(j, k), 0);
            add_boundary_face(mesh, xmax_nodes[0], xmax_nodes[1], xmax_nodes[2], xmax_nodes[3], xmax_oriented,
                              xmax_type(j, k), xmax_name(j, k), 1);
        }
    }

    for (Index i = 0; i < nx - 1; ++i) {
        for (Index k = 0; k < nz - 1; ++k) {
            const std::array<Index, 4> ymin_nodes = {
                node_id(i, 0, k, nx, ny), node_id(i, 0, k + 1, nx, ny), node_id(i + 1, 0, k + 1, nx, ny), node_id(i + 1, 0, k, nx, ny)};
            const std::array<Index, 4> ymax_nodes = {
                node_id(i, ny - 1, k, nx, ny), node_id(i + 1, ny - 1, k, nx, ny), node_id(i + 1, ny - 1, k + 1, nx, ny), node_id(i, ny - 1, k + 1, nx, ny)};
            const Vec3 ymin_normal = quad_area_vector(
                mapping(i, 0, k), mapping(i, 0, k + 1), mapping(i + 1, 0, k + 1), mapping(i + 1, 0, k));
            const Vec3 ymax_normal = quad_area_vector(
                mapping(i, ny - 1, k), mapping(i + 1, ny - 1, k), mapping(i + 1, ny - 1, k + 1), mapping(i, ny - 1, k + 1));
            const Vec3 ymin_center = 0.25 * (mapping(i, 0, k) + mapping(i, 0, k + 1) + mapping(i + 1, 0, k + 1) + mapping(i + 1, 0, k));
            const Vec3 ymax_center =
                0.25 * (mapping(i, ny - 1, k) + mapping(i + 1, ny - 1, k) + mapping(i + 1, ny - 1, k + 1) + mapping(i, ny - 1, k + 1));
            const Vec3 ymin_oriented = dot(ymin_normal, ymin_center - centroid) >= 0.0 ? ymin_normal : -1.0 * ymin_normal;
            const Vec3 ymax_oriented = dot(ymax_normal, ymax_center - centroid) >= 0.0 ? ymax_normal : -1.0 * ymax_normal;
            add_boundary_face(mesh, ymin_nodes[0], ymin_nodes[1], ymin_nodes[2], ymin_nodes[3], ymin_oriented,
                              ymin_type(i, k), ymin_name(i, k), 2);
            add_boundary_face(mesh, ymax_nodes[0], ymax_nodes[1], ymax_nodes[2], ymax_nodes[3], ymax_oriented,
                              ymax_type(i, k), ymax_name(i, k), 3);
        }
    }

    for (Index i = 0; i < nx - 1; ++i) {
        for (Index j = 0; j < ny - 1; ++j) {
            const std::array<Index, 4> zmin_nodes = {
                node_id(i, j, 0, nx, ny), node_id(i + 1, j, 0, nx, ny), node_id(i + 1, j + 1, 0, nx, ny), node_id(i, j + 1, 0, nx, ny)};
            const std::array<Index, 4> zmax_nodes = {
                node_id(i, j, nz - 1, nx, ny), node_id(i, j + 1, nz - 1, nx, ny), node_id(i + 1, j + 1, nz - 1, nx, ny), node_id(i + 1, j, nz - 1, nx, ny)};
            const Vec3 zmin_normal = quad_area_vector(
                mapping(i, j, 0), mapping(i + 1, j, 0), mapping(i + 1, j + 1, 0), mapping(i, j + 1, 0));
            const Vec3 zmax_normal = quad_area_vector(
                mapping(i, j, nz - 1), mapping(i, j + 1, nz - 1), mapping(i + 1, j + 1, nz - 1), mapping(i + 1, j, nz - 1));
            const Vec3 zmin_center = 0.25 * (mapping(i, j, 0) + mapping(i + 1, j, 0) + mapping(i + 1, j + 1, 0) + mapping(i, j + 1, 0));
            const Vec3 zmax_center =
                0.25 * (mapping(i, j, nz - 1) + mapping(i, j + 1, nz - 1) + mapping(i + 1, j + 1, nz - 1) + mapping(i + 1, j, nz - 1));
            const Vec3 zmin_oriented = dot(zmin_normal, zmin_center - centroid) >= 0.0 ? zmin_normal : -1.0 * zmin_normal;
            const Vec3 zmax_oriented = dot(zmax_normal, zmax_center - centroid) >= 0.0 ? zmax_normal : -1.0 * zmax_normal;
            add_boundary_face(mesh, zmin_nodes[0], zmin_nodes[1], zmin_nodes[2], zmin_nodes[3], zmin_oriented,
                              zmin_type(i, j), zmin_name(i, j), 4);
            add_boundary_face(mesh, zmax_nodes[0], zmax_nodes[1], zmax_nodes[2], zmax_nodes[3], zmax_oriented,
                              zmax_type(i, j), zmax_name(i, j), 5);
        }
    }

    for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
        const Real x = mesh.nodes.x[i];
        const Real y = mesh.nodes.y[i];
        const Real z = mesh.nodes.z[i];
        Real min_dist = std::numeric_limits<Real>::max();
        for (Index f = 0; f < static_cast<Index>(mesh.boundary_faces.count); ++f) {
            const Index n = mesh.boundary_faces.n1[f];
            const Vec3 face_origin {mesh.nodes.x[n], mesh.nodes.y[n], mesh.nodes.z[n]};
            const Vec3 normal = {mesh.boundary_faces.nx[f], mesh.boundary_faces.ny[f], mesh.boundary_faces.nz[f]};
            const Vec3 unit = normal / std::max(mesh.boundary_faces.area[f], 1.0e-14);
            const Real dist = std::abs(dot(Vec3 {x - face_origin.x, y - face_origin.y, z - face_origin.z}, unit));
            min_dist = std::min(min_dist, dist);
        }
        mesh.nodes.wall_dist[i] = min_dist;
    }

    finalize_mesh(mesh);
    return mesh;
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

Real total_nodal_volume(const Mesh& mesh) {
    Real total = 0.0;
    for (Real v : mesh.nodes.vol) {
        total += v;
    }
    return total;
}

}  // namespace nssolver
