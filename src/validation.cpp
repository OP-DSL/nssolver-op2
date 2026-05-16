#include "nssolver/validation.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

namespace nssolver {

namespace {

struct WallSample {
    Real x {};
    Real cf {};
    Real cp {};
    Real rex {};
};

std::set<Index> unique_face_nodes(const Mesh& mesh, const std::string& face_name) {
    std::set<Index> nodes;
    for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
        if (mesh.boundary_faces.name[f] != face_name) {
            continue;
        }
        const auto face_nodes = boundary_face_nodes(mesh.boundary_faces, f);
        const std::size_t face_node_count = boundary_face_num_nodes(mesh.boundary_faces, f);
        for (std::size_t local = 0; local < face_node_count; ++local) {
            nodes.insert(face_nodes[local]);
        }
    }
    return nodes;
}

Real nearest_positive_wall_distance(const Mesh& mesh, Index node) {
    Real best = std::numeric_limits<Real>::max();
    for (Index e : mesh.node_to_edges[node]) {
        const Index other = mesh.edges.node_L[e] == node ? mesh.edges.node_R[e] : mesh.edges.node_L[e];
        const Real dy = mesh.nodes.y[other] - mesh.nodes.y[node];
        if (dy > 1.0e-14) {
            best = std::min(best, dy);
        }
    }
    return best;
}

Index nearest_interior_neighbor(const Mesh& mesh, Index node) {
    Index best_node = node;
    Real best = std::numeric_limits<Real>::max();
    for (Index e : mesh.node_to_edges[node]) {
        const Index other = mesh.edges.node_L[e] == node ? mesh.edges.node_R[e] : mesh.edges.node_L[e];
        const Real dy = mesh.nodes.y[other] - mesh.nodes.y[node];
        if (dy > 1.0e-14 && dy < best) {
            best = dy;
            best_node = other;
        }
    }
    return best_node;
}

void ensure_stream(std::ofstream& out, const std::string& path) {
    if (!out) {
        throw std::runtime_error("Failed to open validation output: " + path);
    }
}

}  // namespace

void write_flat_plate_benchmark_outputs(const std::string& prefix,
                                        const Mesh& mesh,
                                        const FlowState& state,
                                        const GasModel& gas,
                                        const BenchmarkOptions& options,
                                        Real leading_edge_x) {
    auto plate_nodes = unique_face_nodes(mesh, "plate");
    if (plate_nodes.empty()) {
        for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
            if (mesh.boundary_faces.type[f] != BoundaryType::NoSlipWall) {
                continue;
            }
            const auto face_nodes = boundary_face_nodes(mesh.boundary_faces, f);
            const std::size_t face_node_count = boundary_face_num_nodes(mesh.boundary_faces, f);
            for (std::size_t local = 0; local < face_node_count; ++local) {
                plate_nodes.insert(face_nodes[local]);
            }
        }
    }
    std::map<Real, WallSample> samples;
    const Real rho_inf = options.freestream.rho;
    const Real u_inf = options.freestream.u;
    const Real mu_inf = dynamic_viscosity(options.freestream.p / (rho_inf * gas.gas_constant), gas);

    for (Index node : plate_nodes) {
        const Index inner = nearest_interior_neighbor(mesh, node);
        if (inner == node) {
            continue;
        }
        const Real x = mesh.nodes.x[node];
        if (x <= leading_edge_x + 1.0e-12) {
            continue;
        }
        const Real dy = nearest_positive_wall_distance(mesh, node);
        const Real mu = dynamic_viscosity(state.T[inner], gas);
        const Real tau_w = mu * state.u[inner] / std::max(dy, 1.0e-14);
        const Real cp = (state.p[inner] - options.freestream.p) / (0.5 * rho_inf * u_inf * u_inf);
        const Real rex = rho_inf * u_inf * (x - leading_edge_x) / mu_inf;
        const Real cf = tau_w / (0.5 * rho_inf * u_inf * u_inf);
        samples[x] = {x, cf, cp, rex};
    }

    {
        const std::string path = prefix + "_wall.csv";
        std::ofstream out(path);
        ensure_stream(out, path);
        out << "x,cf,cp,re_x\n";
        for (const auto& [x, sample] : samples) {
            out << x << ',' << sample.cf << ',' << sample.cp << ',' << sample.rex << '\n';
        }
    }

    const std::vector<Real> fractions = {0.2, 0.5, 0.8};
    const Real plate_end = mesh.nodes.x.back();
    for (Real fraction : fractions) {
        const Real target_x = leading_edge_x + fraction * (plate_end - leading_edge_x);
        std::vector<std::pair<Real, Index>> column;
        Real nearest_x = std::numeric_limits<Real>::max();
        for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
            nearest_x = std::min(nearest_x, std::abs(mesh.nodes.x[i] - target_x));
        }
        for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
            if (std::abs(mesh.nodes.x[i] - target_x) < 0.5 * (plate_end / std::max<std::size_t>(mesh.nodes.count, 2))) {
                column.emplace_back(mesh.nodes.y[i], static_cast<Index>(i));
            }
            if (std::abs(mesh.nodes.x[i] - target_x) <= nearest_x + 1.0e-12) {
                column.emplace_back(mesh.nodes.y[i], static_cast<Index>(i));
            }
        }
        if (column.empty()) {
            continue;
        }
        std::sort(column.begin(), column.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        const std::string path = prefix + "_profile_" + std::to_string(static_cast<int>(fraction * 100.0)) + ".csv";
        std::ofstream out(path);
        ensure_stream(out, path);
        out << "y,u_over_uinf,x\n";
        for (const auto& [y, idx] : column) {
            if (y <= 1.0e-14) {
                continue;
            }
            out << y << ',' << state.u[idx] / u_inf << ',' << mesh.nodes.x[idx] << '\n';
        }
    }
}

void write_flat_plate_benchmark_outputs_from_wall_type(const std::string& prefix,
                                                       const Mesh& mesh,
                                                       const FlowState& state,
                                                       const GasModel& gas,
                                                       const BenchmarkOptions& options,
                                                       Real leading_edge_x,
                                                       BoundaryType wall_type) {
    Mesh copy = mesh;
    for (std::size_t f = 0; f < copy.boundary_faces.count; ++f) {
        if (copy.boundary_faces.type[f] == wall_type) {
            copy.boundary_faces.name[f] = "plate";
        }
    }
    write_flat_plate_benchmark_outputs(prefix, copy, state, gas, options, leading_edge_x);
}

}  // namespace nssolver
