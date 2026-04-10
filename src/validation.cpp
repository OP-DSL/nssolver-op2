#include "nssolver/validation.hpp"

#include <algorithm>
#include <array>
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
        nodes.insert(mesh.boundary_faces.n1[f]);
        nodes.insert(mesh.boundary_faces.n2[f]);
        nodes.insert(mesh.boundary_faces.n3[f]);
        nodes.insert(mesh.boundary_faces.n4[f]);
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

std::vector<std::pair<Real, Real>> blasius_table() {
    std::vector<std::pair<Real, Real>> table;
    Real f = 0.0;
    Real fp = 0.0;
    Real fpp = 0.332057336215;
    const Real step = 0.002;
    for (Real eta = 0.0; eta <= 10.0 + 1.0e-12; eta += step) {
        table.emplace_back(eta, fp);
        auto rhs = [](Real f_in, Real fp_in, Real fpp_in) {
            return std::array<Real, 3> {fp_in, fpp_in, -0.5 * f_in * fpp_in};
        };
        const auto k1 = rhs(f, fp, fpp);
        const auto k2 = rhs(f + 0.5 * step * k1[0], fp + 0.5 * step * k1[1], fpp + 0.5 * step * k1[2]);
        const auto k3 = rhs(f + 0.5 * step * k2[0], fp + 0.5 * step * k2[1], fpp + 0.5 * step * k2[2]);
        const auto k4 = rhs(f + step * k3[0], fp + step * k3[1], fpp + step * k3[2]);
        f += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0;
        fp += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0;
        fpp += step * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0;
    }
    return table;
}

Real interpolate_table(const std::vector<std::pair<Real, Real>>& table, Real x) {
    if (x <= table.front().first) {
        return table.front().second;
    }
    if (x >= table.back().first) {
        return table.back().second;
    }
    const auto upper = std::lower_bound(table.begin(), table.end(), std::pair<Real, Real> {x, -1.0},
                                        [](const auto& a, const auto& b) { return a.first < b.first; });
    const auto lower = std::prev(upper);
    const Real t = (x - lower->first) / (upper->first - lower->first);
    return lower->second * (1.0 - t) + upper->second * t;
}

}  // namespace

void write_residual_history_csv(const std::string& path, const SolverHistory& history) {
    std::ofstream out(path);
    ensure_stream(out, path);
    out << "iteration,l2_rho,l2_rhoE,linf_rho\n";
    for (std::size_t i = 0; i < history.l2_rho.size(); ++i) {
        out << i << ',' << history.l2_rho[i] << ',' << history.l2_rhoE[i] << ',' << history.linf_rho[i] << '\n';
    }
}

void initialize_blasius_flat_plate_state(const Mesh& mesh,
                                         FlowState& state,
                                         const GasModel& gas,
                                         const SolverOptions& options,
                                         Real leading_edge_x) {
    state.resize(mesh.nodes.count);
    const auto table = blasius_table();
    const Real rho_inf = options.freestream.primitive.rho;
    const Real u_inf = options.freestream.primitive.u;
    const Real p_inf = options.freestream.primitive.p;
    const Real mu_inf = dynamic_viscosity(p_inf / (rho_inf * gas.gas_constant), gas);

    for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
        const Real x_rel = mesh.nodes.x[i] - leading_edge_x;
        Real u = u_inf;
        if (x_rel > 1.0e-12) {
            const Real rex = rho_inf * u_inf * x_rel / mu_inf;
            const Real eta = mesh.nodes.y[i] * std::sqrt(rex) / x_rel;
            u = u_inf * interpolate_table(table, eta);
        }
        const Primitive primitive {.rho = rho_inf, .u = u, .v = 0.0, .w = 0.0, .p = p_inf, .nu_tilde = 0.0};
        const Conservative conservative = primitive_to_conservative(primitive, gas);
        state.rho[i] = conservative.rho;
        state.rhou[i] = conservative.rhou;
        state.rhov[i] = conservative.rhov;
        state.rhow[i] = conservative.rhow;
        state.rhoE[i] = conservative.rhoE;
        state.rhoNu[i] = conservative.rhoNu;
    }
    update_primitives(state, gas);
}

void write_flat_plate_benchmark_outputs(const std::string& prefix,
                                        const Mesh& mesh,
                                        const FlowState& state,
                                        const GasModel& gas,
                                        const SolverOptions& options,
                                        Real leading_edge_x) {
    auto plate_nodes = unique_face_nodes(mesh, "plate");
    if (plate_nodes.empty()) {
        for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
            if (mesh.boundary_faces.type[f] != BoundaryType::NoSlipWall) {
                continue;
            }
            plate_nodes.insert(mesh.boundary_faces.n1[f]);
            plate_nodes.insert(mesh.boundary_faces.n2[f]);
            plate_nodes.insert(mesh.boundary_faces.n3[f]);
            plate_nodes.insert(mesh.boundary_faces.n4[f]);
        }
    }
    std::map<Real, WallSample> samples;
    const Real rho_inf = options.freestream.primitive.rho;
    const Real u_inf = options.freestream.primitive.u;
    const Real mu_inf = dynamic_viscosity(options.freestream.primitive.p / (rho_inf * gas.gas_constant), gas);

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
        const Real cp = (state.p[inner] - options.freestream.primitive.p) / (0.5 * rho_inf * u_inf * u_inf);
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
                                                       const SolverOptions& options,
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

void write_bump_benchmark_outputs(const std::string& prefix,
                                  const Mesh& mesh,
                                  const FlowState& state,
                                  const SolverOptions& options) {
    const auto lower_nodes = unique_face_nodes(mesh, "lower_wall");
    std::map<Real, std::pair<Real, int>> cp_by_x;
    const Real rho_inf = options.freestream.primitive.rho;
    const Real u_inf = options.freestream.primitive.u;
    const Real p_inf = options.freestream.primitive.p;

    for (Index node : lower_nodes) {
        const Real x = mesh.nodes.x[node];
        const Real cp = (state.p[node] - p_inf) / (0.5 * rho_inf * u_inf * u_inf);
        auto& entry = cp_by_x[x];
        entry.first += cp;
        entry.second += 1;
    }

    const std::string path = prefix + "_wall_cp.csv";
    std::ofstream out(path);
    ensure_stream(out, path);
    out << "x,cp\n";
    for (const auto& [x, aggregate] : cp_by_x) {
        out << x << ',' << aggregate.first / std::max(aggregate.second, 1) << '\n';
    }
}

}  // namespace nssolver
