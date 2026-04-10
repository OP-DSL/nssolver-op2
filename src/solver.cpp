#include "nssolver/solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace nssolver {

namespace {

Vec3 node_position(const Mesh& mesh, Index i) {
    return {mesh.nodes.x[i], mesh.nodes.y[i], mesh.nodes.z[i]};
}

void add_gradient_contribution(std::vector<Real>& gx,
                               std::vector<Real>& gy,
                               std::vector<Real>& gz,
                               Index i,
                               const Vec3& area_vector,
                               Real scalar) {
    gx[i] += scalar * area_vector.x;
    gy[i] += scalar * area_vector.y;
    gz[i] += scalar * area_vector.z;
}

Real face_length_scale(const Mesh& mesh, const std::array<Index, 4>& face_nodes, Real area) {
    Real volume_sum = 0.0;
    for (Index n : face_nodes) {
        volume_sum += mesh.nodes.vol[n];
    }
    return std::max(volume_sum / (4.0 * area), 1.0e-12);
}

Real vorticity_magnitude(const FlowState& state, std::size_t i) {
    const Real wx = state.grad_w_y[i] - state.grad_v_z[i];
    const Real wy = state.grad_u_z[i] - state.grad_w_x[i];
    const Real wz = state.grad_v_x[i] - state.grad_u_y[i];
    return std::sqrt(wx * wx + wy * wy + wz * wz);
}

Real spalart_allmaras_source(const Mesh& mesh, const FlowState& state, std::size_t i, const GasModel& gas) {
    constexpr Real cb1 = 0.1355;
    constexpr Real cb2 = 0.622;
    constexpr Real sigma = 2.0 / 3.0;
    constexpr Real kappa = 0.41;
    constexpr Real cw2 = 0.3;
    constexpr Real cw3 = 2.0;
    constexpr Real cv1 = 7.1;
    const Real cw1 = cb1 / (kappa * kappa) + (1.0 + cb2) / sigma;

    const Primitive primitive = sample_primitive(state, static_cast<Index>(i));
    const Real temperature = primitive.p / (primitive.rho * gas.gas_constant);
    const Real nu = dynamic_viscosity(temperature, gas) / primitive.rho;
    const Real nu_tilde = std::max(primitive.nu_tilde, 0.0);
    const Real d = std::max(mesh.nodes.wall_dist[i], 1.0e-8);
    const Real omega = vorticity_magnitude(state, i);

    if (nu_tilde <= 0.0) {
        return 0.0;
    }

    const Real chi = nu_tilde / std::max(nu, 1.0e-14);
    const Real chi3 = chi * chi * chi;
    const Real fv1 = chi3 / (chi3 + cv1 * cv1 * cv1);
    const Real fv2 = 1.0 - chi / (1.0 + chi * fv1);
    const Real s_tilde = std::max(omega + nu_tilde * fv2 / (kappa * kappa * d * d), 0.3 * omega + 1.0e-14);
    const Real r = std::min(nu_tilde / (s_tilde * kappa * kappa * d * d), 10.0);
    const Real g = r + cw2 * (std::pow(r, 6) - r);
    const Real fw = g * std::pow((1.0 + std::pow(cw3, 6)) / (std::pow(g, 6) + std::pow(cw3, 6)), 1.0 / 6.0);

    const Real production = cb1 * s_tilde * nu_tilde;
    const Real destruction = cw1 * fw * (nu_tilde * nu_tilde) / (d * d);
    return primitive.rho * (production - destruction);
}

Real scalar_edge_limiter(Real projected_delta, Real edge_delta) {
    if (std::abs(projected_delta) < 1.0e-14) {
        return 1.0;
    }
    if (projected_delta * edge_delta <= 0.0) {
        return 0.0;
    }
    return std::clamp(edge_delta / projected_delta, 0.0, 1.0);
}

Real edge_limiter_scale(const FlowState& state, Index center, Index neighbor, const Vec3& delta_r) {
    const std::array<Real, 6> center_values = {
        state.rho[center], state.u[center], state.v[center], state.w[center], state.p[center], state.nu_tilde[center]};
    const std::array<Real, 6> neighbor_values = {state.rho[neighbor], state.u[neighbor], state.v[neighbor], state.w[neighbor],
                                                  state.p[neighbor], state.nu_tilde[neighbor]};
    const std::array<Real, 6> projected = {
        state.grad_rho_x[center] * delta_r.x + state.grad_rho_y[center] * delta_r.y + state.grad_rho_z[center] * delta_r.z,
        state.grad_u_x[center] * delta_r.x + state.grad_u_y[center] * delta_r.y + state.grad_u_z[center] * delta_r.z,
        state.grad_v_x[center] * delta_r.x + state.grad_v_y[center] * delta_r.y + state.grad_v_z[center] * delta_r.z,
        state.grad_w_x[center] * delta_r.x + state.grad_w_y[center] * delta_r.y + state.grad_w_z[center] * delta_r.z,
        state.grad_p_x[center] * delta_r.x + state.grad_p_y[center] * delta_r.y + state.grad_p_z[center] * delta_r.z,
        state.grad_nu_x[center] * delta_r.x + state.grad_nu_y[center] * delta_r.y + state.grad_nu_z[center] * delta_r.z,
    };

    Real phi = 1.0;
    for (int m = 0; m < 6; ++m) {
        phi = std::min(phi, scalar_edge_limiter(projected[m], neighbor_values[m] - center_values[m]));
    }
    return phi;
}

}  // namespace

void initialize_uniform_state(const Mesh& mesh, const Freestream& freestream, const GasModel& gas, FlowState& state) {
    state.resize(mesh.nodes.count);
    const Conservative conservative = primitive_to_conservative(freestream.primitive, gas);
    for (std::size_t i = 0; i < mesh.nodes.count; ++i) {
        state.rho[i] = conservative.rho;
        state.rhou[i] = conservative.rhou;
        state.rhov[i] = conservative.rhov;
        state.rhow[i] = conservative.rhow;
        state.rhoE[i] = conservative.rhoE;
        state.rhoNu[i] = conservative.rhoNu;
    }
    update_primitives(state, gas);
}

void enforce_boundary_state(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state) {
    std::vector<char> is_dirichlet(mesh.nodes.count, 0);
    std::vector<char> is_wall(mesh.nodes.count, 0);
    std::vector<char> is_slip(mesh.nodes.count, 0);
    std::vector<Vec3> slip_normal(mesh.nodes.count);
    for (std::size_t f = 0; f < mesh.boundary_faces.count; ++f) {
        const std::array<Index, 4> nodes = {
            mesh.boundary_faces.n1[f], mesh.boundary_faces.n2[f], mesh.boundary_faces.n3[f], mesh.boundary_faces.n4[f]};
        if (mesh.boundary_faces.type[f] == BoundaryType::Farfield || mesh.boundary_faces.type[f] == BoundaryType::Inlet) {
            for (Index node : nodes) {
                is_dirichlet[node] = 1;
            }
            continue;
        }
        if (mesh.boundary_faces.type[f] != BoundaryType::NoSlipWall && mesh.boundary_faces.type[f] != BoundaryType::SlipWall) {
            continue;
        }
        const Vec3 normal {mesh.boundary_faces.nx[f], mesh.boundary_faces.ny[f], mesh.boundary_faces.nz[f]};
        for (Index node : nodes) {
            if (mesh.boundary_faces.type[f] == BoundaryType::NoSlipWall) {
                is_wall[node] = 1;
            } else {
                is_slip[node] = 1;
                slip_normal[node] = slip_normal[node] + normal;
            }
        }
    }

    for (std::size_t i = 0; i < state.count; ++i) {
        if (!is_dirichlet[i] && !is_wall[i] && !is_slip[i]) {
            continue;
        }
        Primitive primitive = conservative_to_primitive(
            {state.rho[i], state.rhou[i], state.rhov[i], state.rhow[i], state.rhoE[i], state.rhoNu[i]}, gas);
        if (is_dirichlet[i]) {
            primitive = options.freestream.primitive;
        }
        if (is_wall[i]) {
            primitive.u = 0.0;
            primitive.v = 0.0;
            primitive.w = 0.0;
        } else {
            const Real normal_length = norm(slip_normal[i]);
            if (normal_length > 1.0e-14) {
                const Vec3 unit_normal = slip_normal[i] / normal_length;
                const Real un = primitive.u * unit_normal.x + primitive.v * unit_normal.y + primitive.w * unit_normal.z;
                primitive.u -= un * unit_normal.x;
                primitive.v -= un * unit_normal.y;
                primitive.w -= un * unit_normal.z;
            }
        }
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

void compute_gradients(const Mesh& mesh, const GasModel&, const SolverOptions&, FlowState& state) {
    state.zero_gradients();

    for (Index e = 0; e < static_cast<Index>(mesh.edges.count); ++e) {
        const Index left = mesh.edges.node_L[e];
        const Index right = mesh.edges.node_R[e];
        const Vec3 area_vector {mesh.edges.nx[e], mesh.edges.ny[e], mesh.edges.nz[e]};

        const Real rho_avg = 0.5 * (state.rho[left] + state.rho[right]);
        const Real u_avg = 0.5 * (state.u[left] + state.u[right]);
        const Real v_avg = 0.5 * (state.v[left] + state.v[right]);
        const Real w_avg = 0.5 * (state.w[left] + state.w[right]);
        const Real p_avg = 0.5 * (state.p[left] + state.p[right]);
        const Real nu_avg = 0.5 * (state.nu_tilde[left] + state.nu_tilde[right]);

        add_gradient_contribution(state.grad_rho_x, state.grad_rho_y, state.grad_rho_z, left, area_vector, rho_avg);
        add_gradient_contribution(state.grad_u_x, state.grad_u_y, state.grad_u_z, left, area_vector, u_avg);
        add_gradient_contribution(state.grad_v_x, state.grad_v_y, state.grad_v_z, left, area_vector, v_avg);
        add_gradient_contribution(state.grad_w_x, state.grad_w_y, state.grad_w_z, left, area_vector, w_avg);
        add_gradient_contribution(state.grad_p_x, state.grad_p_y, state.grad_p_z, left, area_vector, p_avg);
        add_gradient_contribution(state.grad_nu_x, state.grad_nu_y, state.grad_nu_z, left, area_vector, nu_avg);

        add_gradient_contribution(state.grad_rho_x, state.grad_rho_y, state.grad_rho_z, right, -1.0 * area_vector, rho_avg);
        add_gradient_contribution(state.grad_u_x, state.grad_u_y, state.grad_u_z, right, -1.0 * area_vector, u_avg);
        add_gradient_contribution(state.grad_v_x, state.grad_v_y, state.grad_v_z, right, -1.0 * area_vector, v_avg);
        add_gradient_contribution(state.grad_w_x, state.grad_w_y, state.grad_w_z, right, -1.0 * area_vector, w_avg);
        add_gradient_contribution(state.grad_p_x, state.grad_p_y, state.grad_p_z, right, -1.0 * area_vector, p_avg);
        add_gradient_contribution(state.grad_nu_x, state.grad_nu_y, state.grad_nu_z, right, -1.0 * area_vector, nu_avg);
    }

    for (Index f = 0; f < static_cast<Index>(mesh.boundary_faces.count); ++f) {
        const std::array<Index, 4> face_nodes = {
            mesh.boundary_faces.n1[f], mesh.boundary_faces.n2[f], mesh.boundary_faces.n3[f], mesh.boundary_faces.n4[f]};
        const Vec3 area_vector {mesh.boundary_faces.nx[f], mesh.boundary_faces.ny[f], mesh.boundary_faces.nz[f]};

        Primitive face_primitive {};
        for (Index n : face_nodes) {
            face_primitive.rho += state.rho[n];
            face_primitive.u += state.u[n];
            face_primitive.v += state.v[n];
            face_primitive.w += state.w[n];
            face_primitive.p += state.p[n];
            face_primitive.nu_tilde += state.nu_tilde[n];
        }
        face_primitive.rho /= 4.0;
        face_primitive.u /= 4.0;
        face_primitive.v /= 4.0;
        face_primitive.w /= 4.0;
        face_primitive.p /= 4.0;
        face_primitive.nu_tilde /= 4.0;

        for (Index n : face_nodes) {
            add_gradient_contribution(state.grad_rho_x, state.grad_rho_y, state.grad_rho_z, n, 0.25 * area_vector, face_primitive.rho);
            add_gradient_contribution(state.grad_u_x, state.grad_u_y, state.grad_u_z, n, 0.25 * area_vector, face_primitive.u);
            add_gradient_contribution(state.grad_v_x, state.grad_v_y, state.grad_v_z, n, 0.25 * area_vector, face_primitive.v);
            add_gradient_contribution(state.grad_w_x, state.grad_w_y, state.grad_w_z, n, 0.25 * area_vector, face_primitive.w);
            add_gradient_contribution(state.grad_p_x, state.grad_p_y, state.grad_p_z, n, 0.25 * area_vector, face_primitive.p);
            add_gradient_contribution(state.grad_nu_x, state.grad_nu_y, state.grad_nu_z, n, 0.25 * area_vector, face_primitive.nu_tilde);
        }
    }

    for (std::size_t i = 0; i < state.count; ++i) {
        const Real inv_vol = 1.0 / std::max(mesh.nodes.vol[i], 1.0e-14);
        state.grad_rho_x[i] *= inv_vol;
        state.grad_rho_y[i] *= inv_vol;
        state.grad_rho_z[i] *= inv_vol;
        state.grad_u_x[i] *= inv_vol;
        state.grad_u_y[i] *= inv_vol;
        state.grad_u_z[i] *= inv_vol;
        state.grad_v_x[i] *= inv_vol;
        state.grad_v_y[i] *= inv_vol;
        state.grad_v_z[i] *= inv_vol;
        state.grad_w_x[i] *= inv_vol;
        state.grad_w_y[i] *= inv_vol;
        state.grad_w_z[i] *= inv_vol;
        state.grad_p_x[i] *= inv_vol;
        state.grad_p_y[i] *= inv_vol;
        state.grad_p_z[i] *= inv_vol;
        state.grad_nu_x[i] *= inv_vol;
        state.grad_nu_y[i] *= inv_vol;
        state.grad_nu_z[i] *= inv_vol;
    }
}

Primitive reconstruct_primitive(const FlowState& state, Index i, const Vec3& delta_r, Real limiter_scale) {
    Primitive p = sample_primitive(state, i);
    p.rho += limiter_scale * (state.grad_rho_x[i] * delta_r.x + state.grad_rho_y[i] * delta_r.y + state.grad_rho_z[i] * delta_r.z);
    p.u += limiter_scale * (state.grad_u_x[i] * delta_r.x + state.grad_u_y[i] * delta_r.y + state.grad_u_z[i] * delta_r.z);
    p.v += limiter_scale * (state.grad_v_x[i] * delta_r.x + state.grad_v_y[i] * delta_r.y + state.grad_v_z[i] * delta_r.z);
    p.w += limiter_scale * (state.grad_w_x[i] * delta_r.x + state.grad_w_y[i] * delta_r.y + state.grad_w_z[i] * delta_r.z);
    p.p += limiter_scale * (state.grad_p_x[i] * delta_r.x + state.grad_p_y[i] * delta_r.y + state.grad_p_z[i] * delta_r.z);
    p.nu_tilde += limiter_scale * (state.grad_nu_x[i] * delta_r.x + state.grad_nu_y[i] * delta_r.y + state.grad_nu_z[i] * delta_r.z);
    p.rho = std::max(p.rho, 1.0e-12);
    p.p = std::max(p.p, 1.0e-12);
    return p;
}

void compute_local_time_step(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state) {
    for (std::size_t i = 0; i < state.count; ++i) {
        Real spectral_sum = 0.0;
        const Primitive primitive = sample_primitive(state, static_cast<Index>(i));
        for (Index e : mesh.node_to_edges[i]) {
            const Vec3 area_vector {mesh.edges.nx[e], mesh.edges.ny[e], mesh.edges.nz[e]};
            const Vec3 unit_normal = area_vector / mesh.edges.area[e];
            const Real un = primitive.u * unit_normal.x + primitive.v * unit_normal.y + primitive.w * unit_normal.z;
            spectral_sum += mesh.edges.area[e] * (std::abs(un) + speed_of_sound(primitive, gas));
            if (options.include_viscous) {
                const Real mu = dynamic_viscosity(state.T[i], gas);
                spectral_sum += 2.0 * mu * mesh.edges.area[e] * mesh.edges.area[e] /
                                (std::max(state.rho[i] * mesh.nodes.vol[i], 1.0e-14));
            }
        }
        state.dt[i] = options.cfl * mesh.nodes.vol[i] / (spectral_sum + options.viscous_penalty + 1.0e-14);
    }
}

void assemble_residual(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state) {
    if (options.second_order || options.include_sa) {
        compute_gradients(mesh, gas, options, state);
    } else {
        state.zero_gradients();
    }

    state.zero_residuals();

    for (Index e = 0; e < static_cast<Index>(mesh.edges.count); ++e) {
        const Index left = mesh.edges.node_L[e];
        const Index right = mesh.edges.node_R[e];
        const Vec3 area_vector {mesh.edges.nx[e], mesh.edges.ny[e], mesh.edges.nz[e]};
        const Vec3 delta_full = node_position(mesh, right) - node_position(mesh, left);

        Primitive left_primitive = sample_primitive(state, left);
        Primitive right_primitive = sample_primitive(state, right);
        if (options.second_order) {
            const Real left_phi = edge_limiter_scale(state, left, right, 0.5 * delta_full);
            const Real right_phi = edge_limiter_scale(state, right, left, -0.5 * delta_full);
            left_primitive = reconstruct_primitive(state, left, 0.5 * delta_full, left_phi);
            right_primitive = reconstruct_primitive(state, right, -0.5 * delta_full, right_phi);
        }

        const Conservative left_conservative = primitive_to_conservative(left_primitive, gas);
        const Conservative right_conservative = primitive_to_conservative(right_primitive, gas);
        const FluxArray inviscid_flux =
            hllc_flux(left_primitive, left_conservative, right_primitive, right_conservative, area_vector, gas);
        FluxArray net_flux = inviscid_flux;

        if (options.include_viscous) {
            const FluxArray viscous_flux = thin_layer_viscous_flux(left_primitive, right_primitive, delta_full, area_vector, gas);
            for (int m = 0; m < 6; ++m) {
                net_flux[m] -= viscous_flux[m];
            }
        }

        state.res_rho[left] += net_flux[0];
        state.res_rhou[left] += net_flux[1];
        state.res_rhov[left] += net_flux[2];
        state.res_rhow[left] += net_flux[3];
        state.res_rhoE[left] += net_flux[4];
        state.res_rhoNu[left] += net_flux[5];

        state.res_rho[right] -= net_flux[0];
        state.res_rhou[right] -= net_flux[1];
        state.res_rhov[right] -= net_flux[2];
        state.res_rhow[right] -= net_flux[3];
        state.res_rhoE[right] -= net_flux[4];
        state.res_rhoNu[right] -= net_flux[5];
    }

    for (Index f = 0; f < static_cast<Index>(mesh.boundary_faces.count); ++f) {
        const std::array<Index, 4> face_nodes = {
            mesh.boundary_faces.n1[f], mesh.boundary_faces.n2[f], mesh.boundary_faces.n3[f], mesh.boundary_faces.n4[f]};
        const Vec3 area_vector {mesh.boundary_faces.nx[f], mesh.boundary_faces.ny[f], mesh.boundary_faces.nz[f]};
        const Vec3 unit_normal = area_vector / mesh.boundary_faces.area[f];

        Primitive interior {};
        for (Index n : face_nodes) {
            interior.rho += state.rho[n];
            interior.u += state.u[n];
            interior.v += state.v[n];
            interior.w += state.w[n];
            interior.p += state.p[n];
            interior.nu_tilde += state.nu_tilde[n];
        }
        interior.rho /= 4.0;
        interior.u /= 4.0;
        interior.v /= 4.0;
        interior.w /= 4.0;
        interior.p /= 4.0;
        interior.nu_tilde /= 4.0;

        const Primitive ghost =
            make_boundary_ghost_state(interior, options.freestream.primitive, unit_normal, mesh.boundary_faces.type[f]);
        const Conservative q_interior = primitive_to_conservative(interior, gas);
        const Conservative q_ghost = primitive_to_conservative(ghost, gas);
        const FluxArray inviscid_flux = hllc_flux(interior, q_interior, ghost, q_ghost, area_vector, gas);
        FluxArray net_flux = inviscid_flux;

        if (options.include_viscous && mesh.boundary_faces.type[f] == BoundaryType::NoSlipWall) {
            Primitive wall_state = interior;
            wall_state.u = 0.0;
            wall_state.v = 0.0;
            wall_state.w = 0.0;
            const Real distance = face_length_scale(mesh, face_nodes, mesh.boundary_faces.area[f]);
            const Vec3 delta_r = distance * unit_normal;
            const FluxArray viscous_flux = thin_layer_viscous_flux(interior, wall_state, delta_r, area_vector, gas);
            for (int m = 0; m < 6; ++m) {
                net_flux[m] -= viscous_flux[m];
            }
        }

        for (Index n : face_nodes) {
            const Real share = 0.25;
            state.res_rho[n] += share * net_flux[0];
            state.res_rhou[n] += share * net_flux[1];
            state.res_rhov[n] += share * net_flux[2];
            state.res_rhow[n] += share * net_flux[3];
            state.res_rhoE[n] += share * net_flux[4];
            state.res_rhoNu[n] += share * net_flux[5];
        }
    }

    if (options.include_sa) {
        for (std::size_t i = 0; i < state.count; ++i) {
            state.res_rhoNu[i] -= mesh.nodes.vol[i] * spalart_allmaras_source(mesh, state, i, gas);
        }
    }
}

void rk4_step(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state) {
    static constexpr Real alpha[4] = {0.25, 1.0 / 3.0, 0.5, 1.0};

    state.save_baseline();
    compute_local_time_step(mesh, gas, options, state);
    enforce_boundary_state(mesh, gas, options, state);

    for (Real stage_alpha : alpha) {
        update_primitives(state, gas);
        enforce_boundary_state(mesh, gas, options, state);
        assemble_residual(mesh, gas, options, state);

        for (std::size_t i = 0; i < state.count; ++i) {
            const Real factor = stage_alpha * state.dt[i] / mesh.nodes.vol[i];
            state.rho[i] = state.rho0[i] - factor * state.res_rho[i];
            state.rhou[i] = state.rhou0[i] - factor * state.res_rhou[i];
            state.rhov[i] = state.rhov0[i] - factor * state.res_rhov[i];
            state.rhow[i] = state.rhow0[i] - factor * state.res_rhow[i];
            state.rhoE[i] = state.rhoE0[i] - factor * state.res_rhoE[i];
            state.rhoNu[i] = state.rhoNu0[i] - factor * state.res_rhoNu[i];

            const Primitive primitive = conservative_to_primitive(
                {state.rho[i], state.rhou[i], state.rhov[i], state.rhow[i], state.rhoE[i], state.rhoNu[i]}, gas);
            if (!std::isfinite(state.rho[i]) || !std::isfinite(state.rhoE[i]) || state.rho[i] < options.rho_floor ||
                primitive.p < options.p_floor) {
                state.rho[i] = state.rho0[i];
                state.rhou[i] = state.rhou0[i];
                state.rhov[i] = state.rhov0[i];
                state.rhow[i] = state.rhow0[i];
                state.rhoE[i] = state.rhoE0[i];
                state.rhoNu[i] = state.rhoNu0[i];
            }
        }
        enforce_boundary_state(mesh, gas, options, state);
    }

    update_primitives(state, gas);
    enforce_boundary_state(mesh, gas, options, state);
}

ResidualNorms compute_residual_norms(const FlowState& state) {
    ResidualNorms norms {};
    for (std::size_t i = 0; i < state.count; ++i) {
        norms.l2_rho += state.res_rho[i] * state.res_rho[i];
        norms.l2_rhoE += state.res_rhoE[i] * state.res_rhoE[i];
        norms.linf_rho = std::max(norms.linf_rho, std::abs(state.res_rho[i]));
    }
    norms.l2_rho = std::sqrt(norms.l2_rho / static_cast<Real>(state.count));
    norms.l2_rhoE = std::sqrt(norms.l2_rhoE / static_cast<Real>(state.count));
    return norms;
}

StateDiagnostics compute_state_diagnostics(const FlowState& state, const GasModel& gas) {
    StateDiagnostics diagnostics;
    diagnostics.min_rho = std::numeric_limits<Real>::max();
    diagnostics.min_p = std::numeric_limits<Real>::max();
    diagnostics.min_rhoE = std::numeric_limits<Real>::max();
    diagnostics.max_mach = 0.0;
    for (std::size_t i = 0; i < state.count; ++i) {
        diagnostics.finite = diagnostics.finite && std::isfinite(state.rho[i]) && std::isfinite(state.rhoE[i]) &&
                             std::isfinite(state.p[i]) && std::isfinite(state.u[i]) && std::isfinite(state.v[i]) &&
                             std::isfinite(state.w[i]);
        diagnostics.min_rho = std::min(diagnostics.min_rho, state.rho[i]);
        diagnostics.min_p = std::min(diagnostics.min_p, state.p[i]);
        diagnostics.min_rhoE = std::min(diagnostics.min_rhoE, state.rhoE[i]);
        const Real velocity = std::sqrt(state.u[i] * state.u[i] + state.v[i] * state.v[i] + state.w[i] * state.w[i]);
        const Real a = speed_of_sound(sample_primitive(state, static_cast<Index>(i)), gas);
        diagnostics.max_mach = std::max(diagnostics.max_mach, velocity / std::max(a, 1.0e-14));
    }
    return diagnostics;
}

SolverHistory run_solver(const Mesh& mesh, const GasModel& gas, const SolverOptions& options, FlowState& state) {
    SolverHistory history;
    Real initial_l2_rho = -1.0;
    for (int iter = 0; iter < options.iterations; ++iter) {
        rk4_step(mesh, gas, options, state);
        assemble_residual(mesh, gas, options, state);
        const ResidualNorms norms = compute_residual_norms(state);
        if (initial_l2_rho < 0.0) {
            initial_l2_rho = norms.l2_rho;
        }
        history.l2_rho.push_back(norms.l2_rho);
        history.l2_rhoE.push_back(norms.l2_rhoE);
        history.linf_rho.push_back(norms.linf_rho);
        history.final_iteration = iter + 1;
        const bool should_report = options.verbose &&
                                   (iter == 0 || iter + 1 == options.iterations ||
                                    norms.l2_rho < options.residual_tolerance ||
                                    ((iter + 1) % std::max(options.progress_interval, 1) == 0));
        if (should_report) {
            write_progress_line(std::cout, iter + 1, options.iterations, norms, initial_l2_rho);
            const StateDiagnostics diagnostics = compute_state_diagnostics(state, gas);
            std::cout << "[state] min_rho=" << std::scientific << std::setprecision(6) << diagnostics.min_rho
                      << " | min_p=" << diagnostics.min_p
                      << " | min_rhoE=" << diagnostics.min_rhoE
                      << " | max_mach=" << diagnostics.max_mach
                      << " | finite=" << (diagnostics.finite ? "yes" : "no") << '\n'
                      << std::flush;
        }
        const StateDiagnostics diagnostics = compute_state_diagnostics(state, gas);
        if (!std::isfinite(norms.l2_rho) || !std::isfinite(norms.l2_rhoE) || !std::isfinite(norms.linf_rho) ||
            !diagnostics.finite) {
            history.stopped_on_nonfinite = true;
            if (options.verbose) {
                std::cout << "[progress] stopping: non-finite residual detected at iteration " << iter + 1 << '\n';
            }
            break;
        }
        if (norms.l2_rho < options.residual_tolerance) {
            history.converged = true;
            break;
        }
    }
    return history;
}

std::string residual_summary(const ResidualNorms& norms) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(6)
           << "L2(rho)=" << norms.l2_rho
           << ", L2(rhoE)=" << norms.l2_rhoE
           << ", Linf(rho)=" << norms.linf_rho;
    return stream.str();
}

void write_progress_line(std::ostream& out, int iteration, int iterations, const ResidualNorms& norms, Real initial_l2_rho) {
    const Real ratio = initial_l2_rho > 0.0 ? norms.l2_rho / initial_l2_rho : 1.0;
    out << "[progress] iter " << iteration << '/' << iterations
        << " | L2(rho)=" << std::scientific << std::setprecision(6) << norms.l2_rho
        << " | L2/L2_0=" << ratio
        << " | Linf(rho)=" << norms.linf_rho << '\n'
        << std::flush;
}

}  // namespace nssolver
