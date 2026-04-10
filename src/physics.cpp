#include "nssolver/physics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nssolver {

void FlowState::resize(std::size_t n) {
    count = n;
    rho.resize(n);
    rhou.resize(n);
    rhov.resize(n);
    rhow.resize(n);
    rhoE.resize(n);
    rhoNu.resize(n);
    u.resize(n);
    v.resize(n);
    w.resize(n);
    p.resize(n);
    T.resize(n);
    nu_tilde.resize(n);
    res_rho.resize(n);
    res_rhou.resize(n);
    res_rhov.resize(n);
    res_rhow.resize(n);
    res_rhoE.resize(n);
    res_rhoNu.resize(n);
    dt.resize(n);
    rho0.resize(n);
    rhou0.resize(n);
    rhov0.resize(n);
    rhow0.resize(n);
    rhoE0.resize(n);
    rhoNu0.resize(n);
    grad_rho_x.resize(n);
    grad_rho_y.resize(n);
    grad_rho_z.resize(n);
    grad_u_x.resize(n);
    grad_u_y.resize(n);
    grad_u_z.resize(n);
    grad_v_x.resize(n);
    grad_v_y.resize(n);
    grad_v_z.resize(n);
    grad_w_x.resize(n);
    grad_w_y.resize(n);
    grad_w_z.resize(n);
    grad_p_x.resize(n);
    grad_p_y.resize(n);
    grad_p_z.resize(n);
    grad_nu_x.resize(n);
    grad_nu_y.resize(n);
    grad_nu_z.resize(n);
}

void FlowState::zero_residuals() {
    std::fill(res_rho.begin(), res_rho.end(), 0.0);
    std::fill(res_rhou.begin(), res_rhou.end(), 0.0);
    std::fill(res_rhov.begin(), res_rhov.end(), 0.0);
    std::fill(res_rhow.begin(), res_rhow.end(), 0.0);
    std::fill(res_rhoE.begin(), res_rhoE.end(), 0.0);
    std::fill(res_rhoNu.begin(), res_rhoNu.end(), 0.0);
}

void FlowState::save_baseline() {
    rho0 = rho;
    rhou0 = rhou;
    rhov0 = rhov;
    rhow0 = rhow;
    rhoE0 = rhoE;
    rhoNu0 = rhoNu;
}

void FlowState::zero_gradients() {
    auto zero = [](std::vector<Real>& values) { std::fill(values.begin(), values.end(), 0.0); };
    zero(grad_rho_x);
    zero(grad_rho_y);
    zero(grad_rho_z);
    zero(grad_u_x);
    zero(grad_u_y);
    zero(grad_u_z);
    zero(grad_v_x);
    zero(grad_v_y);
    zero(grad_v_z);
    zero(grad_w_x);
    zero(grad_w_y);
    zero(grad_w_z);
    zero(grad_p_x);
    zero(grad_p_y);
    zero(grad_p_z);
    zero(grad_nu_x);
    zero(grad_nu_y);
    zero(grad_nu_z);
}

Conservative primitive_to_conservative(const Primitive& primitive, const GasModel& gas) {
    Conservative c;
    c.rho = primitive.rho;
    c.rhou = primitive.rho * primitive.u;
    c.rhov = primitive.rho * primitive.v;
    c.rhow = primitive.rho * primitive.w;
    const Real kinetic = 0.5 * (primitive.u * primitive.u + primitive.v * primitive.v + primitive.w * primitive.w);
    const Real internal = primitive.p / ((gas.gamma - 1.0) * primitive.rho);
    c.rhoE = primitive.rho * (internal + kinetic);
    c.rhoNu = primitive.rho * primitive.nu_tilde;
    return c;
}

Primitive conservative_to_primitive(const Conservative& conservative, const GasModel& gas) {
    constexpr Real rho_floor = 1.0e-12;
    constexpr Real p_floor = 1.0e-12;

    Primitive p;
    p.rho = std::max(conservative.rho, rho_floor);
    p.u = conservative.rhou / p.rho;
    p.v = conservative.rhov / p.rho;
    p.w = conservative.rhow / p.rho;
    const Real kinetic = 0.5 * (p.u * p.u + p.v * p.v + p.w * p.w);
    const Real specific_energy = conservative.rhoE / p.rho;
    p.p = std::max((gas.gamma - 1.0) * p.rho * (specific_energy - kinetic), p_floor);
    p.nu_tilde = conservative.rhoNu / p.rho;
    return p;
}

Real speed_of_sound(const Primitive& primitive, const GasModel& gas) {
    return std::sqrt(gas.gamma * primitive.p / primitive.rho);
}

Real total_enthalpy(const Primitive& primitive, const GasModel& gas) {
    const Real kinetic = 0.5 * (primitive.u * primitive.u + primitive.v * primitive.v + primitive.w * primitive.w);
    return gas.gamma * primitive.p / ((gas.gamma - 1.0) * primitive.rho) + kinetic;
}

Real dynamic_viscosity(Real temperature, const GasModel& gas) {
    const Real ratio = temperature / gas.t_ref;
    return gas.mu_ref * std::pow(ratio, 1.5) * (gas.t_ref + gas.sutherland) / (temperature + gas.sutherland);
}

Real thermal_conductivity(Real temperature, const GasModel& gas) {
    return dynamic_viscosity(temperature, gas) * gas.gamma * gas.cv() / gas.prandtl;
}

Real eddy_viscosity(const Primitive& primitive, const GasModel& gas) {
    const Real temperature = primitive.p / (primitive.rho * gas.gas_constant);
    const Real nu = dynamic_viscosity(temperature, gas) / primitive.rho;
    const Real nu_tilde = std::max(primitive.nu_tilde, 0.0);
    if (nu <= 0.0 || nu_tilde <= 0.0) {
        return 0.0;
    }
    constexpr Real cv1 = 7.1;
    const Real chi = nu_tilde / nu;
    const Real chi3 = chi * chi * chi;
    const Real fv1 = chi3 / (chi3 + cv1 * cv1 * cv1);
    return primitive.rho * nu_tilde * fv1;
}

void update_primitives(FlowState& state, const GasModel& gas) {
    for (std::size_t i = 0; i < state.count; ++i) {
        const Primitive primitive = conservative_to_primitive(
            {state.rho[i], state.rhou[i], state.rhov[i], state.rhow[i], state.rhoE[i], state.rhoNu[i]}, gas);
        state.u[i] = primitive.u;
        state.v[i] = primitive.v;
        state.w[i] = primitive.w;
        state.p[i] = primitive.p;
        state.nu_tilde[i] = primitive.nu_tilde;
        state.T[i] = primitive.p / (primitive.rho * gas.gas_constant);
    }
}

Primitive sample_primitive(const FlowState& state, Index i) {
    return {state.rho[i], state.u[i], state.v[i], state.w[i], state.p[i], state.nu_tilde[i]};
}

Conservative sample_conservative(const FlowState& state, Index i) {
    return {state.rho[i], state.rhou[i], state.rhov[i], state.rhow[i], state.rhoE[i], state.rhoNu[i]};
}

}  // namespace nssolver
