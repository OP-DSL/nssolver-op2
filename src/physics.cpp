#include "nssolver/physics.hpp"

#include <cmath>

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

Real dynamic_viscosity(Real temperature, const GasModel& gas) {
    const Real ratio = temperature / gas.t_ref;
    return gas.mu_ref * std::pow(ratio, 1.5) * (gas.t_ref + gas.sutherland) / (temperature + gas.sutherland);
}

}  // namespace nssolver
