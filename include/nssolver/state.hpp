#pragma once

#include <vector>

#include "nssolver/types.hpp"

namespace nssolver {

struct FlowState {
    std::size_t count {};

    std::vector<Real> rho, rhou, rhov, rhow, rhoE, rhoNu;
    std::vector<Real> u, v, w, p, T, nu_tilde;

    std::vector<Real> res_rho, res_rhou, res_rhov, res_rhow, res_rhoE, res_rhoNu;
    std::vector<Real> dt;

    std::vector<Real> rho0, rhou0, rhov0, rhow0, rhoE0, rhoNu0;

    std::vector<Real> grad_rho_x, grad_rho_y, grad_rho_z;
    std::vector<Real> grad_u_x, grad_u_y, grad_u_z;
    std::vector<Real> grad_v_x, grad_v_y, grad_v_z;
    std::vector<Real> grad_w_x, grad_w_y, grad_w_z;
    std::vector<Real> grad_p_x, grad_p_y, grad_p_z;
    std::vector<Real> grad_nu_x, grad_nu_y, grad_nu_z;

    void resize(std::size_t n);
    void zero_residuals();
    void save_baseline();
    void zero_gradients();
};

struct Primitive {
    Real rho {};
    Real u {};
    Real v {};
    Real w {};
    Real p {};
    Real nu_tilde {};
};

struct Conservative {
    Real rho {};
    Real rhou {};
    Real rhov {};
    Real rhow {};
    Real rhoE {};
    Real rhoNu {};
};

}  // namespace nssolver
