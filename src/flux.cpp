#include "nssolver/flux.hpp"

#include <algorithm>
#include <cmath>

namespace nssolver {

namespace {

Real normal_velocity(const Primitive& primitive, const Vec3& unit_normal) {
    return primitive.u * unit_normal.x + primitive.v * unit_normal.y + primitive.w * unit_normal.z;
}

FluxArray conservative_array(const Conservative& q) {
    return {q.rho, q.rhou, q.rhov, q.rhow, q.rhoE, q.rhoNu};
}

}  // namespace

FluxArray physical_flux(const Primitive& primitive, const Conservative& conservative, const Vec3& normal, const GasModel&) {
    const Real un = primitive.u * normal.x + primitive.v * normal.y + primitive.w * normal.z;
    return {
        conservative.rho * un,
        conservative.rhou * un + primitive.p * normal.x,
        conservative.rhov * un + primitive.p * normal.y,
        conservative.rhow * un + primitive.p * normal.z,
        (conservative.rhoE + primitive.p) * un,
        conservative.rhoNu * un,
    };
}

FluxArray hllc_flux(const Primitive& left_p,
                    const Conservative& left_q,
                    const Primitive& right_p,
                    const Conservative& right_q,
                    const Vec3& area_vector,
                    const GasModel& gas) {
    const Real area = norm(area_vector);
    const Vec3 unit_normal = area_vector / area;

    const Real un_l = normal_velocity(left_p, unit_normal);
    const Real un_r = normal_velocity(right_p, unit_normal);
    const Real a_l = speed_of_sound(left_p, gas);
    const Real a_r = speed_of_sound(right_p, gas);

    const Real s_l = std::min(un_l - a_l, un_r - a_r);
    const Real s_r = std::max(un_l + a_l, un_r + a_r);

    const Real numerator = right_p.p - left_p.p + left_q.rho * un_l * (s_l - un_l) - right_q.rho * un_r * (s_r - un_r);
    const Real denominator = left_q.rho * (s_l - un_l) - right_q.rho * (s_r - un_r);
    const Real s_m = numerator / denominator;

    const FluxArray f_l = physical_flux(left_p, left_q, area_vector, gas);
    const FluxArray f_r = physical_flux(right_p, right_q, area_vector, gas);
    const FluxArray q_l = conservative_array(left_q);
    const FluxArray q_r = conservative_array(right_q);

    const Real p_star_l = left_p.p + left_q.rho * (s_l - un_l) * (s_m - un_l);
    const Real p_star_r = right_p.p + right_q.rho * (s_r - un_r) * (s_m - un_r);

    auto star_state = [&](const Primitive& p, const Conservative& q, Real s_k, Real un_k, Real p_star) {
        const Real factor = q.rho * (s_k - un_k) / (s_k - s_m);
        const Vec3 velocity {p.u, p.v, p.w};
        const Vec3 tangential = velocity - un_k * unit_normal;
        const Vec3 star_velocity = tangential + s_m * unit_normal;
        const Real e_total = q.rhoE / q.rho;
        const Real star_energy = factor * (e_total + (s_m - un_k) * (p_star / (q.rho * (s_k - un_k)) + s_m));

        return FluxArray {
            factor,
            factor * star_velocity.x,
            factor * star_velocity.y,
            factor * star_velocity.z,
            star_energy,
            factor * p.nu_tilde,
        };
    };

    const FluxArray q_star_l = star_state(left_p, left_q, s_l, un_l, p_star_l);
    const FluxArray q_star_r = star_state(right_p, right_q, s_r, un_r, p_star_r);

    FluxArray flux {};
    if (0.0 <= s_l) {
        flux = f_l;
    } else if (s_l <= 0.0 && 0.0 <= s_m) {
        for (int m = 0; m < 6; ++m) {
            flux[m] = f_l[m] + s_l * area * (q_star_l[m] - q_l[m]);
        }
    } else if (s_m <= 0.0 && 0.0 <= s_r) {
        for (int m = 0; m < 6; ++m) {
            flux[m] = f_r[m] + s_r * area * (q_star_r[m] - q_r[m]);
        }
    } else {
        flux = f_r;
    }
    return flux;
}

FluxArray thin_layer_viscous_flux(const Primitive& left_p,
                                  const Primitive& right_p,
                                  const Vec3& delta_r,
                                  const Vec3& area_vector,
                                  const GasModel& gas) {
    const Real distance = std::max(norm(delta_r), 1.0e-14);
    const Real area = norm(area_vector);
    const Real mu = 0.5 * (dynamic_viscosity(left_p.p / (left_p.rho * gas.gas_constant), gas) +
                           dynamic_viscosity(right_p.p / (right_p.rho * gas.gas_constant), gas)) +
                    0.5 * (eddy_viscosity(left_p, gas) + eddy_viscosity(right_p, gas));
    const Real k = 0.5 * (thermal_conductivity(left_p.p / (left_p.rho * gas.gas_constant), gas) +
                          thermal_conductivity(right_p.p / (right_p.rho * gas.gas_constant), gas));

    const Real du_dn = (right_p.u - left_p.u) / distance;
    const Real dv_dn = (right_p.v - left_p.v) / distance;
    const Real dw_dn = (right_p.w - left_p.w) / distance;
    const Real dT_dn = ((right_p.p / (right_p.rho * gas.gas_constant)) -
                        (left_p.p / (left_p.rho * gas.gas_constant))) /
                       distance;
    const Real dnu_dn = (right_p.nu_tilde - left_p.nu_tilde) / distance;

    const Real tau_xn = mu * du_dn;
    const Real tau_yn = mu * dv_dn;
    const Real tau_zn = mu * dw_dn;
    const Primitive midpoint {
        0.5 * (left_p.rho + right_p.rho),
        0.5 * (left_p.u + right_p.u),
        0.5 * (left_p.v + right_p.v),
        0.5 * (left_p.w + right_p.w),
        0.5 * (left_p.p + right_p.p),
        0.5 * (left_p.nu_tilde + right_p.nu_tilde),
    };
    const Real heat_flux = -k * dT_dn;

    return {
        0.0,
        area * tau_xn,
        area * tau_yn,
        area * tau_zn,
        area * (midpoint.u * tau_xn + midpoint.v * tau_yn + midpoint.w * tau_zn - heat_flux),
        area * mu * dnu_dn,
    };
}

Primitive make_boundary_ghost_state(const Primitive& interior,
                                    const Primitive& freestream,
                                    const Vec3& unit_normal,
                                    BoundaryType boundary_type) {
    if (boundary_type == BoundaryType::Farfield || boundary_type == BoundaryType::Inlet) {
        return freestream;
    }
    if (boundary_type == BoundaryType::Outlet) {
        Primitive ghost = interior;
        ghost.p = freestream.p;
        return ghost;
    }

    const Real un = interior.u * unit_normal.x + interior.v * unit_normal.y + interior.w * unit_normal.z;
    Primitive ghost = interior;
    if (boundary_type == BoundaryType::SlipWall || boundary_type == BoundaryType::NoSlipWall) {
        ghost.u = interior.u - 2.0 * un * unit_normal.x;
        ghost.v = interior.v - 2.0 * un * unit_normal.y;
        ghost.w = interior.w - 2.0 * un * unit_normal.z;
        if (boundary_type == BoundaryType::NoSlipWall) {
            ghost.nu_tilde = -interior.nu_tilde;
        }
    }
    return ghost;
}

}  // namespace nssolver
