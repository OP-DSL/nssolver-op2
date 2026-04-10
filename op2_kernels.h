#pragma once

#include <algorithm>
#include <cmath>

static constexpr int NVAR_OP2 = 6;
static constexpr int NPRIM_OP2 = 6;
static constexpr int NGRAD_OP2 = 18;

static constexpr int IDX_RHO = 0;
static constexpr int IDX_U = 1;
static constexpr int IDX_V = 2;
static constexpr int IDX_W = 3;
static constexpr int IDX_P = 4;
static constexpr int IDX_NU = 5;

static constexpr int BTYPE_FARFIELD = 0;
static constexpr int BTYPE_INLET = 1;
static constexpr int BTYPE_OUTLET = 2;
static constexpr int BTYPE_SLIPWALL = 3;
static constexpr int BTYPE_NOSLIPWALL = 4;

extern double op2_gamma;
extern double op2_gas_constant;
extern double op2_prandtl;
extern double op2_mu_ref;
extern double op2_t_ref;
extern double op2_sutherland;
extern double op2_freestream[6];
extern double op2_rho_floor;
extern double op2_p_floor;
extern int op2_second_order;
extern int op2_include_viscous;
extern int op2_include_sa;

inline double cv_op2() { return op2_gas_constant / (op2_gamma - 1.0); }

inline void primitive_to_conservative_op2(const double *p, double *q) {
  q[0] = p[0];
  q[1] = p[0] * p[1];
  q[2] = p[0] * p[2];
  q[3] = p[0] * p[3];
  const double kinetic = 0.5 * (p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
  const double internal = p[4] / ((op2_gamma - 1.0) * p[0]);
  q[4] = p[0] * (internal + kinetic);
  q[5] = p[0] * p[5];
}

inline void conservative_to_primitive_op2(const double *q, double *p) {
  const double rho = std::max(q[0], 1.0e-12);
  p[0] = rho;
  p[1] = q[1] / rho;
  p[2] = q[2] / rho;
  p[3] = q[3] / rho;
  const double kinetic = 0.5 * (p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
  const double specific_energy = q[4] / rho;
  p[4] = std::max((op2_gamma - 1.0) * rho * (specific_energy - kinetic), 1.0e-12);
  p[5] = q[5] / rho;
}

inline double speed_of_sound_op2(const double *p) { return std::sqrt(op2_gamma * p[4] / p[0]); }

inline double vec_norm3(const double *v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

inline double normal_velocity_op2(const double *p, const double *unit_normal) {
  return p[1] * unit_normal[0] + p[2] * unit_normal[1] + p[3] * unit_normal[2];
}

inline double dynamic_viscosity_op2(double temperature) {
  const double ratio = temperature / op2_t_ref;
  return op2_mu_ref * std::pow(ratio, 1.5) * (op2_t_ref + op2_sutherland) / (temperature + op2_sutherland);
}

inline double thermal_conductivity_op2(double temperature) {
  return dynamic_viscosity_op2(temperature) * op2_gamma * cv_op2() / op2_prandtl;
}

inline double eddy_viscosity_op2(const double *p) {
  const double temperature = p[4] / (p[0] * op2_gas_constant);
  const double nu = dynamic_viscosity_op2(temperature) / p[0];
  const double nu_tilde = std::max(p[5], 0.0);
  if (nu <= 0.0 || nu_tilde <= 0.0) return 0.0;
  constexpr double cv1 = 7.1;
  const double chi = nu_tilde / nu;
  const double chi3 = chi * chi * chi;
  const double fv1 = chi3 / (chi3 + cv1 * cv1 * cv1);
  return p[0] * nu_tilde * fv1;
}

inline void physical_flux_op2(const double *p, const double *q, const double *normal, double *f) {
  const double un = p[1] * normal[0] + p[2] * normal[1] + p[3] * normal[2];
  f[0] = q[0] * un;
  f[1] = q[1] * un + p[4] * normal[0];
  f[2] = q[2] * un + p[4] * normal[1];
  f[3] = q[3] * un + p[4] * normal[2];
  f[4] = (q[4] + p[4]) * un;
  f[5] = q[5] * un;
}

inline void hllc_flux_op2(const double *pl, const double *ql, const double *pr, const double *qr, const double *area_vec,
                          double *flux) {
  const double area = vec_norm3(area_vec);
  const double unit_normal[3] = {area_vec[0] / area, area_vec[1] / area, area_vec[2] / area};

  const double un_l = normal_velocity_op2(pl, unit_normal);
  const double un_r = normal_velocity_op2(pr, unit_normal);
  const double a_l = speed_of_sound_op2(pl);
  const double a_r = speed_of_sound_op2(pr);

  const double s_l = std::min(un_l - a_l, un_r - a_r);
  const double s_r = std::max(un_l + a_l, un_r + a_r);
  const double numerator = pr[4] - pl[4] + ql[0] * un_l * (s_l - un_l) - qr[0] * un_r * (s_r - un_r);
  const double denominator = ql[0] * (s_l - un_l) - qr[0] * (s_r - un_r);
  const double s_m = numerator / denominator;

  double f_l[NVAR_OP2];
  double f_r[NVAR_OP2];
  physical_flux_op2(pl, ql, area_vec, f_l);
  physical_flux_op2(pr, qr, area_vec, f_r);

  const double p_star_l = pl[4] + ql[0] * (s_l - un_l) * (s_m - un_l);
  const double p_star_r = pr[4] + qr[0] * (s_r - un_r) * (s_m - un_r);

  auto star_state = [&](const double *p, const double *q, double s_k, double un_k, double p_star, double *q_star) {
    const double factor = q[0] * (s_k - un_k) / (s_k - s_m);
    const double tangential[3] = {
        p[1] - un_k * unit_normal[0],
        p[2] - un_k * unit_normal[1],
        p[3] - un_k * unit_normal[2],
    };
    const double star_velocity[3] = {
        tangential[0] + s_m * unit_normal[0],
        tangential[1] + s_m * unit_normal[1],
        tangential[2] + s_m * unit_normal[2],
    };
    const double e_total = q[4] / q[0];
    const double star_energy = factor * (e_total + (s_m - un_k) * (p_star / (q[0] * (s_k - un_k)) + s_m));
    q_star[0] = factor;
    q_star[1] = factor * star_velocity[0];
    q_star[2] = factor * star_velocity[1];
    q_star[3] = factor * star_velocity[2];
    q_star[4] = star_energy;
    q_star[5] = factor * p[5];
  };

  double q_star_l[NVAR_OP2];
  double q_star_r[NVAR_OP2];
  star_state(pl, ql, s_l, un_l, p_star_l, q_star_l);
  star_state(pr, qr, s_r, un_r, p_star_r, q_star_r);

  if (0.0 <= s_l) {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = f_l[m];
  } else if (s_l <= 0.0 && 0.0 <= s_m) {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = f_l[m] + s_l * area * (q_star_l[m] - ql[m]);
  } else if (s_m <= 0.0 && 0.0 <= s_r) {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = f_r[m] + s_r * area * (q_star_r[m] - qr[m]);
  } else {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = f_r[m];
  }
}

inline void thin_layer_viscous_flux_op2(const double *pl, const double *pr, const double *delta_r, const double *area_vector,
                                        double *flux) {
  const double distance = std::max(vec_norm3(delta_r), 1.0e-14);
  const double area = vec_norm3(area_vector);
  const double t_l = pl[4] / (pl[0] * op2_gas_constant);
  const double t_r = pr[4] / (pr[0] * op2_gas_constant);
  const double mu = 0.5 * (dynamic_viscosity_op2(t_l) + dynamic_viscosity_op2(t_r)) +
                    0.5 * (eddy_viscosity_op2(pl) + eddy_viscosity_op2(pr));
  const double k = 0.5 * (thermal_conductivity_op2(t_l) + thermal_conductivity_op2(t_r));
  const double du_dn = (pr[1] - pl[1]) / distance;
  const double dv_dn = (pr[2] - pl[2]) / distance;
  const double dw_dn = (pr[3] - pl[3]) / distance;
  const double dT_dn = (t_r - t_l) / distance;
  const double dnu_dn = (pr[5] - pl[5]) / distance;
  const double tau_xn = mu * du_dn;
  const double tau_yn = mu * dv_dn;
  const double tau_zn = mu * dw_dn;
  const double um = 0.5 * (pl[1] + pr[1]);
  const double vm = 0.5 * (pl[2] + pr[2]);
  const double wm = 0.5 * (pl[3] + pr[3]);
  const double heat_flux = -k * dT_dn;

  flux[0] = 0.0;
  flux[1] = area * tau_xn;
  flux[2] = area * tau_yn;
  flux[3] = area * tau_zn;
  flux[4] = area * (um * tau_xn + vm * tau_yn + wm * tau_zn - heat_flux);
  flux[5] = area * mu * dnu_dn;
}

inline double scalar_edge_limiter_op2(double projected_delta, double edge_delta) {
  if (std::abs(projected_delta) < 1.0e-14) return 1.0;
  if (projected_delta * edge_delta <= 0.0) return 0.0;
  return std::clamp(edge_delta / projected_delta, 0.0, 1.0);
}

inline void reconstruct_primitive_op2(const double *prim, const double *grad, const double *delta_r, double limiter_scale,
                                      double *out) {
  for (int m = 0; m < NPRIM_OP2; ++m) {
    const int base = 3 * m;
    out[m] = prim[m] + limiter_scale * (grad[base + 0] * delta_r[0] + grad[base + 1] * delta_r[1] + grad[base + 2] * delta_r[2]);
  }
  out[0] = std::max(out[0], 1.0e-12);
  out[4] = std::max(out[4], 1.0e-12);
}

inline void make_boundary_ghost_state_op2(const double *interior, const double *unit_normal, int boundary_type, double *ghost) {
  for (int m = 0; m < NVAR_OP2; ++m) ghost[m] = interior[m];
  if (boundary_type == BTYPE_FARFIELD || boundary_type == BTYPE_INLET) {
    for (int m = 0; m < NVAR_OP2; ++m) ghost[m] = op2_freestream[m];
    return;
  }
  if (boundary_type == BTYPE_OUTLET) {
    ghost[4] = op2_freestream[4];
    return;
  }

  const double un = interior[1] * unit_normal[0] + interior[2] * unit_normal[1] + interior[3] * unit_normal[2];
  ghost[1] = interior[1] - 2.0 * un * unit_normal[0];
  ghost[2] = interior[2] - 2.0 * un * unit_normal[1];
  ghost[3] = interior[3] - 2.0 * un * unit_normal[2];
  if (boundary_type == BTYPE_NOSLIPWALL) {
    ghost[5] = -interior[5];
  }
}

inline void initialize_q_kernel(double *q) { primitive_to_conservative_op2(op2_freestream, q); }

inline void copy_q_kernel(const double *q, double *q0) {
  for (int m = 0; m < NVAR_OP2; ++m) q0[m] = q[m];
}

inline void zero_var_kernel(double *values) {
  for (int m = 0; m < NVAR_OP2; ++m) values[m] = 0.0;
}

inline void zero_grad_kernel(double *values) {
  for (int i = 0; i < NGRAD_OP2; ++i) values[i] = 0.0;
}

inline void zero_scalar_kernel(double *value) { value[0] = 0.0; }

inline void q_to_primitive_kernel(const double *q, double *prim) { conservative_to_primitive_op2(q, prim); }

inline void enforce_boundary_node_kernel(const int *is_dirichlet, const int *is_wall, const int *is_slip,
                                         const double *normal, double *q) {
  if (!op2_include_viscous) {
    return;
  }
  double p[NPRIM_OP2];
  conservative_to_primitive_op2(q, p);
  if (is_dirichlet[0]) {
    for (int m = 0; m < NPRIM_OP2; ++m) p[m] = op2_freestream[m];
  }
  if (is_wall[0]) {
    p[1] = 0.0;
    p[2] = 0.0;
    p[3] = 0.0;
  } else if (is_slip[0]) {
    const double nmag = vec_norm3(normal);
    if (nmag > 1.0e-14) {
      const double un = (p[1] * normal[0] + p[2] * normal[1] + p[3] * normal[2]) / nmag;
      const double unit_normal[3] = {normal[0] / nmag, normal[1] / nmag, normal[2] / nmag};
      p[1] -= un * unit_normal[0];
      p[2] -= un * unit_normal[1];
      p[3] -= un * unit_normal[2];
    }
  }
  primitive_to_conservative_op2(p, q);
}

inline void edge_spectral_kernel(const double *prim_l, const double *prim_r, const double *edge_weight, const double *vol_l,
                                 const double *vol_r, double *spec_l, double *spec_r) {
  const double area = vec_norm3(edge_weight);
  const double unit_normal[3] = {edge_weight[0] / area, edge_weight[1] / area, edge_weight[2] / area};
  const double un_l = std::abs(normal_velocity_op2(prim_l, unit_normal));
  const double un_r = std::abs(normal_velocity_op2(prim_r, unit_normal));
  spec_l[0] += area * (un_l + speed_of_sound_op2(prim_l));
  spec_r[0] += area * (un_r + speed_of_sound_op2(prim_r));
  if (op2_include_viscous) {
    const double t_l = prim_l[4] / (prim_l[0] * op2_gas_constant);
    const double t_r = prim_r[4] / (prim_r[0] * op2_gas_constant);
    const double mu_l = dynamic_viscosity_op2(t_l);
    const double mu_r = dynamic_viscosity_op2(t_r);
    spec_l[0] += 2.0 * mu_l * area * area / std::max(prim_l[0] * vol_l[0], 1.0e-14);
    spec_r[0] += 2.0 * mu_r * area * area / std::max(prim_r[0] * vol_r[0], 1.0e-14);
  }
}

inline void compute_dt_scaled_kernel(const double *volume, const double *spectral, const double *cfl, double *dt) {
  dt[0] = cfl[0] * volume[0] / (spectral[0] + 1.0e-14);
}

inline void edge_grad_kernel(const double *prim_l, const double *prim_r, const double *edge_weight, double *grad_l, double *grad_r) {
  // Green-Gauss edge contribution: the edge average is multiplied by the dual
  // face vector and added with opposite signs to the endpoint control volumes.
  for (int m = 0; m < NPRIM_OP2; ++m) {
    const double avg = 0.5 * (prim_l[m] + prim_r[m]);
    const int base = 3 * m;
    grad_l[base + 0] += avg * edge_weight[0];
    grad_l[base + 1] += avg * edge_weight[1];
    grad_l[base + 2] += avg * edge_weight[2];
    grad_r[base + 0] -= avg * edge_weight[0];
    grad_r[base + 1] -= avg * edge_weight[1];
    grad_r[base + 2] -= avg * edge_weight[2];
  }
}

inline void bface_grad_kernel(const double *normal, const double *prim1, const double *prim2, const double *prim3, const double *prim4,
                              double *grad1, double *grad2, double *grad3, double *grad4) {
  // Boundary faces contribute a face-averaged primitive value projected along
  // the outward face normal, shared equally across the four incident nodes.
  double face_prim[NPRIM_OP2];
  for (int m = 0; m < NPRIM_OP2; ++m) {
    face_prim[m] = 0.25 * (prim1[m] + prim2[m] + prim3[m] + prim4[m]);
    const int base = 3 * m;
    const double cx = 0.25 * normal[0] * face_prim[m];
    const double cy = 0.25 * normal[1] * face_prim[m];
    const double cz = 0.25 * normal[2] * face_prim[m];
    grad1[base + 0] += cx; grad1[base + 1] += cy; grad1[base + 2] += cz;
    grad2[base + 0] += cx; grad2[base + 1] += cy; grad2[base + 2] += cz;
    grad3[base + 0] += cx; grad3[base + 1] += cy; grad3[base + 2] += cz;
    grad4[base + 0] += cx; grad4[base + 1] += cy; grad4[base + 2] += cz;
  }
}

inline void normalize_grad_kernel(const double *volume, double *grad) {
  const double inv_vol = 1.0 / std::max(volume[0], 1.0e-14);
  for (int i = 0; i < NGRAD_OP2; ++i) grad[i] *= inv_vol;
}

inline double edge_limiter_scale_op2(const double *prim_center, const double *prim_neighbor, const double *grad,
                                     const double *delta_r) {
  // The second-order path uses a purely edge-local limiter so the OP2 loop can
  // stay on OP_READ/OP_INC data access patterns without node-global extrema.
  double phi = 1.0;
  for (int m = 0; m < NPRIM_OP2; ++m) {
    const int base = 3 * m;
    const double projected = grad[base + 0] * delta_r[0] + grad[base + 1] * delta_r[1] + grad[base + 2] * delta_r[2];
    phi = std::min(phi, scalar_edge_limiter_op2(projected, prim_neighbor[m] - prim_center[m]));
  }
  return phi;
}

inline void edge_flux_kernel(const double *coord_l, const double *coord_r, const double *prim_l_center, const double *prim_r_center,
                             const double *grad_l, const double *grad_r, const double *edge_weight, double *res_l, double *res_r) {
  // Reconstruct left and right edge states, compute the inviscid HLLC flux, and
  // optionally subtract the thin-layer viscous correction before accumulating
  // equal-and-opposite residuals at the edge endpoints.
  double prim_l[NPRIM_OP2];
  double prim_r[NPRIM_OP2];
  if (op2_second_order) {
    const double delta_lr[3] = {0.5 * (coord_r[0] - coord_l[0]), 0.5 * (coord_r[1] - coord_l[1]), 0.5 * (coord_r[2] - coord_l[2])};
    const double delta_rl[3] = {-delta_lr[0], -delta_lr[1], -delta_lr[2]};
    reconstruct_primitive_op2(prim_l_center, grad_l, delta_lr, edge_limiter_scale_op2(prim_l_center, prim_r_center, grad_l, delta_lr), prim_l);
    reconstruct_primitive_op2(prim_r_center, grad_r, delta_rl, edge_limiter_scale_op2(prim_r_center, prim_l_center, grad_r, delta_rl), prim_r);
  } else {
    for (int m = 0; m < NPRIM_OP2; ++m) {
      prim_l[m] = prim_l_center[m];
      prim_r[m] = prim_r_center[m];
    }
  }

  double ql[NVAR_OP2];
  double qr[NVAR_OP2];
  primitive_to_conservative_op2(prim_l, ql);
  primitive_to_conservative_op2(prim_r, qr);
  double flux[NVAR_OP2];
  hllc_flux_op2(prim_l, ql, prim_r, qr, edge_weight, flux);

  if (op2_include_viscous) {
    const double delta_full[3] = {coord_r[0] - coord_l[0], coord_r[1] - coord_l[1], coord_r[2] - coord_l[2]};
    double viscous[NVAR_OP2];
    thin_layer_viscous_flux_op2(prim_l, prim_r, delta_full, edge_weight, viscous);
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] -= viscous[m];
  }

  for (int m = 0; m < NVAR_OP2; ++m) {
    res_l[m] += flux[m];
    res_r[m] -= flux[m];
  }
}

inline void boundary_flux_kernel(const int *btype, const double *normal, const double *area, const double *vol1, const double *vol2,
                                 const double *vol3, const double *vol4, const double *prim1, const double *prim2, const double *prim3,
                                 const double *prim4, double *r1, double *r2, double *r3, double *r4) {
  // Boundary faces use a face-averaged interior state, a type-specific ghost
  // state, and an equal split of the resulting flux back to the face nodes.
  double p_int[NPRIM_OP2] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int m = 0; m < NPRIM_OP2; ++m) p_int[m] = 0.25 * (prim1[m] + prim2[m] + prim3[m] + prim4[m]);

  const double amag = std::max(area[0], 1.0e-14);
  const double unit_normal[3] = {normal[0] / amag, normal[1] / amag, normal[2] / amag};
  double p_ghost[NPRIM_OP2];
  make_boundary_ghost_state_op2(p_int, unit_normal, btype[0], p_ghost);
  double q_int[NVAR_OP2];
  double q_ghost[NVAR_OP2];
  primitive_to_conservative_op2(p_int, q_int);
  primitive_to_conservative_op2(p_ghost, q_ghost);

  double flux[NVAR_OP2];
  hllc_flux_op2(p_int, q_int, p_ghost, q_ghost, normal, flux);

  if (op2_include_viscous && btype[0] == BTYPE_NOSLIPWALL) {
    double wall_state[NPRIM_OP2];
    for (int m = 0; m < NPRIM_OP2; ++m) wall_state[m] = p_int[m];
    wall_state[1] = 0.0;
    wall_state[2] = 0.0;
    wall_state[3] = 0.0;
    const double distance = std::max((vol1[0] + vol2[0] + vol3[0] + vol4[0]) / (4.0 * amag), 1.0e-12);
    const double delta_r[3] = {distance * unit_normal[0], distance * unit_normal[1], distance * unit_normal[2]};
    double viscous[NVAR_OP2];
    thin_layer_viscous_flux_op2(p_int, wall_state, delta_r, normal, viscous);
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] -= viscous[m];
  }

  for (int m = 0; m < NVAR_OP2; ++m) {
    const double share = 0.25 * flux[m];
    r1[m] += share;
    r2[m] += share;
    r3[m] += share;
    r4[m] += share;
  }
}

inline void sa_source_kernel(const double *prim, const double *grad, const double *wall_dist, const double *volume, double *res) {
  // The SA source term is accumulated directly into the turbulence residual
  // component at each node after the flux terms have been assembled.
  if (!op2_include_sa) return;
  constexpr double cb1 = 0.1355;
  constexpr double cb2 = 0.622;
  constexpr double sigma = 2.0 / 3.0;
  constexpr double kappa = 0.41;
  constexpr double cw2 = 0.3;
  constexpr double cw3 = 2.0;
  constexpr double cv1 = 7.1;
  const double cw1 = cb1 / (kappa * kappa) + (1.0 + cb2) / sigma;

  const double temperature = prim[4] / (prim[0] * op2_gas_constant);
  const double nu = dynamic_viscosity_op2(temperature) / prim[0];
  const double nu_tilde = std::max(prim[5], 0.0);
  const double d = std::max(wall_dist[0], 1.0e-8);
  if (nu_tilde <= 0.0) return;

  const double wx = grad[3 * IDX_W + 1] - grad[3 * IDX_V + 2];
  const double wy = grad[3 * IDX_U + 2] - grad[3 * IDX_W + 0];
  const double wz = grad[3 * IDX_V + 0] - grad[3 * IDX_U + 1];
  const double omega = std::sqrt(wx * wx + wy * wy + wz * wz);
  const double chi = nu_tilde / std::max(nu, 1.0e-14);
  const double chi3 = chi * chi * chi;
  const double fv1 = chi3 / (chi3 + cv1 * cv1 * cv1);
  const double fv2 = 1.0 - chi / (1.0 + chi * fv1);
  const double s_tilde = std::max(omega + nu_tilde * fv2 / (kappa * kappa * d * d), 0.3 * omega + 1.0e-14);
  const double r = std::min(nu_tilde / (s_tilde * kappa * kappa * d * d), 10.0);
  const double g = r + cw2 * (std::pow(r, 6) - r);
  const double fw = g * std::pow((1.0 + std::pow(cw3, 6)) / (std::pow(g, 6) + std::pow(cw3, 6)), 1.0 / 6.0);
  const double production = cb1 * s_tilde * nu_tilde;
  const double destruction = cw1 * fw * nu_tilde * nu_tilde / (d * d);
  res[5] -= volume[0] * prim[0] * (production - destruction);
}

inline void rk_update_kernel(const int *rk_stage, const double *dt, const double *volume, const double *q0, const double *res,
                             double *q) {
  static const double alpha[4] = {0.25, 1.0 / 3.0, 0.5, 1.0};
  const double factor = alpha[*rk_stage] * dt[0] / volume[0];
  for (int m = 0; m < NVAR_OP2; ++m) q[m] = q0[m] - factor * res[m];
  double prim[NPRIM_OP2];
  conservative_to_primitive_op2(q, prim);
  if (!std::isfinite(q[0]) || !std::isfinite(q[4]) || q[0] < op2_rho_floor || prim[4] < op2_p_floor) {
    for (int m = 0; m < NVAR_OP2; ++m) q[m] = q0[m];
  }
}

inline void residual_norm_kernel(const double *res, double *l2rho, double *l2rhoE, double *linf_rho) {
  l2rho[0] += res[0] * res[0];
  l2rhoE[0] += res[4] * res[4];
  linf_rho[0] = std::max(linf_rho[0], std::abs(res[0]));
}
