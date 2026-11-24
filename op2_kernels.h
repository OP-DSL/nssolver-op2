#pragma once

#include <cmath>
#include <math.h>

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
static constexpr int BTYPE_CHARACTERISTIC = 5;
static constexpr int BTYPE_TOTAL_INLET = 6;

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
extern int op2_limiter;
extern int op2_include_viscous;
extern int op2_include_sa;
extern int op2_outlet_backflow;
// SA variants (TMR's names): the compressibility correction, Edwards and Chandra's form.
extern int op2_sa_comp;
extern int op2_sa_edwards;
// limiter 3: the pressure sensor value from which van Albada's limiter starts to act.
extern double op2_limiter_sensor;
// The reference state (rho, rho a, rho a, rho a, rho a^2, rho nu) of the free stream.
extern double op2_scale[6];

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

inline void cleanup_conservative_op2(double *q) {
  const double rho = fmax(fabs(q[0]), 1.0e-12);
  const double momentum_floor = 1.0e-7 * rho;
  for (int m = 1; m <= 3; ++m) {
    if (fabs(q[m]) <= momentum_floor) q[m] = 0.0;
  }
  if (fabs(q[5]) <= 1.0e-14 * rho) q[5] = 0.0;
}

inline void conservative_to_primitive_op2(const double *q, double *p) {
  const double rho = fmax(q[0], 1.0e-12);
  p[0] = rho;
  p[1] = q[1] / rho;
  p[2] = q[2] / rho;
  p[3] = q[3] / rho;
  const double kinetic = 0.5 * (p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
  const double specific_energy = q[4] / rho;
  p[4] = fmax((op2_gamma - 1.0) * rho * (specific_energy - kinetic), 1.0e-12);
  p[5] = q[5] / rho;
}

inline double speed_of_sound_op2(const double *p) { return sqrt(op2_gamma * p[4] / p[0]); }

inline double vec_norm3(const double *v) { return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

inline double normal_velocity_op2(const double *p, const double *unit_normal) {
  return p[1] * unit_normal[0] + p[2] * unit_normal[1] + p[3] * unit_normal[2];
}

inline double dynamic_viscosity_op2(double temperature) {
  const double ratio = temperature / op2_t_ref;
  return op2_mu_ref * pow(ratio, 1.5) * (op2_t_ref + op2_sutherland) / (temperature + op2_sutherland);
}

inline double thermal_conductivity_op2(double temperature) {
  return dynamic_viscosity_op2(temperature) * op2_gamma * cv_op2() / op2_prandtl;
}

inline double eddy_viscosity_op2(const double *p) {
  const double temperature = p[4] / (p[0] * op2_gas_constant);
  const double nu = dynamic_viscosity_op2(temperature) / p[0];
  const double nu_tilde = fmax(p[5], 0.0);
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

inline void rusanov_flux_op2(const double *pl, const double *ql, const double *pr, const double *qr, const double *area_vec,
                             double *flux) {
  const double area = vec_norm3(area_vec);
  if (area <= 1.0e-300) {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = 0.0;
    return;
  }
  const double unit_normal[3] = {area_vec[0] / area, area_vec[1] / area, area_vec[2] / area};
  double f_l[NVAR_OP2];
  double f_r[NVAR_OP2];
  physical_flux_op2(pl, ql, area_vec, f_l);
  physical_flux_op2(pr, qr, area_vec, f_r);

  const double lambda = area * fmax(fabs(normal_velocity_op2(pl, unit_normal)) + speed_of_sound_op2(pl),
                                        fabs(normal_velocity_op2(pr, unit_normal)) + speed_of_sound_op2(pr));
  for (int m = 0; m < NVAR_OP2; ++m) {
    flux[m] = 0.5 * (f_l[m] + f_r[m]) - 0.5 * lambda * (qr[m] - ql[m]);
  }
}

inline void hll_flux_op2(const double *pl, const double *ql, const double *pr, const double *qr, const double *area_vec,
                         double s_l, double s_r, double *flux) {
  double f_l[NVAR_OP2];
  double f_r[NVAR_OP2];
  physical_flux_op2(pl, ql, area_vec, f_l);
  physical_flux_op2(pr, qr, area_vec, f_r);

  if (0.0 <= s_l) {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = f_l[m];
    return;
  }
  if (0.0 >= s_r) {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = f_r[m];
    return;
  }

  const double area = vec_norm3(area_vec);
  const double denom = s_r - s_l;
  const double scale = fmax(fabs(s_l) + fabs(s_r), 1.0);
  if (!isfinite(denom) || fabs(denom) <= 1.0e-12 * scale) {
    rusanov_flux_op2(pl, ql, pr, qr, area_vec, flux);
    return;
  }
  for (int m = 0; m < NVAR_OP2; ++m) {
    flux[m] = (s_r * f_l[m] - s_l * f_r[m] + s_l * s_r * area * (qr[m] - ql[m])) / denom;
  }
}

inline void hllc_flux_op2(const double *pl, const double *ql, const double *pr, const double *qr, const double *area_vec,
                          double *flux) {
  const double area = vec_norm3(area_vec);
  if (area <= 1.0e-300) {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = 0.0;
    return;
  }
  const double unit_normal[3] = {area_vec[0] / area, area_vec[1] / area, area_vec[2] / area};

  const double un_l = normal_velocity_op2(pl, unit_normal);
  const double un_r = normal_velocity_op2(pr, unit_normal);
  const double a_l = speed_of_sound_op2(pl);
  const double a_r = speed_of_sound_op2(pr);

  const double s_l = fmin(un_l - a_l, un_r - a_r);
  const double s_r = fmax(un_l + a_l, un_r + a_r);

  double f_l[NVAR_OP2];
  double f_r[NVAR_OP2];
  physical_flux_op2(pl, ql, area_vec, f_l);
  physical_flux_op2(pr, qr, area_vec, f_r);

  if (0.0 <= s_l) {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = f_l[m];
    return;
  }
  if (0.0 >= s_r) {
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] = f_r[m];
    return;
  }

  const double numerator = pr[4] - pl[4] + ql[0] * un_l * (s_l - un_l) - qr[0] * un_r * (s_r - un_r);
  const double denominator = ql[0] * (s_l - un_l) - qr[0] * (s_r - un_r);
  const double denominator_scale = fmax(fabs(ql[0] * (s_l - un_l)) + fabs(qr[0] * (s_r - un_r)), 1.0);
  if (!isfinite(denominator) || fabs(denominator) <= 1.0e-12 * denominator_scale) {
    hll_flux_op2(pl, ql, pr, qr, area_vec, s_l, s_r, flux);
    return;
  }
  const double s_m = numerator / denominator;
  if (!isfinite(s_m) || s_m <= s_l || s_m >= s_r) {
    hll_flux_op2(pl, ql, pr, qr, area_vec, s_l, s_r, flux);
    return;
  }
  // The left star flux for s_m > 0, the right one below, blended linearly
  // across a band around the contact: the two agree to O(s_m), so the flux
  // stays continuous (and differentiable to the band's width) where a sign
  // test would flip with round-off.
  const double contact_scale = fmax(fabs(s_l) + fabs(s_r) + fabs(s_m), 1.0);
  const double band = 1.0e-8 * contact_scale;
  const double weight_l = fmin(1.0, fmax(0.0, 0.5 + 0.5 * s_m / band));

  const double p_star_l = pl[4] + ql[0] * (s_l - un_l) * (s_m - un_l);
  const double p_star_r = pr[4] + qr[0] * (s_r - un_r) * (s_m - un_r);

  auto star_state = [&](const double *p, const double *q, double s_k, double un_k, double p_star, double *q_star) {
    const double speed_gap = s_k - s_m;
    const double acoustic_gap = s_k - un_k;
    const double gap_scale = fmax(fabs(s_k) + fabs(s_m) + fabs(un_k), 1.0);
    if (fabs(speed_gap) <= 1.0e-12 * gap_scale || fabs(acoustic_gap) <= 1.0e-12 * gap_scale) return false;
    const double factor = q[0] * acoustic_gap / speed_gap;
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
    const double star_energy = factor * (e_total + (s_m - un_k) * (p_star / (q[0] * acoustic_gap) + s_m));
    q_star[0] = factor;
    q_star[1] = factor * star_velocity[0];
    q_star[2] = factor * star_velocity[1];
    q_star[3] = factor * star_velocity[2];
    q_star[4] = star_energy;
    q_star[5] = factor * p[5];
    for (int m = 0; m < NVAR_OP2; ++m) {
      if (!isfinite(q_star[m])) return false;
    }
    return true;
  };

  double q_star_l[NVAR_OP2];
  double q_star_r[NVAR_OP2];
  if ((weight_l > 0.0 && !star_state(pl, ql, s_l, un_l, p_star_l, q_star_l)) ||
      (weight_l < 1.0 && !star_state(pr, qr, s_r, un_r, p_star_r, q_star_r))) {
    hll_flux_op2(pl, ql, pr, qr, area_vec, s_l, s_r, flux);
    return;
  }
  for (int m = 0; m < NVAR_OP2; ++m) {
    const double left = weight_l > 0.0 ? f_l[m] + s_l * area * (q_star_l[m] - ql[m]) : 0.0;
    const double right = weight_l < 1.0 ? f_r[m] + s_r * area * (q_star_r[m] - qr[m]) : 0.0;
    flux[m] = weight_l * left + (1.0 - weight_l) * right;
  }
}

// A node's gradients of u, v, w and T from its primitives' (rho, u, v, w, p,
// nu_tilde) gradient.
inline void viscous_gradients_op2(const double *p, const double *grad, double g[4][3]) {
  const double t = p[4] / (p[0] * op2_gas_constant);
  for (int d = 0; d < 3; ++d) {
    g[0][d] = grad[3 * IDX_U + d];
    g[1][d] = grad[3 * IDX_V + d];
    g[2][d] = grad[3 * IDX_W + d];
    g[3][d] = (grad[3 * IDX_P + d] / op2_gas_constant - t * grad[3 * IDX_RHO + d]) / p[0];
  }
}

// The stress tensor of velocity gradient g (g[i][j] = du_i/dx_j), viscosity mu.
inline void stress_op2(double mu, const double g[4][3], double tau[3][3]) {
  const double divergence = g[0][0] + g[1][1] + g[2][2];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) tau[i][j] = mu * (g[i][j] + g[j][i] - (i == j ? 2.0 / 3.0 * divergence : 0.0));
}

// SA's diffusion (1/sigma) [div((mu + rho nu) grad nu) + cb2 rho |grad nu|^2]
// is taken, as node-based codes do, as (1 + cb2) / sigma div((mu + rho nu)
// grad nu) - cb2 / sigma (mu + rho nu)_node lap(nu), both through compact edge
// differences: every edge's coefficient stays positive, where cb2 |grad nu|^2
// from node gradients feeds on their noise. The second part's coefficient for
// a node of state p on an edge (delta, S); the first is viscous_flux_op2's.
inline double sa_cb2_coefficient_op2(const double *p, const double *delta, const double *S) {
  constexpr double cb2_over_sigma = 0.622 * 1.5;
  const double l2 = fmax(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2], 1.0e-28);
  const double mu = dynamic_viscosity_op2(p[4] / (p[0] * op2_gas_constant));
  return cb2_over_sigma * (mu + p[0] * fmax(p[5], 0.0)) * (delta[0] * S[0] + delta[1] * S[1] + delta[2] * S[2]) /
         l2;
}

// The viscous flux through area vector S from node l to node r (delta = x_r -
// x_l): the full stress tensor, heat flux with the turbulent part, and SA's
// conservative diffusion part (sa_cb2_coefficient_op2). Each gradient of u, v,
// w and T is the edge's average with its component along the edge replaced by
// the edge difference; with zero gradients that difference alone, the compact
// part the first-order Jacobian and the wall's flux take.
inline void viscous_flux_op2(const double *pl, const double *pr, const double *gl, const double *gr,
                             const double *delta, const double *S, double *flux) {
  constexpr double prandtl_turbulent = 0.9;
  constexpr double one_plus_cb2_over_sigma = 1.622 * 1.5;
  const double length = fmax(vec_norm3(delta), 1.0e-14);
  const double e[3] = {delta[0] / length, delta[1] / length, delta[2] / length};
  const double t_l = pl[4] / (pl[0] * op2_gas_constant);
  const double t_r = pr[4] / (pr[0] * op2_gas_constant);
  const double phi_l[4] = {pl[1], pl[2], pl[3], t_l};
  const double phi_r[4] = {pr[1], pr[2], pr[3], t_r};

  double g[4][3], g_l[4][3], g_r[4][3];
  viscous_gradients_op2(pl, gl, g_l);
  viscous_gradients_op2(pr, gr, g_r);
  for (int v = 0; v < 4; ++v) {
    for (int d = 0; d < 3; ++d) g[v][d] = 0.5 * (g_l[v][d] + g_r[v][d]);
    const double along = g[v][0] * e[0] + g[v][1] * e[1] + g[v][2] * e[2];
    const double jump = (phi_r[v] - phi_l[v]) / length - along;
    for (int d = 0; d < 3; ++d) g[v][d] += jump * e[d];
  }

  const double mu_lam = 0.5 * (dynamic_viscosity_op2(t_l) + dynamic_viscosity_op2(t_r));
  const double mu_t = 0.5 * (eddy_viscosity_op2(pl) + eddy_viscosity_op2(pr));
  const double k = op2_gamma * cv_op2() * (mu_lam / op2_prandtl + mu_t / prandtl_turbulent);
  double tau[3][3];
  stress_op2(mu_lam + mu_t, g, tau);

  flux[0] = 0.0;
  flux[4] = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double tau_s = tau[i][0] * S[0] + tau[i][1] * S[1] + tau[i][2] * S[2];
    flux[1 + i] = tau_s;
    flux[4] += 0.5 * (pl[1 + i] + pr[1 + i]) * tau_s;
  }
  flux[4] += k * (g[3][0] * S[0] + g[3][1] * S[1] + g[3][2] * S[2]);
  const double rho_nu = 0.5 * (pl[0] * fmax(pl[5], 0.0) + pr[0] * fmax(pr[5], 0.0));
  flux[5] = one_plus_cb2_over_sigma * (mu_lam + rho_nu) * (pr[5] - pl[5]) / length *
            (e[0] * S[0] + e[1] * S[1] + e[2] * S[2]);
}

inline double scalar_edge_limiter_op2(double projected_delta, double edge_delta) {
  if (fabs(projected_delta) < 1.0e-14) return 1.0;
  if (projected_delta * edge_delta <= 0.0) return 0.0;
  return fmax(0.0, fmin(1.0, edge_delta / projected_delta));
}

inline void reconstruct_primitive_op2(const double *prim, const double *grad, const double *delta_r, double limiter_scale,
                                      double *out) {
  for (int m = 0; m < NPRIM_OP2; ++m) {
    const int base = 3 * m;
    out[m] = prim[m] + limiter_scale * (grad[base + 0] * delta_r[0] + grad[base + 1] * delta_r[1] + grad[base + 2] * delta_r[2]);
  }
  out[0] = fmax(out[0], 1.0e-12);
  out[4] = fmax(out[4], 1.0e-12);
}

// The half-edge increments of a node's primitives toward its neighbour, by
// van Albada's limiter: each of the gradient's projection u and the edge's
// half difference e blend smoothly into (u (e^2 + eps^2) + e (u^2 + eps^2)) /
// (u^2 + e^2 + 2 eps^2), u where they agree, toward 0 where they differ in
// sign. eps is 1% of the free stream's density, speed of sound and pressure:
// differentiable where the edge limiter's minimum switches and holds Newton's
// iterations in a cycle. weight < 1 blends toward (u + e) / 2, the limiter's own
// value for small differences, unlimited (limiter 3 away from shocks).
inline void van_albada_reconstruct_op2(const double *prim, const double *neighbour, const double *grad,
                                       const double *delta_r, double weight, double *out) {
  const double rho = op2_scale[0];
  const double a = op2_scale[1] / op2_scale[0];
  const double scale[NPRIM_OP2] = {rho, a, a, a, op2_scale[4], 1.0};
  for (int m = 0; m < NPRIM_OP2; ++m) {
    const int base = 3 * m;
    const double u = grad[base + 0] * delta_r[0] + grad[base + 1] * delta_r[1] + grad[base + 2] * delta_r[2];
    const double e = 0.5 * (neighbour[m] - prim[m]);
    const double eps2 = 1.0e-4 * scale[m] * scale[m];
    const double limited = (u * (e * e + eps2) + e * (u * u + eps2)) / (u * u + e * e + 2.0 * eps2);
    out[m] = prim[m] + weight * limited + (1.0 - weight) * 0.5 * (u + e);
  }
  out[0] = fmax(out[0], 1.0e-12);
  out[4] = fmax(out[4], 1.0e-12);
}

// The boundary state of a Riemann far field: the invariants un +- 2a/(gamma -
// 1) along the outward normal, the outgoing one from the interior and the
// incoming one from the free stream, and the tangential velocity, entropy and
// nu_tilde from the side the flow comes from.
inline void characteristic_state_op2(const double *interior, const double *n, double *b) {
  const double *inf = op2_freestream;
  const double g = op2_gamma;
  const double un_i = normal_velocity_op2(interior, n);
  const double un_inf = normal_velocity_op2(inf, n);
  const double a_i = speed_of_sound_op2(interior);
  const double a_inf = speed_of_sound_op2(inf);
  if (un_inf <= -a_inf || un_i >= a_i) {
    for (int m = 0; m < NPRIM_OP2; ++m) b[m] = un_i >= a_i ? interior[m] : inf[m];
    return;
  }
  const double r_out = un_i + 2.0 * a_i / (g - 1.0);
  const double r_in = un_inf - 2.0 * a_inf / (g - 1.0);
  const double un = 0.5 * (r_out + r_in);
  const double a = 0.25 * (g - 1.0) * (r_out - r_in);
  // The side's weight blends across |un| < 0.05 a: where the flow runs along
  // the boundary a switch would flip with round-off and stall Newton.
  const double w = fmin(1.0, fmax(0.0, 0.5 + 10.0 * un / a));
  const double entropy = w * interior[4] / pow(interior[0], g) + (1.0 - w) * inf[4] / pow(inf[0], g);
  b[0] = pow(a * a / (g * entropy), 1.0 / (g - 1.0));
  b[4] = b[0] * a * a / g;
  for (int d = 0; d < 3; ++d)
    b[1 + d] = w * (interior[1 + d] - un_i * n[d]) + (1.0 - w) * (inf[1 + d] - un_inf * n[d]) + un * n[d];
  b[5] = w * interior[5] + (1.0 - w) * inf[5];
}

// The boundary state of a subsonic inflow at the free stream's total pressure
// and temperature, along its direction: the static pressure from the interior.
inline void total_inlet_state_op2(const double *interior, double *b) {
  const double *inf = op2_freestream;
  const double g = op2_gamma;
  const double speed_inf = sqrt(inf[1] * inf[1] + inf[2] * inf[2] + inf[3] * inf[3]);
  const double t_inf = inf[4] / (inf[0] * op2_gas_constant);
  const double ratio_inf = 1.0 + 0.5 * (g - 1.0) * speed_inf * speed_inf / (g * op2_gas_constant * t_inf);
  const double t_total = t_inf * ratio_inf;
  const double p_total = inf[4] * pow(ratio_inf, g / (g - 1.0));
  const double p = fmin(interior[4], p_total);
  const double ratio = pow(p_total / p, (g - 1.0) / g);
  const double t = t_total / ratio;
  const double speed = sqrt(fmax(2.0 * g / (g - 1.0) * op2_gas_constant * t * (ratio - 1.0), 0.0));
  b[0] = p / (op2_gas_constant * t);
  b[4] = p;
  for (int d = 0; d < 3; ++d) b[1 + d] = speed_inf > 0.0 ? speed * inf[1 + d] / speed_inf : 0.0;
  b[5] = inf[5];
}

inline void make_boundary_ghost_state_op2(const double *interior, const double *unit_normal, int boundary_type, double *ghost) {
  for (int m = 0; m < NVAR_OP2; ++m) ghost[m] = interior[m];
  if (boundary_type == BTYPE_FARFIELD || boundary_type == BTYPE_INLET) {
    for (int m = 0; m < NVAR_OP2; ++m) ghost[m] = op2_freestream[m];
    return;
  }
  if (boundary_type == BTYPE_CHARACTERISTIC) {
    characteristic_state_op2(interior, unit_normal, ghost);
    return;
  }
  if (boundary_type == BTYPE_TOTAL_INLET) {
    total_inlet_state_op2(interior, ghost);
    return;
  }
  const double un = interior[1] * unit_normal[0] + interior[2] * unit_normal[1] + interior[3] * unit_normal[2];
  if (boundary_type == BTYPE_OUTLET) {
    // With outlet_backflow = freestream, backflow takes the free stream as an
    // inlet would: extrapolated, a reversed outflow can feed itself.
    if (op2_outlet_backflow && un < 0.0) {
      for (int m = 0; m < NVAR_OP2; ++m) ghost[m] = op2_freestream[m];
      return;
    }
    ghost[4] = op2_freestream[4];
    return;
  }

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

// The viscous scheme's strong conditions: the free stream at Dirichlet nodes
// and no velocity at no-slip ones. Slip walls are weak, through the reflected
// ghost's flux: projecting their nodes' velocity would undo part of each
// update the residual asks for (and a node on two slip planes has no single
// normal to project on).
inline void enforce_boundary_node_kernel(const int *is_dirichlet, const int *is_wall, double *q) {
  if (!op2_include_viscous || (!is_dirichlet[0] && !is_wall[0])) {
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
    if (op2_include_sa) p[5] = 0.0;
  }
  primitive_to_conservative_op2(p, q);
}

inline void edge_spectral_kernel(const double *prim_l, const double *prim_r, const double *edge_weight, const double *vol_l,
                                 const double *vol_r, double *spec_l, double *spec_r) {
  const double area = vec_norm3(edge_weight);
  const double unit_normal[3] = {edge_weight[0] / area, edge_weight[1] / area, edge_weight[2] / area};
  const double un_l = fabs(normal_velocity_op2(prim_l, unit_normal));
  const double un_r = fabs(normal_velocity_op2(prim_r, unit_normal));
  spec_l[0] += area * (un_l + speed_of_sound_op2(prim_l));
  spec_r[0] += area * (un_r + speed_of_sound_op2(prim_r));
  if (op2_include_viscous) {
    const double t_l = prim_l[4] / (prim_l[0] * op2_gas_constant);
    const double t_r = prim_r[4] / (prim_r[0] * op2_gas_constant);
    const double mu_l = dynamic_viscosity_op2(t_l) + eddy_viscosity_op2(prim_l);
    const double mu_r = dynamic_viscosity_op2(t_r) + eddy_viscosity_op2(prim_r);
    spec_l[0] += 2.0 * mu_l * area * area / fmax(prim_l[0] * vol_l[0], 1.0e-14);
    spec_r[0] += 2.0 * mu_r * area * area / fmax(prim_r[0] * vol_r[0], 1.0e-14);
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

// Weight of face node k in the average over node j's median sub-face of the
// face's (bi)linear interpolant: exact for linear fields on flat faces.
inline double subface_weight_op2(int nodes, int j, int k) {
  if (k == j) return nodes == 4 ? 9.0 / 16.0 : 22.0 / 36.0;
  if (nodes == 4 && (k - j + 4) % 4 == 2) return 1.0 / 16.0;
  return nodes == 4 ? 3.0 / 16.0 : 7.0 / 36.0;
}

inline void bface_grad_kernel(const int *num_nodes, const double *shares, const double *prim1, const double *prim2,
                              const double *prim3, const double *prim4, double *grad1, double *grad2, double *grad3,
                              double *grad4) {
  // Each node's median sub-face of the face contributes the face's
  // interpolant averaged over it, times the sub-face's area vector, which
  // closes the node's Green-Gauss surface.
  const int nodes = num_nodes[0] == 4 ? 4 : 3;
  double p[4][NPRIM_OP2];
  for (int m = 0; m < NPRIM_OP2; ++m) {
    p[0][m] = prim1[m];
    p[1][m] = prim2[m];
    p[2][m] = prim3[m];
    p[3][m] = prim4[m];
  }
  double g[4][NGRAD_OP2];
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < NGRAD_OP2; ++i) g[j][i] = 0.0;
  for (int j = 0; j < nodes; ++j)
    for (int m = 0; m < NPRIM_OP2; ++m) {
      double value = 0.0;
      for (int k = 0; k < nodes; ++k) value += subface_weight_op2(nodes, j, k) * p[k][m];
      for (int d = 0; d < 3; ++d) g[j][3 * m + d] = value * shares[3 * j + d];
    }
  for (int i = 0; i < NGRAD_OP2; ++i) {
    grad1[i] += g[0][i];
    grad2[i] += g[1][i];
    grad3[i] += g[2][i];
    if (nodes == 4) grad4[i] += g[3][i];
  }
}

// The boundary's share of a node's spectral radius, as the edges' in
// edge_spectral_kernel: each node by its own state and sub-face.
inline void bface_spectral_kernel(const int *num_nodes, const double *shares, const double *prim1, const double *prim2,
                                  const double *prim3, const double *prim4, const double *vol1, const double *vol2,
                                  const double *vol3, const double *vol4, double *spec1, double *spec2, double *spec3,
                                  double *spec4) {
  const int nodes = num_nodes[0] == 4 ? 4 : 3;
  double s[4] = {0.0, 0.0, 0.0, 0.0};
  for (int j = 0; j < nodes; ++j) {
    double p[NPRIM_OP2];
    for (int m = 0; m < NPRIM_OP2; ++m)
      p[m] = j == 0 ? prim1[m] : j == 1 ? prim2[m] : j == 2 ? prim3[m] : prim4[m];
    const double vol = j == 0 ? vol1[0] : j == 1 ? vol2[0] : j == 2 ? vol3[0] : vol4[0];
    const double S[3] = {shares[3 * j], shares[3 * j + 1], shares[3 * j + 2]};
    const double area = vec_norm3(S);
    if (area <= 1.0e-300) continue;
    const double unit_normal[3] = {S[0] / area, S[1] / area, S[2] / area};
    s[j] = area * (fabs(normal_velocity_op2(p, unit_normal)) + speed_of_sound_op2(p));
    if (op2_include_viscous) {
      const double mu = dynamic_viscosity_op2(p[4] / (p[0] * op2_gas_constant)) + eddy_viscosity_op2(p);
      s[j] += 2.0 * mu * area * area / fmax(p[0] * vol, 1.0e-14);
    }
  }
  spec1[0] += s[0];
  spec2[0] += s[1];
  spec3[0] += s[2];
  if (nodes == 4) spec4[0] += s[3];
}

inline void normalize_grad_kernel(const double *volume, double *grad) {
  const double inv_vol = 1.0 / fmax(volume[0], 1.0e-14);
  for (int i = 0; i < NGRAD_OP2; ++i) grad[i] *= inv_vol;
}

inline double edge_limiter_scale_op2(const double *prim_center, const double *prim_neighbor, const double *grad,
                                     const double *delta_r) {
  // The second-order path uses a purely edge-local limiter so the OP2 loop can
  // stay on OP_READ/OP_INC data access patterns without node-global extrema.
  // nu_tilde, advected first-order, does not limit the mean flow.
  double phi = 1.0;
  for (int m = 0; m < IDX_NU; ++m) {
    const int base = 3 * m;
    const double projected = grad[base + 0] * delta_r[0] + grad[base + 1] * delta_r[1] + grad[base + 2] * delta_r[2];
    phi = fmin(phi, scalar_edge_limiter_op2(projected, prim_neighbor[m] - prim_center[m]));
  }
  return phi;
}

// A node's pressure sensor |sum_j (p_j - p_i)| / sum_j (p_j + p_i) over its edges
// (Jameson's), as sums: on the OAT15A grid about 1e-5 in boundary layers, up
// to 3e-4 at the leading edge, 1e-3 to 0.15 through the shock.
inline void zero_sensor_kernel(double *sensor) {
  sensor[0] = 0.0;
  sensor[1] = 0.0;
}

inline void edge_sensor_kernel(const double *prim_l, const double *prim_r, double *sensor_l, double *sensor_r) {
  const double jump = prim_r[IDX_P] - prim_l[IDX_P];
  const double sum = prim_r[IDX_P] + prim_l[IDX_P];
  sensor_l[0] += jump;
  sensor_l[1] += sum;
  sensor_r[0] -= jump;
  sensor_r[1] += sum;
}

// The limiter's weight at a node: 0 up to limiter_sensor, 1 from twice it.
inline double sensor_weight_op2(const double *sensor) {
  const double psi = fabs(sensor[0]) / fmax(sensor[1], 1.0e-300);
  return fmin(1.0, fmax(0.0, psi / op2_limiter_sensor - 1.0));
}

inline void edge_flux_kernel(const double *coord_l, const double *coord_r, const double *prim_l_center, const double *prim_r_center,
                             const double *grad_l, const double *grad_r, const double *sensor_l, const double *sensor_r,
                             const double *edge_weight, double *res_l, double *res_r) {
  // Reconstruct left and right edge states, compute the inviscid HLLC flux, and
  // subtract the viscous flux of the nodal states before accumulating
  // equal-and-opposite residuals at the edge endpoints.
  double prim_l[NPRIM_OP2];
  double prim_r[NPRIM_OP2];
  if (op2_second_order) {
    const double delta_lr[3] = {0.5 * (coord_r[0] - coord_l[0]), 0.5 * (coord_r[1] - coord_l[1]), 0.5 * (coord_r[2] - coord_l[2])};
    const double delta_rl[3] = {-delta_lr[0], -delta_lr[1], -delta_lr[2]};
    if (op2_limiter >= 2) {
      const double weight =
          op2_limiter == 3 ? fmax(sensor_weight_op2(sensor_l), sensor_weight_op2(sensor_r)) : 1.0;
      van_albada_reconstruct_op2(prim_l_center, prim_r_center, grad_l, delta_lr, weight, prim_l);
      van_albada_reconstruct_op2(prim_r_center, prim_l_center, grad_r, delta_rl, weight, prim_r);
    } else {
      const double phi_l = op2_limiter ? edge_limiter_scale_op2(prim_l_center, prim_r_center, grad_l, delta_lr) : 1.0;
      const double phi_r = op2_limiter ? edge_limiter_scale_op2(prim_r_center, prim_l_center, grad_r, delta_rl) : 1.0;
      reconstruct_primitive_op2(prim_l_center, grad_l, delta_lr, phi_l, prim_l);
      reconstruct_primitive_op2(prim_r_center, grad_r, delta_rl, phi_r, prim_r);
    }
    // First-order advection of nu_tilde, as the SA reference solutions use.
    prim_l[IDX_NU] = prim_l_center[IDX_NU];
    prim_r[IDX_NU] = prim_r_center[IDX_NU];
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
    viscous_flux_op2(prim_l_center, prim_r_center, grad_l, grad_r, delta_full, edge_weight, viscous);
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] -= viscous[m];
  }

  for (int m = 0; m < NVAR_OP2; ++m) {
    res_l[m] += flux[m];
    res_r[m] -= flux[m];
  }
  if (op2_include_sa) {
    const double delta[3] = {coord_r[0] - coord_l[0], coord_r[1] - coord_l[1], coord_r[2] - coord_l[2]};
    const double jump = prim_r_center[IDX_NU] - prim_l_center[IDX_NU];
    res_l[IDX_NU] += sa_cb2_coefficient_op2(prim_l_center, delta, edge_weight) * jump;
    res_r[IDX_NU] -= sa_cb2_coefficient_op2(prim_r_center, delta, edge_weight) * jump;
  }
}

// The flux through a node's sub-face S of a boundary face of type btype
// (unit normal n), from the node's own state p and its type-specific ghost,
// and at a no-slip wall the compact viscous flux over distance (adiabatic).
inline void boundary_node_flux_op2(const double *p, const double *S, const double *n, int btype, double distance,
                                   double *flux) {
  double p_ghost[NPRIM_OP2];
  make_boundary_ghost_state_op2(p, n, btype, p_ghost);
  double q[NVAR_OP2];
  double q_ghost[NVAR_OP2];
  primitive_to_conservative_op2(p, q);
  primitive_to_conservative_op2(p_ghost, q_ghost);
  hllc_flux_op2(p, q, p_ghost, q_ghost, S, flux);

  if (op2_include_viscous && btype == BTYPE_NOSLIPWALL) {
    double wall_state[NPRIM_OP2];
    for (int m = 0; m < NPRIM_OP2; ++m) wall_state[m] = p[m];
    wall_state[1] = 0.0;
    wall_state[2] = 0.0;
    wall_state[3] = 0.0;
    const double delta_r[3] = {distance * n[0], distance * n[1], distance * n[2]};
    double viscous[NVAR_OP2];
    const double zero[NGRAD_OP2] = {};
    viscous_flux_op2(p, wall_state, zero, zero, delta_r, S, viscous);
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] -= viscous[m];
  }
}

inline void boundary_flux_kernel(const int *btype, const double *normal, const double *area, const int *num_nodes,
                                 const double *shares, const double *vol1, const double *vol2, const double *vol3,
                                 const double *vol4, const double *prim1, const double *prim2, const double *prim3,
                                 const double *prim4, double *r1, double *r2, double *r3, double *r4) {
  // Each node of a boundary face takes the flux through its median sub-face
  // from its own state and its ghost: a face-averaged state would leave the
  // boundary blind to node-to-node oscillations along it.
  const int nodes = num_nodes[0] == 4 ? 4 : 3;
  const double amag = fmax(area[0], 1.0e-14);
  const double unit_normal[3] = {normal[0] / amag, normal[1] / amag, normal[2] / amag};
  double f[4][NVAR_OP2];
  for (int j = 0; j < 4; ++j)
    for (int m = 0; m < NVAR_OP2; ++m) f[j][m] = 0.0;
  for (int j = 0; j < nodes; ++j) {
    double p[NPRIM_OP2];
    for (int m = 0; m < NPRIM_OP2; ++m)
      p[m] = j == 0 ? prim1[m] : j == 1 ? prim2[m] : j == 2 ? prim3[m] : prim4[m];
    const double vol = j == 0 ? vol1[0] : j == 1 ? vol2[0] : j == 2 ? vol3[0] : vol4[0];
    const double S[3] = {shares[3 * j], shares[3 * j + 1], shares[3 * j + 2]};
    const double distance = fmax(vol / (nodes * fmax(vec_norm3(S), 1.0e-300)), 1.0e-12);
    boundary_node_flux_op2(p, S, unit_normal, btype[0], distance, f[j]);
  }
  for (int m = 0; m < NVAR_OP2; ++m) {
    r1[m] += f[0][m];
    r2[m] += f[1][m];
    r3[m] += f[2][m];
    if (nodes == 4) r4[m] += f[3][m];
  }
}

// A slip wall, a symmetry plane, takes the normal stress -2/3 mu div_t u of
// each node's gradient through its sub-face, div_t the divergence less the
// normal velocity's normal derivative: it balances the interior edges' -2/3
// mu div u of the flow along the plane. (With that derivative, all of n.tau.n,
// it would cancel the viscous damping of a velocity through the plane, which
// then grows on a one-cell extrusion.)
inline void symmetry_stress_kernel(const int *btype, const double *normal, const double *area, const int *num_nodes,
                                   const double *shares, const double *prim1, const double *prim2,
                                   const double *prim3, const double *prim4, const double *grad1, const double *grad2,
                                   const double *grad3, const double *grad4, double *r1, double *r2, double *r3,
                                   double *r4) {
  if (btype[0] != BTYPE_SLIPWALL) return;
  const int nodes = num_nodes[0] == 4 ? 4 : 3;
  const double amag = fmax(area[0], 1.0e-14);
  const double n[3] = {normal[0] / amag, normal[1] / amag, normal[2] / amag};
  double f[4][4];
  for (int j = 0; j < 4; ++j)
    for (int m = 0; m < 4; ++m) f[j][m] = 0.0;
  for (int j = 0; j < nodes; ++j) {
    double p[NPRIM_OP2], grad[NGRAD_OP2];
    for (int m = 0; m < NPRIM_OP2; ++m)
      p[m] = j == 0 ? prim1[m] : j == 1 ? prim2[m] : j == 2 ? prim3[m] : prim4[m];
    for (int i = 0; i < NGRAD_OP2; ++i)
      grad[i] = j == 0 ? grad1[i] : j == 1 ? grad2[i] : j == 2 ? grad3[i] : grad4[i];
    double g[4][3];
    viscous_gradients_op2(p, grad, g);
    double normal_derivative = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) normal_derivative += n[a] * g[a][b] * n[b];
    const double mu = dynamic_viscosity_op2(p[4] / (p[0] * op2_gas_constant)) + eddy_viscosity_op2(p);
    const double tau_nn = -2.0 / 3.0 * mu * (g[0][0] + g[1][1] + g[2][2] - normal_derivative);
    for (int i = 0; i < 3; ++i) {
      const double t = tau_nn * shares[3 * j + i];
      f[j][i] = t;
      f[j][3] += t * p[1 + i];
    }
  }
  for (int i = 0; i < 3; ++i) {
    r1[1 + i] -= f[0][i];
    r2[1 + i] -= f[1][i];
    r3[1 + i] -= f[2][i];
    if (nodes == 4) r4[1 + i] -= f[3][i];
  }
  r1[4] -= f[0][3];
  r2[4] -= f[1][3];
  r3[4] -= f[2][3];
  if (nodes == 4) r4[4] -= f[3][3];
}

// Whether the viscous scheme enforces component m at a node
// (enforce_boundary_node_kernel): bc counts the node's Dirichlet and no-slip
// boundary entries (node_bc_kernel).
inline bool constrained_op2(const int *bc, int m) {
  if (!op2_include_viscous) return false;
  if (bc[0] > 0) return true;
  return bc[1] > 0 && ((m >= 1 && m <= 3) || (m == IDX_NU && op2_include_sa));
}

inline void zero_node_bc_kernel(int *bc) {
  bc[0] = 0;
  bc[1] = 0;
}

inline void node_bc_kernel(const int *is_dirichlet, const int *is_wall, int *bc) {
  bc[0] += is_dirichlet[0] ? 1 : 0;
  bc[1] += is_wall[0] ? 1 : 0;
}

// Enforced components have no residual: the enforcement sets them.
inline void zero_constrained_kernel(const int *bc, double *res) {
  for (int m = 0; m < NVAR_OP2; ++m)
    if (constrained_op2(bc, m)) res[m] = 0.0;
}

// The SA production and destruction of nu_tilde at a node; false where
// nu_tilde <= 0, which has neither.
inline bool sa_terms_op2(const double *prim, const double *grad, const double *wall_dist, double *production,
                         double *destruction) {
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
  const double nu_tilde = fmax(prim[5], 0.0);
  const double d = fmax(wall_dist[0], 1.0e-8);
  if (nu_tilde <= 0.0) return false;

  const double wx = grad[3 * IDX_W + 1] - grad[3 * IDX_V + 2];
  const double wy = grad[3 * IDX_U + 2] - grad[3 * IDX_W + 0];
  const double wz = grad[3 * IDX_V + 0] - grad[3 * IDX_U + 1];
  const double omega = sqrt(wx * wx + wy * wy + wz * wz);
  const double chi = nu_tilde / fmax(nu, 1.0e-14);
  const double chi3 = chi * chi * chi;
  const double fv1 = chi3 / (chi3 + cv1 * cv1 * cv1);
  // du_i/dx_j, for the strain of Edwards' form and the compressibility term.
  double du[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) du[i][j] = grad[3 * (IDX_U + i) + j];
  double s_tilde, r;
  if (op2_sa_edwards) {
    // Edwards and Chandra (TMR's SA-noft2-Edwards): S_tilde from the strain,
    // S^1/2 (1/chi + fv1), and r through tanh.
    const double divergence = du[0][0] + du[1][1] + du[2][2];
    double strain2 = -2.0 / 3.0 * divergence * divergence;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) strain2 += (du[i][j] + du[j][i]) * du[i][j];
    s_tilde = fmax(sqrt(fmax(strain2, 0.0)) * (1.0 / chi + fv1), 1.0e-14);
    r = tanh(nu_tilde / (s_tilde * kappa * kappa * d * d)) / tanh(1.0);
  } else {
    const double fv2 = 1.0 - chi / (1.0 + chi * fv1);
    // S_tilde kept positive smoothly (TMR's SA, note 1(c)): a hard clip of it
    // switches in the buffer layer and holds the iteration in a limit cycle.
    constexpr double c2 = 0.7, c3 = 0.9;
    const double s_bar = nu_tilde * fv2 / (kappa * kappa * d * d);
    s_tilde = fmax(s_bar >= -c2 * omega
                       ? omega + s_bar
                       : omega + omega * (c2 * c2 * omega + c3 * s_bar) / ((c3 - 2.0 * c2) * omega - s_bar),
                   1.0e-14);
    r = fmin(nu_tilde / (s_tilde * kappa * kappa * d * d), 10.0);
  }
  const double g = r + cw2 * (pow(r, 6) - r);
  const double fw = g * pow((1.0 + pow(cw3, 6)) / (pow(g, 6) + pow(cw3, 6)), 1.0 / 6.0);
  production[0] = cb1 * s_tilde * nu_tilde;
  destruction[0] = cw1 * fw * nu_tilde * nu_tilde / (d * d);
  if (op2_sa_comp) {
    // TMR's SA-comp (Shur et al., mixing layers): C5 nu~^2 / a^2 du_i/dx_j du_i/dx_j.
    constexpr double c5 = 3.5;
    double gradient2 = 0.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) gradient2 += du[i][j] * du[i][j];
    destruction[0] += c5 * nu_tilde * nu_tilde * gradient2 * prim[0] / (op2_gamma * prim[4]);
  }
  return true;
}

inline void sa_source_kernel(const double *prim, const double *grad, const double *wall_dist, const double *volume, double *res) {
  // The SA source term is accumulated directly into the turbulence residual
  // component at each node after the flux terms have been assembled.
  if (!op2_include_sa) return;
  double production, destruction;
  if (!sa_terms_op2(prim, grad, wall_dist, &production, &destruction)) return;
  res[5] -= volume[0] * prim[0] * (production - destruction);
}

inline void rk_update_kernel(const int *rk_stage, const double *dt, const double *volume, const double *q0, const double *res,
                             double *q) {
  static const double alpha[4] = {0.25, 1.0 / 3.0, 0.5, 1.0};
  const double factor = alpha[*rk_stage] * dt[0] / volume[0];
  for (int m = 0; m < NVAR_OP2; ++m) q[m] = q0[m] - factor * res[m];
  cleanup_conservative_op2(q);
  double prim[NPRIM_OP2];
  conservative_to_primitive_op2(q, prim);
  if (!isfinite(q[0]) || !isfinite(q[4]) || q[0] < op2_rho_floor || prim[4] < op2_p_floor) {
    for (int m = 0; m < NVAR_OP2; ++m) q[m] = q0[m];
  }
}

inline void residual_norm_kernel(const double *res, double *l2rho, double *l2rhoE, double *linf_rho,
                                 double *l2nu) {
  l2rho[0] += res[0] * res[0];
  l2rhoE[0] += res[4] * res[4];
  linf_rho[0] = fmax(linf_rho[0], fabs(res[0]));
  l2nu[0] += res[5] * res[5];
}

#include "op2_implicit_kernels.h"
