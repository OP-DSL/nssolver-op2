#pragma once

// Kernels of the implicit pseudo-time scheme (docs/OP2_SOLVER.md, section 17):
// each iteration solves (V / dt + dR/dq) dq = -R for the update, R the
// residual the explicit scheme marches with, through OP2's solver layer.
//
// The matrix is a first-order approximation of dR/dq: each edge's analytic
// Euler flux Jacobians with Rusanov dissipation (and a thin-layer viscous
// one), each boundary face's lumped onto the diagonal blocks of its nodes, and
// the SA destruction term. Nodes whose state the viscous scheme enforces keep
// those components: their rows are the identity and their right-hand side 0.
//
// The conservative variables span five orders of magnitude, so the system is
// solved in scaled ones: S^-1 A S x = S^-1 b with dq = S x, S the reference
// state op2_scale, which balances Krylov norms and the matrix-free operator's
// differences.

static constexpr int NBLOCK_OP2 = NVAR_OP2 * NVAR_OP2;

extern double op2_scale[6];

// d(f . S)/dq: the inviscid flux through area vector S by the conservative
// state, at primitive state p.
inline void flux_jacobian_op2(const double *p, const double *S, double *J) {
  const double g1 = op2_gamma - 1.0;
  const double vel[3] = {p[1], p[2], p[3]};
  const double v2 = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
  const double vn = vel[0] * S[0] + vel[1] * S[1] + vel[2] * S[2];
  const double phi = 0.5 * g1 * v2;
  const double h = op2_gamma * p[4] / (g1 * p[0]) + 0.5 * v2;
  for (int k = 0; k < NBLOCK_OP2; ++k) J[k] = 0.0;
  for (int b = 0; b < 3; ++b) J[1 + b] = S[b];
  for (int a = 0; a < 3; ++a) {
    double *row = J + (1 + a) * NVAR_OP2;
    row[0] = phi * S[a] - vel[a] * vn;
    for (int b = 0; b < 3; ++b) row[1 + b] = vel[a] * S[b] - g1 * vel[b] * S[a] + (a == b ? vn : 0.0);
    row[4] = g1 * S[a];
  }
  double *energy = J + 4 * NVAR_OP2;
  energy[0] = vn * (phi - h);
  for (int b = 0; b < 3; ++b) energy[1 + b] = h * S[b] - g1 * vel[b] * vn;
  energy[4] = op2_gamma * vn;
  double *turb = J + 5 * NVAR_OP2;
  turb[0] = -p[5] * vn;
  for (int b = 0; b < 3; ++b) turb[1 + b] = p[5] * S[b];
  turb[5] = vn;
}

// The largest wave speed through area vector S at either state, times its area.
inline double wave_speed_op2(const double *pl, const double *pr, const double *S, double area) {
  const double n[3] = {S[0] / area, S[1] / area, S[2] / area};
  return area * fmax(fabs(normal_velocity_op2(pl, n)) + speed_of_sound_op2(pl),
                     fabs(normal_velocity_op2(pr, n)) + speed_of_sound_op2(pr));
}

// The thin-layer viscous flux's derivative, as a scalar times the identity.
inline double viscous_coefficient_op2(const double *pl, const double *pr, double area, double distance) {
  const double t_l = pl[4] / (pl[0] * op2_gas_constant);
  const double t_r = pr[4] / (pr[0] * op2_gas_constant);
  const double mu = 0.5 * (dynamic_viscosity_op2(t_l) + dynamic_viscosity_op2(t_r)) +
                    0.5 * (eddy_viscosity_op2(pl) + eddy_viscosity_op2(pr));
  const double rho = 0.5 * (pl[0] + pr[0]);
  return fmax(1.0, op2_gamma / op2_prandtl) * mu * area / (rho * fmax(distance, 1.0e-14));
}

// Whether the viscous scheme enforces component m at a node: bc counts the
// node's Dirichlet and no-slip boundary entries (node_bc_kernel).
inline bool constrained_op2(const int *bc, int m) {
  if (!op2_include_viscous) return false;
  if (bc[0] > 0) return true;
  return bc[1] > 0 && m >= 1 && m <= 3;
}

inline void zero_node_bc_kernel(int *bc) {
  bc[0] = 0;
  bc[1] = 0;
}

inline void node_bc_kernel(const int *is_dirichlet, const int *is_wall, int *bc) {
  bc[0] += is_dirichlet[0] ? 1 : 0;
  bc[1] += is_wall[0] ? 1 : 0;
}

// An edge's four blocks: its flux F leaves l and enters r, so row l takes
// dF/dq and row r -dF/dq. decoupled keeps each block's diagonal only.
inline void jac_edge_kernel(const double *coord_l, const double *coord_r, const double *prim_l, const double *prim_r,
                            const double *edge_weight, const int *bc_l, const int *bc_r, const int *decoupled,
                            double *A_ll, double *A_lr, double *A_rl, double *A_rr) {
  const double area = vec_norm3(edge_weight);
  if (area <= 1.0e-300) return;
  double dissipation = 0.5 * wave_speed_op2(prim_l, prim_r, edge_weight, area);
  if (op2_include_viscous) {
    const double delta[3] = {coord_r[0] - coord_l[0], coord_r[1] - coord_l[1], coord_r[2] - coord_l[2]};
    dissipation += viscous_coefficient_op2(prim_l, prim_r, area, vec_norm3(delta));
  }
  double J_l[NBLOCK_OP2];
  double J_r[NBLOCK_OP2];
  flux_jacobian_op2(prim_l, edge_weight, J_l);
  flux_jacobian_op2(prim_r, edge_weight, J_r);
  for (int r = 0; r < NVAR_OP2; ++r) {
    const bool keep_l = !constrained_op2(bc_l, r);
    const bool keep_r = !constrained_op2(bc_r, r);
    for (int c = 0; c < NVAR_OP2; ++c) {
      if (decoupled[0] && r != c) continue;
      const int k = r * NVAR_OP2 + c;
      const double scale = op2_scale[c] / op2_scale[r];
      const double d_l = scale * (0.5 * J_l[k] + (r == c ? dissipation : 0.0));
      const double d_r = scale * (0.5 * J_r[k] - (r == c ? dissipation : 0.0));
      if (keep_l) {
        A_ll[k] += d_l;
        A_lr[k] += d_r;
      }
      if (keep_r) {
        A_rl[k] -= d_l;
        A_rr[k] -= d_r;
      }
    }
  }
}

// A boundary face's flux F by its face-averaged interior state, of which each
// of its n nodes takes F / n, so the block of its nodes (j, k) is D / n^2:
// D = dF/dq_int = (J_int + J_ghost G + lambda (I - G)) / 2 with G the ghost
// state's derivative by the interior one, scaled. jac_bface_row_kernel adds
// the blocks, a row of nodes at a time.
inline void jac_bface_kernel(const int *btype, const double *normal, const double *area, const int *num_nodes,
                             const double *vol1, const double *vol2, const double *vol3, const double *vol4,
                             const double *prim1, const double *prim2, const double *prim3, const double *prim4,
                             const int *decoupled, double *D) {
  const int nodes = num_nodes[0] > 1 ? num_nodes[0] : 1;
  const double inv_nodes = 1.0 / nodes;
  double p_int[NPRIM_OP2];
  for (int m = 0; m < NPRIM_OP2; ++m)
    p_int[m] = inv_nodes * (prim1[m] + prim2[m] + prim3[m] + (nodes == 4 ? prim4[m] : 0.0));

  const double amag = fmax(area[0], 1.0e-14);
  const double unit_normal[3] = {normal[0] / amag, normal[1] / amag, normal[2] / amag};
  double p_ghost[NPRIM_OP2];
  make_boundary_ghost_state_op2(p_int, unit_normal, btype[0], p_ghost);

  double G[NBLOCK_OP2];
  for (int k = 0; k < NBLOCK_OP2; ++k) G[k] = 0.0;
  if (btype[0] == BTYPE_OUTLET) {
    // The ghost keeps rho, u and nu_tilde, and takes the far-field pressure.
    for (int m = 0; m < NVAR_OP2; ++m)
      if (m != 4) G[m * NVAR_OP2 + m] = 1.0;
    const double *u = p_int + 1;
    G[4 * NVAR_OP2 + 0] = -0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    for (int b = 0; b < 3; ++b) G[4 * NVAR_OP2 + 1 + b] = u[b];
  } else if (btype[0] == BTYPE_SLIPWALL || btype[0] == BTYPE_NOSLIPWALL) {
    // The ghost reflects the momentum through the wall.
    for (int m = 0; m < NVAR_OP2; ++m) G[m * NVAR_OP2 + m] = 1.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) G[(1 + a) * NVAR_OP2 + 1 + b] -= 2.0 * unit_normal[a] * unit_normal[b];
    if (btype[0] == BTYPE_NOSLIPWALL) G[5 * NVAR_OP2 + 5] = -1.0;
  }

  double J_int[NBLOCK_OP2];
  double J_ghost[NBLOCK_OP2];
  flux_jacobian_op2(p_int, normal, J_int);
  flux_jacobian_op2(p_ghost, normal, J_ghost);
  const double lambda = wave_speed_op2(p_int, p_ghost, normal, amag);
  double viscous = 0.0;
  if (op2_include_viscous && btype[0] == BTYPE_NOSLIPWALL) {
    const double volume_sum = vol1[0] + vol2[0] + vol3[0] + (nodes == 4 ? vol4[0] : 0.0);
    const double distance = fmax(volume_sum / ((double)nodes * amag), 1.0e-12);
    viscous = viscous_coefficient_op2(p_int, p_int, amag, distance);
  }

  for (int r = 0; r < NVAR_OP2; ++r)
    for (int c = 0; c < NVAR_OP2; ++c) {
      const int k = r * NVAR_OP2 + c;
      double ghost = 0.0;
      for (int j = 0; j < NVAR_OP2; ++j) ghost += J_ghost[r * NVAR_OP2 + j] * G[j * NVAR_OP2 + c];
      double d = 0.5 * (J_int[k] + ghost + lambda * ((r == c ? 1.0 : 0.0) - G[k]));
      if (r == c && r >= 1 && r <= 4) d += viscous;
      D[k] = decoupled[0] && r != c ? 0.0 : d * inv_nodes * inv_nodes * op2_scale[c] / op2_scale[r];
    }
}

// Node `row` of a boundary face: its blocks with each node of the face.
inline void jac_bface_row_kernel(const int *num_nodes, const int *row, const double *D, const int *bc, double *A0,
                                 double *A1, double *A2, double *A3) {
  if (row[0] >= num_nodes[0]) return;
  for (int r = 0; r < NVAR_OP2; ++r) {
    if (constrained_op2(bc, r)) continue;
    for (int c = 0; c < NVAR_OP2; ++c) {
      const int k = r * NVAR_OP2 + c;
      A0[k] += D[k];
      A1[k] += D[k];
      A2[k] += D[k];
      if (num_nodes[0] == 4) A3[k] += D[k];
    }
  }
}

// y += A x through a quad face's diagonals (nodes 0-2 and 1-3), the blocks no
// edge reaches; mat_diag_kernel and mat_edge_kernel take the rest.
inline void mat_face_kernel(const int *num_nodes, const double *A02, const double *A20, const double *A13,
                            const double *A31, const double *x0, const double *x1, const double *x2,
                            const double *x3, double *y0, double *y1, double *y2, double *y3) {
  if (num_nodes[0] != 4) return;
  for (int r = 0; r < NVAR_OP2; ++r) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int c = 0; c < NVAR_OP2; ++c) {
      const int k = r * NVAR_OP2 + c;
      s0 += A02[k] * x2[c];
      s2 += A20[k] * x0[c];
      s1 += A13[k] * x3[c];
      s3 += A31[k] * x1[c];
    }
    y0[r] += s0;
    y1[r] += s1;
    y2[r] += s2;
    y3[r] += s3;
  }
}

// A node's diagonal block: V / dt from the pseudo-time step (spectral / CFL,
// as the explicit step's dt = CFL V / spectral) and the SA destruction's
// derivative; constrained rows the identity.
inline void jac_node_kernel(const int *bc, const double *spectral, const double *prim, const double *grad,
                            const double *wall_dist, const double *volume, const double *cfl, const int *decoupled,
                            double *A) {
  const double time = spectral[0] / cfl[0];
  double sa = 0.0;
  double production, destruction;
  if (op2_include_sa && sa_terms_op2(prim, grad, wall_dist, &production, &destruction))
    sa = 2.0 * volume[0] * destruction / fmax(prim[5], 1.0e-30);
  for (int r = 0; r < NVAR_OP2; ++r) {
    const bool fixed = constrained_op2(bc, r);
    for (int c = 0; c < NVAR_OP2; ++c) {
      if (decoupled[0] && r != c) continue;
      const int k = r * NVAR_OP2 + c;
      if (fixed) {
        if (r == c) A[k] += 1.0;
        continue;
      }
      A[k] += (r == c ? time : 0.0) + (r == 5 && c == 5 ? sa : 0.0);
    }
  }
}

// b = -S^-1 R but on constrained components, and a zero initial guess.
inline void implicit_rhs_kernel(const int *bc, const double *res, double *b, double *dq) {
  for (int m = 0; m < NVAR_OP2; ++m) {
    b[m] = constrained_op2(bc, m) ? 0.0 : -res[m] / op2_scale[m];
    dq[m] = 0.0;
  }
}

// q += relax S dq, kept where the result is admissible as the explicit
// update's.
inline void implicit_update_kernel(const double *dq, const double *relax, double *q, int *rejected) {
  double q_new[NVAR_OP2];
  for (int m = 0; m < NVAR_OP2; ++m) q_new[m] = q[m] + relax[0] * op2_scale[m] * dq[m];
  cleanup_conservative_op2(q_new);
  double prim[NPRIM_OP2];
  conservative_to_primitive_op2(q_new, prim);
  if (!isfinite(q_new[0]) || !isfinite(q_new[4]) || q_new[0] < op2_rho_floor || prim[4] < op2_p_floor) {
    rejected[0] += 1;
    return;
  }
  for (int m = 0; m < NVAR_OP2; ++m) q[m] = q_new[m];
}

// --- Matrix-free operator, scaled: y = V / dt x + S^-1 (R(q + eps S x) - R(q)) / eps

// The squared norms of x and of the scaled state S^-1 q.
inline void sum_squares_kernel(const double *x, const double *q, double *x2, double *q2) {
  for (int m = 0; m < NVAR_OP2; ++m) {
    x2[0] += x[m] * x[m];
    q2[0] += (q[m] / op2_scale[m]) * (q[m] / op2_scale[m]);
  }
}

inline void perturb_kernel(const double *q, const double *x, const double *eps, double *qp) {
  for (int m = 0; m < NVAR_OP2; ++m) qp[m] = q[m] + eps[0] * op2_scale[m] * x[m];
}

inline void matfree_combine_kernel(const int *bc, const double *spectral, const double *cfl, const double *eps,
                                   const double *x, const double *res_p, const double *res, double *y) {
  const double time = spectral[0] / cfl[0];
  for (int m = 0; m < NVAR_OP2; ++m)
    y[m] = constrained_op2(bc, m) ? x[m] : time * x[m] + (res_p[m] - res[m]) / (eps[0] * op2_scale[m]);
}

// --- Par-loop preconditioners, reading the assembled matrix ---------------

// Block Jacobi: the inverse of each diagonal block in solver memory, by
// Gauss-Jordan with partial pivoting.
inline void invert_block_kernel(const double *A, double *dinv) {
  double a[NBLOCK_OP2];
  for (int k = 0; k < NBLOCK_OP2; ++k) {
    a[k] = A[k];
    dinv[k] = 0.0;
  }
  for (int k = 0; k < NVAR_OP2; ++k) dinv[k * NVAR_OP2 + k] = 1.0;
  for (int k = 0; k < NVAR_OP2; ++k) {
    int pivot = k;
    for (int r = k + 1; r < NVAR_OP2; ++r)
      if (fabs(a[r * NVAR_OP2 + k]) > fabs(a[pivot * NVAR_OP2 + k])) pivot = r;
    if (pivot != k)
      for (int c = 0; c < NVAR_OP2; ++c) {
        double t = a[k * NVAR_OP2 + c];
        a[k * NVAR_OP2 + c] = a[pivot * NVAR_OP2 + c];
        a[pivot * NVAR_OP2 + c] = t;
        t = dinv[k * NVAR_OP2 + c];
        dinv[k * NVAR_OP2 + c] = dinv[pivot * NVAR_OP2 + c];
        dinv[pivot * NVAR_OP2 + c] = t;
      }
    const double p = 1.0 / a[k * NVAR_OP2 + k];
    for (int c = 0; c < NVAR_OP2; ++c) {
      a[k * NVAR_OP2 + c] *= p;
      dinv[k * NVAR_OP2 + c] *= p;
    }
    for (int r = 0; r < NVAR_OP2; ++r) {
      if (r == k) continue;
      const double f = a[r * NVAR_OP2 + k];
      for (int c = 0; c < NVAR_OP2; ++c) {
        a[r * NVAR_OP2 + c] -= f * a[k * NVAR_OP2 + c];
        dinv[r * NVAR_OP2 + c] -= f * dinv[k * NVAR_OP2 + c];
      }
    }
  }
}

inline void apply_block_kernel(const double *dinv, const double *r, double *z) {
  for (int i = 0; i < NVAR_OP2; ++i) {
    double s = 0.0;
    for (int j = 0; j < NVAR_OP2; ++j) s += dinv[i * NVAR_OP2 + j] * r[j];
    z[i] = s;
  }
}

// y = A x through the matrix in solver memory: each node's diagonal block,
// then each edge's off-diagonal blocks.
inline void mat_diag_kernel(const double *A, const double *x, double *y) {
  for (int r = 0; r < NVAR_OP2; ++r) {
    double s = 0.0;
    for (int c = 0; c < NVAR_OP2; ++c) s += A[r * NVAR_OP2 + c] * x[c];
    y[r] = s;
  }
}

inline void mat_edge_kernel(const double *A_lr, const double *A_rl, const double *x_l, const double *x_r, double *y_l,
                            double *y_r) {
  for (int r = 0; r < NVAR_OP2; ++r) {
    double s_l = 0.0;
    double s_r = 0.0;
    for (int c = 0; c < NVAR_OP2; ++c) {
      s_l += A_lr[r * NVAR_OP2 + c] * x_r[c];
      s_r += A_rl[r * NVAR_OP2 + c] * x_l[c];
    }
    y_l[r] += s_l;
    y_r[r] += s_r;
  }
}

// One Neumann step: z += Dinv (r - y), y = A z.
inline void neumann_step_kernel(const double *dinv, const double *r, const double *y, double *z) {
  for (int i = 0; i < NVAR_OP2; ++i) {
    double s = 0.0;
    for (int j = 0; j < NVAR_OP2; ++j) s += dinv[i * NVAR_OP2 + j] * (r[j] - y[j]);
    z[i] += s;
  }
}
