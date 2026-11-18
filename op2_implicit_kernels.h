#pragma once

// Kernels of the implicit pseudo-time scheme (docs/OP2_SOLVER.md, section 17):
// each iteration solves (V / dt + dR/dq) dq = -R for the update, R the
// residual the explicit scheme marches with, through OP2's solver layer.
//
// The matrix is dR/dq of the first-order residual: each edge's flux (HLLC
// and thin-layer viscous) and each boundary sub-face's, differentiated by
// one-sided differences of the same flux functions, and the SA destruction
// term. Nodes whose state the viscous scheme enforces keep those components:
// their rows are the identity and their right-hand side 0.
//
// The conservative variables span five orders of magnitude, so the system is
// solved in scaled ones: S^-1 A S x = S^-1 b with dq = S x, S the reference
// state op2_scale, which balances Krylov norms and the differences.

static constexpr int NBLOCK_OP2 = NVAR_OP2 * NVAR_OP2;

extern double op2_scale[6];

// The step of the one-sided differences in component c of the state q.
inline double jacobian_step_op2(const double *q, int c) {
  return 1.0e-7 * fmax(fabs(q[c]), op2_scale[c]);
}

// An edge's first-order flux from primitive states pl to pr through area
// vector S: HLLC less the thin-layer viscous flux over delta.
inline void edge_flux_first_order_op2(const double *pl, const double *pr, const double *delta, const double *S,
                                      double *flux) {
  double ql[NVAR_OP2];
  double qr[NVAR_OP2];
  primitive_to_conservative_op2(pl, ql);
  primitive_to_conservative_op2(pr, qr);
  hllc_flux_op2(pl, ql, pr, qr, S, flux);
  if (op2_include_viscous) {
    double viscous[NVAR_OP2];
    thin_layer_viscous_flux_op2(pl, pr, delta, S, viscous);
    for (int m = 0; m < NVAR_OP2; ++m) flux[m] -= viscous[m];
  }
}

// An edge's four blocks: its flux F leaves l and enters r, so row l takes
// dF/dq and row r -dF/dq. decoupled keeps each block's diagonal only.
inline void jac_edge_kernel(const double *coord_l, const double *coord_r, const double *prim_l, const double *prim_r,
                            const double *edge_weight, const int *bc_l, const int *bc_r, const int *decoupled,
                            double *A_ll, double *A_lr, double *A_rl, double *A_rr) {
  if (vec_norm3(edge_weight) <= 1.0e-300) return;
  const double S[3] = {edge_weight[0], edge_weight[1], edge_weight[2]};
  const double delta[3] = {coord_r[0] - coord_l[0], coord_r[1] - coord_l[1], coord_r[2] - coord_l[2]};
  double pl[NPRIM_OP2];
  double pr[NPRIM_OP2];
  for (int m = 0; m < NPRIM_OP2; ++m) {
    pl[m] = prim_l[m];
    pr[m] = prim_r[m];
  }
  double ql[NVAR_OP2];
  double qr[NVAR_OP2];
  primitive_to_conservative_op2(pl, ql);
  primitive_to_conservative_op2(pr, qr);
  double flux[NVAR_OP2];
  edge_flux_first_order_op2(pl, pr, delta, S, flux);

  // dF/dq_l and dF/dq_r, column c by column.
  double D_l[NBLOCK_OP2];
  double D_r[NBLOCK_OP2];
  for (int c = 0; c < NVAR_OP2; ++c) {
    double q[NVAR_OP2];
    double p[NPRIM_OP2];
    double perturbed[NVAR_OP2];
    const double h_l = jacobian_step_op2(ql, c);
    for (int m = 0; m < NVAR_OP2; ++m) q[m] = ql[m] + (m == c ? h_l : 0.0);
    conservative_to_primitive_op2(q, p);
    edge_flux_first_order_op2(p, pr, delta, S, perturbed);
    for (int r = 0; r < NVAR_OP2; ++r) D_l[r * NVAR_OP2 + c] = (perturbed[r] - flux[r]) / h_l;
    const double h_r = jacobian_step_op2(qr, c);
    for (int m = 0; m < NVAR_OP2; ++m) q[m] = qr[m] + (m == c ? h_r : 0.0);
    conservative_to_primitive_op2(q, p);
    edge_flux_first_order_op2(pl, p, delta, S, perturbed);
    for (int r = 0; r < NVAR_OP2; ++r) D_r[r * NVAR_OP2 + c] = (perturbed[r] - flux[r]) / h_r;
  }

  for (int r = 0; r < NVAR_OP2; ++r) {
    const bool keep_l = !constrained_op2(bc_l, r);
    const bool keep_r = !constrained_op2(bc_r, r);
    for (int c = 0; c < NVAR_OP2; ++c) {
      if (decoupled[0] && r != c) continue;
      const int k = r * NVAR_OP2 + c;
      const double scale = op2_scale[c] / op2_scale[r];
      const double d_l = scale * D_l[k];
      const double d_r = scale * D_r[k];
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

// Each node's boundary flux depends on its own state alone, so a boundary
// face adds to its nodes' diagonal blocks only, through bdiag: the
// derivative of boundary_node_flux_op2 by the node's state.
inline void jac_bface_kernel(const int *btype, const double *normal, const double *area, const int *num_nodes,
                             const double *shares, const double *vol1, const double *vol2, const double *vol3,
                             const double *vol4, const double *prim1, const double *prim2, const double *prim3,
                             const double *prim4, double *b1, double *b2, double *b3, double *b4) {
  const int nodes = num_nodes[0] == 4 ? 4 : 3;
  const double amag = fmax(area[0], 1.0e-14);
  const double unit_normal[3] = {normal[0] / amag, normal[1] / amag, normal[2] / amag};
  double D[4][NBLOCK_OP2];
  for (int j = 0; j < 4; ++j)
    for (int k = 0; k < NBLOCK_OP2; ++k) D[j][k] = 0.0;
  for (int j = 0; j < nodes; ++j) {
    double p[NPRIM_OP2];
    for (int m = 0; m < NPRIM_OP2; ++m)
      p[m] = j == 0 ? prim1[m] : j == 1 ? prim2[m] : j == 2 ? prim3[m] : prim4[m];
    const double vol = j == 0 ? vol1[0] : j == 1 ? vol2[0] : j == 2 ? vol3[0] : vol4[0];
    const double S[3] = {shares[3 * j], shares[3 * j + 1], shares[3 * j + 2]};
    const double distance = fmax(vol / (nodes * fmax(vec_norm3(S), 1.0e-300)), 1.0e-12);
    double q0[NVAR_OP2];
    primitive_to_conservative_op2(p, q0);
    double flux[NVAR_OP2];
    boundary_node_flux_op2(p, S, unit_normal, btype[0], distance, flux);
    for (int c = 0; c < NVAR_OP2; ++c) {
      double q[NVAR_OP2];
      double pp[NPRIM_OP2];
      double perturbed[NVAR_OP2];
      const double h = jacobian_step_op2(q0, c);
      for (int m = 0; m < NVAR_OP2; ++m) q[m] = q0[m] + (m == c ? h : 0.0);
      conservative_to_primitive_op2(q, pp);
      boundary_node_flux_op2(pp, S, unit_normal, btype[0], distance, perturbed);
      for (int r = 0; r < NVAR_OP2; ++r) D[j][r * NVAR_OP2 + c] = (perturbed[r] - flux[r]) / h;
    }
  }
  for (int k = 0; k < NBLOCK_OP2; ++k) {
    b1[k] += D[0][k];
    b2[k] += D[1][k];
    b3[k] += D[2][k];
    if (nodes == 4) b4[k] += D[3][k];
  }
}

inline void zero_block_kernel(double *block) {
  for (int k = 0; k < NBLOCK_OP2; ++k) block[k] = 0.0;
}

// A node's diagonal block: V / dt from the pseudo-time step (spectral / CFL,
// as the explicit step's dt = CFL V / spectral), its boundary faces' blocks
// (bdiag) and the SA destruction's derivative; constrained rows the identity.
inline void jac_node_kernel(const int *bc, const double *spectral, const double *bdiag, const double *prim,
                            const double *grad, const double *wall_dist, const double *volume, const double *cfl,
                            const int *decoupled, double *A) {
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
      A[k] += bdiag[k] * op2_scale[c] / op2_scale[r] + (r == c ? time : 0.0) + (r == 5 && c == 5 ? sa : 0.0);
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
