#!/usr/bin/env python3
"""Writes a solver sweep for apps/c/solve_hdf5 (-config, -cases): each library's
matrix storage and a grid of Krylov solvers, preconditioners and settings, as
OP2's solver configuration names them.

  make_sweep.py OUTDIR [--rtol 1e-3] [--maxit 300] [--libs petsc,hypre,ginkgo]
                [--set basic|full|hard|memory] [--lean]

OUTDIR/config.json holds the matrices and solvers sections, OUTDIR/cases.txt
the "label matrix solver" lines, grouped by matrix, and cases.<matrix>.txt each
matrix's own. Edit SPECS to add settings;
a setting a library rejects ends only its own case (run_sweep.sh resumes).
"""
import argparse
import json
import os


def hard(rtol, maxit):
    """The strong preconditioners, for the systems the basic set cannot solve."""
    out = []
    for restart in (100, 200):
        for levels in (0, 1, 2):
            out.append(("petsc", "petsc", f"gmres{restart}-bjacobi-ilu{levels}",
                        {"ksp_type": "gmres", "ksp_gmres_restart": restart, "ksp_rtol": rtol,
                         "ksp_max_it": maxit, "ksp_pc_side": "right", "pc_type": "bjacobi",
                         "sub_pc_type": "ilu", "sub_pc_factor_levels": levels}))
    for overlap, levels in ((1, 1), (2, 1), (1, 2), (2, 2)):
        out.append(("petsc", "petsc", f"gmres100-asm{overlap}-ilu{levels}",
                    {"ksp_type": "gmres", "ksp_gmres_restart": 100, "ksp_rtol": rtol,
                     "ksp_max_it": maxit, "ksp_pc_side": "right", "pc_type": "asm",
                     "pc_asm_overlap": overlap, "sub_pc_type": "ilu",
                     "sub_pc_factor_levels": levels}))
    for levels in (1, 2):
        out.append(("petsc", "petsc", f"bcgs-bjacobi-ilu{levels}",
                    {"ksp_type": "bcgs", "ksp_rtol": rtol, "ksp_max_it": maxit,
                     "pc_type": "bjacobi", "sub_pc_type": "ilu", "sub_pc_factor_levels": levels}))
    for levels in (0, 1, 2):
        out.append(("hypre", "hypre", f"gmres100-ilu-fill{levels}",
                    {"solver": "GMRES", "KDim": 100, "Tol": rtol, "MaxIter": maxit,
                     "precond": "ILU", "ILU": {"LevelOfFill": levels}}))
        out.append(("hypre", "hypre", f"bicgstab-ilu-fill{levels}",
                    {"solver": "BiCGSTAB", "Tol": rtol, "MaxIter": maxit,
                     "precond": "ILU", "ILU": {"LevelOfFill": levels}}))
    for label, local in (("jacobi6", {"type": "preconditioner::Jacobi", "max_block_size": 6}),
                         ("ilu", {"type": "preconditioner::Ilu"})):
        out.append(("ginkgo", "ginkgo", f"gmres100-schwarz-{label}",
                    {"type": "solver::Gmres", "krylov_dim": 100,
                     "criteria": {"iteration": maxit, "relative_residual_norm": rtol},
                     "preconditioner": {"type": "preconditioner::Schwarz", "local_solver": local}}))
    return out


def memory(rtol, maxit):
    """Matrix storage that is lean on the device, against the defaults."""
    out = []
    for matrix in ("petsc", "petsc_owned"):
        for label, o in (("pbjacobi", {"pc_type": "pbjacobi"}),
                         ("bjacobi-ilu0", {"pc_type": "bjacobi", "sub_pc_type": "ilu"})):
            out.append(("petsc", matrix, f"{matrix}-gmres30-{label}",
                        {"ksp_type": "gmres", "ksp_gmres_restart": 30, "ksp_rtol": rtol,
                         "ksp_max_it": maxit, "ksp_pc_side": "right", **o}))
    schwarz = {"type": "preconditioner::Schwarz", "local_solver": "op2_block_jacobi"}
    for matrix in ("ginkgo", "ginkgo_fbcsr"):
        out.append(("ginkgo", matrix, f"{matrix}-bicgstab-blockjacobi",
                    {"type": "solver::Bicgstab", "preconditioner": schwarz,
                     "criteria": {"iteration": maxit, "relative_residual_norm": rtol}}))
        out.append(("ginkgo", matrix, f"{matrix}-gmres30-blockjacobi",
                    {"type": "solver::Gmres", "krylov_dim": 30, "preconditioner": schwarz,
                     "criteria": {"iteration": maxit, "relative_residual_norm": rtol}}))
    return out


def specs(rtol, maxit, level):
    """(library, matrix, label, solver-section) in the library's terms."""
    if level == "hard":
        return hard(rtol, maxit)
    if level == "memory":
        return memory(rtol, maxit)
    out = []

    def petsc(label, **o):
        s = {"ksp_type": "gmres", "ksp_gmres_restart": 30, "ksp_rtol": rtol,
             "ksp_max_it": maxit, "ksp_pc_side": "right"}
        s.update(o)
        if s["ksp_type"] not in ("gmres", "fgmres"):
            del s["ksp_gmres_restart"]      # PETSc rejects options nothing reads
        out.append(("petsc", "petsc", label, s))

    petsc("gmres30-pbjacobi", pc_type="pbjacobi")
    petsc("gmres30-jacobi", pc_type="jacobi")
    petsc("gmres30-bjacobi-ilu0", pc_type="bjacobi", sub_pc_type="ilu", sub_pc_factor_levels=0)
    petsc("gmres30-bjacobi-ilu1", pc_type="bjacobi", sub_pc_type="ilu", sub_pc_factor_levels=1)
    petsc("gmres30-asm1-ilu0", pc_type="asm", pc_asm_overlap=1, sub_pc_type="ilu")
    petsc("gmres30-gamg", pc_type="gamg")
    petsc("gmres60-bjacobi-ilu0", pc_type="bjacobi", sub_pc_type="ilu", ksp_gmres_restart=60)
    petsc("bcgs-bjacobi-ilu0", ksp_type="bcgs", pc_type="bjacobi", sub_pc_type="ilu")
    if level == "full":
        petsc("gmres100-bjacobi-ilu0", pc_type="bjacobi", sub_pc_type="ilu", ksp_gmres_restart=100)
        petsc("gmres30-asm2-ilu1", pc_type="asm", pc_asm_overlap=2, sub_pc_type="ilu",
              sub_pc_factor_levels=1)
        petsc("gmres30-gamg-thr", pc_type="gamg", pc_gamg_threshold=0.02)
        petsc("fgmres30-pbjacobi", ksp_type="fgmres", pc_type="pbjacobi")

    def hypre(label, precond, solver="GMRES", **o):
        s = {"solver": solver, "Tol": rtol, "MaxIter": maxit, "precond": precond}
        if solver in ("GMRES", "FlexGMRES"):
            s["KDim"] = 30
        s.update(o)
        out.append(("hypre", "hypre", label, s))

    hypre("gmres30-ilu", "ILU")
    hypre("gmres30-diagscale", "DiagScale")
    hypre("gmres30-boomeramg", "BoomerAMG")
    hypre("gmres30-boomeramg-hmis", "BoomerAMG", BoomerAMG={"CoarsenType": 10, "StrongThreshold": 0.5})
    hypre("gmres30-boomeramg-nf6", "BoomerAMG", BoomerAMG={"NumFunctions": 6})
    hypre("bicgstab-ilu", "ILU", solver="BiCGSTAB")
    if level == "full":
        hypre("gmres60-ilu", "ILU", KDim=60)
        hypre("flexgmres30-boomeramg", "BoomerAMG", solver="FlexGMRES")

    def ginkgo(label, pre, solver="solver::Gmres", matrix="ginkgo", **o):
        s = {"type": solver, "criteria": {"iteration": maxit, "relative_residual_norm": rtol}}
        if solver == "solver::Gmres":
            s["krylov_dim"] = 30
        if pre is not None:
            s["preconditioner"] = pre
        s.update(o)
        out.append(("ginkgo", matrix, label, s))

    ginkgo("gmres30-none", None)
    schwarz = lambda local: {"type": "preconditioner::Schwarz", "local_solver": local}
    ginkgo("gmres30-schwarz-jacobi6", schwarz({"type": "preconditioner::Jacobi", "max_block_size": 6}))
    ginkgo("gmres30-schwarz-blockjacobi",
           {"type": "preconditioner::Schwarz", "local_solver": "op2_block_jacobi"})
    ginkgo("gmres30-schwarz-ilu", schwarz({"type": "preconditioner::Ilu"}))
    ginkgo("bicgstab-schwarz-blockjacobi", schwarz("op2_block_jacobi"), solver="solver::Bicgstab")
    ginkgo("gmres30-schwarz-blockjacobi-fbcsr", schwarz("op2_block_jacobi"), matrix="ginkgo_fbcsr")
    return out


MATRICES = {
    "petsc": {"library": "petsc"},
    "petsc_owned": {"library": "petsc", "format": "aij", "owned": True},
    "hypre": {"library": "hypre", "format": "csr"},
    "ginkgo": {"library": "ginkgo", "format": "csr"},
    "ginkgo_fbcsr": {"library": "ginkgo", "format": "fbcsr"},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--maxit", type=int, default=300)
    ap.add_argument("--libs", default="petsc,hypre,ginkgo")
    ap.add_argument("--lean", action="store_true",
                    help="PETSc ILU/ASM/GAMG cases on borrowed device storage (petsc_owned), "
                         "which holds the matrix once: pbjacobi/jacobi read host values and need coo")
    ap.add_argument("--set", dest="level", default="basic", choices=["basic", "full", "hard", "memory"])
    a = ap.parse_args()
    libs = a.libs.split(",")
    chosen = [s for s in specs(a.rtol, a.maxit, a.level) if s[0] in libs]
    if a.lean:
        chosen = [(lib, "petsc_owned" if lib == "petsc" and sec.get("pc_type") in ("bjacobi", "asm", "gamg")
                   else matrix, label, sec) for lib, matrix, label, sec in chosen]
    os.makedirs(a.outdir, exist_ok=True)
    solvers, lines = {}, []
    for lib, matrix, label, section in chosen:
        name = f"{lib}_{label}".replace("-", "_")
        solvers[name] = {lib: section}
        lines.append((matrix, f"{lib}/{label}", name))
    lines.sort(key=lambda t: list(MATRICES).index(t[0]))
    used = {m for m, _, _ in lines}
    config = {"matrices": {m: MATRICES[m] for m in MATRICES if m in used}, "solvers": solvers}
    with open(os.path.join(a.outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    with open(os.path.join(a.outdir, "cases.txt"), "w") as f:
        f.write("# label matrix solver\n")
        for matrix, label, name in lines:
            f.write(f"{label} {matrix} {name}\n")
    # one list per matrix: run_sweep.sh launches the driver once for each, which
    # then frees the matrix's values from the device after building it
    for matrix in dict.fromkeys(m for m, _, _ in lines):
        with open(os.path.join(a.outdir, f"cases.{matrix}.txt"), "w") as f:
            f.write("# label matrix solver\n")
            for m, label, name in lines:
                if m == matrix:
                    f.write(f"{label} {m} {name}\n")
    print(f"{len(lines)} cases in {a.outdir} (rtol {a.rtol:g}, maxit {a.maxit})")
