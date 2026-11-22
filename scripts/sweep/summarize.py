#!/usr/bin/env python3
"""Tabulates sweep CSVs: per system, every case's iterations, true residual and
times, then one table across systems (iterations; F failed, N not converged)
and the fastest converged case of each system.

  summarize.py OUT.csv [OUT2.csv ...]  [--tol 1e-3]
"""
import argparse
import csv
import sys

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--tol", type=float, default=2e-3,
                help="true residual a converged case must reach (the libraries' own flags "
                     "can claim convergence a breakdown away from it; 0: trust them)")
a = ap.parse_args()
rows = []
for path in a.csv:
    with open(path) as f:
        rows += list(csv.DictReader(f))


def status(r):
    if r["converged"] in ("-1", "-2"):
        return "F" if r["converged"] == "-1" else "T"
    ok = r["converged"] == "1" and (not a.tol or float(r["true_residual"]) <= a.tol)
    return "" if ok else "N"


systems = list(dict.fromkeys(r["system"] for r in rows))
labels = list(dict.fromkeys(r["case"] for r in rows))
by = {(r["system"], r["case"]): r for r in rows}

for s in systems:
    print(f"\n### {s}\n\n| case | its | true res | first s | best s | solver MiB | matrix MiB | |\n|---|---:|---:|---:|---:|---:|---:|---|")
    for r in (r for r in rows if r["system"] == s):
        if r["converged"] in ("-1", "-2"):
            print(f"| {r['case']} | | | | | | | {'failed' if r['converged'] == '-1' else 'timed out'} |")
        else:
            print(f"| {r['case']} | {r['iterations']} | {float(r['true_residual']):.1e} | "
                  f"{float(r['first_s']):.3f} | {float(r['best_s']):.3f} | "
                  f"{float(r.get('solver_device_mib') or 0):.0f} | "
                  f"{float(r.get('matrix_device_mib') or 0):.0f} | {status(r)} |")

if len(systems) > 1:
    print("\n### iterations across systems (F failed, T timed out, N not converged by the true residual)\n")
    print("| case | " + " | ".join(systems) + " |\n|---|" + "---:|" * len(systems))
    for l in labels:
        cells = []
        for s in systems:
            r = by.get((s, l))
            cells.append("" if r is None else status(r) if status(r) in ("F", "T") else
                         r["iterations"] + ("N" if status(r) == "N" else ""))
        print(f"| {l} | " + " | ".join(cells) + " |")

print("\n### fastest converged case per system (best solve time)\n")
for s in systems:
    ok = [r for r in rows if r["system"] == s and not status(r)]
    if ok:
        r = min(ok, key=lambda r: float(r["best_s"]))
        print(f"- {s}: {r['case']} ({r['iterations']} its, {float(r['best_s']):.3f} s)")
    else:
        print(f"- {s}: none converged")
