"""Mean and RMS surface pressure of a unit-chord airfoil's buffet over whole
periods, from a dual-time run's wall snapshots (scripts/cl/wall_series.sbatch
writes WALLDIR/TAG_NNNNNN_wall.csv), against an experiment's mean Cp and Cp RMS
files (Tecplot-style zones "... a = ALPHA", as the buffet workshop's OAT15A
Cp.dat and Cp_rms.dat). The window runs between the first and last lift maxima
the snapshots cover. Prints the lift's mean and swing, the Strouhal number, the
shock's travel (steepest pressure rise on the upper surface) and the peak RMS;
with matplotlib also a plot.

    python3 cpstats.py HISTORY.csv WALLDIR TAG MACH RE T ALPHA DT CP.dat CP_RMS.dat [PLOT.png]
"""
import csv
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp import experiment_sides, shock, wall_sides, zone  # noqa: E402
from freestream import freestream  # noqa: E402


def main():
    history, walldir, tag = sys.argv[1:4]
    mach, re_, t, alpha, dt = map(float, sys.argv[4:9])
    cp_path, rms_path = sys.argv[9:11]
    fs = freestream(mach, re_, t, alpha)
    p_inf, q = fs["p"], 0.5 * fs["rho"] * fs["speed"] ** 2
    a = math.radians(alpha)
    rows = list(csv.DictReader(open(history)))
    # The force acts on the mesh's extrusion SPAN (0.01 by default, as in run.sh).
    span = float(os.environ.get("SPAN", "0.01"))
    cl = [(-float(r["fx"]) * math.sin(a) + float(r["fy"]) * math.cos(a)) / (q * span) for r in rows]

    files = sorted(glob.glob(os.path.join(walldir, f"{tag}_[0-9][0-9][0-9][0-9][0-9][0-9]_wall.csv")))
    if not files:
        sys.exit(f"no {tag}_NNNNNN_wall.csv in {walldir}")
    steps = [int(re.search(r"_(\d{6})_wall\.csv$", f).group(1)) for f in files]
    # Lift maxima (1-based steps) the snapshots cover: a whole number of periods.
    maxima = [i + 1 for i in range(1, len(cl) - 1)
              if cl[i] > cl[i - 1] and cl[i] >= cl[i + 1] and steps[0] <= i + 1 <= steps[-1]]
    if len(maxima) < 2:
        sys.exit(f"fewer than two lift maxima within steps {steps[0]}-{steps[-1]}")
    first, last = maxima[0], maxima[-1]
    periods = len(maxima) - 1
    period = (last - first) / periods * dt * fs["speed"]
    window = cl[first - 1:last - 1]
    used = [f for s, f in zip(steps, files) if first <= s < last]
    print(f"steps {first}-{last}: {periods} periods, {len(used)} snapshots; Strouhal number {1 / period:.4f}; "
          f"Cl mean {sum(window) / len(window):.4f}, swing {max(window) - min(window):.4f}")

    snaps = [wall_sides(f, p_inf, q) for f in used]
    sides = []
    for side in range(2):
        xs = [x for x, _ in snaps[0][side]]
        values = [[s[side][i][1] for s in snaps] for i in range(len(xs))]
        mean = [sum(v) / len(v) for v in values]
        rms = [math.sqrt(sum((c - m) ** 2 for c in v) / len(v)) for v, m in zip(values, mean)]
        sides.append((xs, mean, rms))
    shocks = [shock([x for x, _ in s[0]], [c for _, c in s[0]]) for s in snaps]
    xs, _, rms = sides[0]
    peak = max(range(len(xs)), key=lambda i: rms[i] if 0.2 < xs[i] < 0.9 else 0.0)
    exp_rms = zone(rms_path, alpha)
    exp_peak = max(exp_rms, key=lambda p: p[1])
    print(f"shock x/c {min(shocks):.3f} to {max(shocks):.3f}; upper Cp RMS peak {rms[peak]:.4f} at x/c "
          f"{xs[peak]:.3f} (experiment {exp_peak[1]:.4f} at {exp_peak[0]:.3f})")

    if len(sys.argv) > 11:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        exp_up, exp_lo = experiment_sides(zone(cp_path, alpha))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        ax[0].plot(sides[0][0], sides[0][1], "b-", lw=1.5, label="nssolver upper")
        ax[0].plot(sides[1][0], sides[1][1], "g-", lw=1.5, label="nssolver lower")
        ax[0].plot(*zip(*(exp_up + exp_lo)), "ko", ms=3, label="experiment")
        ax[0].invert_yaxis()
        ax[0].set_ylabel("mean Cp")
        ax[1].plot(sides[0][0], sides[0][2], "b-", lw=1.5, label="nssolver upper")
        ax[1].plot(*zip(*exp_rms), "ko", ms=4, label="experiment")
        ax[1].set_ylabel("Cp RMS")
        ax[1].set_xlim(0.0, 1.0)
        for axis in ax:
            axis.set_xlabel("x/c")
            axis.legend()
        fig.suptitle(f"M {mach}, alpha {alpha}: {periods} periods")
        plt.tight_layout()
        plt.savefig(sys.argv[11], dpi=110)


if __name__ == "__main__":
    main()
