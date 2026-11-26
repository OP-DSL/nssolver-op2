"""Strouhal number, mean drag and lift amplitude from a dual-time run's
history (output_file's .history.csv): coefficients on the diameter D and the
span, periods between the lift's upward zero crossings over the last
PERIODS of them.

    python3 forces.py HISTORY.csv RHO U D SPAN [PERIODS]
"""
import csv
import sys


def main():
    path = sys.argv[1]
    rho, u, d, span = map(float, sys.argv[2:6])
    periods = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    q = 0.5 * rho * u * u * d * span
    rows = [(float(r["time"]), float(r["fx"]) / q, float(r["fy"]) / q) for r in csv.DictReader(open(path))]
    # A diverged run's force stops crossing zero, which would leave a window before it.
    bad = next((r for r in rows if not abs(r[1]) + abs(r[2]) < 100.0), None)
    if bad:
        sys.exit(f"{path}: diverged by t = {bad[0]:.4g} (Cd {bad[1]:.3g}, Cl {bad[2]:.3g})")
    t = [r[0] for r in rows]
    cd = [r[1] for r in rows]
    cl = [r[2] for r in rows]
    ups = []
    for i in range(1, len(cl)):
        if cl[i - 1] < 0.0 <= cl[i]:
            ups.append(t[i - 1] + (t[i] - t[i - 1]) * (-cl[i - 1]) / (cl[i] - cl[i - 1]))
    if len(ups) < periods + 1:
        print(f"{path}: {len(ups)} lift up-crossings, not yet {periods + 1}; last Cd {cd[-1]:.4f} Cl {cl[-1]:+.4f}")
        return
    t0, t1 = ups[-periods - 1], ups[-1]
    window = [i for i in range(len(t)) if t0 <= t[i] <= t1]
    period = (t1 - t0) / periods
    strouhal = d / (u * period)
    cd_mean = sum(cd[i] for i in window) / len(window)
    cl_amp = 0.5 * (max(cl[i] for i in window) - min(cl[i] for i in window))
    cd_amp = 0.5 * (max(cd[i] for i in window) - min(cd[i] for i in window))
    print(f"St {strouhal:.4f}  Cd mean {cd_mean:.4f} (+-{cd_amp:.4f})  Cl amplitude {cl_amp:.4f}  "
          f"({periods} periods to t = {t1:.4g}, {len(ups)} up-crossings)")


if __name__ == "__main__":
    main()
