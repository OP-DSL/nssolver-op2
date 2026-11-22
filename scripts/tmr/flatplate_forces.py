"""Skin friction at x = 0.97 and the drag coefficient of the TMR flat plate
from a wall CSV of the post-processing helper (x, ..., tau_w), against the
TMR reference values (SA, CFL3D and FUN3D on the 545x385 grid).

    python3 flatplate_forces.py WALL.csv RHO U [LABEL]
"""
import csv
import sys

REFERENCE = {"cf_097": (0.270562153e-2, 0.270540473e-2), "cd": (0.285985288e-2, 0.2852469e-2)}


def main():
    path, rho, u = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else path
    q = 0.5 * rho * u * u
    rows = sorted((float(r["x"]), float(r["tau_w"]) / q) for r in csv.DictReader(open(path)))
    rows = [(x, cf) for x, cf in rows if 0.0 < x <= 2.0 + 1e-9]
    xs = [x for x, _ in rows]
    cfs = [cf for _, cf in rows]
    # The plate's leading-edge node has no defined shear: the integral starts
    # from its first neighbour's value held back to x = 0.
    cd = cfs[0] * xs[0]
    for i in range(1, len(xs)):
        cd += 0.5 * (cfs[i] + cfs[i - 1]) * (xs[i] - xs[i - 1])
    cd /= 2.0
    target = 0.97
    for i in range(1, len(xs)):
        if xs[i] >= target:
            t = (target - xs[i - 1]) / (xs[i] - xs[i - 1])
            cf097 = cfs[i - 1] + t * (cfs[i] - cfs[i - 1])
            break
    dcf = 100.0 * (cf097 / REFERENCE["cf_097"][0] - 1.0)
    dcd = 100.0 * (cd / REFERENCE["cd"][0] - 1.0)
    print(f"{label:24s} Cf(0.97) {cf097:.6e} ({dcf:+.2f}% vs CFL3D)  CD {cd:.6e} ({dcd:+.2f}% vs CFL3D)")


if __name__ == "__main__":
    main()
