"""ONERA M6 wing sections against Schmitt and Charpin's experiment (AGARD
AR-138, test 2308): Cp of the wall cut at y/b 0.2-0.99
(nssolver_onera_m6_validation_helper's sections file) and the experiment's
Tecplot file (NPARC's ONERAb114.tec). Prints the RMS difference on each side of
each section; with matplotlib also a plot.

    python3 onera_m6_sections.py SECTIONS.csv ONERAb114.tec [PLOT.png]
"""
import csv
import math
import re
import sys

LOWER, UPPER = 1, 3  # SU2 marker order: LOWER_SIDE, TIP, UPPER_SIDE


def interp(xs, ys, x):
    for i in range(1, len(xs)):
        if xs[i] >= x:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def computed(path):
    """{station: (upper, lower)}, each [(x/c, Cp)] by x/c over the section's chord."""
    rows = {}
    for r in csv.DictReader(open(path)):
        rows.setdefault(round(float(r["station"]), 4), []).append(
            (int(r["group_id"]), float(r["x"]), float(r["cp"])))
    sections = {}
    for station, pts in rows.items():
        pts = [p for p in pts if p[0] in (LOWER, UPPER)]
        le, te = min(p[1] for p in pts), max(p[1] for p in pts)
        side = lambda g: sorted(((x - le) / (te - le), c) for gg, x, c in pts if gg == g)
        sections[station] = side(UPPER), side(LOWER)
    return sections


def experiment(path):
    """{station: (upper, lower)}, each [(x/c, Cp)] by x/c; the upper surface has z > 0."""
    sections, zone = {}, None
    for line in open(path):
        if re.match(r"\s*ZONE", line):
            zone = []
            continue
        f = line.split()
        if zone is None or len(f) != 5 or not re.match(r"[-0-9.]", f[0]):
            continue
        _, x, y, z, cp = map(float, f)
        up, lo = sections.setdefault(round(y, 4), ([], []))
        (up if z > 0 else lo).append((x, cp))
    return {s: (sorted(u), sorted(l)) for s, (u, l) in sections.items()}


def rms(sim, ref):
    xs, cs = [x for x, _ in sim], [c for _, c in sim]
    d = [interp(xs, cs, x) - c for x, c in ref]
    return math.sqrt(sum(v * v for v in d) / len(d))


def main():
    sim, exp = computed(sys.argv[1]), experiment(sys.argv[2])
    print(" y/b   Cp RMS difference upper  lower")
    for s in sorted(exp):
        print(f"{s:5.2f}   {rms(sim[s][0], exp[s][0]):21.3f}  {rms(sim[s][1], exp[s][1]):.3f}")
    if len(sys.argv) > 3:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey=True)
        for ax, s in zip(axes.flat, sorted(exp)):
            for side, colour in ((0, "tab:blue"), (1, "tab:green")):
                ax.plot(*zip(*sim[s][side]), "-", color=colour, lw=1.2)
                ax.plot(*zip(*exp[s][side]), "o", color="k", ms=3, mfc="none")
            ax.set_title(f"y/b = {s}", fontsize=10)
            ax.grid(alpha=0.3)
        axes.flat[0].invert_yaxis()
        last = axes.flat[-1]
        last.axis("off")
        last.plot([], [], "-", color="tab:blue", label="nssolver, upper")
        last.plot([], [], "-", color="tab:green", label="nssolver, lower")
        last.plot([], [], "o", color="k", mfc="none", label="experiment")
        last.legend(loc="center")
        for ax in axes[1]:
            ax.set_xlabel("x/c")
        for ax in axes[:, 0]:
            ax.set_ylabel("Cp")
        fig.suptitle("ONERA M6, M 0.8395, α 3.06°: Euler against Schmitt and Charpin")
        fig.tight_layout()
        fig.savefig(sys.argv[3], dpi=110)


if __name__ == "__main__":
    main()
