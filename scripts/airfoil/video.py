"""A video of a planar case's flow from a dual-time run's snapshots
(snapshot_interval writes TAG_NNNNNN.h5 beside the output): the extruded mesh's
lower symmetry plane, whose boundary quads are the section's cells, interpolated
linearly onto the frame's pixels; each frame shows the Mach number with the sonic
line (or numerical schlieren, or Cp) above the lift's history. Needs numpy, h5py,
matplotlib and, for the video, ffmpeg.

    python3 video.py render MESH.h5 SNAPDIR TAG HISTORY.csv MACH RE T ALPHA OUT.mp4 [options]
    python3 video.py check MESH.h5 SNAPDIR TAG HISTORY.csv MACH RE T ALPHA

check compares the two symmetry planes' states (a planar solution has the same on
both), interpolates an exact linear field through the frame's pixels, reads the free
stream far away, and sets each snapshot's pressure lift against the history's lift at
the snapshot's step and its neighbours: a snapshot numbered off by a step shows as a
phase error.
"""
import argparse
import csv
import glob
import math
import multiprocessing
import os
import re
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freestream import GAMMA, freestream  # noqa: E402

NO_SLIP_WALL = 4  # nssolver::BoundaryType


class Mesh:
    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.xyz = f["node_coordinates"][:]
            self.faces = f["bface-->node"][:]
            self.count = f["bface_num_nodes"][:, 0]
            self.type = f["bface_type"][:, 0]
            self.subfaces = f["bface_node_normals"][:].reshape(-1, 4, 3)
        z = self.xyz[:, 2]
        self.span = np.ptp(z)

    def plane(self, top=False):
        """The nodes of the lower (or upper) z plane and its cells as triangles of them."""
        z = self.xyz[:, 2]
        on = np.abs(z - (z.max() if top else z.min())) <= 1e-9 * max(1.0, self.span)
        quads = self.faces[(self.count == 4) & on[self.faces].all(axis=1)]
        trias = self.faces[(self.count == 3) & on[self.faces[:, :3]].all(axis=1), :3]
        nodes = np.flatnonzero(on)
        local = np.full(len(z), -1)
        local[nodes] = np.arange(len(nodes))
        tris = local[np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]], trias])]
        if (tris < 0).any() or len(np.unique(tris)) != len(nodes):
            sys.exit("the lowest z plane's boundary faces do not cover its nodes: not an extruded planar mesh")
        return nodes, tris

    def wall_force(self, p):
        """The pressure force on the no-slip walls, sum of p dS over their nodes'
        sub-faces (the history's force less the viscous traction)."""
        wall = self.type == NO_SLIP_WALL
        return np.einsum("fk,fkd->d", p[self.faces[wall]], self.subfaces[wall])


class Raster:
    """Linear interpolation of node values onto WIDTH x HEIGHT pixel centres over
    WINDOW (x0, x1, y0, y1); pixels outside every cell are NaN."""

    def __init__(self, x, y, tris, window, width, height):
        from matplotlib.tri import Triangulation
        x0, x1, y0, y1 = window
        tx, ty = x[tris], y[tris]
        tris = tris[(tx.max(1) >= x0) & (tx.min(1) <= x1) & (ty.max(1) >= y0) & (ty.min(1) <= y1)]
        px = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
        py = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height
        X, Y = (a.ravel() for a in np.meshgrid(px, py))
        cell = Triangulation(x, y, tris).get_trifinder()(X, Y)
        self.inside = cell >= 0
        v = tris[cell[self.inside]]
        X, Y = X[self.inside], Y[self.inside]
        (xa, xb, xc), (ya, yb, yc) = x[v].T, y[v].T
        det = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        la = ((yb - yc) * (X - xc) + (xc - xb) * (Y - yc)) / det
        lb = ((yc - ya) * (X - xc) + (xa - xc) * (Y - yc)) / det
        self.nodes, self.weights = v, np.stack([la, lb, 1.0 - la - lb], axis=1)
        self.shape = (height, width)
        self.window = window

    def __call__(self, values):
        image = np.full(self.shape[0] * self.shape[1], np.nan)
        image[self.inside] = np.einsum("pk,pk->p", values[self.nodes], self.weights)
        return image.reshape(self.shape)


def snapshots(snapdir, tag):
    files = sorted(glob.glob(os.path.join(snapdir, f"{tag}_[0-9][0-9][0-9][0-9][0-9][0-9].h5")))
    if not files:
        sys.exit(f"no {tag}_NNNNNN.h5 in {snapdir}")
    return [(int(re.search(r"_(\d{6})\.h5$", f).group(1)), f) for f in files]


def primitive(path):
    with h5py.File(path, "r") as f:
        return f["primitive"][:]


def lift(history, fs, alpha, span):
    """Each step's time and lift coefficient from a dual-time run's history."""
    a = math.radians(alpha)
    q = 0.5 * fs["rho"] * fs["speed"] ** 2 * span
    rows = list(csv.DictReader(open(history)))
    time = np.array([float(r["time"]) for r in rows])
    cl = np.array([(-float(r["fx"]) * math.sin(a) + float(r["fy"]) * math.cos(a)) / q for r in rows])
    return time, cl


def check(args):
    fs = freestream(args.mach, args.re, args.t, args.alpha)
    q = 0.5 * fs["rho"] * fs["speed"] ** 2
    mesh = Mesh(args.mesh)
    lower, tris = mesh.plane()
    upper, _ = mesh.plane(top=True)
    print(f"{len(mesh.xyz)} nodes, planes of {len(lower)} and {len(upper)}, {len(tris)} triangles; span {mesh.span:g}")
    # Pair the planes' nodes by their (x, y).
    a = lower[np.lexsort(mesh.xyz[lower, :2].T[::-1])]
    b = upper[np.lexsort(mesh.xyz[upper, :2].T[::-1])]
    print(f"planes' nodes paired to {np.abs(mesh.xyz[a, :2] - mesh.xyz[b, :2]).max():.2e} in x, y")

    x, y = mesh.xyz[lower, 0], mesh.xyz[lower, 1]
    raster = Raster(x, y, tris, (-0.25, 1.45, -0.35, 0.45), 1700, 800)
    x0, x1, y0, y1 = raster.window
    px = x0 + (np.arange(1700) + 0.5) * (x1 - x0) / 1700
    py = y1 - (np.arange(800) + 0.5) * (y1 - y0) / 800
    X, Y = np.meshgrid(px, py)
    image = raster(0.3 + 1.7 * x - 0.9 * y)
    inside = ~np.isnan(image)
    print(f"linear field through {inside.sum()} of {inside.size} pixels: "
          f"max error {np.abs(image - (0.3 + 1.7 * X - 0.9 * Y))[inside].max():.2e}")

    radius = np.hypot(x - 0.5, y)
    far = lower[radius > 0.9 * radius.max()]
    time, cl = lift(args.history, fs, args.alpha, mesh.span)
    alpha = math.radians(args.alpha)
    shifts = range(-2, 3)
    offsets = {k: [] for k in shifts}
    # rho, |u, v|, p by the free stream's, nu~ by its own largest; w, which a slip
    # plane imposes weakly only, by the free stream's speed. A shock a little apart
    # on the two planes differs at its few nodes: the pressure's 99.9th percentile
    # shows the rest.
    planes, spanwise, bulk = np.zeros(4), 0.0, 0.0
    snaps = snapshots(args.snapdir, args.tag)
    for i, (step, path) in enumerate(snaps):
        w = primitive(path)
        d = np.abs(w[a] - w[b])
        planes = np.maximum(planes, [d[:, 0].max() / fs["rho"], d[:, 1:3].max() / fs["speed"],
                                     d[:, 4].max() / fs["p"], d[:, 5].max() / np.abs(w[:, 5]).max()])
        bulk = max(bulk, np.percentile(d[:, 4], 99.9) / fs["p"])
        spanwise = max(spanwise, np.abs(w[a, 3]).max() / fs["speed"], np.abs(w[b, 3]).max() / fs["speed"])
        if i == 0:
            mach = np.linalg.norm(w[far, 1:4], axis=1) / np.sqrt(GAMMA * w[far, 4] / w[far, 0])
            cp = (w[far, 4] - fs["p"]) / q
            print(f"far field ({len(far)} nodes beyond {0.9 * radius.max():.0f} chords): "
                  f"Mach {mach.min():.4f} to {mach.max():.4f}, Cp {cp.min():+.4f} to {cp.max():+.4f}")
        f = mesh.wall_force(w[:, 4])
        cl_p = (-f[0] * math.sin(alpha) + f[1] * math.cos(alpha)) / (q * mesh.span)
        for k in shifts:
            if 1 <= step + k <= len(cl):
                offsets[k].append(cl[step + k - 1] - cl_p)
    print(f"{len(snaps)} snapshots, steps {snaps[0][0]}-{snaps[-1][0]}; the planes differ by at most: density "
          f"{planes[0]:.1e}, velocity {planes[1]:.1e}, pressure {planes[2]:.1e} (at 99.9% of the nodes "
          f"{bulk:.1e}) of the free stream's, nu~ {planes[3]:.1e} of its largest; spanwise velocity on them up "
          f"to {spanwise:.1e} of the free stream's")
    print("history lift less snapshot pressure lift, by the history's step offset:")
    for k in shifts:
        d = np.array(offsets[k])
        print(f"  {k:+d}: mean {d.mean():+.5f}, spread (std) {d.std():.5f}")


FIELDS = {
    "mach": "Mach number",
    "schlieren": "numerical schlieren",
    "cp": "pressure coefficient",
}

worker = {}


def setup(args, raster, fs, time, cl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    width, height = raster.shape[1], raster.shape[0]
    fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100)
    # The field at one pixel a value: its axes exactly the raster's size.
    left, bottom = 20, args.height - 20 - height
    ax = fig.add_axes([left / args.width, bottom / args.height, width / args.width, height / args.height])
    ax.set_axis_off()
    x0, x1, y0, y1 = raster.window
    if args.field == "mach":
        cmap, norm = plt.get_cmap("RdBu_r").copy(), TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=1.6)
    elif args.field == "cp":
        cmap, norm = plt.get_cmap("RdBu").copy(), TwoSlopeNorm(vmin=-1.6, vcenter=0.0, vmax=1.1)
    else:
        cmap, norm = plt.get_cmap("gray").copy(), None
    cmap.set_bad("#b8b8b8")
    image = ax.imshow(np.zeros(raster.shape), extent=(x0, x1, y0, y1), origin="upper", cmap=cmap,
                      norm=norm, interpolation="none", vmin=None if norm else 0.0, vmax=None if norm else 1.0)
    if args.field != "schlieren":
        cax = fig.add_axes([(left + width + 15) / args.width, bottom / args.height, 18 / args.width,
                            height / args.height])
        fig.colorbar(image, cax=cax).set_label(FIELDS[args.field])
    label = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=15,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    lax = fig.add_axes([90 / args.width, 45 / args.height, (args.width - 140) / args.width,
                        (bottom - 85) / args.height])
    tu = time * fs["speed"]
    lax.plot(tu, cl, color="#1f4e8c", lw=1.5)
    lax.set_xlim(tu[0], tu[-1])
    lax.set_xlabel("t U / c")
    lax.set_ylabel("lift coefficient")
    lax.grid(alpha=0.3)
    cursor = lax.axvline(tu[0], color="#c0392b", lw=1.5)
    dot, = lax.plot([tu[0]], [cl[0]], "o", color="#c0392b", ms=7)
    worker.update(fig=fig, ax=ax, image=image, label=label, cursor=cursor, dot=dot, contour=None, plt=plt)


def frame(job):
    args, raster, fs, time, cl, nodes = (worker[k] for k in ("args", "raster", "fs", "time", "cl", "nodes"))
    step, path = job
    if "fig" not in worker:
        setup(args, raster, fs, time, cl)
    w = primitive(path)[nodes]
    if args.field == "cp":
        value = raster((w[:, 4] - fs["p"]) / (0.5 * fs["rho"] * fs["speed"] ** 2))
    else:
        mach = raster(np.linalg.norm(w[:, 1:4], axis=1) / np.sqrt(GAMMA * w[:, 4] / w[:, 0]))
        if args.field == "mach":
            value = mach
        else:
            # Quirk's numerical schlieren of the density's gradient, at a fixed scale.
            dx = (raster.window[1] - raster.window[0]) / raster.shape[1]
            gy, gx = np.gradient(raster(w[:, 0]), dx)
            value = np.exp(-args.schlieren * np.hypot(gx, gy) / (fs["rho"] / 0.05))
    worker["image"].set_data(value)
    if worker["contour"] is not None:
        worker["contour"].remove()
        worker["contour"] = None
    if args.field == "mach" and np.nanmax(value) > 1.0:
        x0, x1, y0, y1 = raster.window
        worker["contour"] = worker["ax"].contour(np.where(np.isnan(value), 0.0, value), levels=[1.0],
                                                 colors="black", linewidths=1.0, origin="upper",
                                                 extent=(x0, x1, y0, y1))
    tu = time[step - 1] * fs["speed"]
    worker["label"].set_text(f"{args.title}\nt U/c = {tu:7.2f}")
    worker["cursor"].set_xdata([tu, tu])
    worker["dot"].set_data([tu], [cl[step - 1]])
    worker["fig"].canvas.draw()
    rgba = np.asarray(worker["fig"].canvas.buffer_rgba())
    if args.stills and step in args.stills:
        worker["plt"].imsave(f"{os.path.splitext(args.out)[0]}_{step:06d}.png", rgba)
    return rgba[:, :, :3].tobytes()


def render(args):
    fs = freestream(args.mach, args.re, args.t, args.alpha)
    mesh = Mesh(args.mesh)
    nodes, tris = mesh.plane()
    x, y = mesh.xyz[nodes, 0], mesh.xyz[nodes, 1]
    x0, x1, yc = args.window
    # The field's panel: the frame less margins and the lift's strip, square pixels.
    width, height = args.width - 140, int(args.height * 0.74)
    span = (x1 - x0) * height / width
    raster = Raster(x, y, tris, (x0, x1, yc - span / 2, yc + span / 2), width, height)
    time, cl = lift(args.history, fs, args.alpha, mesh.span)
    snaps = snapshots(args.snapdir, args.tag)[::args.stride]
    worker.update(args=args, raster=raster, fs=fs, time=time, cl=cl, nodes=nodes)
    print(f"{len(snaps)} frames, steps {snaps[0][0]}-{snaps[-1][0]}, {width}x{height} field pixels")
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)
    workers = args.jobs or len(os.sched_getaffinity(0))
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for i, data in enumerate(pool.imap(frame, snaps)):
            encoder.stdin.write(data)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1} frames", flush=True)
    encoder.stdin.close()
    if encoder.wait():
        sys.exit("ffmpeg failed")
    print(f"wrote {args.out}: {len(snaps)} frames at {args.fps} per second")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("render", "check"):
        p = sub.add_parser(mode)
        p.add_argument("mesh")
        p.add_argument("snapdir")
        p.add_argument("tag")
        p.add_argument("history")
        for name in ("mach", "re", "t", "alpha"):
            p.add_argument(name, type=float)
    p = sub.choices["render"]
    p.add_argument("out")
    p.add_argument("--field", choices=FIELDS, default="mach")
    p.add_argument("--window", type=lambda s: tuple(map(float, s.split(","))), default=(-0.35, 1.88, 0.35),
                   help="X0,X1,YC: the field's x range and its centre's y (default -0.35,1.88,0.35)")
    p.add_argument("--size", default="1920x1080", help="the frame's pixels")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1, help="every N-th snapshot")
    p.add_argument("--title", default="")
    p.add_argument("--schlieren", type=float, default=10.0, help="the schlieren's gain")
    p.add_argument("--stills", type=lambda s: {int(v) for v in s.split(",")}, default=set(),
                   help="steps whose frames are also written as OUT_NNNNNN.png")
    p.add_argument("--jobs", type=int, default=0, help="frames rendered at once (default: the CPUs)")
    args = parser.parse_args()
    if args.mode == "check":
        check(args)
    else:
        args.width, args.height = map(int, args.size.split("x"))
        render(args)


if __name__ == "__main__":
    main()
