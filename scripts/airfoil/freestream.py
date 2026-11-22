"""The free stream of a unit-chord case from Mach, Reynolds number (on the
chord), static temperature and angle of attack, as nssolver options:

    python3 freestream.py MACH RE T ALPHA_DEG
    -> --rho .. --u .. --v .. --p .. --nu-tilde ..   (nu_tilde = 3 nu)
"""
import math
import sys

GAMMA, R = 1.4, 287.05
MU_REF, T_REF, SUTHERLAND = 1.7894e-5, 288.15, 110.4


def freestream(mach, re, t, alpha_deg):
    a = math.sqrt(GAMMA * R * t)
    speed = mach * a
    mu = MU_REF * (t / T_REF) ** 1.5 * (T_REF + SUTHERLAND) / (t + SUTHERLAND)
    rho = re * mu / speed
    alpha = math.radians(alpha_deg)
    return {"rho": rho, "u": speed * math.cos(alpha), "v": speed * math.sin(alpha), "p": rho * R * t,
            "nu-tilde": 3.0 * mu / rho, "speed": speed}


if __name__ == "__main__":
    fs = freestream(*map(float, sys.argv[1:5]))
    print(" ".join(f"--{k} {fs[k]:.12g}" for k in ("rho", "u", "v", "p", "nu-tilde")))
