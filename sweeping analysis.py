import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cos_m(u, v):
    A = 1 + (v/2.0) * np.cos(u/2)
    return A * np.cos(u)


def sin_m(u, v):
    A = 1 + (v/2.0) * np.cos(u/2)
    return A * np.sin(u)


def zeta_m(u, v):
    return (v/2.0) * np.sin(u/2)


T = 4 * np.pi
N = 20001
u = np.linspace(0, T, N, endpoint=True)
v_values = np.linspace(-1.0, 1.0, 41)
max_n = 6


def analyze_function(f_values, u, max_n=6):
    energy = np.trapezoid(f_values**2, u)
    rms = np.sqrt(energy / (u[-1] - u[0]))
    fmax = np.max(f_values)
    fmin = np.min(f_values)

    a_n, b_n = [], []

    for n in range(1, max_n + 1):
        coef_a = (2.0 / (u[-1] - u[0])) * np.trapezoid(
            f_values * np.cos(2*np.pi*n*u/(u[-1] - u[0])), u
        )
        coef_b = (2.0 / (u[-1] - u[0])) * np.trapezoid(
            f_values * np.sin(2*np.pi*n*u/(u[-1] - u[0])), u
        )
        a_n.append(coef_a)
        b_n.append(coef_b)

    total_energy = energy / (u[-1] - u[0])

    frac_fundamental = (
        0.5 * (a_n[1]**2 + b_n[1]**2) / total_energy
        if total_energy > 0 else np.nan
    )

    frac_sidebands = (
        0.5 * (
            (a_n[0]**2 + b_n[0]**2) +
            (a_n[2]**2 + b_n[2]**2)
        ) / total_energy
        if total_energy > 0 else np.nan
    )

    return {
        "RMS": rms,
        "peak_to_peak": fmax - fmin,
        "a1": a_n[0], "a2": a_n[1], "a3": a_n[2],
        "b1": b_n[0], "b2": b_n[1], "b3": b_n[2],
        "frac_fundamental": frac_fundamental,
        "frac_sidebands": frac_sidebands
    }


results = {
    "v": [],
    "coord": [],
    "RMS": [],
    "peak_to_peak": [],
    "frac_fundamental": [],
    "frac_sidebands": []
}

coords = {
    "cos": cos_m,
    "sin": sin_m,
    "zeta": zeta_m
}


for coord_name, func in coords.items():
    for v in v_values:
        f_vals = func(u, v)
        metrics = analyze_function(f_vals, u, max_n=max_n)

        results["v"].append(v)
        results["coord"].append(coord_name)
        results["RMS"].append(metrics["RMS"])
        results["peak_to_peak"].append(metrics["peak_to_peak"])
        results["frac_fundamental"].append(metrics["frac_fundamental"])
        results["frac_sidebands"].append(metrics["frac_sidebands"])


df = pd.DataFrame(results)
print(df.round(6))


plt.figure(figsize=(12,4))
for coord_name in coords.keys():
    subset = df[df["coord"] == coord_name]
    plt.plot(subset["v"], subset["RMS"], marker='o', label=f"RMS-{coord_name}")

plt.title("RMS vs v")
plt.xlabel("v")
plt.ylabel("RMS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,4))
for coord_name in coords.keys():
    subset = df[df["coord"] == coord_name]
    plt.plot(
        subset["v"],
        subset["peak_to_peak"],
        marker='o',
        label=f"Peak-to-Peak-{coord_name}"
    )

plt.title("Peak-to-Peak amplitude vs v")
plt.xlabel("v")
plt.ylabel("Peak-to-Peak")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,4))
for coord_name in coords.keys():
    subset = df[df["coord"] == coord_name]
    plt.plot(
        subset["v"],
        subset["frac_fundamental"],
        marker='o',
        label=f"Fundamental-{coord_name}"
    )
    plt.plot(
        subset["v"],
        subset["frac_sidebands"],
        marker='s',
        linestyle='--',
        label=f"Sidebands-{coord_name}"
    )

plt.title("Energy fraction: fundamental vs sidebands")
plt.xlabel("v")
plt.ylabel("Fraction of total energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
