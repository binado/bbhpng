"""
Galaxy/halo power spectrum with local PNG (k^-2) scale-dependent bias.

Notation (consistent throughout):
  h             = H0 / (100 km s^-1 Mpc^-1)
  Omega_b, Omega_c, Omega_m = dimensionless density fractions today
  omega_b, omega_c          = physical densities = Omega_* * h^2
  CLASS expects 'omega_b' and 'omega_cdm' (physical densities).

PNG bias model (Dalal et al. 2008):
  Δb_PNG(k, z) = 2 f_NL (b1 - 1) δ_c / M(k, z)
  M(k, z) = (2/3) * k^2 * T(k) * D(z) / (Omega_m * H0^2)
where k is in 1/Mpc inside M, T(k) is the (matter) transfer function,
and D(z) is normalized to D(0)=1.

Outputs:
  Pg_results[(z, fNL)] = (k_h, P_g(k,z; fNL)) with k in h/Mpc.
No plots are produced.
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# User controls
# -----------------------------
z_list = [0.0, 1.0, 2.0]
f_nl_list = [-50, 0, 50]
h = 0.6736
kmin_h, kmax_h, nk = 5e-4 * h, 1e-1 * h, 400
delta_c = 1.686
p_value = 2.0
b1_value = 1.0  ## fixed in z
nbar = 2e-7


def b1_of_z(z: float) -> float:
    # Example tracer bias; customize as needed
    return b1_value
    # return 1.5 + 0.8*z


# -----------------------------
# Cosmology (Planck-like) & notation
# -----------------------------
Omega_b = 0.0493
Omega_c = 0.2649
Omega_m = Omega_b + Omega_c

# Physical densities (this is the key fix):
omega_b = Omega_b * h**2  # = Ω_b h^2
omega_c = Omega_c * h**2  # = Ω_c h^2

n_s = 0.9649
A_s = 2.1e-9
tau = 0.0543

# --- Replace the H0 constants with these (units fixed) ---
c_km_s = 299792.458
H0_km_s_Mpc = 100.0 * h  # km/s/Mpc
H0_over_c = H0_km_s_Mpc / c_km_s  # 1/Mpc
H0_over_c_sq = H0_over_c**2  # 1/Mpc^2

# k-grids
k_h = np.logspace(np.log10(kmin_h), np.log10(kmax_h), nk)  # h/Mpc
k_1Mpc = k_h * h  # 1/Mpc for CLASS / M(k)

# -----------------------------
# Try CLASS for P_m, D(z), and T(k); else fall back
# -----------------------------
_use_classy = False
try:
    from classy import Class

    _use_classy = True
except Exception:
    pass

Pm_results: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
D_of_z: Dict[float, float] = {}
T_of_k: np.ndarray = None


def compute_with_classy():
    cosmo = Class()
    params = {
        # Provide PHYSICAL densities to CLASS:
        "h": h,
        "omega_b": omega_b,  # correct: ω_b = Ω_b h^2
        "omega_cdm": omega_c,  # correct: ω_c = Ω_c h^2
        "A_s": A_s,
        "n_s": n_s,
        "tau_reio": tau,
        "output": "mPk,mTk",
        "P_k_max_h/Mpc": 50.0,
        "z_pk": ",".join(str(z) for z in sorted(set(z_list))),
    }
    cosmo.set(params)
    cosmo.compute()

    # Matter P(k,z) (CLASS returns in Mpc^3) -> convert to (h/Mpc)^3 by multiplying h^3
    for z in z_list:
        pk = np.array([cosmo.pk_lin(kk, z) for kk in k_1Mpc]) * (h**3)
        Pm_results[(z, 0.0)] = (k_h.copy(), pk)

    # Growth factor normalized to D(0)=1
    for z in z_list:
        D_of_z[z] = cosmo.scale_independent_growth_factor(z)

    # Transfer function T(k) (shape). We de-grow from some z_T to get ~z=0 convention.
    z_T = min(z_list)
    tr = cosmo.get_transfer(z_T)  # dict; includes 'k' in 1/Mpc and various components
    # Prefer a total matter-like key
    for key in ["d_cb", "d_m", "d_tot"]:
        if key in tr:
            dm = tr[key]
            break
    else:
        if "d_cdm" in tr and "d_b" in tr:
            dm = tr["d_cdm"] + tr["d_b"]
        else:
            raise KeyError("No suitable matter transfer found in CLASS output.")

    Dz = D_of_z[z_T]
    T_z0_shape = dm / Dz  # remove growth to approximate z=0 shape
    T_interp = np.interp(k_1Mpc, tr["k"], T_z0_shape)
    cosmo.struct_cleanup()
    cosmo.empty()
    return T_interp


def T_eh98(k_h, omh2, obh2, h):
    """Simplified Eisenstein–Hu 98 no-wiggle transfer (shape only)."""
    om = omh2 / h**2
    ob = obh2 / h**2
    fb = ob / om
    theta = 2.725 / 2.7
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10.0 * (obh2**0.75))  # Mpc/h
    alpha = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * (fb**2)
    gamma_eff = omh2 * (alpha + (1 - alpha) / (1 + (0.43 * k_h * s) ** 4))
    q = k_h * theta**2 / gamma_eff
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


def growth_factor_approx(z, Om=Omega_m, Ol=1.0 - Omega_m):
    """Carroll–Press–Turner D(z) normalized to D(0)=1."""
    a = 1 / (1 + z)
    Ez = np.sqrt(Om * (1 + z) ** 3 + Ol)
    Omz = Om * (1 + z) ** 3 / Ez**2
    Olz = Ol / Ez**2
    gz = 5 * Omz / (2 * (Omz ** (4 / 7) - Olz + (1 + 0.5 * Omz) * (1 + Olz / 70)))
    Ez0 = np.sqrt(Om + Ol)
    Om0 = Om / Ez0**2
    Ol0 = Ol / Ez0**2
    g0 = 5 * Om0 / (2 * (Om0 ** (4 / 7) - Ol0 + (1 + 0.5 * Om0) * (1 + Ol0 / 70)))
    return a * gz / g0


def Pm_eh98_shape(k_h, ns=n_s):
    """Unnormalized P_m shape using EH98 T(k) and primordial tilt."""
    T = T_eh98(k_h, omh2=Omega_m * h**2, obh2=omega_b, h=h)
    return (k_h**ns) * (T**2)


# Compute ingredients
if _use_classy:
    try:
        T_of_k = compute_with_classy()
    except Exception:
        T_of_k = None

if len(Pm_results) == 0:
    # Fallback: build P_m shape and add growth; normalize at k=0.2 h/Mpc for convenience
    P0 = Pm_eh98_shape(k_h)
    P0 *= 1.0 / np.interp(0.2, k_h, P0)
    for z in z_list:
        D = growth_factor_approx(z)
        D_of_z[z] = D
        Pm_results[(z, 0.0)] = (k_h.copy(), (D**2) * P0)

if T_of_k is None:
    # Fallback T(k) shape
    T_of_k = T_eh98(k_h, omh2=Omega_m * h**2, obh2=omega_b, h=h)


# -----------------------------
# PNG scale-dependent bias and galaxy/halo P_g
# -----------------------------
# --- Replace delta_b_PNG with this corrected version ---
def delta_b_PNG(k_h, z, fNL, b1, T_k, D_z, k_floor_h=1e-4):
    """
    Δb_PNG(k,z) = 2 fNL (b1 - 1) δ_c / M(k,z),
    M(k,z)      = (2/3) * k^2 * T(k) * D(z) / (Ω_m * H0^2)

    Here:
      - k_h is in h/Mpc; we convert to 1/Mpc via k = k_h * h
      - H0 is used as H0/c in 1/Mpc to keep M dimensionless
      - T(k) is dimensionless and de-grown to z=0 (we still multiply by D(z))
      - D(z) is normalized to D(0)=1
    """
    # Avoid pathologies at extremely small k:
    k_h_eff = np.maximum(k_h, k_floor_h)
    k_1Mpc = k_h_eff * h  # 1/Mpc

    # M(k,z) with consistent units
    M_kz = (2.0 / 3.0) * (k_1Mpc**2) * T_k * D_z / (Omega_m * H0_over_c_sq)

    # Guard against any numerical zeros:
    M_kz = np.where(M_kz > 0, M_kz, np.inf)

    return 2.0 * fNL * (b1 - p_value) * delta_c / M_kz


Pg_results: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
for z in z_list:
    k_grid, Pm_k = Pm_results[(z, 0.0)]
    b1 = b1_of_z(z)
    D_z = D_of_z[z]
    for fNL in f_nl_list:
        dB = delta_b_PNG(k_grid, z, fNL, b1, T_of_k, D_z)
        Pg = (b1 + dB) ** 2 * Pm_k
        Pg_results[(z, fNL)] = (k_grid, Pg)

# Minimal confirmation printout
print("Done. Available keys for Pg_results (z, fNL):")
for key in sorted(Pg_results.keys()):
    print("  ", key)

# Example for saving one case:
# z0, f0 = 0.0, 50
# k_out, Pg_out = Pg_results[(z0, f0)]
# np.savez("Pg_z{:.1f}_fNL{:+d}.npz".format(z0, f0), k=k_out, Pg=Pg_out, z=z0, fNL=f0)

# --- Add this plotting section at the END of the previous script ---


# --- Add this at the end of the script (after computing Pg_results) ---


# Define colors per redshift and linestyles per fNL
color_map = {0.0: "tab:blue", 1.0: "tab:orange", 2.0: "tab:green"}
linestyles = {-50: "dashed", 0: "solid", 50: "dotted"}

plt.figure(figsize=(7, 5))
for z in z_list:
    color = color_map.get(z, None)
    for fNL in f_nl_list:
        ls = linestyles.get(fNL, "solid")
        k_plot, Pg_plot = Pg_results[(z, fNL)]
        plt.loglog(
            k_plot,
            Pg_plot,
            color=color,
            linestyle=ls,
            label=f"z={z:.1f}, f_NL={fNL:+d}",
        )

shot_noise = 1 / nbar / h**3
plt.axhline(y=shot_noise, ls="--", color="k", label="Shot noise")
plt.xlabel(r"$k \,[h\,\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$P_g(k)\,[(h^{-1}\mathrm{Mpc})^3]$")
plt.title("Galaxy/halo power spectrum with PNG bias")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

# Save and show
outpath = Path("Pg_fnl_colored.png")
plt.savefig(outpath, dpi=180)
plt.show()

print(f"Saved combined PNG bias plot to: {outpath}")
