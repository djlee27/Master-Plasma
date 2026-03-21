#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print DC boundary conditions from data ONLY (no model).

- Reads:
  data/grids.pkl
  data/results_last_iters.pkl

- Extracts BC at endpoints (first and last grid point) for:
  n_i, n_e, V, Gamma_i, Gamma_e

- Prints BC in:
  (A) SI units (as stored in results_last_iters.pkl)
  (B) Nondimensional units (same normalization as DC training code)
"""

from pathlib import Path
import pickle
import numpy as np
import scipy.constants

# ===========================
# USER CONFIG
# ===========================
DATA_DIR = Path("./data")     # contains grids.pkl, results_last_iters.pkl
T_INDEX = -1                  # use last iteration / last time index

# --- DC normalization constants (same as your DC code) ---
u_e = 20.0
D_e_SI = 50.0
D_i_SI = 0.01
gam = 0.01
V_dc_SI = -1000.0

eps0 = scipy.constants.epsilon_0
qe = scipy.constants.elementary_charge

L_SI = 2e-2
T_SI = 1e-8

n0d = 1e16
E0d = 1e5

# ===========================
# DC nondimensionalization (same as training)
# ===========================
t0d = eps0 / qe / n0d / u_e
R0d = eps0 / qe * E0d / n0d
alpha0d = 1.0 / R0d

def to_nondim_bc(ni_SI, ne_SI, V_SI, Gi_SI, Ge_SI):
    """
    Convert SI boundary values to nondimensional values used in training.
    """
    ni = ni_SI / n0d
    ne = ne_SI / n0d
    V  = V_SI  / (E0d * R0d)
    Gi = Gi_SI / (u_e * E0d * n0d)
    Ge = Ge_SI / (u_e * E0d * n0d)
    return ni, ne, V, Gi, Ge

# ===========================
# Load data
# ===========================
def load_dc_data(data_dir: Path):
    with open(data_dir / "grids.pkl", "rb") as f:
        grids = pickle.load(f)

    z_nV = np.asarray(grids["nV"])  # SI grid for n,V
    z_GE = np.asarray(grids["GE"])  # SI grid for Gamma,E

    with open(data_dir / "results_last_iters.pkl", "rb") as f:
        res = pickle.load(f)

    # Take last index
    n_i = np.asarray(res["n_i"][T_INDEX])
    n_e = np.asarray(res["n_e"][T_INDEX])
    V   = np.asarray(res["V"][T_INDEX])
    Gi  = np.asarray(res["Gamma_i"][T_INDEX])
    Ge  = np.asarray(res["Gamma_e"][T_INDEX])

    return z_nV, z_GE, n_i, n_e, V, Gi, Ge

def print_bc():
    z_nV, z_GE, n_i, n_e, V, Gi, Ge = load_dc_data(DATA_DIR)

    # Endpoints in SI
    bc_si = {
        "z0_nV": float(z_nV[0]),
        "zL_nV": float(z_nV[-1]),
        "z0_GE": float(z_GE[0]),
        "zL_GE": float(z_GE[-1]),
        "n_i_0": float(n_i[0]),
        "n_i_L": float(n_i[-1]),
        "n_e_0": float(n_e[0]),
        "n_e_L": float(n_e[-1]),
        "V_0": float(V[0]),
        "V_L": float(V[-1]),
        "Gamma_i_0": float(Gi[0]),
        "Gamma_i_L": float(Gi[-1]),
        "Gamma_e_0": float(Ge[0]),
        "Gamma_e_L": float(Ge[-1]),
    }

    # Convert to nondimensional BC
    ni0, ne0, V0, Gi0, Ge0 = to_nondim_bc(bc_si["n_i_0"], bc_si["n_e_0"], bc_si["V_0"], bc_si["Gamma_i_0"], bc_si["Gamma_e_0"])
    niL, neL, VL, GiL, GeL = to_nondim_bc(bc_si["n_i_L"], bc_si["n_e_L"], bc_si["V_L"], bc_si["Gamma_i_L"], bc_si["Gamma_e_L"])

    bc_nd = {
        "n_i_0": float(ni0), "n_i_L": float(niL),
        "n_e_0": float(ne0), "n_e_L": float(neL),
        "V_0": float(V0),   "V_L": float(VL),
        "Gamma_i_0": float(Gi0), "Gamma_i_L": float(GiL),
        "Gamma_e_0": float(Ge0), "Gamma_e_L": float(GeL),
    }

    print("\n==============================")
    print("DC Boundary Conditions (SI)")
    print("==============================")
    print(f"z_nV:  z0={bc_si['z0_nV']:.6e} m, zL={bc_si['zL_nV']:.6e} m")
    print(f"z_GE:  z0={bc_si['z0_GE']:.6e} m, zL={bc_si['zL_GE']:.6e} m")
    print(f"n_i:   n_i(0)={bc_si['n_i_0']:.6e},   n_i(L)={bc_si['n_i_L']:.6e}")
    print(f"n_e:   n_e(0)={bc_si['n_e_0']:.6e},   n_e(L)={bc_si['n_e_L']:.6e}")
    print(f"V:     V(0)  ={bc_si['V_0']:.6e} V,   V(L)  ={bc_si['V_L']:.6e} V")
    print(f"Gamma_i: Γ_i(0)={bc_si['Gamma_i_0']:.6e}, Γ_i(L)={bc_si['Gamma_i_L']:.6e}")
    print(f"Gamma_e: Γ_e(0)={bc_si['Gamma_e_0']:.6e}, Γ_e(L)={bc_si['Gamma_e_L']:.6e}")

    print("\n=========================================")
    print("DC Boundary Conditions (NON-DIM, training)")
    print("=========================================")
    print(f"(scales) n0={n0d:.3e}, E0={E0d:.3e}, R0={R0d:.3e}, u_e={u_e:.3e}")
    print(f"n_i:     n_i(0)={bc_nd['n_i_0']:.6e},     n_i(L)={bc_nd['n_i_L']:.6e}")
    print(f"n_e:     n_e(0)={bc_nd['n_e_0']:.6e},     n_e(L)={bc_nd['n_e_L']:.6e}")
    print(f"V:       V(0)  ={bc_nd['V_0']:.6e},       V(L)  ={bc_nd['V_L']:.6e}")
    print(f"Gamma_i: Γ_i(0)={bc_nd['Gamma_i_0']:.6e}, Γ_i(L)={bc_nd['Gamma_i_L']:.6e}")
    print(f"Gamma_e: Γ_e(0)={bc_nd['Gamma_e_0']:.6e}, Γ_e(L)={bc_nd['Gamma_e_L']:.6e}")


#%%
print_bc()
# %%
