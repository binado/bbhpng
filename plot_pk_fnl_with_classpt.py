"""
Galaxy/halo power spectrum with local PNG using Model class infrastructure.

This script computes and plots galaxy power spectra for different redshifts
and f_NL values using the Model class from modelclass.py, similar to how
fisher.py uses it. It emulates the functionality of plot_pk_of_z_fnl.py
but leverages the full model infrastructure.

Outputs:
  Pg_results[(z, fNL)] = (k_h, P_g(k,z; fNL)) with k in h/Mpc.
  Saves plot to Pg_fnl_model.png
"""

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

import fiducial_cosmo
import modellib
from config import Config

if __name__ == "__main__":
    # Load configuration from config.toml
    config = Config.from_cli()

    # Define redshifts and f_NL values to compute
    z_list = [0.0, 1.0, 2.0]
    f_nl_list = [-50, 0, 50]

    # Extract configuration parameters
    # Helper function to safely get config values with defaults
    def get_config(key, default=None):
        try:
            return config[key]
        except:
            return default

    config_args = {
        "ntracers": 1,  # Single tracer for auto-correlation
        "eps": config["tracers.eps"],
        "noise_amplitude": config["tracers.noise_amplitude"],
        "kmin_value": config["tracers.kmin_value"],
        "kmax_value": config["tracers.kmax_value"],
        "V1440_factor": config["tracers.V1440_factor"],
        "nbar_value": config["tracers.nbar_value"],
        "b1set": config["bias.b1set"],
        "b2set": config["bias.b2set"],
        "bG2set": config["bias.bG2set"],
        "bGamma3set": config["bias.bGamma3set"],
        "bNabla2set": config["bias.bNabla2set"],
        "b_phiset": get_config("bias.b_phiset", []),
        "cct_2_0_set": config["bias.cct_2_0_set"],
        "cct_2_2_set": config["bias.cct_2_2_set"],
        "cct_2_4_set": config["bias.cct_2_4_set"],
        "cct_4_4_set": config["bias.cct_4_4_set"],
        "cct_4_6_set": config["bias.cct_4_6_set"],
        "use_universality_relation": config["bias.use_universality_relation"],
        "universality_relation_p": config["bias.universality_relation_p"],
        "omega_cdm_fid": config["cosmology.omega_cdm_fid"],
        "h_fid": config["cosmology.h_fid"],
        "As_fid": config["cosmology.As_fid"],
        "includeAs": config["cosmology.includeAs"],
        "nonlin": False,  # Linear theory only
        "nocross_stoch": config["power_spectrum.nocross_stoch"],
        "RSD": False,  # Real-space only
        "multipoles": [True, False, False],  # Only monopole
        "active_bias": config["power_spectrum.active_bias"],
        "active_stoch": config["power_spectrum.active_stoch"],
        "open_precomputed_files": False,  # Compute CLASS on-the-fly for all redshifts
        "save_precomputed_files": False,  # Don't save precomputed files
    }

    # Default fiducial cosmology values
    default_args = {
        "omega_cdm_fid": fiducial_cosmo.OMEGA_CDM_FID,
        "h_fid": fiducial_cosmo.H_FID,
        "As_fid": fiducial_cosmo.AS_FID,
    }

    # Merge default and config args
    merged_args = {**default_args, **config_args}

    # Get verbose flag
    verbose = config["plotting.verbose"]

    if verbose:
        print("Configuration parameters:")
        pprint(merged_args)

    # Dictionary to store power spectra results
    Pg_results = {}

    # Compute power spectra for each (z, f_NL) combination
    if verbose:
        print("\nComputing power spectra...")
    for z in z_list:
        for fNL in f_nl_list:
            if verbose:
                print(f"\nProcessing z={z:.1f}, f_NL={fNL:+d}")

            # Create Model instance with specific z and f_NL
            model = modellib.set_model(
                **merged_args,
                redshift=z,
                f_NL_fid=fNL,
            )

            # Extract k-grid (in h/Mpc)
            k_h = model.k_loc

            # Extract power spectrum (in (h/Mpc)^-3)
            # For single tracer, model.Pdata is 1D array of length len_k
            Pg = model.Pdata

            # Store results
            Pg_results[(z, fNL)] = (k_h, Pg)

            if verbose:
                print(f"  Computed: k range = [{k_h[0]:.4f}, {k_h[-1]:.4f}] h/Mpc")
                print(f"  P_g range = [{Pg.min():.2e}, {Pg.max():.2e}] (h/Mpc)^-3")

    # Print summary
    if verbose:
        print("\nDone. Available keys for Pg_results (z, fNL):")
        for key in sorted(Pg_results.keys()):
            print("  ", key)

    # Plotting
    if verbose:
        print("\nGenerating plot...")

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

    # Add shot noise reference line
    h = merged_args["h_fid"]
    nbar = merged_args["nbar_value"]
    if isinstance(nbar, list):
        nbar = nbar[0]  # Take first tracer if list
    shot_noise = 1 / nbar / h**3
    plt.axhline(y=shot_noise, ls="--", color="k", label="Shot noise")

    plt.xlabel(r"$k \,[h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P_g(k)\,[(h^{-1}\mathrm{Mpc})^3]$")
    plt.title("Galaxy/halo power spectrum with PNG bias (Model class)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    # Save and show
    outpath = Path("Pg_fnl_model.png")
    plt.savefig(outpath, dpi=180)
    plt.show()

    print(f"\nSaved plot to: {outpath}")
