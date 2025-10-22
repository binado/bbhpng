# /// script
# dependencies = [
#   "numpy",
# ]
# ///
"""
Generate redshift binning data from an input file containing redshift samples.

Reads a catalog file containing redshift data (z) and generates binning
information based on the specified binning method. The results are written
to a CSV file with columns for bin labels, left edges, right edges, and
the number of sources in each bin.
"""

import argparse
import csv
from pathlib import Path
from typing import Literal

import numpy as np
from classy import Class
from numpy.typing import ArrayLike

from fiducial_cosmo import AS_FID, H_FID, OMEGA_CDM_FID

DEFAULT_PARAMS = {
    "A_s": AS_FID * 1e-9,
    "n_s": 0.966,
    "tau_reio": 0.054,
    "omega_b": 0.02238,
    "omega_cdm": OMEGA_CDM_FID,
    "h": H_FID,
    "N_ur": 3.044,
    "YHe": 0.2425,
    "output": "mPk",
}


class Formatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


def get_binning_data(
    a: ArrayLike,
    amin: float,
    amax: float,
    nbins: int,
    method: Literal["fixed", "equal"] = "fixed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split an array of samples into bins

    Parameters
    ----------
    a : array_like
        Array of samples
    amin : float
        Left edge of first bin
    amax : float
        Right edge of last bin
    nbins : int
        Number of bins
    method : {"fixed", "equal"}
        If "fixed", all bins have the same width.
        If "equal", all bins have the same number of samples.

    Returns
    -------
    ndarray (N,)
        bins labelled 1 to N
    ndarray (nbins + 1,)
        array of bin edges
    ndarray (nbins,)
        number of samples per bin
    """
    a_sorted = np.array(a, copy=True)
    a_sorted.sort()
    bin_labels = np.arange(1, nbins + 1, dtype=np.int64)
    if method == "fixed":
        bin_edges = np.linspace(amin, amax, nbins + 1)
    elif method == "equal":
        quantiles = np.linspace(0, 1, nbins + 1)
        bin_edges = np.quantile(a_sorted, quantiles)

    bin_edges_indices = np.searchsorted(a_sorted, bin_edges)
    num_objects_per_bin = np.diff(bin_edges_indices).astype(np.int64)
    return bin_labels, bin_edges, num_objects_per_bin


def main(
    input_file: str,
    output_file: str,
    bias: ArrayLike,
    k_pivot: float,
    zmin: float = 0,
    zmax: float = np.inf,
    method: Literal["fixed", "equal"] = "fixed",
    **kwargs,
) -> None:
    """
    Generate redshift binning data from an input file containing redshift samples.

    Reads a catalog file containing redshift data (z) and generates binning
    information based on the specified binning method. The results are written
    to a {.npy,.csv} file with columns for bin labels, left edges, right edges, and
    the number of sources in each bin.

    Parameters
    ----------
    input_file : str
        Path to the input .npy file containing redshift data.
    output_file : str
        Path to the output file where binning results will be written {.npy, .csv}
    bias : array_like
        Bias values for each bin, determines the number of bins.
    k_pivot : float
        Pivot wavenumber for computing effective volume.
    zmin : float
        Left edge of first bin. Defaults to 0
    zmax : float
        Right edge of last bin. Defaults to np.inf, so that the maximum is
        inferred from the redshift array
    method : {"fixed", "equal"}, optional
        Binning method to use. "fixed" creates bins with fixed width,
        while "equal" creates bins with equal numbers of sources (quantile-based).
        Default is "fixed".
    kwargs:
        Parameters passed to the Class instance.

    Returns
    -------
    None
    """
    z = np.load(input_file)
    zmin = max(zmin, z.min())
    zmax = min(zmax, z.max())
    bias = np.asarray(bias)
    nbins = bias.shape[0]
    labels, edges, number_of_sources = get_binning_data(
        z, zmin, zmax, nbins, method=method
    )
    left, right = edges[:-1], edges[1:]
    centers = 0.5 * (left + right)

    cosmo = Class()
    zmax_pk = max(zmax, edges[-1])
    cosmo.set(z_max_pk=zmax_pk, **DEFAULT_PARAMS, **kwargs)
    cosmo.compute()

    # Compute volume, kmin, nbar
    dl_fn = np.vectorize(cosmo.luminosity_distance)
    dl = dl_fn(edges)
    dm = dl / (1 + edges)
    volume = dm**3 * 4 * np.pi / 3  # Flat space only
    volume_diff = np.diff(volume)
    kmin = 2 * np.pi / np.cbrt(volume_diff)
    nbar = number_of_sources.astype(np.float64) / volume_diff

    # Compute effective volume following Eq.(18) of https://arxiv.org/pdf/2504.18245
    # Vectorize pk computation over z
    pk = np.vectorize(cosmo.pk_lin, excluded=[0])
    pk_centers = pk(k_pivot, centers)
    fkp_factor_numerator = bias**2 * pk_centers / nbar
    fkp_factor = fkp_factor_numerator / (fkp_factor_numerator + 1)
    effective_volume = np.sum(volume_diff * fkp_factor**2)

    data = {
        "bin": labels,
        "left_edge": left,
        "right_edge": right,
        "number_of_sources": number_of_sources,
        "volume": volume_diff,
        "nbar": nbar,
        "kmin": kmin,
    }
    print(f"Total volume up to redshift {zmax:.1f}: {volume[-1]:.2f} Mpc^3")
    print(f"Effective volume: {effective_volume:.2f} Mpc^3")

    output_filepath = Path(output_file)
    if output_filepath.suffix == ".npz":
        np.save(output_filepath, **data)
    else:
        with open(output_filepath, mode="w") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writeheader()
            for i in range(nbins):
                writer.writerow({k: v[i] for k, v in data.items()})


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate redshift binning data from an HDF5 catalog file.",
        formatter_class=Formatter,
        epilog=__doc__,
    )

    # Positional arguments
    parser.add_argument(
        "input_file",
        help="Path to the input .npy file containing redshift data.",
    )
    parser.add_argument(
        "output_file",
        help="Path to the output file where binning results will be written {.npy, .csv}",
    )

    # Optional arguments
    parser.add_argument(
        "--bias",
        type=float,
        nargs="+",
        required=True,
        help="Bias values for each bin (space-separated floats).",
    )
    parser.add_argument(
        "--k-pivot",
        type=float,
        required=True,
        help="Pivot wavenumber for computing effective volume.",
    )
    parser.add_argument(
        "--zmin", type=float, default=0, help="Left edge of first redshift bin"
    )
    parser.add_argument(
        "--zmax", type=float, default=np.inf, help="Right edge of last redshift bin"
    )
    parser.add_argument(
        "--method",
        choices=["fixed", "equal"],
        default="fixed",
        help=(
            "Binning method to use. 'fixed' creates bins with fixed width, "
            "'equal' creates bins with equal numbers of sources (default: fixed)."
        ),
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(
        input_file=args.input_file,
        output_file=args.output_file,
        bias=args.bias,
        k_pivot=args.k_pivot,
        zmin=args.zmin,
        zmax=args.zmax,
        method=args.method,
    )
