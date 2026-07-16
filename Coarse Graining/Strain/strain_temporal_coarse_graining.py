# Temporal Coarse graining strain Fields 
import torch
import numpy as np
from tqdm import tqdm


def strain_based_coarse_graining(strain_scalar, strain_rate, dump_freq, time_step,
                                  strain_half_window=0.020, overlap_fraction=0.89):
    """
    Temporally coarse-grain the stress tensor using a Gaussian window centred
    at uniformly spaced strain values.

    The window is specified entirely in STRAIN space so that results are
    independent of dump frequency and timestep once those are fixed.

    Parameters
    ----------
    strain_scalar : ndarray, shape (N_dumps, N_grid, 3, 3)
    strain_rate   : float   [1/s]
    dump_freq     : int     [timesteps per dump]
    time_step     : float   [s]
    strain_half_window : float
        Half-width of the averaging window in strain units.
        Default = 0.5 (average over Δγ = 1.0, centred on each output point).
        The Gaussian sigma is set to strain_half_window / 2, so that weights
        fall to e^{-2} ≈ 0.14 at the window edges.
    overlap_fraction : float in [0, 1)
        Fraction of the window that adjacent output centres share.
        0 = non-overlapping, 0.5 (default) = 50% overlap.
        For ML training, 0 or at most 0.5 is recommended to limit
        autocorrelation between samples.

    Returns
    -------
    cg_strain  : ndarray, shape (N_out, N_grid, 3, 3)
    out_strains: ndarray, shape (N_out,)   strain value at each output centre
    """

    N_dumps = strain_scalar.shape[0]
    dt_dump  = dump_freq * time_step                          # s per dump
    strain_per_dump = dt_dump * strain_rate                   # Δγ per dump

    # Total strain spanned by the simulation
    total_strain = N_dumps * strain_per_dump

    # Gaussian sigma in STRAIN units (not frames)
    #   sigma = half_window / 2  so weights at edges are e^{-2}
    sigma_strain = strain_half_window / 2.0

    # Step between output centres in strain units
    stride_strain = strain_half_window * (1.0 - overlap_fraction)

    print(f"Temporal CG parameters")
    print(f"  dt_dump            = {dt_dump:.4f} s")
    print(f"  strain per dump    = {strain_per_dump:.4f}")
    print(f"  total strain       = {total_strain:.2f}")
    print(f"  window half-width  = {strain_half_window:.3f}  (strain units)")
    print(f"  Gaussian sigma     = {sigma_strain:.3f}  (strain units)")
    print(f"  output stride      = {stride_strain:.3f}  (strain units)")
    print(f"  frames in window   = {strain_half_window / strain_per_dump:.1f}")

    # Output centre strains: from half_window to (total - half_window)
    out_strains = np.arange(strain_half_window,
                            total_strain - strain_half_window + stride_strain,
                            stride_strain)
    print(f"  N output frames    = {len(out_strains)}")

    cg_strain = np.zeros((len(out_strains),) + strain_scalar.shape[1:],
                         dtype=np.float64)

    # Strain value at each dump
    dump_strains = np.arange(N_dumps) * strain_per_dump   # shape (N_dumps,)

    for k, gamma_centre in enumerate(
            tqdm(out_strains, desc="Temporal coarse graining")):

        #1. Gaussian weights in STRAIN space 
        delta_gamma = dump_strains - gamma_centre           # shape (N_dumps,)
        weights     = np.exp(-0.5 * (delta_gamma / sigma_strain) ** 2)

        # Zero out frames outside 3-sigma support to avoid negligible contributions
        weights[np.abs(delta_gamma) > 3.0 * sigma_strain] = 0.0

        w_sum = weights.sum()
        if w_sum < 1e-12:
            # No frames in window — copy nearest available frame
            nearest = int(np.round(gamma_centre / strain_per_dump))
            nearest = np.clip(nearest, 0, N_dumps - 1)
            cg_strain[k] = strain_scalar[nearest]
            continue

        weights /= w_sum

        # Weighted sum over all contributing dumps
        # tensordot contracts axis 0 of weights (N_dumps,) with axis 0 of
        # strain_scalar (N_dumps, N_grid, 3, 3) -> (N_grid, 3, 3)
        cg_strain[k] = np.tensordot(weights, strain_scalar, axes=(0, 0))

    print("Temporal CG complete")
    return cg_strain
     

import matplotlib.pyplot as plt
from pathlib import Path
import os

os.makedirs(
    "/Users/kavyanshrajsingh/Desktop/Data/strain_temporal_cgs",
    exist_ok=True
)

os.makedirs(
    "/Users/kavyanshrajsingh/Desktop/Data/fabric_temporal_cgs",
    exist_ok=True
)
import re


print("Loading coarse-grained strain...")

s = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/strain_tensor_cg.pt",
    weights_only=False
)

print("Loading coarse-grained fabric...")

f = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/fabric_tensor_cg.pt",
    weights_only=False
)

print("Loaded.")

print("Strain shape :", s.shape)
print("Fabric shape :", f.shape)
#===================================================================================
strain_rate = 0.065249326754243 
dump_freq = 4500
timeStep = 5.77e-7
strain_ = s.squeeze(-1).numpy()
fabric_ = f.numpy()
#=======================================================================================
#=======================================================================================

cg_strain = strain_based_coarse_graining(
    strain_,
    strain_rate,
    dump_freq,
    timeStep
)

torch.save(
    torch.from_numpy(cg_strain),
    "/Users/kavyanshrajsingh/Desktop/Data/strain_temporal_cgs/strain_temporal_cg.pt"
)

print("✓ Saved strain_temporal_cg.pt")

cg_fabric = strain_based_coarse_graining(
    fabric_,
    strain_rate,
    dump_freq,
    timeStep
)

torch.save(
    torch.from_numpy(cg_fabric),
    "/Users/kavyanshrajsingh/Desktop/Data/fabric_temporal_cgs/fabric_temporal_cg.pt"
)

print("✓ Saved fabric_temporal_cg.pt")

print()
print("Final output shapes")
print(np.array(cg_strain).shape)
print(np.array(cg_fabric).shape)