#Temporal Coarse Graining....
import numpy as np
from tqdm import tqdm
def strain_based_coarse_graining(phi, strain_rate, dump_freq, time_step,
                                  strain_half_window=0.02, overlap_fraction=0.89):
    """
    Temporally coarse-grain the stress tensor using a Gaussian window centred
    at uniformly spaced strain values.

    The window is specified entirely in STRAIN space so that results are
    independent of dump frequency and timestep once those are fixed.

    Parameters
    ----------
    phi : ndarray, shape (N_dumps, N_grid, 3, 3)
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
    cg_stress  : ndarray, shape (N_out, N_grid, 3, 3)
    out_strains: ndarray, shape (N_out,)   strain value at each output centre
    """

    N_dumps = phi.shape[0]
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

    cg_phi = np.zeros((len(out_strains),) + phi.shape[1:],
                         dtype=np.float64)

    # Strain value at each dump
    dump_strains = np.arange(N_dumps) * strain_per_dump   # shape (N_dumps,)

    for k, gamma_centre in enumerate(
        tqdm(out_strains,
             desc="Temporal coarse graining",
             unit="window")):

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
            cg_phi[k] = phi[nearest]
            continue

        weights /= w_sum

        # Weighted sum over all contributing dumps
        # tensordot contracts axis 0 of weights (N_dumps,) with axis 0 of
        # phi (N_dumps, N_grid, 3, 3) -> (N_grid, 3, 3)
        cg_phi[k] = np.tensordot(weights, phi, axes=(0, 0))

    print("Temporal CG complete")
    return cg_phi, out_strains

import torch
import os


if __name__ == "__main__":

    # -------------------------------------------------
    # Load spatially coarse-grained volume fraction
    # -------------------------------------------------

    phi = torch.load(
        "/Users/kavyanshrajsingh/Desktop/Data/volume_fraction_cgs/phi_smooth_timeseries.pt"
    )

    if torch.is_tensor(phi):
        phi = phi.numpy()

    # -------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------

    strain_rate = 0.065249326754243     # your simulation
    time_step   = 5.77e-7
    dump_freq   = 4500

    # -------------------------------------------------
    # Temporal coarse graining
    # -------------------------------------------------

    cg_phi, out_strains = strain_based_coarse_graining(
        phi,
        strain_rate,
        dump_freq,
        time_step,
        strain_half_window=0.02,
        overlap_fraction=0.89
    )

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    output_dir = "volume_fraction_temporal_cgs"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        torch.from_numpy(cg_phi),
        os.path.join(output_dir, "phi_temporal_cg.pt")
    )

    torch.save(
        torch.from_numpy(out_strains),
        os.path.join(output_dir, "output_strains.pt")
    )

    print("\n✓ Temporal coarse graining finished.")
    print(f"Saved {len(out_strains)} output frames.")