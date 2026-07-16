"""
Temporal coarse-graining of spatially CG'd stress tensors.

Bug fixes relative to original:
  1. Gaussian kernel units mismatch  (gaussian_width was in frames, argument
     was in strain -> the kernel was nearly flat for ALL vt cases)
  2. Output step was increment/5  ->  5x over-sampled, 80% autocorrelated
     windows for slow cases (vt003). Step is now frames_to_be_averaged/2
     (50% overlap) which is a standard choice for sliding estimators.
  3. Guard against int() truncating step to 0 on very short dumps.
  4. Removed the inconsistent `if frames_to_be_averaged < 5` branch;
     behaviour is now uniform across all vt cases.
"""
#==================================================================================================================#
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import re


# ======================================================================
# Core temporal CG
# ======================================================================

def strain_based_coarse_graining(stress_tensor, strain_rate, dump_freq, time_step,
                                  strain_half_window=0.5, overlap_fraction=0.5):
    """
    Temporally coarse-grain the stress tensor using a Gaussian window centred
    at uniformly spaced strain values.

    The window is specified entirely in STRAIN space so that results are
    independent of dump frequency and timestep once those are fixed.

    Parameters
    ----------
    stress_tensor : ndarray, shape (N_dumps, N_grid, 3, 3)
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

    N_dumps = stress_tensor.shape[0]
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

    cg_stress = np.zeros((len(out_strains),) + stress_tensor.shape[1:],
                         dtype=np.float64)

    # Strain value at each dump
    dump_strains = np.arange(N_dumps) * strain_per_dump   # shape (N_dumps,)

    from tqdm import tqdm

    for k, gamma_centre in enumerate(
        tqdm(
            out_strains,
            desc="Temporal coarse graining",
            unit="window"
        )
    ):

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
            cg_stress[k] = stress_tensor[nearest]
            continue

        weights /= w_sum

        # Weighted sum over all contributing dumps
        # tensordot contracts axis 0 of weights (N_dumps,) with axis 0 of
        # stress_tensor (N_dumps, N_grid, 3, 3) -> (N_grid, 3, 3)
        cg_stress[k] = np.tensordot(weights, stress_tensor, axes=(0, 0))

    print("Temporal CG complete")
    return cg_stress, out_strains


# ======================================================================
# I/O
# ======================================================================

def load_stress_tensors(folderpath, pattern='stress_tensor_t*.pt'):
    folderpath = Path(folderpath)
    files      = list(folderpath.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {folderpath}")

    print(f"Found {len(files)} tensor files")

    pairs = []
    for f in files:
        m = re.search(r't(\d+)\.pt', f.name)
        if m:
            pairs.append((f, int(m.group(1))))

    pairs.sort(key=lambda x: x[1])
    sorted_files, sorted_timesteps = zip(*pairs)

    print(f"Timestep range: {sorted_timesteps[0]} to {sorted_timesteps[-1]}")

    first = torch.load(sorted_files[0], weights_only=False)
    n_grid = first['stress_tensor'].shape[0]
    print(f"Grid size : {n_grid}   shape: {first['grid_shape']}")

    n   = len(sorted_files)
    out = torch.zeros(n, n_grid, 3, 3)

    for i, (f, ts) in enumerate(zip(sorted_files, sorted_timesteps)):
        data   = torch.load(f, weights_only=False)
        out[i] = data['stress_tensor']

    print(f"Loaded {n} files   output shape: {out.shape}")
    return out, list(sorted_timesteps)


# ======================================================================
# Diagnostics
# ======================================================================

def plot_temporal_cg_diagnostics(raw_stress, cg_stress, out_strains,
                                  grid_shape, component=(0, 1)):
    """
    Compare raw vs CG stress for one component, averaged over the grid.
    Also shows the effective number of independent samples used.
    """
    i, j = component
    raw_mean = raw_stress[:, :, i, j].mean(axis=1)
    cg_mean  = cg_stress[:, :, i, j].mean(axis=1)

    n_raw = raw_stress.shape[0]
    strain_raw = np.linspace(0, out_strains[-1], n_raw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(strain_raw, raw_mean, alpha=0.4, linewidth=0.8, label='raw (per dump)')
    axes[0].plot(out_strains, cg_mean, linewidth=1.5, label='CG (strain averaged)')
    axes[0].set_xlabel('Strain γ')
    axes[0].set_ylabel(f'Mean σ_{["x","y","z"][i]}{["x","y","z"][j]}  [Pa]')
    axes[0].set_title('Raw vs temporally CG stress')
    axes[0].legend()

    # Show spatial distribution at middle of run
    mid_t  = cg_stress.shape[0] // 2
    nx, ny, nz = grid_shape
    mid_z  = nz // 2
    sfield = cg_stress[mid_t, :, i, j].reshape(grid_shape)[:, :, mid_z]

    im = axes[1].imshow(sfield.T, origin='lower', cmap='RdBu_r',
                        vmax=np.percentile(np.abs(sfield), 95),
                        vmin=-np.percentile(np.abs(sfield), 95),
                        aspect='auto')
    plt.colorbar(im, ax=axes[1], label='Pa')
    axes[1].set_title(f'σ_{["x","y","z"][i]}{["x","y","z"][j]}  at γ={out_strains[mid_t]:.2f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    plt.tight_layout()
    return fig

# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":

    folder = "/Users/kavyanshrajsingh/Desktop/Data/stress_tensors_CG"
    strain_rate = 0.065249326754243


    stress_raw, time_steps = load_stress_tensors(folderpath=folder)

    # Convert to numpy for CG (keep float64 for precision)
    stress_np = stress_raw.numpy().astype(np.float64)

    time_step  = 5.77e-7
    dump_freq  = 4500

    cg_stress, out_strains = strain_based_coarse_graining(
        stress_np,
        strain_rate  = strain_rate,
        dump_freq    = dump_freq,
        time_step    = time_step,
        strain_half_window = 0.02,   # average over Δγ = 0.04
        overlap_fraction   = 0.89,   # 50% overlap between output frames
    )
    print(f"Output CG stress shape: {cg_stress.shape}")

    grid_shape = (50, 77, 1)
    from pathlib import Path

    output_dir = Path("/Users/kavyanshrajsingh/Desktop/Data/stress_temporal_cgs")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        torch.from_numpy(cg_stress),
        output_dir/"stress_temporal_cg.pt"
    )
    # Diagnostics
    fig = plot_temporal_cg_diagnostics(stress_np, cg_stress, out_strains,
                                        grid_shape, component=(0, 1))
    # plt.savefig('temporal_cg_diagnostics.png', dpi=120)
    plt.show()
     
     