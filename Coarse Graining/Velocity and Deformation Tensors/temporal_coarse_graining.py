# Temporal Coarse Graining
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

os.makedirs(
    "/Users/kavyanshrajsingh/Desktop/Data/velocity_temporal_cgs",
    exist_ok=True
)
import torch
import re
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter  # ADD THIS
import matplotlib

def strain_based_coarse_graining(velocity, strain_rate, dump_freq, time_step,
                                  strain_half_window=0.02, overlap_fraction=0.89):
    """
    Temporally coarse-grain the stress tensor using a Gaussian window centred
    at uniformly spaced strain values.

    The window is specified entirely in STRAIN space so that results are
    independent of dump frequency and timestep once those are fixed.

    Parameters
    ----------
    velocity : ndarray, shape (N_dumps, N_grid, 3)
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

    N_dumps = velocity.shape[0]
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

    cg_velocity = np.zeros((len(out_strains),) + velocity.shape[1:],
                         dtype=np.float64)

    # Strain value at each dump
    dump_strains = np.arange(N_dumps) * strain_per_dump   # shape (N_dumps,)

    for k, gamma_centre in enumerate(out_strains):

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
            cg_velocity[k] = velocity[nearest]
            continue

        weights /= w_sum

        # Weighted sum over all contributing dumps
        # tensordot contracts axis 0 of weights (N_dumps,) with axis 0 of
        # phi (N_dumps, N_grid, 3, 3) -> (N_grid, 3, 3)
        cg_velocity[k] = np.tensordot(weights, velocity, axes=(0, 0))

    print("Temporal CG complete")
    return cg_velocity, out_strains
    
def load_velocity_tensors(folderpath, pattern='velocity_tensor_cg_*.pt', 
                         return_metadata=False):
    """
    Load velocity tensor files and return them sorted by timestep
    
    Parameters:
    -----------
    folderpath : str or Path
        Directory containing tensor files
    pattern : str
        Glob pattern for matching files
    return_metadata : bool
        If True, return timesteps as metadata
        
    Returns:
    --------
    velocity_tensors : torch.Tensor
        Shape (n_timesteps, n_grid_points, 3)
    sorted_timesteps : list
        List of timesteps corresponding to each frame
    """
    folderpath = Path(folderpath)
    
    # Find all matching files
    files = list(folderpath.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {folderpath}")
    
    print(f"Found {len(files)} tensor files")
    
    # Extract timesteps and sort files
    file_timestep_pairs = []
    for file in files:
        match = re.search(r'_cg_(\d+)\.pt$', file.name)
        if match:
            timestep = int(match.group(1))
            file_timestep_pairs.append((file, timestep))
        else:
            print(f"Warning: Could not extract timestep from {file.name}, skipping")
    
    if not file_timestep_pairs:
        raise ValueError("No valid timestep information found in filenames")
    
    # Sort by timestep
    file_timestep_pairs.sort(key=lambda x: x[1])
    sorted_files = [f for f, t in file_timestep_pairs]
    sorted_timesteps = [t for f, t in file_timestep_pairs]
    
    print(f"Timestep range: {sorted_timesteps[0]} to {sorted_timesteps[-1]}")
    if len(sorted_timesteps) > 1:
        timestep_interval = sorted_timesteps[1] - sorted_timesteps[0]
        print(f"Timestep interval: {timestep_interval}")
    
    # Get dimensions from first file
    print("\nLoading first file to check dimensions...")
    first_tensor = torch.load(sorted_files[0], weights_only=False)
    
    # Validate tensor structure
    if not isinstance(first_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(first_tensor)}")
    
    velocity_shape = first_tensor.shape # (n_grid_points, 3)
    
    n_grid = velocity_shape[0]
    
    print(f"Grid size: {n_grid} points")
    print(f"Tensor shape per timestep: {velocity_shape}")
    
    # Initialize storage
    num_files = len(sorted_files)
    velocity_tensors = torch.zeros(num_files, n_grid, 3, 
                                   dtype=first_tensor.dtype)
    
    # Store first file data (already loaded)
    velocity_tensors[0] = first_tensor
    
    # Load remaining files
    print("\nLoading remaining files...")
    for i in range(1, num_files):
        # print(f"  Loading {i+1}/{num_files}: timestep {sorted_timesteps[i]}")
        tensor = torch.load(sorted_files[i], weights_only=False)
        # Validate consistency
        if tensor.shape != velocity_shape:
            raise ValueError(f"Shape mismatch at timestep {sorted_timesteps[i]}: "
                           f"expected {velocity_shape}, got {tensor.shape}")
        
        velocity_tensors[i] = tensor
    
    print(f"\n✓ Successfully loaded {num_files} files")
    return velocity_tensors, sorted_timesteps

def plot_velocity_animation_gif(
        velocity_data,
        output_filename="velocity_animation.gif",
        grid_shape=(50,77,1),
        slice_dim=2,
        slice_index=0,
        fps=10,
        dpi=100
    ):
    """
    Create GIF animation of velocity field evolution
    
    Parameters:
    -----------
    velocity_data : torch.Tensor or numpy.ndarray
        Shape: (n_timesteps, n_grid, 3)
        Example: (14, 27000, 3) for velocity components [vx, vy, vz]
    output_filename : str
        Output GIF filename
    grid_shape : tuple
        3D grid dimensions (nx, ny, nz)
        Example: (30, 30, 30) for 27000 points
    slice_dim : int
        Dimension for 2D slice (0=x, 1=y, 2=z)
    slice_index : int or None
        Index for slice (None = middle)
    fps : int
        Frames per second
    dpi : int
        Resolution
    """
    
    # Convert to numpy if needed
    if torch.is_tensor(velocity_data):
        velocity_data = velocity_data.numpy()
    
    n_timesteps = velocity_data.shape[0]
    
    print(f"Creating animation with {n_timesteps} timesteps")
    print(f"Grid shape: {grid_shape}")
    print(f"Velocity data shape: {velocity_data.shape}")
    
    # Validate shape
    expected_grid_size = np.prod(grid_shape)
    if velocity_data.shape[1] != expected_grid_size:
        raise ValueError(f"Grid size mismatch: expected {expected_grid_size}, got {velocity_data.shape[1]}")
    
    # Auto-detect middle slice
    if slice_index is None:
        if grid_shape[slice_dim] == 1:
            slice_index = 0
        else:
            slice_index = grid_shape[slice_dim] // 2
        print(f"Using middle slice: dim={slice_dim}, index={slice_index}")
    
    # Velocity component names
    components = ['vx', 'vy', 'vz']
    
    # Compute global color scales for consistency
    print("Computing color scales...")
    vmax_list = []
    for comp_idx in range(3):
        all_vals = velocity_data[:, :, comp_idx].flatten()
        non_zero = all_vals[all_vals != 0]
        if len(non_zero) > 0:
            vmax = np.percentile(np.abs(non_zero), 99)
            vmin = np.percentile(non_zero, 1)
        else:
            vmax = np.max(np.abs(all_vals))
            vmin = np.min(all_vals)
        vmax_list.append((vmin, vmax))
        print(f"  {components[comp_idx]}: [{vmin:.6f}, {vmax:.6f}]")
    
    # Compute velocity magnitude for overall visualization
    velocity_mag = np.linalg.norm(velocity_data, axis=2)  # Shape: (n_timesteps, n_grid)
    mag_vmax = np.percentile(velocity_mag[velocity_mag > 0], 99) if np.any(velocity_mag > 0) else 1.0
    print(f"  |v| magnitude: [0, {mag_vmax:.6f}]")
    
    # Determine axis labels based on slice dimension
    if slice_dim == 0:
        xlabel, ylabel = 'y', 'z'
    elif slice_dim == 1:
        xlabel, ylabel = 'x', 'z'
    else:  # slice_dim == 2
        xlabel, ylabel = 'x', 'y'
    
    # Create figure and axes that will be reused
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    # Store image objects and colorbars for updating
    images = []
    colorbars = []
    
    # Initialize plots with first frame
    print("Initializing plots...")
    
    # Get first frame data
    velocity_field = velocity_data[0]
    velocity_magnitude = velocity_mag[0]
    
    # Reshape to 3D grid
    velocity_field_3d = velocity_field.reshape(grid_shape + (3,))
    velocity_mag_3d = velocity_magnitude.reshape(grid_shape)
    
    # Extract 2D slice
    if slice_dim == 0:
        velocity_slice = velocity_field_3d[slice_index, :, :, :]
        velocity_mag_slice = velocity_mag_3d[slice_index, :, :]
    elif slice_dim == 1:
        velocity_slice = velocity_field_3d[:, slice_index, :, :]
        velocity_mag_slice = velocity_mag_3d[:, slice_index, :]
    else:  # slice_dim == 2
        velocity_slice = velocity_field_3d[:, :, slice_index, :]
        velocity_mag_slice = velocity_mag_3d[:, :, slice_index]
    
    # Create initial plots for each component
    for ax_idx, comp_name in enumerate(components):
        comp_data = velocity_slice[:, :, ax_idx]
        vmin, vmax = vmax_list[ax_idx]
        
        # Determine colormap and limits
        if vmin < 0 and vmax > 0:
            cmap = 'RdBu_r'
            vmax_sym = max(abs(vmin), abs(vmax))
            vmin_plot, vmax_plot = -vmax_sym, vmax_sym
        else:
            cmap = 'viridis'
            vmin_plot, vmax_plot = vmin, vmax
        
        im = axes[ax_idx].imshow(comp_data.T, 
                                 cmap=cmap, 
                                 origin='lower',
                                 vmin=vmin_plot, 
                                 vmax=vmax_plot,
                                 aspect='auto')
        
        axes[ax_idx].set_title(f'{comp_name} (t=1/{n_timesteps})', fontsize=12)
        axes[ax_idx].set_xlabel(xlabel)
        axes[ax_idx].set_ylabel(ylabel)
        
        cbar = plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
        images.append(im)
        colorbars.append(cbar)
    
    # Create velocity magnitude plot
    im = axes[3].imshow(velocity_mag_slice.T, 
                       cmap='plasma', 
                       origin='lower',
                       vmin=0, 
                       vmax=mag_vmax,
                       aspect='auto')
    
    axes[3].set_title(f'|v| Magnitude (t=1/{n_timesteps})', fontsize=12)
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel(ylabel)
    
    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    images.append(im)
    colorbars.append(cbar)
    
    fig.suptitle(f'Velocity Field - Timestep 1/{n_timesteps}', fontsize=14, y=0.98)
    plt.tight_layout()
    
    def update_frame(timestep_idx):
        """Update plots for given timestep"""
        
        # Get velocity field for this timestep
        velocity_field = velocity_data[timestep_idx]
        velocity_magnitude = velocity_mag[timestep_idx]
        
        # Reshape to 3D grid
        velocity_field_3d = velocity_field.reshape(grid_shape + (3,))
        velocity_mag_3d = velocity_magnitude.reshape(grid_shape)
        
        # Extract 2D slice
        if slice_dim == 0:
            velocity_slice = velocity_field_3d[slice_index, :, :, :]
            velocity_mag_slice = velocity_mag_3d[slice_index, :, :]
        elif slice_dim == 1:
            velocity_slice = velocity_field_3d[:, slice_index, :, :]
            velocity_mag_slice = velocity_mag_3d[:, slice_index, :]
        else:  # slice_dim == 2
            velocity_slice = velocity_field_3d[:, :, slice_index, :]
            velocity_mag_slice = velocity_mag_3d[:, :, slice_index]
        
        # Update each velocity component
        for ax_idx, comp_name in enumerate(components):
            comp_data = velocity_slice[:, :, ax_idx]
            images[ax_idx].set_data(comp_data.T)
            axes[ax_idx].set_title(f'{comp_name} (t={timestep_idx+1}/{n_timesteps})', fontsize=12)
        
        # Update velocity magnitude
        images[3].set_data(velocity_mag_slice.T)
        axes[3].set_title(f'|v| Magnitude (t={timestep_idx+1}/{n_timesteps})', fontsize=12)
        
        # Update main title
        fig.suptitle(f'Velocity Field - Timestep {timestep_idx+1}/{n_timesteps}', 
                     fontsize=14, y=0.98)
        
        return images
    
    # Create animation using FuncAnimation
    print(f"\nGenerating animation with {n_timesteps} frames...")
    anim = FuncAnimation(fig, update_frame, frames=n_timesteps, 
                        interval=1000/fps, blit=False, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_filename, writer=writer, dpi=dpi)
    
    plt.close(fig)
    
    print(f"\n✓ Animation saved: {output_filename}")
    print(f"  File size: {Path(output_filename).stat().st_size / (1024**2):.2f} MB")
    print(f"  Duration: {n_timesteps/fps:.1f} seconds")
    
    return output_filename
    
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load coarse-grained velocity tensors
    # ------------------------------------------------------------
    velocity_tensor, time_steps = load_velocity_tensors(
        folderpath="/Users/kavyanshrajsingh/Desktop/Data/velocity_cgs"
    )

    # ------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------
    strain_rate = 0.065249326754243      # Change only if your simulation uses a different shear rate
    time_step = 5.77e-7       # LAMMPS timestep (s)
    dump_freq = 4500          # Dump every 4500 LAMMPS steps

    # ------------------------------------------------------------
    # Temporal coarse graining
    # ------------------------------------------------------------
    cg_velocity, out_strains = strain_based_coarse_graining(
        velocity_tensor,
        strain_rate,
        dump_freq,
        time_step
    )
    print()

    print("CG velocity shape =", cg_velocity.shape)
    torch.save(
        torch.from_numpy(cg_velocity),
        "/Users/kavyanshrajsingh/Desktop/Data/velocity_temporal_cgs/velocity_temporal_cg.pt"
    )

    print("✓ Saved velocity_temporal_cg.pt")

    # ------------------------------------------------------------
    # Generate GIF
    # ------------------------------------------------------------
    plot_velocity_animation_gif(
       cg_velocity,
        output_filename="velocity_animation.gif",
        grid_shape=(50, 77, 1),
        slice_dim=2,
        slice_index=None,
        fps=10,
        dpi=100
    )

    print("\nFinished.")