import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
import gzip
import re
from scipy.spatial import cKDTree
import torch

class VelocityCoarseGrainer:
    """
    Goldhirsch-Weinhart coarse-graining implementation for DEM Simulation data.
    
    Velocities are computed via finite differences of particle positions between
    consecutive dump frames:
        v_i = (x_{i+1} - x_i) / delta_t
    
    The first frame (index 0) is assigned zero velocity (initially at rest).
    If 1001 frames are read, 1000 FD velocities are computed; frame 0 gets v=0,
    frames 1..1000 get the FD velocity computed from the preceding pair.
    """
    BULK_TYPE   = 1      # LAMMPS atom type for granular (fluid) particles

    def __init__(self, folder_path, pattern='Dump_Shear.*',
                 max_frames=np.inf, n_cores=None,
                 dt=5.77e-7):
        """
        Parameters
        ----------
        folder_path : str | Path
        pattern : str
            Glob pattern for dump files.
        max_frames : int | float
            Maximum number of frames to load.
        n_cores : int | None
            Number of CPU cores for parallel work.
        dt : float
            Physical time represented by one LAMMPS timestep (seconds).
            delta_t between two frames = (timestep_j - timestep_i) * dt.
        timestep_to_seconds : float
            Alias for dt; if both are given dt takes precedence.
            Kept for backwards compatibility.
        """
        self.folder_path = Path(folder_path)
        self.pattern = pattern
        self.max_frames = max_frames
        self.n_cores = n_cores or cpu_count()
        self.dt = dt  # physical seconds per LAMMPS step

        # Particle type densities (kg/m³)
        self.type_density = {1: 2600.0, 2: 2600.0, 3: 2600.0}

        # Coarse-graining parameters
        self.w = None
        self.support_fac = 3.0
        self.support = None
        self.dx = None

        print(f"Coarse-grainer initialized with folder: {self.folder_path}")
        print(f"Physical dt per LAMMPS step: {self.dt}")

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def gaussian_kernel(self, r, w):
        """3-D Gaussian kernel φ(r) = (1/√(2πw²))³ exp(-r²/(2w²))"""
        norm = 1.0 / (np.sqrt(2 * np.pi) * w) ** 3
        return norm * np.exp(-r ** 2 / (2 * w ** 2))

    # ------------------------------------------------------------------
    # File discovery & reading
    # ------------------------------------------------------------------

    def find_and_sort_files(self):
        """Find and sort LAMMPS dump files by timestep."""
        print(f"Looking for files in: {self.folder_path}")

        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {self.folder_path}")

        files = list(self.folder_path.glob(self.pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {self.pattern}")

        print(f"Found {len(files)} files matching pattern")

        patterns = [
            r'\.(\d+)$', r'\.(\d+)\.gz$', r'\.(\d+)\..*$',
            r'wall\.(\d+)', r'_(\d+)$', r'(\d+)$'
        ]

        timesteps, valid_files = [], []
        for file in files:
            for pat in patterns:
                match = re.search(pat, file.name)
                if match:
                    timesteps.append(int(match.group(1)))
                    valid_files.append(file)
                    break

        if not valid_files:
            raise ValueError("No files with extractable timesteps found")

        sorted_indices = np.argsort(timesteps)
        self.files = [valid_files[i] for i in sorted_indices]
        self.timesteps = [timesteps[i] for i in sorted_indices]

        print(f"Processing {len(self.files)} files with valid timesteps")
        return self.files[:int(self.max_frames)]

    def read_lammps_dump(self, filename):
        """Read a single LAMMPS dump file."""
        filepath = Path(filename)
        opener = gzip.open if filepath.suffix == '.gz' else open

        with opener(filepath, 'rt') as f:
            line = f.readline()
            if not line.startswith('ITEM: TIMESTEP'):
                raise ValueError("Expected TIMESTEP header")
            timestep = int(f.readline().strip())

            line = f.readline()
            if not line.startswith('ITEM: NUMBER OF ATOMS'):
                raise ValueError("Expected NUMBER OF ATOMS header")
            N = int(f.readline().strip())

            line = f.readline()
            if not line.startswith('ITEM: BOX BOUNDS'):
                raise ValueError("Expected BOX BOUNDS header")
            boundary_types = line.split()[3:6] if len(line.split()) > 3 else ['pp', 'pp', 'pp']

            bounds = []
            for _ in range(3):
                line = f.readline().strip()
                if line.startswith('ITEM:'):
                    break
                vals = list(map(float, line.split()))
                if len(vals) >= 2:
                    bounds.append(vals[:2])

            bounds = np.array(bounds) 
            dim = len(bounds)

            if not line.startswith('ITEM: ATOMS'):
                line = f.readline()
            cols = line.split()[2:]

            data = []
            for _ in range(N):
                line = f.readline().strip()
                if line:
                    data.append(list(map(float, line.split())))
            data = np.array(data)

        frame = {
            'timestep': timestep,
            'N': N,
            'box_bounds': bounds,
            'boundary_types': boundary_types,
            'dim': dim,
            'x': np.zeros((N, max(2, dim))),
            # 'v' here stores the dump-file velocities (kept for reference).
            # Finite-difference velocities are stored in 'v_fd' after calling
            # compute_finite_difference_velocities().
            'v': np.zeros((N, max(2, dim))),
            'type': np.ones(N, dtype=int),
            'radius': np.zeros(N) * 0.005,
            'mass': None,
            'id': None,
        }

        for i, col in enumerate(cols):
            col = col.lower()
            if col == 'id':
                frame['id'] = data[:, i].astype(int)
            elif col == 'type':
                frame['type'] = data[:, i].astype(int)
            elif col in ['x', 'xu']:
                frame['x'][:, 0] = data[:, i]
            elif col in ['y', 'yu']:
                frame['x'][:, 1] = data[:, i]
            elif col in ['z', 'zu'] and dim >= 3:
                frame['x'][:, 2] = data[:, i]
            elif col == 'vx':
                frame['v'][:, 0] = data[:, i]
            elif col == 'vy':
                frame['v'][:, 1] = data[:, i]
            elif col == 'vz' and dim >= 3:
                frame['v'][:, 2] = data[:, i]
            elif col in ['mass', 'm']:
                frame['mass'] = data[:, i]
            elif col == "radius":
                frame["radius"] = data[:, i]

        return frame

    def read_all_frames(self):
        """
        Read all LAMMPS dump files and compute finite-difference velocities.

        After this call every frame has a 'v_fd' key (shape N x dim) containing
        the position-derived velocity:
            frame[0]['v_fd'] = 0   (at rest initially)
            frame[i]['v_fd'] = (x_{i} - x_{i-1}) / delta_t   for i >= 1
        """
        files = self.find_and_sort_files()
        frames = []

        for i, file in enumerate(files):
            try:
                frame = self.read_lammps_dump(file)
                frames.append(frame)
            except Exception as e:
                print(f"  ✗ Error reading {file.name}: {e}")
                import traceback
                traceback.print_exc()

        if not frames:
            raise RuntimeError("No frames successfully read")

        print(f"\n✓ Successfully read {len(frames)} frames")
        self.frames = frames

        # Compute finite-difference velocities across all frames
        self.compute_finite_difference_velocities(frames)

        return frames

    # ------------------------------------------------------------------
    # Finite-difference velocity computation  ← NEW
    # ------------------------------------------------------------------

    def compute_finite_difference_velocities(self, frames):
        print("\nUsing velocities directly from LAMMPS dump files...")
  
        for frame in frames:

               # keep particles ordered by ID exactly as before
            if frame["id"] is not None:
                order = np.argsort(frame["id"])
      
                frame["id"] = frame["id"][order]
                frame["x"] = frame["x"][order]
                frame["v"] = frame["v"][order]
                frame["type"] = frame["type"][order]
                frame["radius"] = frame["radius"][order]

                if frame["mass"] is not None:
                   frame["mass"] = frame["mass"][order]

                # use LAMMPS velocities directly
                frame["v_fd"] = frame["v"].copy()
        
    # ------------------------------------------------------------------
    # Coarse-graining (uses v_fd by default)
    # ------------------------------------------------------------------

    def compute_particle_mass(self, frame):
        """Compute particle masses if not provided."""
        if frame['mass'] is not None:
            return frame['mass']
        volume = (4.0 / 3.0) * np.pi * frame['radius'] ** 3
        return np.array([volume[i] * self.type_density.get(frame['type'][i], 2600.0)
                         for i in range(frame['N'])])

    def estimate_coarse_graining_width(self, frame, xi_mode='particle'):
        """
        Compute CG kernel width ξ.

        Parameters
        ----------
        xi_mode : 'particle' (default) or 'smooth'
            'particle'  ξ = 2·dp = 2R̄
                IKH near-wall profile mode.  Resolves layering at 1dp.
                erf Z(y) correction essential for first ~2 wall nodes.

            'smooth'    ξ = 3·dp = 6R̄
                Bulk diagnostic mode.  Suppresses force-chain noise so
                that ‖∇·σ‖ diagnostics are reliable.  Correction needed
                for first ~9 nodes near wall.
        """
        mean_r = np.mean(frame['radius'])
        dp     = 2.0 * mean_r

        if xi_mode == 'particle':
            w     = 2.0 * dp
            label = "2·dp  (particle-scale IKH)"
        elif xi_mode == 'smooth':
            w     = 3.0 * dp
            label = "3·dp  (smooth, momentum-balance)"
        else:
            raise ValueError(f"xi_mode must be 'particle' or 'smooth', got '{xi_mode}'")

        print(f"Coarse-graining width:")
        print(f"  Mean radius R̄   : {mean_r:.6f} m")
        print(f"  Particle diam dp : {dp:.6f} m")
        print(f"  ξ = {label} : {w:.6f} m")
        print(f"  Support {self.support_fac}·ξ     : {self.support_fac * w:.6f} m")
        print(f"  erf correction needed for first ~{self.support_fac * w / dp:.1f} nodes from wall")
        return w
    

    def coarse_grain_velocity(self, frame, grid_points, w=None, use_fd_velocity=True):
        """
        Coarse-grain velocity field using Goldhirsch-Weinhart method.

        Parameters
        ----------
        frame : dict
            Frame data (must contain 'v_fd' if use_fd_velocity=True).
        grid_points : ndarray  (N_grid × dim)
        w : float | None
            Coarse-graining width; auto-estimated if None.
        use_fd_velocity : bool
            If True (default) use finite-difference velocities ('v_fd').
            If False use the velocities stored in the dump file ('v').

        Returns
        -------
        velocity_field : ndarray  (N_grid × dim)
        density_field  : ndarray  (N_grid,)
        """
        if w is None:
            w = self.estimate_coarse_graining_width(frame)
        else:
            print(f"Using provided coarse-graining width: {w}")

        support = self.support_fac * w
        dim = frame['dim']
        positions = frame['x'][:, :dim]

        # Choose velocity source
        if use_fd_velocity:
            if 'v_fd' not in frame:
                raise KeyError("'v_fd' not found in frame. "
                               "Call compute_finite_difference_velocities() first "
                               "or set use_fd_velocity=False.")
            velocities = frame['v_fd'][:, :dim]
            vel_label = "finite-difference"
        else:
            velocities = frame['v'][:, :dim]
            vel_label = "dump-file"

        print(f"Velocity source: {vel_label}")

        mass = self.compute_particle_mass(frame)
        tree = cKDTree(positions)

        n_grid = len(grid_points)
        velocity_field = np.zeros((n_grid, dim))
        density_field = np.zeros(n_grid)

        print(f"Coarse-graining {n_grid} grid points …")
        for i, grid_pt in enumerate(grid_points):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{n_grid} points")

            indices = tree.query_ball_point(grid_pt[:dim], support)
            if not indices:
                continue

            r_vec = positions[indices] - grid_pt[:dim]
            r = np.linalg.norm(r_vec, axis=1)
            weights = self.gaussian_kernel(r, w) * mass[indices]

            total_weight = np.sum(weights)
            if total_weight > 1e-12:
                velocity_field[i] = (np.sum(weights[:, None] * velocities[indices], axis=0)
                                     / total_weight)
                density_field[i] = total_weight

        print("  ✓ Coarse-graining complete")
        return velocity_field, density_field

    def compute_velocity_profile(self, frame, direction='y', velocity_component='x',
                                 n_bins=50, w=None, use_fd_velocity=True):
        """
        Compute 1D velocity profile along a direction (useful for shear flows).

        Parameters
        ----------
        frame : dict
        direction : str | int
            Direction to bin along ('x', 'y', 'z' or 0, 1, 2).
        velocity_component : str | int
            Velocity component to average.
        n_bins : int
        w : float | None
            Coarse-graining width.
        use_fd_velocity : bool
            If True (default) use 'v_fd'; otherwise use dump-file 'v'.

        Returns
        -------
        bin_centers     : ndarray
        velocity_profile: ndarray
        velocity_std    : ndarray
        particle_count  : ndarray
        """
        dir_map = {'x': 0, 'y': 1, 'z': 2}
        dir_idx = dir_map[direction.lower()] if isinstance(direction, str) else direction
        vel_idx = (dir_map[velocity_component.lower()]
                   if isinstance(velocity_component, str) else velocity_component)

        positions = frame['x'][:, dir_idx]

        if use_fd_velocity:
            if 'v_fd' not in frame:
                raise KeyError("'v_fd' not found. Run compute_finite_difference_velocities first.")
            velocities = frame['v_fd'][:, vel_idx]
            vel_label = "finite-difference"
        else:
            velocities = frame['v'][:, vel_idx]
            vel_label = "dump-file"

        if w is None:
            w = self.estimate_coarse_graining_width(frame)

        bounds = frame['box_bounds'][dir_idx]
        bin_edges = np.linspace(bounds[0]+w, bounds[1]-w, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        velocity_profile = np.zeros(n_bins)
        velocity_std = np.zeros(n_bins)
        particle_count = np.zeros(n_bins)

        for i, center in enumerate(bin_centers):
            distances = np.abs(positions - center)
            mask = distances < (self.support_fac * w)

            if np.sum(mask) > 0:
                weights = self.gaussian_kernel(distances[mask], w)
                velocity_profile[i] = np.average(velocities[mask], weights=weights)
                velocity_std[i] = np.std(velocities[mask])
                particle_count[i] = np.sum(mask)

        print(f"Velocity profile computed ({vel_label}):")
        print(f"  Direction: {direction} (index {dir_idx})")
        print(f"  Velocity component: {velocity_component} (index {vel_idx})")
        print(f"  Bins with particles: {np.sum(particle_count > 0)}/{n_bins}")

        return bin_centers, velocity_profile, velocity_std, particle_count

    # ------------------------------------------------------------------
    # Grid creation
    # ------------------------------------------------------------------


    def create_grid(self, frame, grid_type='uniform', n_points=None, dx=None):
        bounds = frame['box_bounds']
        dim = frame['dim']

        # 1. Ensure 'w' is always defined for the boundary logic
        w = self.estimate_coarse_graining_width(frame)

        # 2. Handle default dx if nothing is provided
        if n_points is None and dx is None:
            dx = w / 2.0
            print(f"Using grid spacing: {dx:.6f}")

        # 3. Calculate n_points based on dx and bounds
        if n_points is None:
            if np.isscalar(dx):
                dx = np.array([dx] * dim)

            n_pts_list = []
            for i in range(dim):
                # Calculate the effective length for this dimension
                if i == 1: # Apply buffer 'w' to the y-dimension
                    length = (bounds[i, 1] - bounds[i, 0]) - 2 * w
                else:
                    length = (bounds[i, 1] - bounds[i, 0])

                # Ensure we have at least 1 point and convert to int
                n_pts_list.append(max(int(length / dx[i]), 1))

            n_points = tuple(n_pts_list)

        # 4. Handle scalar n_points input
        if isinstance(n_points, int):
            n_points = tuple([n_points] * dim)

        # 5. Generate the grid
        if dim == 2:
            x = np.linspace(bounds[0, 0], bounds[0, 1], n_points[0])
            y = np.linspace(bounds[1, 0] + w, bounds[1, 1] - w, n_points[1])
            xx, yy = np.meshgrid(x, y)
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])
            grid_shape = (n_points[1], n_points[0])
        else:
            x = np.linspace(bounds[0, 0], bounds[0, 1], n_points[0])
            y = np.linspace(bounds[1, 0] + w, bounds[1, 1] - w, n_points[1])
            z = np.linspace(bounds[2, 0], bounds[2, 1], n_points[2])
            # Use indexing='ij' to keep (x, y, z) order consistent
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
            grid_shape = n_points

        print(f"Created {grid_type} grid: {n_points} points")
        return grid_points, grid_shape

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_velocity_profile(self, frame, velocity_field, grid_points, grid_shape,
                              slice_dim=None, slice_val=None, component=0, support=None):
        """Create scatter/image plot of velocity profiles."""
        dim = frame['dim']
        if support is None:
            support = self.estimate_coarse_graining_width(frame)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: raw particle velocities (from v_fd if available, else dump v)
        ax1 = axes[0]
        if 'v_fd' in frame:
            particle_vel = frame['v_fd'][:, component]
            ax1.set_title(f'FD Particle Velocities (Component {component})')
        else:
            particle_vel = frame['v'][:, component]
            ax1.set_title(f'Dump Particle Velocities (Component {component})')

        vmin_p = np.nanpercentile(particle_vel, 1)
        vmax_p = np.nanpercentile(particle_vel, 99)

        if dim == 2:
            sc = ax1.scatter(frame['x'][:, 0], frame['x'][:, 1],
                             c=particle_vel, s=1, cmap='viridis',
                             alpha=0.7, vmin=vmin_p, vmax=vmax_p)
        else:
            pts_x, pts_y, vel_plot = frame['x'][:, 0], frame['x'][:, 1], particle_vel
            if slice_dim is not None and slice_val is not None:
                mask = np.abs(frame['x'][:, slice_dim] - slice_val) < support
                pts_x, pts_y, vel_plot = pts_x[mask], pts_y[mask], vel_plot[mask]
            sc = ax1.scatter(pts_x, pts_y, c=vel_plot, s=2, cmap='viridis',
                             alpha=0.7, vmin=vmin_p, vmax=vmax_p)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(sc, ax=ax1, label=f'v_{component}')

        # Right: coarse-grained field
        ax2 = axes[1]
        vel_component = velocity_field[:, component]
        valid_mask = ~np.isnan(vel_component) & (vel_component != 0)
        if np.sum(valid_mask) > 0:
            vmin_c = np.percentile(vel_component[valid_mask], 1)
            vmax_c = np.percentile(vel_component[valid_mask], 99)
        else:
            vmin_c, vmax_c = vel_component.min(), vel_component.max()

        if vmin_c == vmax_c:
            vmin_c -= 0.1 * abs(vmin_c) if vmin_c != 0 else 0.1
            vmax_c += 0.1 * abs(vmax_c) if vmax_c != 0 else 0.1

        if dim == 2:
            vel_grid = vel_component.reshape(grid_shape)
            im = ax2.imshow(vel_grid,
                            extent=[frame['box_bounds'][0, 0], frame['box_bounds'][0, 1],
                                    frame['box_bounds'][1, 0], frame['box_bounds'][1, 1]],
                            origin='lower', cmap='viridis', aspect='auto',
                            vmin=vmin_c, vmax=vmax_c, interpolation='bilinear')
            plt.colorbar(im, ax=ax2, label=f'v_{component}')
        else:
            gx, gy = grid_points[:, 0], grid_points[:, 1]
            vc = vel_component
            if slice_dim is not None and slice_val is not None:
                mask = np.abs(grid_points[:, slice_dim] - slice_val) < support
                gx, gy, vc = gx[mask], gy[mask], vc[mask]
            sc2 = ax2.scatter(gx, gy, c=vc, s=50, cmap='viridis',
                              vmin=vmin_c, vmax=vmax_c, edgecolors='none')
            plt.colorbar(sc2, ax=ax2, label=f'v_{component}')

        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'Coarse-Grained Velocity Field (Component {component})')

        plt.tight_layout()
        plt.show()
        return fig

    def plot_1d_velocity_profile(self, bin_centers, velocity_profile, velocity_std=None,
                                 particle_count=None, direction='y', velocity_component='x'):
        """Plot 1D velocity profile."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.plot(bin_centers, velocity_profile, 'b-', linewidth=2, label='Coarse-grained')
        if velocity_std is not None:
            ax1.fill_between(bin_centers,
                             velocity_profile - velocity_std,
                             velocity_profile + velocity_std,
                             alpha=0.3, label='±1 std dev')
        ax1.set_xlabel(f'{direction} position')
        ax1.set_ylabel(f'v_{velocity_component}')
        ax1.set_title(f'Velocity Profile: v_{velocity_component} vs {direction}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        if particle_count is not None:
            ax2 = axes[1]
            ax2.bar(bin_centers, particle_count,
                    width=bin_centers[1] - bin_centers[0], alpha=0.6, color='green')
            ax2.set_xlabel(f'{direction} position')
            ax2.set_ylabel('Particle count in kernel support')
            ax2.set_title('Particles per Bin')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

if __name__ == "__main__":

    vcg = VelocityCoarseGrainer(
        folder_path="/Users/kavyanshrajsingh/Desktop/Data/Dump",
        pattern="Dump_Shear.*",
        dt=5.77e-7,
        max_frames=20001
    )

    frames = vcg.read_all_frames()

    grid_points, grid_shape = vcg.create_grid(frames[0])

    print(f"\nGrid shape = {grid_shape}")
    print(f"Number of grid points = {len(grid_points)}")
    print(f"Product of grid shape = {np.prod(grid_shape)}")

    output_dir = Path("velocity_cgs")
    output_dir.mkdir(exist_ok=True)

    for i, frame in enumerate(frames):

        print(f"\nProcessing frame {i+1}/{len(frames)}")
        print(f"Timestep: {frame['timestep']}")

        velocity_field, density_field = vcg.coarse_grain_velocity(
            frame,
            grid_points,
            use_fd_velocity=True
        )

        torch.save(
            torch.from_numpy(velocity_field.astype(np.float32)),
            output_dir / f"velocity_tensor_cg_{frame['timestep']}.pt"
        )

    print("\nFinished.")