#====================================================================#
# COARSE GRAINING OF STRESS TENSOR from per particle to uniform grids
#====================================================================#

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
import re
from scipy.spatial import cKDTree
from scipy.special import erf
import torch
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib


class StressCoarseGrainer:
    """
    Irving-Kirkwood-Hardy (IKH) coarse-graining of per-particle stress tensors.

    CG formula
    ----------
        σ(r) = Σ_i [ σ_i · φ(|r−r_i|, ξ) ]
               ─────────────────────────────────────────
               Z(y_r) · Σ_i [ V_i · φ(|r−r_i|, ξ) ]

    where:
        σ_i    = per-particle stress (stress × volume) from LAMMPS
        V_i    = particle volume  (4π R_i³ / 3)
        φ      = 3-D normalised Gaussian kernel, width ξ
        Z(y_r) = erf-based renormalisation for boundary truncation (see below)

    Kernel width ξ
    --------------
    Two operating modes, set via the `xi_mode` argument of
    `estimate_coarse_graining_width`:

        'particle'  (default, IKH near-wall profile mode)
            ξ = 1·dp = 2R̄  — resolves layering oscillations at 1 dp
            Truncation correction is essential for first ~2 nodes near wall.
            Wall nodes (n=0, n=N-1) are skipped automatically.

        'smooth'    (bulk momentum-balance diagnostic mode)
            ξ = 3·dp = 6R̄  — suppresses force-chain noise enough that
            ∇·σ diagnostics are reliable.  Truncation matters for first ~6
            nodes near wall.

    Boundary normalisation Z(y)
    ---------------------------
    For a Gaussian kernel centred at grid node y_n, the accessible-domain
    kernel integral is:

        Z(y_n) = ∫_{y_lo}^{y_hi} φ(y − y_n ; ξ) dy
               = 0.5 · [ erf((y_n − y_lo)/(√2·ξ)) + erf((y_hi − y_n)/(√2·ξ)) ]

    This is exactly 1 in the bulk (node > ~3ξ from either wall) and < 1 near
    the walls.  Dividing the denominator by Z(y_n) restores correct
    normalisation without adding any non-physical particles or forces.

    Note: ghost/mirror particles are NOT used.  At particle-scale resolution
    (ξ ≈ 1 dp) ghost particles would double-count near-wall particles already
    within the support and introduce artefacts.  The erf renormalisation is the
    analytically exact correction for a Gaussian kernel with a flat wall.

    Wall particle exclusion
    -----------------------
    Wall particles (type != BULK_TYPE, i.e. the constrained boundary atoms)
    are excluded from all summations.  Their wall-fluid contact stress is
    already captured in the LAMMPS virial of the bulk particle on the other
    side of each contact, so no force contribution is lost.

    Grid convention
    ---------------
    The y-grid spans [y_lo + Δy, y_hi − Δy] — the first and last fluid
    layers, skipping the wall-node positions themselves.  With Δy = 1 dp
    this gives nodes n = 1 … N-1 (0-indexed), consistent with the IKH
    recommendation to skip n=0 and n=N.
    """

    BULK_TYPE   = 1      # LAMMPS atom type for granular (fluid) particles
    SUPPORT_FAC = 3.0    # Gaussian truncation at SUPPORT_FAC · ξ

    def __init__(self, folder_path, pattern='dump.stress.*', max_frames=np.inf):
        self.folder_path = Path(folder_path)
        self.pattern     = pattern
        self.max_frames  = max_frames
        self.w           = None   # CG width ξ, set by estimate_coarse_graining_width

        print("Initialising StressCoarseGrainer  (IKH, erf renormalisation)")
        print(f"  Bulk particle type  : {self.BULK_TYPE}")
        print(f"  Support factor      : {self.SUPPORT_FAC}·ξ")
        print(f"  Boundary fix        : erf Z(y) renormalisation — exact for Gaussian/flat wall")

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def gaussian_kernel(self, r, w):
        """Normalised 3-D Gaussian  φ(r) = (1/√(2π) w)³ exp(−r²/2w²)."""
        norm = 1.0 / ((np.sqrt(2.0 * np.pi) * w) ** 3)
        return norm * np.exp(-r ** 2 / (2.0 * w ** 2))

    # ------------------------------------------------------------------
    # Boundary renormalisation  Z(y)
    # ------------------------------------------------------------------

    def boundary_normalization(self, y_pts, y_lo, y_hi, w):
        """
        Fraction of the 1-D Gaussian kernel mass that lies within [y_lo, y_hi].

            Z(y) = 0.5 · [ erf((y − y_lo)/(√2·w)) + erf((y_hi − y)/(√2·w)) ]

        Properties:
            Z = 1.0  when node is > ~3w from both walls  (bulk, no correction needed)
            Z < 1.0  near walls  (kernel spills outside domain → must renormalise)
            Z → 0    if node is placed AT or outside the wall  (avoid this)

        Dividing the weighted-volume denominator by Z(y) is the analytically
        exact renormalisation for a Gaussian kernel truncated by a flat wall.
        No approximation is involved — it is simply the ratio of the actual
        kernel integral over the accessible domain to the full-space integral.

        Parameters
        ----------
        y_pts : array_like  y-coordinates of grid nodes
        y_lo  : float       bottom wall y-coordinate
        y_hi  : float       top wall y-coordinate
        w     : float       CG width ξ

        Returns
        -------
        Z : ndarray, same shape as y_pts, values in (0, 1]
        """
        sq2w = np.sqrt(2.0) * w
        Z    = 0.5 * (erf((y_pts - y_lo) / sq2w) + erf((y_hi - y_pts) / sq2w))
        return np.clip(Z, 1e-6, 1.0)   # guard against node placed on/outside wall

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def find_and_sort_files(self):
        print(f"Looking for files in: {self.folder_path}")
        print(f"Pattern: {self.pattern}")
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {self.folder_path}")

        files = list(self.folder_path.glob(self.pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching: {self.pattern}")

        print(f"Found {len(files)} files")
        timesteps, valid_files = [], []
        patterns = [r'\.(\d+)$', r'\.(\d+)\.gz$', r'\.(\d+)\..*$',
                    r'stress\.(\d+)', r'_(\d+)$', r'(\d+)$']

        for file in files:
            for pat in patterns:
                m = re.search(pat, file.name)
                if m:
                    timesteps.append(int(m.group(1)))
                    valid_files.append(file)
                    break

        if not valid_files:
            raise ValueError("No files with extractable timesteps found")

        idx            = np.argsort(timesteps)
        self.files     = [valid_files[i] for i in idx]
        self.timesteps = [timesteps[i]   for i in idx]

        print(f"Will process {len(self.files)} files")
        return self.files[:int(self.max_frames)]

    # ------------------------------------------------------------------
    # Config reader
    # ------------------------------------------------------------------

    def read_radius_from_config(self, config_file):
        with open(config_file, 'r') as f:
            lines = f.readlines()

        start = None
        for i, line in enumerate(lines):
            if 'Atoms' in line:
                start = i + 2
                break
        if start is None:
            raise ValueError("Atoms section not found")

        id_to_radius = {}
        for line in lines[start:]:
            if not line.strip():
                break
            parts = line.split()
            if len(parts) >= 3:
                id_to_radius[int(parts[0])] = float(parts[2]) / 2.0

        return id_to_radius

    # ------------------------------------------------------------------
    # Dump reader
    # ------------------------------------------------------------------

    def read_lammps_dump_with_stress(self, dump_filename, stress_filename):
        # ==========================================================
        # READ DUMP FILE (coordinates, type, velocity)
        # ==========================================================

        dump_path = Path(dump_filename)
        opener = gzip.open if dump_path.suffix == ".gz" else open

        with opener(dump_path, "rt") as f:

            assert f.readline().startswith("ITEM: TIMESTEP")
            timestep = int(f.readline())

            assert f.readline().startswith("ITEM: NUMBER OF ATOMS")
            N = int(f.readline())

            line = f.readline()
            assert line.startswith("ITEM: BOX BOUNDS")

            bounds = []
            for _ in range(3):
                vals = list(map(float, f.readline().split()))
                bounds.append(vals[:2])

            bounds = np.array(bounds)
            dim = len(bounds)

            line = f.readline()
            assert line.startswith("ITEM: ATOMS")

            dump_cols = line.split()[2:]

            dump_data = []
            for _ in range(N):
                dump_data.append(list(map(float, f.readline().split())))

        dump_data = np.asarray(dump_data)
        
        # ==========================================================
        # READ STRESS FILE
        # ==========================================================

        stress_path = Path(stress_filename)
        opener = gzip.open if stress_path.suffix == ".gz" else open

        with opener(stress_path, "rt") as f:

            assert f.readline().startswith("ITEM: TIMESTEP")
            stress_timestep = int(f.readline())

            if stress_timestep != timestep:
                raise ValueError(
                    f"Timestep mismatch: dump={timestep}, stress={stress_timestep}"
                )

            assert f.readline().startswith("ITEM: NUMBER OF ATOMS")
            Ns = int(f.readline())

            if Ns != N:
                raise ValueError(
                    f"Atom count mismatch: dump={N}, stress={Ns}"
                )
        
            f.readline()

            for _ in range(3):
                f.readline()

            line = f.readline()
            assert line.startswith("ITEM: ATOMS")

            stress_cols = line.split()[2:]

            stress_data = []
            for _ in range(N):
                stress_data.append(list(map(float, f.readline().split())))

        stress_data = np.asarray(stress_data)
        
        dump_ids = dump_data[:, dump_cols.index("id")].astype(int)
        stress_ids = stress_data[:, stress_cols.index("id")].astype(int)

        #print("\nFirst 10 dump IDs:")
        #print(dump_ids[:10])

        #print("First 10 stress IDs:")
        #print(stress_ids[:10])

        #print("IDs identical?", np.array_equal(dump_ids, stress_ids))

        frame = {
            'timestep'  : timestep,
            'N'         : N,
            'box_bounds': bounds,
            'dim'       : dim,
            'x'         : np.zeros((N, max(2, dim))),
            'v'         : np.zeros((N, max(2, dim))),
            'type'      : np.ones(N, dtype=int),
            'radius'    : np.zeros(N),
            'id'        : np.arange(1, N + 1),
            'stress'    : np.zeros((N, 6)),
        }
        
        #print("Dump columns :", dump_cols)
        #print("Stress columns:", stress_cols)
    

        for i, col in enumerate(dump_cols):
            cl = col.lower()
            if   cl == 'id':                                  frame['id'][:]     = dump_data[:, i].astype(int)
            elif cl == 'type':                                frame['type'][:]   = dump_data[:, i].astype(int)
            elif cl in ('x', 'xu'):                           frame['x'][:, 0]   = dump_data[:, i]
            elif cl in ('y', 'yu'):                           frame['x'][:, 1]   = dump_data[:, i]
            elif cl in ('z', 'zu') and dim >= 3:              frame['x'][:, 2]   = dump_data[:, i]
            elif cl == 'vx':                                  frame['v'][:, 0]   = dump_data[:, i]
            elif cl == 'vy':                                  frame['v'][:, 1]   = dump_data[:, i]
            elif cl == 'vz' and dim >= 3:                     frame['v'][:, 2]   = dump_data[:, i]
            
        id_to_r = self.read_radius_from_config(
            "/Users/kavyanshrajsingh/Desktop/Data/Dump/Shear_Boundary_50x40.txt"
        )

        fallback_r = np.mean(list(id_to_r.values()))

        for i, col in enumerate(stress_cols):

            m = re.search(r"\[(\d+)\]", col)

            if m is None:
                continue

            idx = int(m.group(1)) - 1

            if 0 <= idx < 6:
                frame["stress"][:, idx] = stress_data[:, i]
        
        for row_idx, atom_id in enumerate(frame['id']):
            frame['radius'][row_idx] = id_to_r.get(int(atom_id), fallback_r)
        
        mask = frame['type'] == self.BULK_TYPE
        for key in ('id', 'type', 'x', 'v', 'radius', 'stress'):
            frame[key] = frame[key][mask]
        frame['N'] = int(mask.sum())
        #print("\n===== Parsed frame =====")
        #print("First 5 IDs:")
        #print(frame["id"][:5])

        #print("First 5 Types:")
        #print(frame["type"][:5])

        #print("First 5 Coordinates:")
        #print(frame["x"][:5])

        #print("First 5 Radii:")
        #print(frame["radius"][:5])
        
        #print("\nFirst five stresses")
        #print(frame["stress"][:5])

        #print("========================\n")
        
        #print("\n===== BEFORE RETURN =====")
        #print(frame["x"][:10])
        #print("=========================\n")
        return frame

    # ------------------------------------------------------------------
    # Coarse-graining
    # ------------------------------------------------------------------

    def compute_particle_volume(self, frame):
        return (4.0 / 3.0) * np.pi * frame['radius'] ** 3

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
        print(f"  Support {self.SUPPORT_FAC}·ξ     : {self.SUPPORT_FAC * w:.6f} m")
        print(f"  erf correction needed for first ~{self.SUPPORT_FAC * w / dp:.1f} nodes from wall")
        return w

    def stress_voigt_to_tensor(self, stress_voigt):
        """[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz] → (N,3,3)."""
        T = np.zeros((len(stress_voigt), 3, 3))
        T[:, 0, 0] = stress_voigt[:, 0]
        T[:, 1, 1] = stress_voigt[:, 1]
        T[:, 2, 2] = stress_voigt[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = stress_voigt[:, 3]
        T[:, 0, 2] = T[:, 2, 0] = stress_voigt[:, 4]
        T[:, 1, 2] = T[:, 2, 1] = stress_voigt[:, 5]
        return T

    def coarse_grain_stress(self, frame, grid_points, w=None, xi_mode='particle'):
        """
        IKH coarse-grain per-particle stress to a continuum field.

        Formula:
            σ(r) = Σ_i [ σ_i · φ(|r−r_i|, ξ) ]
                   ──────────────────────────────────────────────
                   Z(y_r) · Σ_i [ V_i · φ(|r−r_i|, ξ) ]

        Z(y_r) is the erf-based renormalisation accounting for the fraction of
        the Gaussian kernel that lies outside [y_lo, y_hi].  In the bulk
        Z = 1 and the formula reduces to the standard volume-weighted average.
        Near the walls Z < 1, which correctly amplifies the denominator to
        compensate for the missing kernel mass — no ghost particles needed.

        Only bulk particles (type == BULK_TYPE) contribute.  Wall-fluid contact
        stress is already in the LAMMPS virial of the bulk particle.

        PBC: x and z periodic;  y wall-bounded (no wrapping).

        Parameters
        ----------
        frame       : dict from read_lammps_dump_with_stress
        grid_points : (N_grid, 3) evaluation coordinates
        w           : CG width ξ [m].  If None, estimated from data using xi_mode.
        xi_mode     : 'particle' (1dp) or 'smooth' (3dp) — used only when w is None.

        Returns
        -------
        stress_field : (N_grid, 3, 3) [Pa]
        """
        if w is None:
            w = self.estimate_coarse_graining_width(frame, xi_mode=xi_mode)
        self.w = w

        support = self.SUPPORT_FAC * w
        print("\n===== ENTERING coarse_grain_stress =====")
        print("frame['x'] shape =", frame['x'].shape)
        #print("First 10 coordinates:")
        #print(frame['x'][:10])
        print("========================================\n")
        pos     = frame['x'][:, :3]
        sigma_p = self.stress_voigt_to_tensor(frame['stress'])
        vol     = self.compute_particle_volume(frame)

        if np.sum(np.abs(frame['stress'])) < 1e-12:
            print("WARNING: all stress values near-zero — check LAMMPS compute")

        x_lo = frame['box_bounds'][0, 0];  Lx  = frame['box_bounds'][0, 1] - x_lo
        y_lo = frame['box_bounds'][1, 0] + w;  y_hi = frame['box_bounds'][1, 1] - w
        z_lo = frame['box_bounds'][2, 0];  Lz  = frame['box_bounds'][2, 1] - z_lo

        # Shift x, z into [0, L) for cKDTree PBC handling; y is absolute
        pos_s       = pos.copy()
        pos_s[:, 0] = (pos[:, 0] - x_lo) % Lx
        pos_s[:, 2] = (pos[:, 2] - z_lo) % Lz

        # Tree on real particles only; y PBC disabled (large boxsize)
        tree = cKDTree(pos_s, boxsize=[Lx, 1e30, Lz])

        n_grid       = len(grid_points)
        stress_field = np.zeros((n_grid, 3, 3))
        has_data     = np.zeros(n_grid, dtype=bool)

        # Pre-compute erf renormalisation Z(y) for every grid node — O(N_grid)
        # This is the ONLY boundary correction needed.  It accounts analytically
        # for the Gaussian kernel mass that falls outside [y_lo, y_hi].
        Z = self.boundary_normalization(grid_points[:, 1], y_lo, y_hi, w)  # (N_grid,)

        # Diagnostic: how many nodes are in the correction zone?
        n_corrected = np.sum(Z < 0.99)
        #print(f"Coarse-graining {n_grid} points  (ξ={w:.5f} m, erf Z(y), PBC x/z)")
        #print(f"  Nodes with Z < 0.99 (boundary correction active): {n_corrected}")

        for k, gpt in enumerate(grid_points):
            if k % 2000 == 0:
                print(f"  {k}/{n_grid}")

            gpt_s    = gpt.copy()
            if k == 0:
                #print("Grid =", gpt)
                #print("Shifted grid =", gpt_s)
                #print("support =", support)

                #print("First five particles")
                #print(pos_s[:5])
            
                d = np.linalg.norm(pos_s - gpt_s, axis=1)
                #print("Distance min =", d.min())
                #print("Distance max =", d.max())
            gpt_s[0] = (gpt[0] - x_lo) % Lx
            gpt_s[2] = (gpt[2] - z_lo) % Lz
            # gpt_s[1] = gpt[1] — y is not wrapped

            nbrs = tree.query_ball_point(gpt_s, support)
            if k % 500 == 0:
                print(f"Grid {k}: neighbours = {len(nbrs)}")
            if not nbrs:
                continue

            # Minimum-image displacement: PBC in x and z only
            disp       = pos_s[nbrs] - gpt_s
            disp[:, 0] -= Lx * np.round(disp[:, 0] / Lx)
            disp[:, 2] -= Lz * np.round(disp[:, 2] / Lz)
            # disp[:, 1] = true y-separation, no wrapping

            r   = np.linalg.norm(disp, axis=1)         # (n_nbr,)
            phi = self.gaussian_kernel(r, w)             # (n_nbr,)
            if k in [0, 500] and len(phi):
                print(
                    f"phi: sum={phi.sum():.3e}, "
                    f"max={phi.max():.3e}"
                )
            #print("Grid point =", gpt)
            #print("Shifted    =", gpt_s)
            #print("Particle min =", pos_s.min(axis=0))
            #print("Particle max =", pos_s.max(axis=0))

            # Numerator:   Σ_i [ σ_i · φ_i ]            shape (3,3)
            numerator = np.einsum('i,ijk->jk', phi, sigma_p[nbrs])

            # Denominator: Z(y) · Σ_i [ V_i · φ_i ]     scalar
            # Z(y) < 1 near walls → denominator is reduced → stress is
            # correctly amplified to account for the missing kernel support.
            denominator = Z[k] * np.dot(vol[nbrs], phi)
            if k % 500 == 0:
                print(
                    f"Z={Z[k]:.3f}, "
                    f"denom={denominator:.3e}"
                )

            if denominator > 1e-30:
                stress_field[k] = numerator / denominator
                has_data[k]     = True

        print(f"  Done. Points with data: {has_data.sum()}/{n_grid}")
        return stress_field

    # ------------------------------------------------------------------
    # Grid creation
    # ------------------------------------------------------------------

    def create_grid(self, frame, n_points=None, dx=None,
                    y_margin=None, xi_mode='particle'):
        """
        Create uniform evaluation grid consistent with IKH CG scheme.

        y_margin convention
        -------------------
        Default: y_margin = 1·dp = 2R̄  (one particle diameter from each wall).

        This places the first grid node at y_lo + dp and the last at
        y_hi − dp, corresponding to nodes n=1 … N-1 in the IKH convention
        (skipping the wall nodes n=0 and n=N).  The erf renormalisation Z(y)
        fully corrects these near-wall nodes — they do NOT need to be
        excluded.  The margin only removes the wall-node itself (y = y_wall)
        where Z → 0 and the estimate would be unreliable.

        Grid spacing
        ------------
        Default: Δ = dp (one particle diameter), giving particle-scale
        resolution.  For the 'smooth' mode you may want Δ = ξ/2 instead.
        Pass dx explicitly to override.

        Parameters
        ----------
        n_points : tuple or int, optional  — explicit grid shape
        dx       : float or array, optional — grid spacing
        y_margin : float, optional — inset from each y-wall [m]
        xi_mode  : 'particle' or 'smooth' — only used when self.w is None
        """
        bounds = frame['box_bounds'].copy()
        dim    = frame['dim']

        if self.w is None:
            self.w = self.estimate_coarse_graining_width(frame, xi_mode=xi_mode)
        w      = self.w
        mean_r = np.mean(frame['radius'])
        dp     = 2.0 * mean_r

        # Default margin: 1dp — skip the wall-node itself, keep all fluid nodes
        if y_margin is None:
            y_margin = dp

        bounds[1, 0] += y_margin
        bounds[1, 1] -= y_margin

        if n_points is None and dx is None:
            # Default grid spacing = 1dp (particle-scale IKH convention)
            dx = dp
            print(f"Auto grid spacing: Δ = dp = {dx:.6f} m")

        if n_points is None:
            dx_arr   = np.full(dim, dx) if np.isscalar(dx) else np.asarray(dx)
            n_points = tuple(max(2, int(round((bounds[i, 1] - bounds[i, 0]) / dx_arr[i])) + 1)
                             for i in range(dim))

        if isinstance(n_points, int):
            n_points = (n_points,) * dim

        axes        = [np.linspace(bounds[i, 0], bounds[i, 1], n_points[i])
                       for i in range(dim)]
        grids       = np.meshgrid(*axes, indexing='ij')
        grid_points = np.column_stack([g.ravel() for g in grids])
        grid_shape  = n_points

        # Warn if grid spacing > ξ/2 (Nyquist-like criterion for CG field)
        actual_dy = (bounds[1, 1] - bounds[1, 0]) / (n_points[1] - 1) if n_points[1] > 1 else 1.0
        nyquist_ok = actual_dy <= w / 2.0
        print(f"Grid shape  : {n_points}")
        print(f"y range     : [{bounds[1,0]:.4f}, {bounds[1,1]:.4f}]  (margin ±{y_margin:.4f} m)")
        print(f"Δy = {actual_dy:.4f} m,  ξ/2 = {w/2:.4f} m  "
              f"{'✓ Nyquist OK' if nyquist_ok else '⚠ Δy > ξ/2: consider finer grid'}")
        return grid_points, grid_shape

    # ------------------------------------------------------------------
    # Stress invariants
    # ------------------------------------------------------------------

    def compute_stress_invariants(self, stress_field):
        pressure  = -(stress_field[:, 0, 0]
                      + stress_field[:, 1, 1]
                      + stress_field[:, 2, 2]) / 3.0
        dev       = stress_field - np.eye(3)[None, :, :] * (-pressure[:, None, None])
        von_mises = np.sqrt(1.5 * np.einsum('ijk,ijk->i', dev, dev))
        return pressure, von_mises

    # ------------------------------------------------------------------
    # Momentum balance diagnostic
    # ------------------------------------------------------------------

    def compute_momentum_residual(self, stress_field, grid_shape, grid_points):
        """
        Compute ‖∇·σ‖ using second-order finite differences.

        x, z : periodic  → np.roll wraps stencil across boundaries.
        y    : non-periodic → second-order one-sided stencil at walls,
               central differences in interior.

        Returns residual (nx, ny, nz) in Pa/m.
        """
        nx, ny, nz = grid_shape

        x_u = np.unique(np.round(grid_points[:, 0], 10))
        y_u = np.unique(np.round(grid_points[:, 1], 10))
        z_u = np.unique(np.round(grid_points[:, 2], 10))
        dx  = float(x_u[1] - x_u[0]) if len(x_u) > 1 else 1.0
        dy  = float(y_u[1] - y_u[0]) if len(y_u) > 1 else 1.0
        dz  = float(z_u[1] - z_u[0]) if len(z_u) > 1 else 1.0

        S   = stress_field.reshape(nx, ny, nz, 3, 3)
        div = np.zeros((nx, ny, nz, 3))

        # ∂σ/∂x — periodic
        dSdx        = (np.roll(S, -1, axis=0) - np.roll(S, 1, axis=0)) / (2.0 * dx)
        div        += dSdx[:, :, :, :, 0]

        # ∂σ/∂y — non-periodic, second-order one-sided at boundaries
        dSdy             = np.zeros_like(S)
        dSdy[:, 1:-1, :] = (S[:, 2:, :] - S[:, :-2, :]) / (2.0 * dy)
        dSdy[:, 0,    :] = (-3*S[:, 0, :] + 4*S[:, 1, :] - S[:, 2,  :]) / (2.0 * dy)
        dSdy[:, -1,   :] = ( 3*S[:,-1, :] - 4*S[:,-2, :] + S[:, -3, :]) / (2.0 * dy)
        div             += dSdy[:, :, :, :, 1]

        # ∂σ/∂z — periodic
        dSdz        = (np.roll(S, -1, axis=2) - np.roll(S, 1, axis=2)) / (2.0 * dz)
        div        += dSdz[:, :, :, :, 2]

        residual = np.linalg.norm(div, axis=-1)   # (nx, ny, nz)

        interior = residual[:, 1:-1, :]
        print(f"Momentum residual  max  (interior) : {interior.max():.4e} Pa/m")
        print(f"Momentum residual  mean (interior) : {interior.mean():.4e} Pa/m")
        return residual

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def save_stress_tensor(self, frame, stress_field, grid_points, grid_shape,
                           output_folder, filename=None):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"stress_tensor_t{frame['timestep']}.pt"
        filepath = output_folder / filename

        data = {
            'stress_tensor': torch.from_numpy(stress_field.astype(np.float32)),
            'grid_points'  : torch.from_numpy(grid_points.astype(np.float32)),
            'grid_shape'   : grid_shape,
            'box_bounds'   : torch.from_numpy(frame['box_bounds'].astype(np.float32)),
            'timestep'     : frame['timestep'],
            'N_particles'  : frame['N'],
            'cg_width_w'   : self.w,
        }
        torch.save(data, filepath)
        print(f"  ✓ Saved {filepath}  shape={stress_field.shape}")
        return filepath

    def load_stress_tensor(self, filepath):
        data = torch.load(filepath)
        print(f"✓ Loaded {filepath}  shape={data['stress_tensor'].shape}")
        return data

    # ------------------------------------------------------------------
    # Frame readers
    # ------------------------------------------------------------------

    def read_complete_frames(self):

        dump_files = sorted(
            Path("/Users/kavyanshrajsingh/Desktop/Data/Dump").glob("Dump_Shear.*"),
            key=lambda f: int(f.name.split(".")[-1])
        )

        stress_files = sorted(
            Path("/Users/kavyanshrajsingh/Desktop/Data/Stress").glob("Stress_Shear.*"),
            key=lambda f: int(f.name.split(".")[-1])
        )

        if len(dump_files) != len(stress_files):
            raise RuntimeError(
                f"Number of dump files ({len(dump_files)}) "
                f"does not match number of stress files ({len(stress_files)})"
            )

        if self.max_frames is not None:
            dump_files = dump_files[:int(self.max_frames)]
            stress_files = stress_files[:int(self.max_frames)]

        frames = []

        print(f"\nReading {len(dump_files)} frame pairs ...")

        for i, (dump_file, stress_file) in enumerate(zip(dump_files, stress_files)):

            print(
                f"  [{i+1}/{len(dump_files)}] "
                f"{dump_file.name}  +  {stress_file.name}"
            )

            frames.append(
                self.read_lammps_dump_with_stress(
                    dump_file,
                    stress_file
                )
            )

        self.frames = frames

        print(f"\n✓ Successfully read {len(frames)} frames")

        return frames

    def read_all_frames(self):
        """Read every 50th frame + last (sparse, for animation)."""
        return self.read_complete_frames()

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_and_save_all_frames(self, frames, output_folder, grid_params=None):
        """
        Compute and save CG stress for every frame.

        w and grid are estimated once from frame 0 and reused for all frames
        (consistent spatial coordinates across the dataset).
        """
        import traceback
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        saved = []

        print(f"\nProcessing {len(frames)} frames → {output_folder}")

        # One-time setup from first frame
        frame0  = frames[0]
        kw      = dict(grid_params or {})          # mutable copy — must exist before pop
        xi_mode = kw.pop('xi_mode', 'particle')    # extract before passing to create_grid
        w       = self.estimate_coarse_graining_width(frame0, xi_mode=xi_mode)
        self.w  = w
        grid_points, grid_shape = self.create_grid(frame0, **kw)
        print(f"Grid: {grid_shape},  w = {w:.6f} m")

        for i, frame in enumerate(frames):
            print(f"\n--- Frame {i+1}/{len(frames)}: t={frame['timestep']} ---")
            try:
                sf   = self.coarse_grain_stress(frame, grid_points, w=w)
                path = self.save_stress_tensor(frame, sf, grid_points,
                                               grid_shape, output_folder)
                saved.append(path)
            except Exception as e:
                print(f"✗ Frame {i} failed: {e}")
                traceback.print_exc()

        print(f"\n✓ Saved {len(saved)}/{len(frames)} frames")
        return saved

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_all_stress_components(self, frame, stress_field, grid_points,
                                   grid_shape, slice_dim=2, slice_index=None):
        components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        comp_map   = {'xx':(0,0),'yy':(1,1),'zz':(2,2),
                      'xy':(0,1),'xz':(0,2),'yz':(1,2)}

        if slice_index is None and frame['dim'] == 3:
            slice_index = grid_shape[slice_dim] // 2

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for ax, comp in zip(axes.ravel(), components):
            i, j  = comp_map[comp]
            sg3   = stress_field[:, i, j].reshape(grid_shape)
            if   slice_dim == 0: sg = sg3[slice_index, :, :]; xl,yl = 'y','z'
            elif slice_dim == 1: sg = sg3[:, slice_index, :]; xl,yl = 'x','z'
            else:                sg = sg3[:, :, slice_index].T; xl,yl = 'x','y'

            nz  = sg[sg != 0]
            vm  = np.percentile(np.abs(nz), 95) if len(nz) else 1.0
            ext = [frame['box_bounds'][0,0], frame['box_bounds'][0,1],
                   frame['box_bounds'][1,0], frame['box_bounds'][1,1]]
            im  = ax.imshow(sg, extent=ext, origin='lower', cmap='RdBu_r',
                            aspect='auto', vmin=-vm, vmax=vm)
            plt.colorbar(im, ax=ax, label=f'σ_{comp} [Pa]')
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f'σ_{comp}')

        plt.suptitle(f'Stress components (t={frame["timestep"]})')
        plt.tight_layout()
        return fig

    def plot_momentum_residual(self, residual, grid_shape, frame,
                               slice_dim=2, slice_index=None):
        if slice_index is None:
            slice_index = grid_shape[slice_dim] // 2

        rg = residual.reshape(grid_shape)
        if   slice_dim == 0: sl = rg[slice_index, :, :]; xl,yl = 'y','z'
        elif slice_dim == 1: sl = rg[:, slice_index, :]; xl,yl = 'x','z'
        else:                sl = rg[:, :, slice_index].T; xl,yl = 'x','y'

        x_min, x_max = 0, 50
        y_min, y_max = 0, 38
        z_min, z_max = 0, 10

        ext = ([y_min, y_max, z_min, z_max] if slice_dim == 0 else
               [x_min, x_max, z_min, z_max] if slice_dim == 1 else
               [x_min, x_max, y_min, y_max])

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(sl, extent=ext, origin='lower', cmap='hot_r', aspect='auto')
        plt.colorbar(im, ax=ax, label='‖∇·σ‖  [Pa/m]')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f'Momentum residual ‖∇·σ‖  (t={frame["timestep"]})')
        plt.tight_layout()
        return fig


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    cg = StressCoarseGrainer(
        folder_path="/Users/kavyanshrajsingh/Desktop/Data/Stress",
        pattern="Stress_Shear.*",
        max_frames=100000,
    )

    frames = cg.read_complete_frames()
    print(f"Loaded {len(frames)} frames")

    # ── Mode A: particle-scale IKH profile  (ξ = 1dp, Δy = 1dp) ──────
    # Use this to resolve wall-normal stress profiles and layering.
    # erf Z(y) correction automatically applied at boundary nodes.
    cg.process_and_save_all_frames(
        frames,
        output_folder='stress_tensors_CG',
        grid_params={
            'xi_mode' : 'particle',   # ξ = 1dp = 2R̄
            'n_points': (50,77,1), # 38 y-nodes ≈ 1dp spacing across 40dp channel
        },
    )
     