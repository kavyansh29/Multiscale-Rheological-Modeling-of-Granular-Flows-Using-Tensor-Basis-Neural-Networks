import glob
import torch
import numpy as np
import os


def get_avg_coordNum(file_path):
    """Get average coordination number from the output file."""
    last_value = None
    
    with open(file_path, 'r') as f:
        for line in f:
            # Check if line starts with a number (timestep line)
            if line.strip() and line.split()[0].isdigit():
                # Get the last column value
                columns = line.split()
                last_value = columns[-1]
    coordination_number = float(last_value)
    return coordination_number

def read_radius_from_config(config_file):
    """
    Read LAMMPS config file and extract radius from diameter column
    Returns array of radius values ordered by atom ID
    """
    with open(config_file, 'r') as f:
        lines = f.readlines()
    
    # Find Atoms section
    particles_start = None
    for i, line in enumerate(lines):
        if 'Atoms' in line:
            particles_start = i + 2  # Skip "Atoms" line and blank line
            break
    
    if particles_start is None:
        raise ValueError("Atoms section not found in config file")
    
    # Extract ID and diameter
    particle_data = []
    for line in lines[particles_start:]:
        if not line.strip():  # Stop at blank line
            break
        parts = line.split()
        if len(parts) >= 3:
            particle_id = int(parts[0])
            diameter = float(parts[2])  # dia is 3rd column
            particle_data.append((particle_id, diameter))
    
    # Sort by particle ID to ensure correct order
    particle_data.sort(key=lambda x: x[0])
    
    # Extract diameters and convert to radius
    diameters = np.array([d for _, d in particle_data])
    radius = diameters / 2.0
    
    return radius

def read_dump_file(file_path):
    """
    Read a LAMMPS dump file and extract particle data
    Returns a dictionary with keys: 'N', 'dim', 'x', 'v', 'type', etc.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    frame_data = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ITEM: TIMESTEP"):
            i += 1
            frame_data['timestep'] = int(lines[i].strip())
        elif line.startswith("ITEM: NUMBER OF ATOMS"):
            i += 1
            frame_data['N'] = int(lines[i].strip())
        elif line.startswith("ITEM: BOX BOUNDS"):
            i += 1
            bounds = []
            for _ in range(3):
                bounds.append(list(map(float, lines[i].strip().split())))
                i += 1
            frame_data['box_bounds'] = np.array(bounds)
            frame_data['dim'] = len(bounds)
            continue  # Skip incrementing i here
        elif line.startswith("ITEM: ATOMS"):
            headers = line.split()[2:]  # Get column headers
            data = []
            for j in range(frame_data['N']):
                i += 1
                parts = lines[i].strip().split()
                data.append([float(part) for part in parts])
            data_array = np.array(data)
            
            # Map headers to data columns
            for idx, header in enumerate(headers):
                frame_data[header] = data_array[:, idx]
        i += 1
    
    return frame_data

def read_all_frames(folder_path, pattern="dump.stress.*", max_frames=1000000000):
    """
    Read all dump files in the specified folder matching the pattern
    Returns a list of frames (dictionaries)
    """
    file_paths = sorted(glob.glob(os.path.join(folder_path, pattern)))
    frames = []
    for file_path in file_paths[:max_frames]:
        frame = read_dump_file(file_path)
        frames.append(frame)
    return frames
     

import torch
import numpy as np
from scipy.spatial import cKDTree


# ── Gaussian kernel ───────────────────────────────────────────────────────────
def gaussian_kernel(r, w):
    """Normalised 3-D Gaussian: phi(r) = (1/sqrt(2pi)*w)^3 * exp(-r^2 / 2w^2)"""
    norm = 1.0 / ((np.sqrt(2.0 * np.pi) * w) ** 3)
    return norm * np.exp(-r ** 2 / (2.0 * w ** 2))


def compute_strain_and_fabric(
    frames_sorted,
    grid_shape = (50, 77, 1),
    x_range    = (0.00, 0.50),
    y_range    = (-0.005, 0.805),
    z_range    = (-0.005, 0.005),
    xi_mode    = 'particle',
    device     = 'cpu',
):
    """
    Single-pass computation of strain and fabric with Gaussian coarse-graining.

    For each contact c with midpoint m_c, its contribution is spread to ALL
    cell centres within a cutoff radius r_cut = 3w, weighted by phi(|x_cell - m_c|, w).

    Gaussian coarse-graining replaces hard voxel assignment:
        BEFORE:  contact c → cell k if midpoint in cell k  (weight = 1 or 0)
        AFTER:   contact c → all cells k within r_cut      (weight = phi(dist, w))

    Outputs
    -------
    strain : [N_frames, Nx, Ny, Nz, 1]        weighted mean eps per voxel
    fabric : [N_frames, Nx, Ny, Nz, 3, 3]     weighted mean n⊗n - (1/3)I per voxel
    """
    Nx, Ny, Nz = grid_shape
    N_frames   = len(frames_sorted)
    N_cells    = Nx * Ny * Nz

    Lx = x_range[1] - x_range[0]
    Lz = z_range[1] - z_range[0]

    # Cell centres
    xc = 0.5 * (np.linspace(*x_range, Nx + 1)[:-1] + np.linspace(*x_range, Nx + 1)[1:])
    yc = 0.5 * (np.linspace(*y_range, Ny + 1)[:-1] + np.linspace(*y_range, Ny + 1)[1:])
    zc = 0.5 * (np.linspace(*z_range, Nz + 1)[:-1] + np.linspace(*z_range, Nz + 1)[1:])

    # Cell centre coordinate array: (N_cells, 3)
    Xc, Yc, Zc  = np.meshgrid(xc, yc, zc, indexing='ij')   # each (Nx, Ny, Nz)
    cell_centres = np.column_stack([Xc.ravel(), Yc.ravel(), Zc.ravel()])  # (N_cells, 3)

    # 9 periodic image shifts in x-z
    shifts = np.array(
        [[dx * Lx, 0.0, dz * Lz] for dx in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=np.float64
    )  # (9, 3)


    strain_frames = []
    fabric_frames = []
    cc_shifts = np.array(
            [[dx * Lx, 0.0, dz * Lz] for dx in (-1, 0, 1) for dz in (-1, 0, 1)],
            dtype=np.float64
        )  # (9, 3)
    aug_cc     = np.vstack([cell_centres + s for s in cc_shifts])  # (9*N_cells, 3)
    aug_cc_idx = np.tile(np.arange(N_cells), 9)                    # original cell idx

    cc_tree = cKDTree(aug_cc)

    for t, frame in enumerate(frames_sorted):

        z = frame['z'] if 'z' in frame else np.zeros_like(frame['x'])
        pos = np.column_stack([frame['x'], frame['y'], z])
        types = frame["type"].astype(int)
        mask = (types == 1)
        types = frame["type"].astype(int)
        mask = (types == 1)

        pos = pos[mask]
        radii = np.asarray(frame["radius"], dtype=np.float64)[mask]
        ids = frame["id"].astype(int)[mask]

        order = np.argsort(ids)

        ids = ids[order]
        pos = pos[order]
        radii = radii[order]
        N     = len(pos)
        r_max = radii.max()

        # ── Bandwidth from mean particle diameter ─────────────────────────
        mean_r = float(np.mean(radii))
        dp     = 2.0 * mean_r
        if xi_mode == 'particle':
            w = 2.0 * dp
        elif xi_mode == 'smooth':
            w = 3.0 * dp
        else:
            raise ValueError(f"xi_mode must be 'particle' or 'smooth', got '{xi_mode}'")
        r_cut = 3.0 * w          # Gaussian is negligible beyond 3w

        # ── 9-image augmented cloud ───────────────────────────────────────
        aug_pos = np.vstack([pos + s for s in shifts])
        img_id  = np.tile(np.arange(N), 9)

        # ── Contact detection ─────────────────────────────────────────────
        tree  = cKDTree(aug_pos)
        pairs = tree.query_pairs(r=2.05 * r_max, output_type='ndarray')

        if len(pairs) == 0:
            print(f"  [{t+1:4d}/{N_frames}]  WARNING: zero pairs found")
            strain_frames.append(np.zeros((Nx, Ny, Nz), dtype=np.float32))
            fabric_frames.append(np.zeros((Nx, Ny, Nz, 3, 3), dtype=np.float32))
            continue

        ai = pairs[:, 0];  aj = pairs[:, 1]
        oi = img_id[ai];   oj = img_id[aj]

        # Drop same-particle pairs (own periodic image)
        mask           = oi != oj
        ai, aj         = ai[mask], aj[mask]
        oi, oj         = oi[mask], oj[mask]

        # Canonicalise: each physical pair once
        swap           = oi > oj
        oi[swap], oj[swap] = oj[swap].copy(), oi[swap].copy()
        ai[swap], aj[swap] = aj[swap].copy(), ai[swap].copy()
        _, uniq        = np.unique(np.column_stack([oi, oj]), axis=0, return_index=True)
        ai, aj         = ai[uniq], aj[uniq]
        oi, oj         = oi[uniq], oj[uniq]

        # ── Branch vector, overlap, unit normal ───────────────────────────
        branch = aug_pos[aj] - aug_pos[ai]              # (M, 3)
        dist   = np.linalg.norm(branch, axis=1)         # (M,)
        r_sum  = radii[oi] + radii[oj]                  # (M,)
        delta  = r_sum - dist

        c = delta > 0.0
        if not c.any():
            print(f"  [{t+1:4d}/{N_frames}]  WARNING: no real contacts")
            strain_frames.append(np.zeros((Nx, Ny, Nz), dtype=np.float32))
            fabric_frames.append(np.zeros((Nx, Ny, Nz, 3, 3), dtype=np.float32))
            continue

        branch_c = branch[c]
        dist_c   = dist[c]
        delta_c  = delta[c]
        r_sum_c  = r_sum[c]
        ai_c     = ai[c];  aj_c = aj[c]

        n   = (branch_c / dist_c[:, None]).astype(np.float32)   # (Mc, 3)
        eps = (delta_c  / r_sum_c).astype(np.float32)           # (Mc,)
        nnT = (n[:, :, None] * n[:, None, :])                   # (Mc, 3, 3)

        # ── Contact midpoints with PBC wrap ───────────────────────────────
        mid = 0.5 * (aug_pos[ai_c] + aug_pos[aj_c])             # (Mc, 3)
        mid[:, 0] = (mid[:, 0] - x_range[0]) % Lx + x_range[0]
        mid[:, 2] = (mid[:, 2] - z_range[0]) % Lz + z_range[0]

        # Domain filter: midpoint must be in bulk y range
        # (x and z are always in range after wrap)
        valid = (mid[:, 1] >= y_range[0]) & (mid[:, 1] < y_range[1])
        mid_v = mid[valid]          # (Mv, 3)
        eps_v = eps[valid]          # (Mv,)
        nnT_v = nnT[valid]          # (Mv, 3, 3)

        if len(eps_v) == 0:
            print(f"  [{t+1:4d}/{N_frames}]  no valid contacts in bulk domain")
            strain_frames.append(np.zeros((Nx, Ny, Nz), dtype=np.float32))
            fabric_frames.append(np.zeros((Nx, Ny, Nz, 3, 3), dtype=np.float32))
            continue

        # ── Gaussian coarse-graining ──────────────────────────────────────
        # For each contact midpoint m_c, find all cell centres within r_cut.
        # Weight each cell by phi(|cell_centre - m_c|, w).
        # Accumulate weighted eps and n⊗n, then normalise by sum of weights.
        #
        # Uses a KDTree on cell centres for fast radius search.
        # PBC in x and z: augment cell centres with periodic images too,
        # then map back to primary cell index.

        # Build cell-centre tree (done once per frame, same w)
        # Augment cell centres with x-z periodic images for PBC-safe search

        # Weighted accumulators
        W_eps = np.zeros(N_cells, dtype=np.float64)          # sum phi * eps
        W_fab = np.zeros((N_cells, 3, 3), dtype=np.float64)  # sum phi * n⊗n
        W_sum = np.zeros(N_cells, dtype=np.float64)          # sum phi (normaliser)

        # Query: for each contact midpoint, find all cell centres within r_cut
        hits = cc_tree.query_ball_point(mid_v, r=r_cut)      # list of lists

        for k, nbr_list in enumerate(hits):
            if len(nbr_list) == 0:
                continue
            nbr      = np.asarray(nbr_list, dtype=np.int64)
            cell_idx = aug_cc_idx[nbr]                        # original cell indices

            # Distance from contact midpoint to each neighbouring cell centre
            dr   = aug_cc[nbr] - mid_v[k]                    # (K, 3)
            r2   = (dr ** 2).sum(axis=1)                      # (K,)
            phi  = gaussian_kernel(np.sqrt(r2), w)            # (K,)

            np.add.at(W_eps, cell_idx, phi * eps_v[k])
            np.add.at(W_sum, cell_idx, phi)
            for a in range(3):
                for b in range(3):
                    np.add.at(W_fab[:, a, b], cell_idx, phi * nnT_v[k, a, b])

        # ── Normalise ─────────────────────────────────────────────────────
        has = W_sum > 0.0

        strain_field = np.zeros(N_cells, dtype=np.float32)
        fabric_field = np.zeros((N_cells, 3, 3), dtype=np.float32)

        strain_field[has] = (W_eps[has] / W_sum[has]).astype(np.float32)

        fabric_field[has] = (W_fab[has] / W_sum[has, None, None]).astype(np.float32)
        # fabric_field[has] -= I3[None, :, :] / 3.0            # traceless: subtract (1/3)I
        
        n_contacts = len(eps_v)

        strain_frames.append(strain_field.reshape(Nx, Ny, Nz))
        fabric_frames.append(fabric_field.reshape(Nx, Ny, Nz, 3, 3))

        del aug_pos
        del pairs
        del branch
        del branch_c
        del n
        del nnT
        del nnT_v
        del W_eps
        del W_fab
        del W_sum
        del tree

        if (t + 1) % max(1, N_frames // 10) == 0 or t == 0:
            mean_aniso = np.sqrt(
                (fabric_frames[-1].reshape(N_cells, 3, 3)[has] ** 2).sum(axis=(1, 2))
            ).mean() if has.any() else 0.0
            print(
                f"  [{t+1:4d}/{N_frames}]"
                f"  w={w:.4f}m  r_cut={r_cut:.4f}m"
                f"  contacts={c.sum():8d}"
                f"  in_domain={valid.sum():8d}"
                f"  cells_with_weight={has.sum():5d}/{N_cells}"
                f"  mean_eps={eps_v.mean():.4e}"
                f"  mean_|A|={mean_aniso:.4e}"
            )

    # ── Stack and return ──────────────────────────────────────────────────
    print("\nStacking frames...")

    strain = torch.from_numpy(
        np.stack(strain_frames, axis=0)
    ).unsqueeze(-1).to(device)                               # [T, Nx, Ny, Nz, 1]

    fabric = torch.from_numpy(
        np.stack(fabric_frames, axis=0)
    ).to(device)                                             # [T, Nx, Ny, Nz, 3, 3]

    print(f"  strain : {list(strain.shape)}  {strain.dtype}")
    print(f"  fabric : {list(fabric.shape)}  {fabric.dtype}")

    return strain, fabric

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    folder_path = "/Users/kavyanshrajsingh/Desktop/Data/Dump"

    print("Reading dump files...")

    frames = read_all_frames(
        folder_path,
        pattern="Dump_Shear.*"
    )

    print(f"Loaded {len(frames)} frames")

    frames_sorted = sorted(
        frames,
        key=lambda x: int(x["timestep"])
    )

    print("Computing strain and fabric...")

    strain, fabric = compute_strain_and_fabric(
        frames_sorted,
        grid_shape=(50,77,1),
        x_range=(0.0,0.5),
        y_range=(-0.005,0.805),
        z_range=(-0.005,0.005),
        xi_mode="particle",
        device="cpu"
    )

    print()

    print("Saving...")

    torch.save(
        strain,
        "strain_tensor_cg.pt"
    )

    torch.save(
        fabric,
        "fabric_tensor_cg.pt"
    )

    print("Done.")
