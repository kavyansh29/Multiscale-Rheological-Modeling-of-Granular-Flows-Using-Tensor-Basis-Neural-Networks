"""
phi_voxel_gaussian.py
=====================
Two-stage local solid volume fraction for DEM granular data.

Device strategy
---------------
  cKDTree        : always CPU (scipy limitation — used only for candidate lookup)
  tensor math    : GPU if available, else CPU
  Rule           : every tensor is created/moved via the module-level DEVICE;
                   .cpu().numpy() is called only when passing to cKDTree.

Gridding
--------
  Full box  : Ly = 0.40 m  (Ny_full = 40 at d = 0.01 m)
  Bulk range: y ∈ [y_lo+d, y_hi−d]  →  Ly_eff = 0.38 m
  Grid      : Nx=50, Ny=77, Nz=1
  PBC       : x (flow), z (vorticity)  →  circular padding
  Walls     : y (gradient)             →  reflect padding
"""
from volume_fraction_coarse_graining_1 import frames_sorted
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List

# ── global device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device]  using {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 0.  TRIMMED BOUNDS
# ─────────────────────────────────────────────────────────────────────────────

def get_bulk_bounds(box_bounds: np.ndarray, particle_diam: float = 0.01) -> torch.Tensor:
    """
    Trim one particle diameter from each y-wall; keep x, z unchanged.
    Returns (3,2) float64 tensor on DEVICE.
    """
    b    = np.asarray(box_bounds, dtype=np.float64)
    bulk = b.copy()
    bulk[1, 0] += particle_diam
    bulk[1, 1] -= particle_diam
    t = torch.from_numpy(bulk).to(DEVICE)
    print(f"[bounds]  y_full=[{b[1,0]:.4f},{b[1,1]:.4f}]  "
          f"y_bulk=[{bulk[1,0]:.4f},{bulk[1,1]:.4f}]  "
          f"Ly_eff={bulk[1,1]-bulk[1,0]:.4f} m")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARTICLE ARRAYS
# ─────────────────────────────────────────────────────────────────────────────

def build_particle_arrays(
        snapshot:  dict,
        bulk_type: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, cKDTree]:
    """
    Returns
    -------
    centers : (Np,3) float64 on DEVICE
    radii   : (Np,)  float64 on DEVICE
    tree    : cKDTree built on CPU numpy array (scipy requirement)
    """
    type_arr = np.asarray(snapshot['type'])
    mask     = (type_arr == bulk_type)

    centers_np = np.column_stack([
        np.asarray(snapshot['x'],      dtype=np.float64)[mask],
        np.asarray(snapshot['y'],      dtype=np.float64)[mask],
        np.asarray(snapshot['z'],      dtype=np.float64)[mask],
    ])                                                        # (Np,3) CPU numpy
    radii_np   = np.asarray(snapshot['radius'], dtype=np.float64)[mask]

    centers = torch.from_numpy(centers_np).to(DEVICE)        # GPU
    radii   = torch.from_numpy(radii_np  ).to(DEVICE)
    tree    = cKDTree(centers_np)                             # CPU only

    print(f"[build]  Np={len(radii_np)}  "
          f"r_mean={radii_np.mean():.5f}  r_max={radii_np.max():.5f}")
    return centers, radii, tree


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MAX OVERLAP → POINT SEPARATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_max_overlap_and_separation(
        centers:  torch.Tensor,   # DEVICE
        radii:    torch.Tensor,   # DEVICE
        tree:     cKDTree,        # CPU
        n_sample: int   = 2000,
        safety:   float = 1.05,
) -> Tuple[float, float]:
    """
    Samples n_sample particles, queries CPU tree, computes distances on GPU.
    """
    n_total  = centers.shape[0]
    n_sample = min(n_sample, n_total)
    idx      = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))[:n_sample]

    # one-time CPU copy for tree queries
    centers_cpu = centers.cpu().numpy()
    r_max_val   = radii.max().item()
    max_overlap = torch.tensor(0.0, dtype=centers.dtype, device=DEVICE)

    for i in idx.tolist():
        r_i  = radii[i]
        nbrs = tree.query_ball_point(centers_cpu[i], r_i.item() + r_max_val)
        nbrs = [j for j in nbrs if j != i]
        if not nbrs:
            continue
        j_idx    = torch.tensor(nbrs, dtype=torch.long, device=DEVICE)
        dists    = torch.linalg.norm(centers[i] - centers[j_idx], dim=1)  # GPU
        overlaps = r_i + radii[j_idx] - dists
        best     = overlaps.max()
        if best > max_overlap:
            max_overlap = best

    lens_radius = max_overlap / 2.0
    separation  = safety * torch.max(lens_radius, radii.max() * 0.05)
    print(f"[overlap]  max_overlap={max_overlap.item():.5f} m  "
          f"sep={separation.item():.5f} m")
    return max_overlap.item(), separation.item()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TEMPLATE VOXEL POINTS
# ─────────────────────────────────────────────────────────────────────────────

def make_template_voxel_points(
        dx: float, dy: float, dz: float,
        separation: float,
) -> Tuple[torch.Tensor, int]:
    """
    Sub-voxel uniform points in [0,dx]×[0,dy]×[0,dz]. Built once, on DEVICE.
    """
    nx_pt = max(1, int(np.floor(dx / separation)))
    ny_pt = max(1, int(np.floor(dy / separation)))
    nz_pt = max(1, int(np.floor(dz / separation)))

    xs = np.linspace(dx/(2*nx_pt), dx - dx/(2*nx_pt), nx_pt)
    ys = np.linspace(dy/(2*ny_pt), dy - dy/(2*ny_pt), ny_pt)
    zs = np.linspace(dz/(2*nz_pt), dz - dz/(2*nz_pt), nz_pt)

    XX, YY, ZZ   = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_np       = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    template_pts = torch.from_numpy(pts_np).to(DEVICE)       # GPU
    Npt          = len(template_pts)

    print(f"[template]  {nx_pt}×{ny_pt}×{nz_pt} = {Npt} pts/voxel  "
          f"voxel=({dx:.4f},{dy:.4f},{dz:.4f})")
    return template_pts, Npt


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SINGLE-VOXEL PHI  (GPU math, CPU tree query)
# ─────────────────────────────────────────────────────────────────────────────

def _phi_one_voxel(
        vox_origin:    torch.Tensor,   # (3,)    DEVICE
        template_pts:  torch.Tensor,   # (Npt,3) DEVICE
        centers:       torch.Tensor,   # (Np,3)  DEVICE
        centers_cpu:   np.ndarray,     # (Np,3)  CPU — tree queries only
        radii:         torch.Tensor,   # (Np,)   DEVICE
        tree:          cKDTree,
        r_max:         float,
        vox_half_diag: float,
) -> float:
    """OR-test point counting. Candidate lookup on CPU tree, distance math on GPU."""
    pts        = vox_origin + template_pts                              # (Npt,3) GPU
    vox_ctr    = (vox_origin + template_pts.mean(dim=0)).cpu().numpy() # CPU for tree
    candidates = tree.query_ball_point(vox_ctr, vox_half_diag + r_max)

    if not candidates:
        return 0.0

    cand_idx   = torch.tensor(candidates, dtype=torch.long, device=DEVICE)
    dists      = torch.cdist(pts, centers[cand_idx], p=2)              # (Npt,Nc) GPU
    inside_any = (dists <= radii[cand_idx]).any(dim=1)
    return inside_any.float().mean().item()

# ─── PBC FUNCTION ──────────────────────────────────────────────────────────────

def _augment_pbc_images(
        centers:     torch.Tensor,   # (Np,3) DEVICE
        radii:       torch.Tensor,   # (Np,)  DEVICE
        bulk_bounds: torch.Tensor,   # (3,2)  DEVICE
        cutoff:      float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add image particles for PBC in x (dim 0) and z (dim 2).
    Only duplicates particles within `cutoff` of each x/z boundary face.
    y is bounded (walls) — no images needed.
    """
    x_lo, x_hi = bulk_bounds[0, 0].item(), bulk_bounds[0, 1].item()
    z_lo, z_hi = bulk_bounds[2, 0].item(), bulk_bounds[2, 1].item()
    Lx = x_hi - x_lo
    Lz = z_hi - z_lo

    cx, cz = centers[:, 0], centers[:, 2]

    near_xlo = (cx - x_lo) < cutoff
    near_xhi = (x_hi - cx) < cutoff
    near_zlo = (cz - z_lo) < cutoff
    near_zhi = (z_hi - cz) < cutoff

    # 4 face images + 4 corner images
    shifts = [
        ( Lx,   0.0,  near_xlo),
        (-Lx,   0.0,  near_xhi),
        ( 0.0,  Lz,   near_zlo),
        ( 0.0, -Lz,   near_zhi),
        ( Lx,   Lz,   near_xlo & near_zlo),
        ( Lx,  -Lz,   near_xlo & near_zhi),
        (-Lx,   Lz,   near_xhi & near_zlo),
        (-Lx,  -Lz,   near_xhi & near_zhi),
    ]

    aug_c = [centers]
    aug_r = [radii]
    for dx, dz, mask in shifts:
        if mask.any():
            c = centers[mask].clone()
            c[:, 0] += dx
            c[:, 2] += dz
            aug_c.append(c)
            aug_r.append(radii[mask])

    centers_aug = torch.cat(aug_c, dim=0)
    radii_aug   = torch.cat(aug_r, dim=0)
    n_img = centers_aug.shape[0] - centers.shape[0]
    print(f"[pbc images]  added {n_img} image particles  "
          f"(total {centers_aug.shape[0]} incl. images)")
    return centers_aug, radii_aug
# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL VOXEL SWEEP
# ─────────────────────────────────────────────────────────────────────────────

# ─── PATCHED compute_phi_voxelwise ───────────────────────────────────────────
# ** augment particles with PBC images BEFORE the sweep,
# then rebuild the KD-tree from the augmented set.

def compute_phi_voxelwise(
        centers:     torch.Tensor,
        radii:       torch.Tensor,
        tree:        cKDTree,           # original tree (used nowhere below now)
        bulk_bounds: torch.Tensor,
        grid_shape:  Tuple[int, int, int],
        separation:  float,
        verbose:     bool = True,
) -> torch.Tensor:
    """Returns phi_voxel (Nx,Ny,Nz) float64 on DEVICE."""
    Nx, Ny, Nz  = grid_shape
    diffs       = bulk_bounds[:, 1] - bulk_bounds[:, 0]
    Lx, Ly, Lz = diffs[0].item(), diffs[1].item(), diffs[2].item()
    dx, dy, dz  = Lx/Nx, Ly/Ny, Lz/Nz

    vox_half_diag = 0.5 * (dx**2 + dy**2 + dz**2)**0.5
    r_max         = radii.max().item()
    cutoff        = r_max + vox_half_diag   # query radius used in _phi_one_voxel

    # ── PBC IMAGE AUGMENTATION (new) ─────────────────────────────────────────
    centers_aug, radii_aug = _augment_pbc_images(
        centers, radii, bulk_bounds, cutoff
    )
    centers_cpu = centers_aug.cpu().numpy()      # rebuild tree on augmented set
    tree_aug    = cKDTree(centers_cpu)
    # ─────────────────────────────────────────────────────────────────────────

    template_pts, _ = make_template_voxel_points(dx, dy, dz, separation)

    phi_voxel  = torch.zeros((Nx, Ny, Nz), dtype=torch.float64, device=DEVICE)
    x0, y0, z0 = bulk_bounds[:, 0].tolist()
    total, done = Nx * Ny * Nz, 0

    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                origin = torch.tensor(
                    [x0 + ix*dx, y0 + iy*dy, z0 + iz*dz],
                    dtype=torch.float64, device=DEVICE,
                )
                phi_voxel[ix, iy, iz] = _phi_one_voxel(
                    origin, template_pts,
                    centers_aug, centers_cpu,   # ← augmented set
                    radii_aug, tree_aug,         # ← augmented tree
                    r_max, vox_half_diag,
                )
                done += 1
                if verbose and done % max(1, total // 20) == 0:
                    filled = phi_voxel[phi_voxel > 0]
                    mean_s = f"phi_mean={filled.mean().item():.4f}" if filled.numel() else "phi_mean=n/a"
                    print(f"[voxel]  {done}/{total} ({100*done//total}%)  {mean_s}")

    if verbose:
        print(f"[voxel done]  mean={phi_voxel.mean():.4f}  "
              f"min={phi_voxel.min():.4f}  max={phi_voxel.max():.4f}")
    return phi_voxel

# ─────────────────────────────────────────────────────────────────────────────
# 6.  GAUSSIAN SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_smooth_phi(
        phi_voxel:   torch.Tensor,
        bulk_bounds: torch.Tensor,
        grid_shape:  Tuple[int, int, int],
        sigma_phys:  Optional[float] = None,
        sigma_vox:   Optional[Tuple[float, float, float]] = None,
        truncate:    float = 3.0,
) -> torch.Tensor:
    """
    Separable Gaussian filter entirely on DEVICE.
      x → circular (PBC)   y → reflect (walls)   z → circular (PBC)
    """
    Nx, Ny, Nz = grid_shape
    diffs = bulk_bounds[:, 1] - bulk_bounds[:, 0]
    dx    = (diffs[0] / Nx).item()
    dy    = (diffs[1] / Ny).item()
    dz    = (diffs[2] / Nz).item()

    if sigma_vox is not None:
        sx, sy, sz = sigma_vox
    elif sigma_phys is not None:
        sx, sy, sz = sigma_phys/dx, sigma_phys/dy, sigma_phys/dz
    else:
        sx = sy = sz = 2.0

    print(f"[smooth]  sigma_vox=({sx:.2f},{sy:.2f},{sz:.2f})  "
          f"voxel=({dx:.4f},{dy:.4f},{dz:.4f})")

    def _kernel1d(sigma: float) -> torch.Tensor:
        r = int(truncate * sigma + 0.5)
        x = torch.arange(-r, r+1, dtype=phi_voxel.dtype, device=DEVICE)
        k = torch.exp(-0.5 * (x / sigma)**2)
        return k / k.sum()

    kx, ky, kz = _kernel1d(sx), _kernel1d(sy), _kernel1d(sz)
    out = phi_voxel.unsqueeze(0).unsqueeze(0)   # (1,1,Nx,Ny,Nz)

    # x — circular (PBC)
    px  = len(kx) // 2
    out = F.pad(out, (0, 0, 0, 0, px, px), mode='circular')
    out = F.conv3d(out, kx.view(1, 1, -1, 1, 1))

    # y — reflect (walls)
    py  = len(ky) // 2
    out = F.pad(out, (0, 0, py, py, 0, 0), mode='reflect')
    out = F.conv3d(out, ky.view(1, 1, 1, -1, 1))

    # z — circular (PBC)
    if Nz > 1:
        pz = len(kz) // 2
        out = F.pad(out, (pz, pz, 0, 0, 0, 0), mode='circular')
        out = F.conv3d(out, kz.view(1, 1, 1, 1, -1))

    phi_smooth = out.squeeze(0).squeeze(0)
    
    print(f"[smooth]  mean={phi_smooth.mean():.4f}  "
          f"min={phi_smooth.min():.4f}  max={phi_smooth.max():.4f}")
    return phi_smooth


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SINGLE-SNAPSHOT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_phi_two_stage(
        snapshot:         dict,
        grid_shape:       Tuple[int, int, int] = (50, 77, 1),
        bulk_type:        int   = 1,
        particle_diam:    float = 0.01,
        n_overlap_sample: int   = 2000,
        safety:           float = 1.05,
        sigma_phys:       Optional[float] = None,
        sigma_vox:        Optional[Tuple[float, float, float]] = None,
        verbose:          bool  = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    phi_voxel  : (Nx,Ny,Nz)       float64  DEVICE
    phi_smooth : (Nx,Ny,Nz)       float64  DEVICE
    phi_tensor : (1,Nx,Ny,Nz,1)   float32  DEVICE  — ML-ready
    """
    centers, radii, tree = build_particle_arrays(snapshot, bulk_type)
    bulk_bounds          = get_bulk_bounds(snapshot['box_bounds'], particle_diam)

    _, separation = compute_max_overlap_and_separation(
        centers, radii, tree, n_sample=n_overlap_sample, safety=safety
    )
    phi_voxel  = compute_phi_voxelwise(
        centers, radii, tree, bulk_bounds, grid_shape, separation, verbose
    )
    phi_smooth = gaussian_smooth_phi(
        phi_voxel, bulk_bounds, grid_shape, sigma_phys, sigma_vox
    )
    phi_tensor = phi_smooth.float().unsqueeze(0).unsqueeze(-1)  # (1,Nx,Ny,Nz,1)
    return phi_voxel, phi_smooth, phi_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TIME-SERIES WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_phi_timeseries(
        frames:        List[dict],
        grid_shape:    Tuple[int, int, int] = (50, 77, 1),
        bulk_type:     int   = 1,
        particle_diam: float = 0.01,
        safety:        float = 1.05,
        sigma_phys:    Optional[float] = None,
        sigma_vox:     Optional[Tuple[float, float, float]] = None,
        verbose:       bool  = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    phi_voxel_ts  : (Nt,Nx,Ny,Nz)    float32  DEVICE
    phi_smooth_ts : (Nt,Nx,Ny,Nz,1)  float32  DEVICE
    """
    Nt         = len(frames)
    Nx, Ny, Nz = grid_shape

    # pre-compute from frame 0 (radii and box are constant)
    cen0, rad0, tree0 = build_particle_arrays(frames[0], bulk_type)
    bulk_bounds       = get_bulk_bounds(frames[0]['box_bounds'], particle_diam)
    _, separation     = compute_max_overlap_and_separation(
        cen0, rad0, tree0, n_sample=2000, safety=safety
    )
    diffs = bulk_bounds[:, 1] - bulk_bounds[:, 0]
    dx, dy, dz = (diffs[0]/Nx).item(), (diffs[1]/Ny).item(), (diffs[2]/Nz).item()
    _, Npt = make_template_voxel_points(dx, dy, dz, separation)
    print(f"\n[timeseries]  Nt={Nt}  grid={Nx}×{Ny}×{Nz}  "
          f"Ly_eff={diffs[1].item():.4f} m  Npt/voxel={Npt}  device={DEVICE}")

    phi_v_all = torch.zeros((Nt, Nx, Ny, Nz), dtype=torch.float32, device=DEVICE)
    phi_s_all = torch.zeros((Nt, Nx, Ny, Nz), dtype=torch.float32, device=DEVICE)

    for t, frame in enumerate(frames):
        print(f"\n[frame {t+1}/{Nt}]  timestep={frame['timestep']}")
        cen, rad, tr = build_particle_arrays(frame, bulk_type)

        phi_v = compute_phi_voxelwise(
            cen, rad, tr, bulk_bounds, grid_shape, separation, verbose=False
        )
        phi_s = gaussian_smooth_phi(
            phi_v, bulk_bounds, grid_shape, sigma_phys, sigma_vox
        )
        phi_v_all[t] = phi_v.float()
        phi_s_all[t] = phi_s.float()
        if t == 0:
            import sys
            sys.exit()
        print(f"  raw:    mean={phi_v.mean():.4f}  min={phi_v.min():.4f}  max={phi_v.max():.4f}")
        print(f"  smooth: mean={phi_s.mean():.4f}  min={phi_s.min():.4f}  max={phi_s.max():.4f}")

    phi_smooth_ts = phi_s_all.unsqueeze(-1)
    print(f"\n[done]  phi_smooth_ts: {list(phi_smooth_ts.shape)}  device={phi_smooth_ts.device}")
    return phi_v_all, phi_smooth_ts

# ─────────────────────────────────────────────────────────────────────────────
# 9.  DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_phi(
        phi_voxel:   torch.Tensor,   # (Nt,Nx,Ny,Nz)
        phi_smooth:  torch.Tensor,   # (Nt,Nx,Ny,Nz) or (Nt,Nx,Ny,Nz,1)
        bulk_bounds: torch.Tensor,   # (3,2)
) -> None:
    if phi_smooth.dim() == 5:
        phi_smooth = phi_smooth.squeeze(-1)

    Ny   = phi_voxel.shape[2]
    Ly   = (bulk_bounds[1, 1] - bulk_bounds[1, 0]).item()
    dy   = Ly / Ny
    y_lo = bulk_bounds[1, 0].item()
    y_c  = [y_lo + (iy + 0.5) * dy for iy in range(Ny)]

    # pull to CPU for printing
    phi_v_y = phi_voxel.float().mean(dim=(0, 1, 3)).cpu()
    phi_s_y = phi_smooth.float().mean(dim=(0, 1, 3)).cpu()

    print("\n── phi(y) bulk profile  [avg over t, x, z] ──────────────────────")
    print(f"  {'y_centre':>9}  {'raw':>7}  {'smooth':>7}  bar")
    for iy in range(Ny):
        bar = '█' * int(phi_s_y[iy].item() * 40)
        print(f"  y={y_c[iy]:.4f}  {phi_v_y[iy]:.4f}  {phi_s_y[iy]:.4f}  {bar}")
    print("─────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    phi_voxel_ts, phi_smooth_ts = compute_phi_timeseries(
        frames=frames_sorted,
        grid_shape=(50, 77, 1),
        bulk_type=1,
        particle_diam=0.01,
        safety=1.05,
        sigma_phys=0.015,
    )

    # ---------------- SAVE RESULTS ----------------

    import os

    output_dir = "volume_fraction_cgs"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        phi_voxel_ts.cpu(),
        os.path.join(output_dir, "phi_voxel_timeseries.pt")
    )

    torch.save(
        phi_smooth_ts.cpu(),
        os.path.join(output_dir, "phi_smooth_timeseries.pt")
    )

    print("\n✓ Saved:")
    print("   volume_fraction_cgs/phi_voxel_timeseries.pt")
    print("   volume_fraction_cgs/phi_smooth_timeseries.pt")

    # ---------------- DIAGNOSTICS ----------------

    bulk_bounds = get_bulk_bounds(
        frames_sorted[0]["box_bounds"],
        particle_diam=0.01
    )

    diagnose_phi(
        phi_voxel_ts,
        phi_smooth_ts,
        bulk_bounds
    )

    print("\nFinished.")