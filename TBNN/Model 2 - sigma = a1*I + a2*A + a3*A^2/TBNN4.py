"""
Granular Constitutive TBNN — A only
sigma* = a1*I + a2*A + a3*A²   (sigma* = sigma * dp/kn)
scalar invariants: I1=tr(A), I2=0.5*(I1²-tr(A²))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

SYM_IJ = [
(0,0),
(1,1),
(0,1)
]

# =============================================================================
# DATA PROCESSOR
# =============================================================================

class GranularDataProcessor:

    def __init__(self):
        self._x_mean    = None
        self._x_std     = None
        self._s_var     = None
        self._s_clip_lo = None
        self._s_clip_hi = None

    def calc_scalar_basis(self, A, phi, epsilon, is_train=True):
        """
        3 scalar invariants of A, z-scored on training set.
        Returns (x_phys [N,3], x_scaled [N,3])
        """
        # I1 = np.einsum('nii->n', A)                              # tr(A)
        I2 = 0.5 * ((np.einsum('nii->n', A))**2 - np.einsum('nij,nij->n', A, A))      # 0.5*(I1²-tr(A²))
        I3 = np.linalg.det(A)                                         # det(A)
        x  = np.stack([I2, I3, phi, epsilon], axis=1)                          # [N,4]
        x  = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if is_train:
            self._x_mean = x.mean(axis=0)
            self._x_std  = x.std(axis=0).clip(1e-8)
        if self._x_mean is None:
            raise RuntimeError("Call with is_train=True first.")

        return x, (x - self._x_mean) / self._x_std

    def calc_tensor_basis(self, A):
        """
        Returns tb [N, 3, 3, 3]: [I, A, A²]
        All 3 bases passed to TBNN (a1*I + a2*A + a3*A²).
        """
        N  = A.shape[0]
        I_ = np.eye(3)[None].repeat(N, axis=0)
        A = np.nan_to_num(A)
        A2 = np.einsum('nij,njk->nik', A, A)
        return np.stack([I_, A, A2], axis=1)                     # [N, 3, 3, 3]

    def calc_output(self, sigma, is_train=True, clip_percentile=99.5):
        """
        Non-dimensionalise: sigma* = sigma * dp/kn
        Returns s3 [N,3]
        """

        kn = 1e5
        dp = 0.01

        s = sigma * (dp / kn)

        s3 = np.stack(
            [
                s[:,0,0],
                s[:,1,1],
                s[:,0,1]
            ],
            axis=1
        )

        s3 = np.nan_to_num(s3, nan=0.0, posinf=0.0, neginf=0.0)

        if is_train:
            self._s_clip_lo = np.percentile(
                s3,
                100 - clip_percentile,
                axis=0
            )
            self._s_clip_hi = np.percentile(
                s3,
                clip_percentile,
                axis=0
            )

        s3 = np.clip(s3, self._s_clip_lo, self._s_clip_hi)

        if is_train:
            self._s_var = s3.var(axis=0).clip(1e-8)

            names = ["xx", "yy", "xy"]

            for k, name in enumerate(names):
                print(
                    f"  sigma*[{name}]: "
                    f"mean={s3[:,k].mean():+.5e}  "
                    f"std={np.sqrt(self._s_var[k]):.5e}"
                )

        return s3

# =============================================================================
# NETWORK STRUCTURE
# =============================================================================

class NetworkStructure:
    def __init__(self):
        self.num_layers       = 3
        self.num_nodes        = 64
        self.max_epochs       = 3000
        self.min_epochs       = 200
        self.interval         = 10
        self.average_interval = 3
        self.learning_rate    = 1e-4
        self.batch_size       = 512
        self.split_fraction   = 0.8
        self.seed             = 42
        self.lambda_coeff     = 0.01
        self.ramp_epochs      = 300

    def set_num_layers(self, n): self.num_layers = n
    def set_num_nodes(self,  n): self.num_nodes  = n
    def set_max_epochs(self, n): self.max_epochs = n
    def set_min_epochs(self, n): self.min_epochs = n


# =============================================================================
# MLP builder
# =============================================================================

def _build_mlp(n_in, n_out, num_layers, num_nodes):
    layers, d = [], n_in
    for _ in range(num_layers):
        layers += [nn.Linear(d, num_nodes), nn.LayerNorm(num_nodes), nn.GELU()]
        d = num_nodes
    layers.append(nn.Linear(d, n_out))
    nn.init.xavier_uniform_(layers[-1].weight, gain=0.05)
    nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


# =============================================================================
# GRANULAR TBNN
# =============================================================================

class GranularTBNN(nn.Module):
    """
    sigma* = a1*I + a2*A + a3*A²
    coeff_mlp: [N, n_inv] → [a1, a2, a3]   [N, 3]
    """
    def __init__(self, n_inv, n_basis, structure):
        super().__init__()
        self.n_basis   = n_basis
        self.coeff_mlp = _build_mlp(n_inv, n_basis, structure.num_layers, structure.num_nodes)

    def forward(self, x, tb):
        """
        x  [N, n_inv]
        tb [N, n_basis, 3, 3]   (I, A, A²)
        Returns: sigma_pred [N,3,3], coeffs [N,n_basis]
        """
        coeffs     = self.coeff_mlp(x)                               # [N, 3]
        sigma_pred = torch.einsum('nb,nbij->nij', coeffs, tb)        # [N, 3, 3]
        return sigma_pred, coeffs


# =============================================================================
# PHYSICS LOSS  (coefficient regularisation only)
# =============================================================================

class GranularPhysicsLoss(nn.Module):
    def __init__(self, struct):
        super().__init__()
        self.lam_coeff = struct.lambda_coeff

    def forward(self, coeffs):
        return {'coeff_loss': self.lam_coeff * (coeffs ** 2).mean()}


# =============================================================================
# TRAINER
# =============================================================================

class GranularTBNNTrainer:

    def __init__(self, model, processor, structure,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model     = model.to(device)
        self.processor = processor
        self.structure = structure
        self.device    = device
        self.phys_loss = GranularPhysicsLoss(structure).to(device)
        self.history   = {k: [] for k in ['total', 'data_s', 'coeff_loss', 'val_total']}

    def _make_loader(self, *arrays, shuffle=True):
        tensors = [torch.from_numpy(np.ascontiguousarray(a)).float() for a in arrays]
        return DataLoader(TensorDataset(*tensors),
                          batch_size=self.structure.batch_size,
                          shuffle=shuffle, num_workers=0, pin_memory=False)

    @staticmethod
    def _huber_scaled(pred, true, var, delta=1.0):
        std = torch.sqrt(var)
        err = (pred - true) / std[None, :]
        return F.huber_loss(err, torch.zeros_like(err), delta=delta, reduction='mean')

    def fit(self, x, tb, s3_true, print_every=1):
        """
        x       [N, 2]     z-scored scalar invariants
        tb      [N, 3, 3, 3]  tensor basis (I, A, A²)
        s3_true [N, 5]     non-dim stress components
        """
        struct  = self.structure
        N       = x.shape[0]
        n_basis = tb.shape[1]

        rng     = np.random.default_rng(struct.seed)
        idx     = rng.permutation(N)
        N_train = int(N * struct.split_fraction)
        tr, va  = idx[:N_train], idx[N_train:]

        tb_flat = tb.reshape(N, n_basis * 9)

        tr_loader = self._make_loader(x[tr], tb_flat[tr], s3_true[tr], shuffle=True)
        va_loader = self._make_loader(x[va], tb_flat[va], s3_true[va], shuffle=False)

        opt      = torch.optim.Adam(self.model.parameters(), lr=struct.learning_rate)
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        s_var = torch.tensor(self.processor._s_var, dtype=torch.float32, device=self.device)

        val_window = []

        best_val = np.inf
        for epoch in range(struct.max_epochs):
            ep_loss   = {k: 0.0 for k in self.history if k != 'val_total'}
            n_batches = 0
            self.model.train()

            for batch in tr_loader:
                xb, tbb, sb = [t.to(self.device) for t in batch]
                tbb = tbb.view(-1, n_basis, 3, 3)

                sigma_pred, coeffs = self.model(xb, tbb)
                sigma_pred_3 = torch.stack(
                [sigma_pred[:,0,0], sigma_pred[:,1,1], sigma_pred[:,0,1] ], dim=1 )

                loss_s = self._huber_scaled(sigma_pred_3, sb, s_var)
                phys   = self.phys_loss(coeffs)
                total  = loss_s + sum(phys.values())

                opt.zero_grad()
                total.backward()

                grad_ok = all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in self.model.parameters() if p.requires_grad)
                if not grad_ok:
                    opt.zero_grad()
                    print(f"  [epoch {epoch+1}] nan grad — batch skipped")
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                ep_loss['total']      += total.item()
                ep_loss['data_s']     += loss_s.item()
                ep_loss['coeff_loss'] += phys['coeff_loss'].item()
                n_batches += 1

            if n_batches == 0:
                print(f"Epoch {epoch+1}: ALL batches had nan — stopping.")
                break

            for k in ep_loss:
                self.history[k].append(ep_loss[k] / n_batches)

            if (epoch + 1) % struct.interval == 0:
                vl = self._eval_loss(va_loader, n_basis, s_var)
                val_window.append(vl)
                self.history['val_total'].append(vl)
                lr_sched.step(vl)
                
                if vl < best_val:
                    best_val = vl

                    torch.save({
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "val_loss": vl,
                        "history": self.history,
                        "x_mean": self.processor._x_mean,
                        "x_std": self.processor._x_std,
                        "s_var": self.processor._s_var,
                        "s_clip_lo": self.processor._s_clip_lo,
                        "s_clip_hi": self.processor._s_clip_hi,
                    }, "best_model.pt")

                    print(f"Saved best model at epoch {epoch+1}")
                
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_loss": vl,
                    "history": self.history,
                    "x_mean": self.processor._x_mean,
                    "x_std": self.processor._x_std,
                    "s_var": self.processor._s_var,
                    "s_clip_lo": self.processor._s_clip_lo,
                    "s_clip_hi": self.processor._s_clip_hi,
                }

                torch.save(checkpoint, "checkpoint_latest.pt")

                if (epoch >= struct.min_epochs
                        and len(val_window) >= struct.average_interval * 2):
                    recent = np.mean(val_window[-struct.average_interval:])
                    older  = np.mean(val_window[-(struct.average_interval*2):-struct.average_interval])
                    if older > 1e-6 and (older - recent) / older < 1e-3:
                        print(f"Early stopping at epoch {epoch+1}.")
                        break

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1:5d} | "
                      f"total={self.history['total'][-1]:.4f} | "
                      f"s={self.history['data_s'][-1]:.4f} | "
                      f"lr={opt.param_groups[0]['lr']:.2e}")

    @torch.no_grad()
    def _eval_loss(self, loader, n_basis, s_var):
        self.model.eval()
        total, n = 0.0, 0
        for xb, tbb, sb in loader:
            xb, tbb, sb = xb.to(self.device), tbb.to(self.device), sb.to(self.device)
            tbb = tbb.view(-1, n_basis, 3, 3)
            sigma_pred, _ = self.model(xb, tbb)
            sigma_pred_3  = torch.stack(
                [sigma_pred[:,0,0], sigma_pred[:,1,1], sigma_pred[:,0,1] ], dim=1 )
            total += self._huber_scaled(sigma_pred_3, sb, s_var).item()
            n += 1
        self.model.train()
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, x, tb):
        self.model.eval()
        n_basis = tb.shape[1]
        xt  = torch.from_numpy(np.ascontiguousarray(x)).float().to(self.device)
        tbt = torch.from_numpy(
            np.ascontiguousarray(tb.reshape(len(x), n_basis * 9))).float().to(self.device)
        tbt = tbt.view(-1, n_basis, 3, 3)
        sigma_pred, coeffs = self.model(xt, tbt)
        s3 = torch.stack(
                [sigma_pred[:,0,0], sigma_pred[:,1,1], sigma_pred[:,0,1] ], dim=1 )
        return s3.cpu().numpy(), sigma_pred.cpu().numpy(), coeffs.cpu().numpy()

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =============================================================================
# MAIN
# =============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    structure = NetworkStructure()
    structure.set_num_layers(3)
    structure.set_num_nodes(64)
    structure.set_max_epochs(3000)
    structure.set_min_epochs(500)

    print("Loading data...")

    sigma_raw = torch.load(
        "/Users/kavyanshrajsingh/Desktop/Data/stress_temporal_cgs/stress_temporal_cg.pt",
        weights_only=False
    ).numpy()

    A_raw = torch.load(
        "/Users/kavyanshrajsingh/Desktop/Data/fabric_temporal_cgs/fabric_temporal_cg.pt",
        weights_only=False
    ).numpy()

    phi_raw = torch.load(
        "/Users/kavyanshrajsingh/Desktop/Data/volume_fraction_cgs/phi_smooth_timeseries.pt",
        weights_only=False
    ).numpy()

    D = torch.load(
        "/Users/kavyanshrajsingh/Desktop/Data/D_tensor.pt",
        weights_only=False
    ).numpy()

    epsilon_raw = D[...,0,0] + D[...,1,1]
    print(f"  Stress : {sigma_raw.shape}")
    print(f"  Fabric Anistropy tensor: {A_raw.shape}")
    print(f"  Strain(epsilon): {epsilon_raw.shape}")
    print(f"  solid vol. frac: {phi_raw.shape}")
    
    # ==========================================================
    # Convert all tensors to a common shape
    # ==========================================================

    # Stress
    sigma_raw = sigma_raw.reshape(1524, 50, 77, 3, 3)

    # Fabric (remove singleton z dimension)
    A_raw = np.squeeze(A_raw, axis=3)          # (1524,50,77,3,3)

    # Volume fraction
    phi_raw = phi_raw[:1524]                   # temporal CG only
    phi_raw = np.squeeze(phi_raw, axis=-1)
    phi_raw = np.squeeze(phi_raw, axis=-1)     # (1524,50,77)

    # epsilon already correct
    # (1524,50,77)

    # ----------------------------------------------------------
    # Flatten
    # ----------------------------------------------------------

    sigma = sigma_raw.reshape(-1,3,3)

    A = A_raw.reshape(-1,3,3)

    phi = phi_raw.reshape(-1)

    epsilon = epsilon_raw.reshape(-1)

    N = len(phi)

    print(f"Total sample points : {N:,}")
   
    print(f"Total sample points: {N:,}")
    print("\n=============================================== ")
    print("\nComputing basis functions...")
    print("\n=============================================== ")
    processor   = GranularDataProcessor()
    x_phys, x   = processor.calc_scalar_basis(A, phi, epsilon, is_train=True)    # [N,4]
    tb           = processor.calc_tensor_basis(A)                   # [N,3,3,3]
    s3_true      = processor.calc_output(sigma, is_train=True)      # [N,5]

    print(f"\n  x       : {x.shape}   (I2 I3 phi epsilon)")
    print(f"  tb      : {tb.shape}   (I, A, A²)")
    print(f"  s3_true : {s3_true.shape}")

    n_inv, n_basis = x.shape[1], tb.shape[1]   # 4, 3
    model = GranularTBNN(n_inv, n_basis, structure)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining...")
    trainer = GranularTBNNTrainer(model, processor, structure)
    trainer.fit(x, tb, s3_true, print_every=1)

    print("\nEvaluating on full dataset...")
    s3_pred, sigma_pred, coeffs = trainer.predict(x, tb)

    for k, (i, j) in enumerate(SYM_IJ):
        r = trainer.rmse(s3_true[:, k], s3_pred[:, k])
        print(f"RMSE sigma*[{i},{j}]: {r:.6f}  (std={np.sqrt(processor._s_var[k]):.6f})")

    print("\nMean basis coefficients:")
    for i, name in enumerate(['a1 (I)', 'a2 (A)', 'a3 (A²)']):
        print(f"  {name}: μ={coeffs[:,i].mean():+.4f}  σ={coeffs[:,i].std():.4f}")

    torch.save({
        'model_state': model.state_dict(),
        'n_inv': n_inv,
        'n_basis': n_basis,
        'x_mean': processor._x_mean,
        'x_std': processor._x_std,
        's_var': processor._s_var,
        's_clip_lo': processor._s_clip_lo,
        's_clip_hi': processor._s_clip_hi,
        'grid_shape': (1524, 50, 77)
    }, "granular_tbnn_A_phi.pt")
    print("\nSaved → granular_tbnn_A_phi.pt")
    
    torch.save(coeffs, "coeffs.pt")
    torch.save(s3_pred, "predictions.pt")

if __name__ == '__main__':
    main()
