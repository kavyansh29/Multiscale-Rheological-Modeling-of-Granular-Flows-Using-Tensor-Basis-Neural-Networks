"""
Granular Constitutive TBNN — A and D 
sigma* = a1*I + a2*A + a3*A^2 + a4*D + a5*D^2   (sigma* = sigma * dp/kn)
scalar invariants: I1=tr(A), I2=0.5*(I1²-tr(A²))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import tempfile

def atomic_torch_save(obj, filename):
    directory = os.path.dirname(filename) or "."
    fd, tmpname = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)

    try:
        torch.save(obj, tmpname)
        os.replace(tmpname, filename)
    finally:
        if os.path.exists(tmpname):
            os.remove(tmpname)

SYM_IJ = [(0,0), (1,1), (0,1), (0,2), (1,2)]  # zz from tracelessness


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
    
    def calc_scalar_basis(self, A, D, epsilon, is_train=True):
        """
        Flow state parameters, the scalar invariants which will be fit into the coefficients of the tensor basis equation
        """
        # Fabric tensor invariants (unnormalized — A is dimensionless)
        # I1_A = np.einsum('nii->n', A)
        I2_A = 0.5 * ((np.einsum('nii->n', A))**2 - np.einsum('nij,nij->n', A, A))
        I3_A = np.linalg.det(A)

        # Deformation rate tensor invariants
        I1_D  = np.einsum('nii->n', D)
        I2_D  = 0.5 * (I1_D**2 - np.einsum('nij,nij->n', D, D))

        trDhA = np.einsum('nii->n', np.einsum('nij,njk->nik', D, A))
# =================================================================================================# 
        x = np.stack([I1_D, I2_D, trDhA, epsilon], axis=1) # 4 invariants
# =================================================================================================# 
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if is_train:
            self._x_mean = x.mean(axis=0)
            self._x_std  = x.std(axis=0).clip(1e-8)
        if self._x_mean is None:
            raise RuntimeError("Call with is_train=True first.")
        return x, (x - self._x_mean) / self._x_std               # [N,6], [N,6]

    def calc_tensor_basis(self, A, D, epsilon):
        """
        Returns tb [N, 6, 3, 3]: [I, A, A², D̂, D̂², {AD+DA}/2]
        D normalized per-sample by γ̇ = epsilon.
        """
        N     = A.shape[0]
        I_    = np.eye(3)[None].repeat(N, axis=0)
        A2    = np.einsum('nij,njk->nik', A, A)
        D_hat = D.astype(np.float64)                         
        D2_hat = np.einsum('nij,njk->nik', D_hat, D_hat)         # D̂²
        DA    = np.einsum('nij,njk->nik', D_hat, A)
        AD    = np.einsum('nij,njk->nik', A, D_hat)
        T5    = 0.5 * (DA + AD)

        # Upto all second order effects have been captured.
        return np.stack([I_, A, A2, D_hat, D2_hat, T5], axis=1)      # [N, 6, 3, 3]
    

    def calc_output(self, sigma, is_train=True, clip_percentile=99.5):
        kn = 1e5
        dp = 0.01
        s  = sigma * (dp / kn)
        s5 = np.stack([s[:, i, j] for i, j in SYM_IJ], axis=1)
        s5 = np.nan_to_num(s5, nan=0.0, posinf=0.0, neginf=0.0)
        if is_train:
            self._s_clip_lo = np.percentile(s5, 100 - clip_percentile, axis=0)
            self._s_clip_hi = np.percentile(s5, clip_percentile,       axis=0)
        s5 = np.clip(s5, self._s_clip_lo, self._s_clip_hi)
        if is_train:
            self._s_var = s5.var(axis=0).clip(1e-8)
            for k, (i, j) in enumerate(SYM_IJ):
                print(f"  sigma*[{i},{j}]: mean={s5[:,k].mean():+.5f}  "
                      f"std={np.sqrt(self._s_var[k]):.5f}")
        return s5

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
        self.batch_size       = 2048
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
    sigma* = a1*I + a2*A + a3*A² + a4*D + a5*D² + a6*AD
    coeff_mlp: [N, n_inv] → [a1, a2, a3, a4, a5, a6]   [N, 6]
    """
    def __init__(self, n_inv, n_basis, structure):
        super().__init__()
        self.n_basis   = n_basis
        self.coeff_mlp = _build_mlp(n_inv, n_basis, structure.num_layers, structure.num_nodes)

    def forward(self, x, tb):
        """
        x  [N, n_inv]
        tb [N, n_basis, 3, 3]   (I, A, A^2, D, D^2, AD)
        Returns: sigma_pred [N,3,3], coeffs [N,n_basis]
        """
        coeffs     = self.coeff_mlp(x)                               # [N, 6]
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

    def fit(self, x, tb, s5_true, print_every=100):
        """
        x       [N, 4]     z-scored scalar invariants
        tb      [N, 6, 3, 3]  tensor basis (I, A, A^2, D, D^2, AD)
        s5_true [N, 5]     non-dim stress components
        """
        struct  = self.structure
        N       = x.shape[0]
        n_basis = tb.shape[1]

        rng     = np.random.default_rng(struct.seed)
        idx     = rng.permutation(N)
        N_train = int(N * struct.split_fraction)
        tr, va  = idx[:N_train], idx[N_train:]

        tb_flat = tb.reshape(N, n_basis * 9)

        tr_loader = self._make_loader(x[tr], tb_flat[tr], s5_true[tr], shuffle=True)
        va_loader = self._make_loader(x[va], tb_flat[va], s5_true[va], shuffle=False)

        opt      = torch.optim.Adam(self.model.parameters(), lr=struct.learning_rate)
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        s_var = torch.tensor(self.processor._s_var, dtype=torch.float32, device=self.device)

        val_window = []

        for epoch in range(struct.max_epochs):
            ep_loss   = {k: 0.0 for k in self.history if k != 'val_total'}
            n_batches = 0
            self.model.train()

            for batch in tr_loader:
                xb, tbb, sb = [t.to(self.device) for t in batch]
                tbb = tbb.view(-1, n_basis, 3, 3)

                sigma_pred, coeffs = self.model(xb, tbb)
                sigma_pred_5 = torch.stack(
                    [sigma_pred[:, i, j] for i, j in SYM_IJ], dim=1)  # [B,5]

                loss_s = self._huber_scaled(sigma_pred_5, sb, s_var)
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
                
                if (epoch + 1) % 50 == 0:
                    try:
                        atomic_torch_save(
                            {
                                "epoch": epoch + 1,
                                "model_state": self.model.state_dict(),
                                "optimizer_state": opt.state_dict(),
                            },
                            "checkpoint_latest.pt",
                        )
                        print(f"Checkpoint saved (epoch {epoch+1})")
                    except Exception as e:
                        print("Checkpoint failed:", e)

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
            sigma_pred_5  = torch.stack(
                [sigma_pred[:, i, j] for i, j in SYM_IJ], dim=1)
            total += self._huber_scaled(sigma_pred_5, sb, s_var).item()
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
        s5 = torch.stack([sigma_pred[:, i, j] for i, j in SYM_IJ], dim=1)
        self.model.train()
        return s5.cpu().numpy(), sigma_pred.cpu().numpy(), coeffs.cpu().numpy()

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
    STRESS_FILE  = "/Users/kavyanshrajsingh/Desktop/Data/stress_temporal_cgs/stress_temporal_cg.pt"
    D_FILE       = "/Users/kavyanshrajsingh/Desktop/Data/D_tensor.pt"
    FABRIC_FILE  = "/Users/kavyanshrajsingh/Desktop/Data/fabric_temporal_cgs/fabric_temporal_cg.pt"
    STRAIN_FILE  = "/Users/kavyanshrajsingh/Desktop/Data/strain_temporal_cgs/strain_temporal_cg.pt"

    sigma_raw      = torch.load(STRESS_FILE, weights_only=False).numpy()
    deformation_raw= torch.load(D_FILE, weights_only=False).numpy()
    D33 = np.zeros((*deformation_raw.shape[:3],3,3))

    D33[...,0,0] = deformation_raw[...,0,0]
    D33[...,0,1] = deformation_raw[...,0,1]
    D33[...,1,0] = deformation_raw[...,1,0]
    D33[...,1,1] = deformation_raw[...,1,1]

    deformation_raw = D33
    A_raw          = torch.load(FABRIC_FILE, weights_only=False).numpy()
    A_raw = np.squeeze(A_raw, axis=3)
    epsilon_raw    = torch.load(STRAIN_FILE, weights_only=False).numpy()
    
    # -----------------------------
    # reshape data to common format
    # -----------------------------

    if epsilon_raw.ndim == 4:
        epsilon_raw = np.squeeze(epsilon_raw, axis=-1)

    T, Gx, Gy = epsilon_raw.shape

    sigma_raw = sigma_raw.reshape(T, Gx, Gy, 1, 3, 3)
    deformation_raw = deformation_raw.reshape(T, Gx, Gy, 1, 3, 3)
    A_raw = A_raw.reshape(T, Gx, Gy, 1, 3, 3)
    epsilon_raw = epsilon_raw.reshape(T, Gx, Gy, 1)

    T2, Gx2, Gy2, Gz2 = A_raw.shape[:4]
    N = T2 * Gx2 * Gy2 * Gz2

    sigma = sigma_raw.reshape(N, 3, 3)
    D = deformation_raw.reshape(N, 3, 3)
    A     = A_raw.reshape(N, 3, 3)
    epsilon     = epsilon_raw.reshape(N)
    # phi = phi_raw.reshape(N)
    print(f"Total sample points: {N:,}")
    print("\n=============================================== ")
    print("\nComputing basis functions...")
    print("\n=============================================== ")
    processor   = GranularDataProcessor()
    x_phys, x   = processor.calc_scalar_basis(A, D, epsilon, is_train=True)    # [N,6]
    tb           = processor.calc_tensor_basis(A, D, epsilon)
    s5_true      = processor.calc_output(sigma, is_train=True)      # [N,5]

    print(f"\n  x       : {x.shape}   (I1(D), I2(D), tr(AD), epsilon)")
    print(f"  tb      : {tb.shape}   (I, A, A^2, D, D^2, AD)")
    print(f"  s5_true : {s5_true.shape}")

    n_inv, n_basis = x.shape[1], tb.shape[1]   # 6, 6
    model = GranularTBNN(n_inv, n_basis, structure)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining...")
    trainer = GranularTBNNTrainer(model, processor, structure)
    trainer.fit(x, tb, s5_true, print_every=1)

    print("\nEvaluating on full dataset...")
    s5_pred, sigma_pred, coeffs = trainer.predict(x, tb)
    
    print("\nSaving outputs...")

    try:
        atomic_torch_save(torch.from_numpy(s5_true.copy()), "truth.pt")
        print("  ✓ truth.pt")
    except Exception as e:
        print("  ✗ truth.pt:", e)

    try:
        atomic_torch_save(torch.from_numpy(s5_pred.copy()), "predictions.pt")
        print("  ✓ predictions.pt")
    except Exception as e:
        print("  ✗ predictions.pt:", e)

    try:
        atomic_torch_save(torch.from_numpy(coeffs.copy()), "coeffs.pt")
        print("  ✓ coeffs.pt")
    except Exception as e:
        print("  ✗ coeffs.pt:", e)


    for k, (i, j) in enumerate(SYM_IJ):
        r = trainer.rmse(s5_true[:, k], s5_pred[:, k])
        print(f"RMSE sigma*[{i},{j}]: {r:.6f}  (std={np.sqrt(processor._s_var[k]):.6f})")

    print("\nMean basis coefficients:")
    for i, name in enumerate(['a1 (I)', 'a2 (A)', 'a3 (A²)', 'a4 (D)', 'a5 (D²)', 'a6 (DA)']):
        print(f"  {name}: μ={coeffs[:,i].mean():+.4f}  σ={coeffs[:,i].std():.4f}") 

    

    

    try:
        atomic_torch_save(model.state_dict(), "best_model.pt")
        print("  ✓ best_model.pt")
    except Exception as e:
       print("  ✗ best_model.pt:", e)
    
    
    metadata = {
        "n_inv": n_inv,
        "n_basis": n_basis,
        "x_mean": processor._x_mean,
        "x_std": processor._x_std,
        "s_var": processor._s_var,
        "s_clip_lo": processor._s_clip_lo,
        "s_clip_hi": processor._s_clip_hi,
        "grid_shape": (T2, Gx2, Gy2, Gz2),
    }

    atomic_torch_save(metadata, "metadata.pt")
    print("✓ metadata.pt")

        print("\n✓ Saved granular_tbnn_AD.pt")

    except Exception as e:
        print("\n✗ Failed saving granular_tbnn_AD.pt")
        print(e)
if __name__ == '__main__':
    main()