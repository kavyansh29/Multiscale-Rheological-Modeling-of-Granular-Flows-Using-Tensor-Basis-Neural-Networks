"""
Granular Constitutive TBNN — 2D (xx, yy, xy only)
Model: sigma = -a1*I + a2*D + a3*D² + a4*Do
       a0, a1, a2, a4 = MLP(I2_D, epsilon)
       I2_D = second invariant of D (2D), epsilon = volumetric strain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

SYM_IJ = [(0, 0), (1, 1), (0, 1)]   # xx, yy, xy


# =============================================================================
# SYMLOG
# =============================================================================

def symlog(x, c):
    """Sign-preserving log-linear transform."""
    return np.sign(x) * np.log1p(np.abs(x) / c)


# =============================================================================
# INERTIAL NUMBER MASK
# =============================================================================

def compute_inertial_mask(D_dim, p_hydro, d_particle=0.01, rho=2600.0, I_threshold=1e-5):
    """
    D_dim   : [N, 2, 2]  dimensional strain rate
    p_hydro : [N]        hydrostatic pressure
    Returns bool mask [N], True = active shear (I > threshold)
    """
    trD2     = D_dim[:, 0, 0]**2 + D_dim[:, 1, 1]**2 + 2.0 * D_dim[:, 0, 1]**2
    gd_local = np.sqrt(2.0 * trD2)
    p_safe   = np.clip(p_hydro, 1e-8, None)
    I        = gd_local * d_particle / np.sqrt(p_safe / rho)
    mask     = I > I_threshold
    print(f"  Inertial mask (I>{I_threshold}): {mask.sum():,} / {len(mask):,} "
          f"({100.0 * mask.mean():.1f}% retained)")
    return mask


# =============================================================================
# DATA PROCESSOR
# =============================================================================

class GranularDataProcessor:

    def __init__(self):
        self._c_I2     = None
        self._c_eps    = None
        self._xa_mean  = None
        self._xa_std   = None
        self._sp_clip_lo = None
        self._sp_clip_hi = None
        self._sp_var     = None

    # ── scalar invariants ───────────────────────────────────────────────────

    def calc_scalar_basis(self, D, epsilon=None, is_train=True):
        """
        Computes the three tensor invariants exactly as in the thesis.

        Parameters
        ----------
        D : [N,2,2]
            Non-dimensional deformation-rate tensor

        Returns
        -------
        x_a_raw
            [I1,I2,I3]

        x_a_sc
            Symlog-normalized invariants
        """

        trD = D[:,0,0] + D[:,1,1]

        D2 = np.einsum('nij,njk->nik', D, D)

        trD2 = D2[:,0,0] + D2[:,1,1]

        detD = (
            D[:,0,0]*D[:,1,1]
            -
            D[:,0,1]*D[:,1,0]
        )

        I1 = trD

        I2 = 0.5*(trD**2 - trD2)

        I3 = detD

        x_a_raw = np.stack([I1,I2,I3],axis=1)

        x_a_raw = np.nan_to_num(x_a_raw)

        if is_train:

            def _c(col):

                vals=np.abs(col[col!=0])

                if len(vals)==0:
                    return 1e-4

                return float(np.percentile(vals,10))

            self._c_I1=_c(x_a_raw[:,0])
            self._c_I2=_c(x_a_raw[:,1])
            self._c_I3=_c(x_a_raw[:,2])

            print(
                f"symlog c : "
                f"I1={self._c_I1:.2e} "
                f"I2={self._c_I2:.2e} "
                f"I3={self._c_I3:.2e}"
            )

        x_a_sym=np.stack([

            symlog(x_a_raw[:,0],self._c_I1),

            symlog(x_a_raw[:,1],self._c_I2),

            symlog(x_a_raw[:,2],self._c_I3)

        ],axis=1)

        if is_train:

            self._xa_mean=x_a_sym.mean(0)

            self._xa_std=x_a_sym.std(0).clip(1e-8)

        x_a_sc=(x_a_sym-self._xa_mean)/self._xa_std

        return x_a_raw,x_a_sc
    # ── tensor basis ────────────────────────────────────────────────────────

    def calc_tensor_basis(self, D, Do):

        """
        Returns

        [I,
         D,
         D²,
        Do]
        """

        N = D.shape[0]

        I = np.eye(2)[None].repeat(N, axis=0)

        D2 = np.einsum(
            'nij,njk->nik',
            D,
            D
        )

        return np.stack(

            [

                I,

                D,

                D2,

                Do

            ],

            axis=1

        )
        
    # ── output: non-dimensionalise sigma ───────────────────────────────────

    def calc_output(self, sigma, is_train=True, clip_percentile=99.5):
        """
        sigma : [N, 2, 2]  raw stress (Pa or simulation units)
        Returns sp3 [N, 3] = (sigma_xx, sigma_yy, sigma_xy) * scale
        No traceless decomposition — pure sigma, only non-dimensionalised.
        """
        kn    = 1e5
        dp    = 0.01
        scale = dp / kn

        sp3 = np.stack([sigma[:, i, j] for i, j in SYM_IJ], axis=1) * scale  # [N, 3]
        sp3 = np.nan_to_num(sp3)

        if is_train:
            self._sp_clip_lo = np.percentile(sp3, 100 - clip_percentile, axis=0)
            self._sp_clip_hi = np.percentile(sp3, clip_percentile,       axis=0)

        sp3 = np.clip(sp3, self._sp_clip_lo, self._sp_clip_hi)

        if is_train:
            self._sp_var = sp3.var(0) + 1e-8
            for k, (i, j) in enumerate(SYM_IJ):
                print(f"  sigma*[{i},{j}]: mean={sp3[:, k].mean():+.5f}  "
                      f"std={np.sqrt(self._sp_var[k]):.5f}")

        return sp3


# =============================================================================
# NETWORK
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


class NetworkStructure:
    def __init__(self):
        self.num_layers       = 3
        self.num_nodes        = 64
        self.max_epochs       = 3000
        self.min_epochs       = 500
        self.interval         = 10
        self.average_interval = 4
        self.learning_rate    = 1e-4
        self.batch_size       = 2048
        self.split_fraction   = 0.8
        self.seed             = 42
        self.lambda_coeff     = 1e-5
        self.w_sp             = 1.0

    def set_num_layers(self, n): self.num_layers = n
    def set_num_nodes(self,  n): self.num_nodes  = n
    def set_max_epochs(self, n): self.max_epochs = n
    def set_min_epochs(self, n): self.min_epochs = n


class GranularTBNN(nn.Module):
    """
    sigma_pred = a0*I + a1*D + a2*D^2
    a0, a1, a2 = MLP(I1, I2, I3)
    """

    def __init__(self, structure):
        super().__init__()
        self.coeff_mlp = _build_mlp(3, 4, structure.num_layers, structure.num_nodes)

    def forward(self, x_a, tb):
        """
        x_a : [N, 3]       scaled scalar inputs (I1, I2, I3)
        tb  : [N, 3, 2, 2] tensor basis [I, D, D^2]
        Returns sigma_pred [N,2,2], coeffs [N,3]
        """
        coeffs = self.coeff_mlp(x_a)                        # [N, 3]

        a0 = coeffs[:,0].view(-1,1,1)

        a1 = coeffs[:,1].view(-1,1,1)

        a2 = coeffs[:,2].view(-1,1,1)

        a3 = coeffs[:,3].view(-1,1,1)

        I_mat = tb[:,0]

        D_mat = tb[:,1]

        D2_mat = tb[:,2]

        Do_mat = tb[:,3]
        

        sigma_pred = (

            -a0*I_mat

            +

            a1*D_mat

            +

            a2*D2_mat

            +

            a3*Do_mat

        )

        return sigma_pred, coeffs


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
        self.history   = {k: [] for k in
                          ['total', 'loss_sp', 'loss_reg', 'val_total']}

    def _make_loader(self, *arrays, shuffle=True):
        tensors = [torch.from_numpy(np.ascontiguousarray(a)).float() for a in arrays]
        return DataLoader(TensorDataset(*tensors),
                          batch_size=self.structure.batch_size,
                          shuffle=shuffle, num_workers=0, pin_memory=False)

    @staticmethod
    def _relative_huber(pred, true, var):
        return F.mse_loss(pred, true)

    def fit(self, x_a, tb, sp3_true, print_every=100):
        struct  = self.structure
        N       = x_a.shape[0]
        tb_flat = tb.reshape(N, -1)                          # [N, 12]

        rng     = np.random.default_rng(struct.seed)
        idx     = rng.permutation(N)
        N_train = int(N * struct.split_fraction)
        tr, va  = idx[:N_train], idx[N_train:]

        # loader has 3 arrays: x_a, tb_flat, sp3_true
        tr_loader = self._make_loader(x_a[tr],  tb_flat[tr],  sp3_true[tr])
        va_loader = self._make_loader(x_a[va],  tb_flat[va],  sp3_true[va], shuffle=False)

        opt      = torch.optim.Adam(self.model.parameters(), lr=struct.learning_rate)
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                       opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        sp_var = torch.tensor(self.processor._sp_var, dtype=torch.float32,
                              device=self.device)  # [3]

        val_window = []

        for epoch in range(struct.max_epochs):
            ep = {k: 0.0 for k in ['total', 'loss_sp', 'loss_reg']}
            nb = 0
            self.model.train()

            for xab, tbb, spb in tr_loader:
                xab = xab.to(self.device)
                tbb = tbb.to(self.device).view(-1, 4, 2, 2)  # [B, 3, 2, 2]
                spb = spb.to(self.device)

                sigma_pred, coeffs = self.model(xab, tbb)

                # extract (xx, yy, xy) from predicted [B,2,2]
                sp3_pred = torch.stack([sigma_pred[:, i, j] for i, j in SYM_IJ], dim=1)

                loss_sp  = self._relative_huber(sp3_pred, spb, sp_var)
                loss_reg = struct.lambda_coeff * (coeffs**2).mean()
                total    = struct.w_sp * loss_sp + loss_reg

                opt.zero_grad()
                total.backward()

                grad_ok = all(
                    p_.grad is not None and torch.isfinite(p_.grad).all()
                    for p_ in self.model.parameters() if p_.requires_grad)
                if not grad_ok:
                    opt.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                ep['total']    += total.item()
                ep['loss_sp']  += loss_sp.item()
                ep['loss_reg'] += loss_reg.item()
                nb += 1

            if nb == 0:
                print(f"Epoch {epoch + 1}: all batches NaN — stopping.")
                break

            for k in ep:
                self.history[k].append(ep[k] / nb)

            if (epoch + 1) % struct.interval == 0:
                vl = self._eval_loss(va_loader, sp_var)
                val_window.append(vl)
                self.history['val_total'].append(vl)
                lr_sched.step(vl)

                if (epoch >= struct.min_epochs
                        and len(val_window) >= struct.average_interval * 2):
                    recent = np.mean(val_window[-struct.average_interval:])
                    older  = np.mean(val_window[-(struct.average_interval * 2)
                                                : -struct.average_interval])
                    if older > 1e-6 and (older - recent) / older < 1e-3:
                        print(f"Early stopping at epoch {epoch + 1}.")
                        break

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1:5d} | total={self.history['total'][-1]:.4f} | "
                      f"sp={self.history['loss_sp'][-1]:.4f} | "
                      f"reg={self.history['loss_reg'][-1]:.4f} | "
                      f"lr={opt.param_groups[0]['lr']:.2e}")

    @torch.no_grad()
    def _eval_loss(self, loader, sp_var):
        self.model.eval()
        total, n = 0.0, 0
        for xab, tbb, spb in loader:                         # ← 3 tensors only
            xab = xab.to(self.device)
            tbb = tbb.to(self.device).view(-1, 4, 2, 2)     # ← [B, 3, 2, 2]
            spb = spb.to(self.device)

            sigma_pred, coeffs = self.model(xab, tbb)
            sp3_pred = torch.stack([sigma_pred[:, i, j] for i, j in SYM_IJ], dim=1)

            loss_sp = self._relative_huber(sp3_pred, spb, sp_var)
            loss_r  = self.structure.lambda_coeff * (coeffs**2).mean()
            total  += (self.structure.w_sp * loss_sp + loss_r).item()
            n      += 1

        self.model.train()
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, x_a, tb):
        self.model.eval()
        N   = x_a.shape[0]
        xat = torch.from_numpy(np.ascontiguousarray(x_a)).float().to(self.device)
        tbt = torch.from_numpy(np.ascontiguousarray(tb.reshape(N, -1))).float().to(self.device)
        tbt = tbt.view(N, 4, 2, 2)                          # ← [N, 3, 2, 2]

        sigma_pred, coeffs = self.model(xat, tbt)
        sp3 = torch.stack([sigma_pred[:, i, j] for i, j in SYM_IJ], dim=1)

        self.model.train()
        return sp3.cpu().numpy(), sigma_pred.cpu().numpy(), coeffs.cpu().numpy()

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred)**2)))


# =============================================================================
# DATA LOADING
# =============================================================================

###############################################################################
# YOUR DATA
###############################################################################

STRESS_FILE = "/Users/kavyanshrajsingh/Desktop/Data/stress_temporal_cgs/stress_temporal_cg.pt"

D_FILE = "/Users/kavyanshrajsingh/Desktop/Data/D_tensor.pt"

DO_FILE = "/Users/kavyanshrajsingh/Desktop/Data/Do_tensor.pt"

GAMMA_DOT = 0.065249326754243

def load_and_nondim():

    print("Loading stress...")

    sigma = torch.load(
        STRESS_FILE,
        weights_only=False
    ).numpy()

    print("Loading D tensor...")

    D = torch.load(
        D_FILE,
        weights_only=False
    ).numpy()
    
    print("Loading Do tensor...")

    Do = torch.load(
        DO_FILE,
        weights_only=False
    ).numpy()

    print("Loaded.")

    print("Stress shape :", sigma.shape)
    print("D shape      :", D.shape)
    print("Do shape     :", Do.shape)

    return sigma, D, Do
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

    print("Loading coarse-grained fields...")

    sigma, D, Do = load_and_nondim()
    
    ###############################################################################
    # RESHAPE
    ###############################################################################

    Nt = sigma.shape[0]

    Nx = 50

    Ny = 77

    Ncells = Nx * Ny

    sigma = sigma.reshape(Nt, Ncells, 3, 3)

    D = D.reshape(Nt, Ncells, 2, 2)
    Do = Do.reshape(Nt, Ncells, 2, 2)

    N = Nt * Ncells

    sigma = sigma[:, :, :2, :2]

    sigma = sigma.reshape(N, 2, 2)

    D = D.reshape(N, 2, 2)
    Do = Do.reshape(N, 2, 2)
    # ---------------------------------------------------------
    # RMS normalization of Do
    # ---------------------------------------------------------

    Do_rms = np.sqrt(np.mean(Do**2))

    print(f"\nRMS(Do) = {Do_rms:.6e}")

    Do = Do / Do_rms

    print()

    print("Training samples =", N)
    
    # ============================================================
    # DEBUG TRAINING
    # ============================================================

    DEBUG = False
    
    print()

    print("Stress tensor shape:", sigma.shape)
    print("D tensor shape     :", D.shape)
    print("Do tensor shape    :", Do.shape)   
    

    print()

    print("No inertial masking applied.")

    mask = np.ones(D.shape[0], dtype=bool)

    print("\nComputing basis functions...")
    processor = GranularDataProcessor()
    x_a_raw, x_a = processor.calc_scalar_basis(
        D,
        is_train=True
    )
    

    print("="*60)

    sp3_true  = processor.calc_output(sigma, is_train=True)  # [N, 3]
    tb = processor.calc_tensor_basis(
        D,
        Do
    )
    

    model = GranularTBNN(structure)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining...")
    trainer = GranularTBNNTrainer(model, processor, structure)
    trainer.fit(
        x_a,
        tb,
        sp3_true,
        print_every=1
    )

    print("\nEvaluating on full training set...")
    sp3_pred, sigma_pred_mat, coeffs = trainer.predict(x_a, tb)

    print("\n── Stress RMSE ──")
    for k, (i, j) in enumerate(SYM_IJ):
        r = trainer.rmse(sp3_true[:, k], sp3_pred[:, k])
        print(f"  RMSE sigma[{i},{j}]: {r:.6f}  (std={np.sqrt(processor._sp_var[k]):.6f})")

    print(f"\n── Coefficient stats ──")
    for k,name in enumerate(['a1 (I)','a2 (D)','a3 (D²)','a4 (Do)']):
        print(f"  {name}: μ={coeffs[:, k].mean():+.4f}  σ={coeffs[:, k].std():.4f}")
        
    torch.save({
        'model_state':   model.state_dict(),
        'xa_mean':       processor._xa_mean,
        'xa_std':        processor._xa_std,
        'c_I1': processor._c_I1,
        'c_I2': processor._c_I2,
        'c_I3': processor._c_I3,
        'sp_var':        processor._sp_var,
        'sp_clip_lo':    processor._sp_clip_lo,
        'sp_clip_hi':    processor._sp_clip_hi,
        'sp3_true':      sp3_true,
        'sp3_pred':      sp3_pred,
        'x_a_raw':       x_a_raw,
        'coeffs':        coeffs,
        'mask':          mask,
    }, '/Users/kavyanshrajsingh/Desktop/Data/granular_tbnn_nott.pt')
    print("\nSaved → granular_tbnn_nott.pt")

if __name__ == '__main__':
    main()
