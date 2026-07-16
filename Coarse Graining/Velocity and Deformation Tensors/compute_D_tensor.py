import torch
import numpy as np

# ============================================================
# LOAD VELOCITY
# ============================================================

velocity = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/velocity_temporal_cgs/velocity_temporal_cg.pt",
    weights_only=False
).numpy()

print("Velocity shape:", velocity.shape)

Nt = velocity.shape[0]

Nx = 50
Ny = 77

dx = 0.5 / Nx
dy = 0.79 / Ny

velocity = velocity.reshape(Nt, Nx, Ny, 3)

u = velocity[...,0]
v = velocity[...,1]

# ============================================================
# VELOCITY GRADIENTS
# ============================================================

du_dx = np.zeros_like(u)
du_dy = np.zeros_like(u)

dv_dx = np.zeros_like(v)
dv_dy = np.zeros_like(v)

# ---------- x direction (periodic) ----------

du_dx[:,1:-1,:] = (u[:,2:,:]-u[:,:-2,:])/(2*dx)
dv_dx[:,1:-1,:] = (v[:,2:,:]-v[:,:-2,:])/(2*dx)

du_dx[:,0,:] = (u[:,1,:]-u[:,-1,:])/(2*dx)
dv_dx[:,0,:] = (v[:,1,:]-v[:,-1,:])/(2*dx)

du_dx[:,-1,:] = (u[:,0,:]-u[:,-2,:])/(2*dx)
dv_dx[:,-1,:] = (v[:,0,:]-v[:,-2,:])/(2*dx)

# ---------- y direction ----------

du_dy[:,:,1:-1] = (u[:,:,2:]-u[:,:,:-2])/(2*dy)
dv_dy[:,:,1:-1] = (v[:,:,2:]-v[:,:,:-2])/(2*dy)

du_dy[:,:,0] = (u[:,:,1]-u[:,:,0])/dy
dv_dy[:,:,0] = (v[:,:,1]-v[:,:,0])/dy

du_dy[:,:,-1] = (u[:,:,-1]-u[:,:,-2])/dy
dv_dy[:,:,-1] = (v[:,:,-1]-v[:,:,-2])/dy

# ============================================================
# BUILD D
# ============================================================

D = np.zeros((Nt,Nx,Ny,2,2),dtype=np.float64)

D[...,0,0] = du_dx

D[...,1,1] = dv_dy

D[...,0,1] = 0.5*(du_dy + dv_dx)

D[...,1,0] = D[...,0,1]

# ============================================================
# DIVERGENCE
# ============================================================

div = du_dx + dv_dy

print()
print("="*60)
print("D Tensor Statistics")
print("="*60)

print("Shape :",D.shape)

print("Min :",D.min())
print("Max :",D.max())
print("Mean:",D.mean())

print()

print("Divergence")

print("Min :",div.min())
print("Max :",div.max())
print("Mean:",div.mean())

print()

print("NaN :",np.isnan(D).any())
print("Inf :",np.isinf(D).any())

# ============================================================
# SAVE
# ============================================================

torch.save(
    torch.from_numpy(D),
    "/Users/kavyanshrajsingh/Desktop/Data/D_tensor.pt"
)

torch.save(
    {
        "du_dx":torch.from_numpy(du_dx),
        "du_dy":torch.from_numpy(du_dy),
        "dv_dx":torch.from_numpy(dv_dx),
        "dv_dy":torch.from_numpy(dv_dy),
    },
    "/Users/kavyanshrajsingh/Desktop/Data/velocity_gradients.pt"
)

print()
print("Saved D_tensor.pt")
print("Saved velocity_gradients.pt")