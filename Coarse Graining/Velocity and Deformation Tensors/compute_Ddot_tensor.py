import torch
import numpy as np

# ============================================================
# LOAD VELOCITY
# ============================================================

velocity = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/velocity_temporal_cgs/velocity_temporal_cg.pt",
    weights_only=False
).numpy()

# ============================================================
# LOAD D
# ============================================================

D = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/D_tensor.pt",
    weights_only=False
).numpy()

print("Velocity shape :", velocity.shape)
print("D shape        :", D.shape)

# ============================================================
# GRID
# ============================================================

Nt = D.shape[0]

Nx = 50
Ny = 77

dx = 0.5/(Nx-1)
dy = 0.79/(Ny-1)

# ------------------------------------------------------------
# Temporal spacing
# ------------------------------------------------------------

strain_rate = 0.065249326754243

strain_half_window = 0.02
overlap_fraction = 0.89

stride_strain = strain_half_window*(1-overlap_fraction)

dt = stride_strain/strain_rate

print()
print("Temporal spacing")
print("dt =",dt,"seconds")

# ============================================================
# RESHAPE VELOCITY
# ============================================================

velocity = velocity.reshape(Nt,Nx,Ny,3)

u = velocity[...,0]
v = velocity[...,1]

# ============================================================
# STORAGE
# ============================================================

dDdt = np.zeros_like(D)

dDdx = np.zeros_like(D)

dDdy = np.zeros_like(D)

# ============================================================
# TIME DERIVATIVE
# ============================================================

print()
print("Computing time derivative...")

# interior

dDdt[1:-1] = (D[2:] - D[:-2])/(2*dt)

# forward

dDdt[0] = (D[1]-D[0])/dt

# backward

dDdt[-1] = (D[-1]-D[-2])/dt

# ============================================================
# X DERIVATIVE (PERIODIC)
# ============================================================

print("Computing x derivatives...")

# interior

dDdx[:,1:-1,:,:,:] = (
    D[:,2:,:,:,:] -
    D[:,:-2,:,:,:]
)/(2*dx)

# first

dDdx[:,0,:,:,:] = (
    D[:,1,:,:,:] -
    D[:,-1,:,:,:]
)/(2*dx)

# last

dDdx[:,-1,:,:,:] = (
    D[:,0,:,:,:] -
    D[:,-2,:,:,:]
)/(2*dx)

# ============================================================
# Y DERIVATIVE
# ============================================================

print("Computing y derivatives...")

# interior

dDdy[:,:,1:-1,:,:] = (
    D[:,:,2:,:,:] -
    D[:,:,:-2,:,:]
)/(2*dy)

# bottom

dDdy[:,:,0,:,:] = (
    D[:,:,1,:,:] -
    D[:,:,0,:,:]
)/dy

# top

dDdy[:,:,-1,:,:] = (
    D[:,:,-1,:,:] -
    D[:,:,-2,:,:]
)/dy

# ============================================================
# MATERIAL DERIVATIVE
# ============================================================

print("Computing material derivative...")

Ddot = np.zeros_like(D)

for i in range(2):
    for j in range(2):

        Ddot[...,i,j] = (

            dDdt[...,i,j]

            +

            u*dDdx[...,i,j]

            +

            v*dDdy[...,i,j]

        )

# ============================================================
# STATISTICS
# ============================================================

print()
print("="*60)
print("Ddot Statistics")
print("="*60)

print("Shape :",Ddot.shape)

print()

print("Min  :",Ddot.min())
print("Max  :",Ddot.max())
print("Mean :",Ddot.mean())
print("Std  :",Ddot.std())

print()

print("NaN :",np.isnan(Ddot).any())
print("Inf :",np.isinf(Ddot).any())

print()

for i in range(2):
    for j in range(2):

        comp = Ddot[...,i,j]

        print(
            f"Ddot[{i},{j}]"
        )

        print(
            "   min :",
            comp.min()
        )

        print(
            "   max :",
            comp.max()
        )

        print(
            "   mean:",
            comp.mean()
        )

        print(
            "   std :",
            comp.std()
        )

        print()

# ============================================================
# SAVE
# ============================================================

torch.save(

    torch.from_numpy(Ddot),

    "/Users/kavyanshrajsingh/Desktop/Data/Ddot_tensor.pt"

)

torch.save(

    {

        "dDdt":torch.from_numpy(dDdt),

        "dDdx":torch.from_numpy(dDdx),

        "dDdy":torch.from_numpy(dDdy)

    },

    "/Users/kavyanshrajsingh/Desktop/Data/Ddot_derivatives.pt"

)

print()
print("Saved Ddot_tensor.pt")
print("Saved Ddot_derivatives.pt")