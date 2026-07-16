import torch
import numpy as np

# ============================================================
# LOAD
# ============================================================

D = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/D_tensor.pt",
    weights_only=False
).numpy().astype(np.float32)

W = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/W_tensor.pt",
    weights_only=False
).numpy().astype(np.float32)

Ddot = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/Ddot_tensor.pt",
    weights_only=False
).numpy().astype(np.float32)

print("D     :", D.shape)
print("W     :", W.shape)
print("Ddot  :", Ddot.shape)

# ============================================================
# SANITY CHECK
# ============================================================

assert D.shape == W.shape == Ddot.shape

Nt, Nx, Ny, _, _ = D.shape

# ============================================================
# MATRIX PRODUCTS
# ============================================================

print()
print("Computing WD ...")

WD = np.matmul(W, D)

print("Computing DW ...")

DW = np.matmul(D, W)

# ============================================================
# OBJECTIVE DERIVATIVE
# ============================================================

print("Computing Do ...")

Do = Ddot - WD + DW

# ============================================================
# CHECKS
# ============================================================

print()
print("="*60)
print("Do Statistics")
print("="*60)

print("Shape :", Do.shape)

print()

print("Min  :", Do.min())
print("Max  :", Do.max())
print("Mean :", Do.mean())
print("Std  :", Do.std())

print()

print("NaN :", np.isnan(Do).any())
print("Inf :", np.isinf(Do).any())

print()

for i in range(2):
    for j in range(2):

        comp = Do[..., i, j]

        print(f"Do[{i},{j}]")

        print("   min :", comp.min())
        print("   max :", comp.max())
        print("   mean:", comp.mean())
        print("   std :", comp.std())

        print()

# ============================================================
# EXTRA SANITY CHECK
# ============================================================

symmetry_error = np.abs(
    Do[...,0,1] - Do[...,1,0]
)

print("="*60)
print("Symmetry Check")
print("="*60)

print("Maximum |Do12-Do21| =", symmetry_error.max())
print("Mean    |Do12-Do21| =", symmetry_error.mean())

print()

# ============================================================
# SAVE
# ============================================================

torch.save(

    torch.from_numpy(Do),

    "/Users/kavyanshrajsingh/Desktop/Data/Do_tensor.pt"

)

torch.save(

    {
        "WD": torch.from_numpy(WD),
        "DW": torch.from_numpy(DW)
    },

    "/Users/kavyanshrajsingh/Desktop/Data/Do_products.pt"

)

print()

print("Saved Do_tensor.pt")

print("Saved Do_products.pt")
