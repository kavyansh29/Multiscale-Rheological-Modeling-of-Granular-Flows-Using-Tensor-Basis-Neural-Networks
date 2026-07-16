import torch
import numpy as np

coeffs = torch.load(
    "coeffs.pt",
    map_location="cpu",
    weights_only=False
)

if isinstance(coeffs, torch.Tensor):
    coeffs = coeffs.numpy()

names = [
    "a1(I)",
    "a2(A)",
    "a3(A²)",
    "a4(D)",
    "a5(D²)",
    "a6(DA)"
]

print("\n==============================")
print("TBNN5 COEFFICIENT SUMMARY")
print("==============================\n")

for i,name in enumerate(names):

    c = coeffs[:,i]

    print(name)
    print(f"Mean   : {np.mean(c): .6f}")
    print(f"Std    : {np.std(c): .6f}")
    print(f"Min    : {np.min(c): .6f}")
    print(f"Max    : {np.max(c): .6f}")
    print(f"Median : {np.median(c): .6f}")
    print(f"5%     : {np.percentile(c,5): .6f}")
    print(f"95%    : {np.percentile(c,95): .6f}")
    print()
