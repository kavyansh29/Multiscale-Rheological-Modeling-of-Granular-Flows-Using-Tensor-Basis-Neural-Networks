import torch
import numpy as np

ck = torch.load(
    "tbnn_representation_theorem.pt",
    map_location="cpu",
    weights_only=False
)

coeffs = ck["coeffs"]

if isinstance(coeffs, torch.Tensor):
    coeffs = coeffs.numpy()

names = [
    "a1(I)",
    "a2(D)",
    "a3(D²)"
]

print("\n==============================")
print("TBNN1 COEFFICIENT SUMMARY")
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
