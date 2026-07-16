import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

# ============================================================
# LOAD PREDICTIONS
# ============================================================

pred = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/TBNN/predictions.pt",
    weights_only=False
)

pred = np.asarray(pred)

print("Prediction shape:", pred.shape)

# ============================================================
# LOAD GROUND TRUTH STRESS
# ============================================================

sigma = torch.load(
    "/Users/kavyanshrajsingh/Desktop/Data/stress_temporal_cgs/stress_temporal_cg.pt",
    weights_only=False
).numpy()

sigma = sigma.reshape(1524, 50, 77, 3, 3)
sigma = sigma.reshape(-1, 3, 3)

kn = 1e5
dp = 0.01

sigma = sigma * (dp / kn)

truth = np.stack(
    [
        sigma[:, 0, 0],
        sigma[:, 1, 1],
        sigma[:, 0, 1]
    ],
    axis=1
)

print("Ground truth shape:", truth.shape)

assert pred.shape == truth.shape

# ============================================================
# METRICS
# ============================================================

names = [
    r"$\sigma_{xx}$",
    r"$\sigma_{yy}$",
    r"$\sigma_{xy}$"
]

print("\n")
print("=" * 60)
print("TBNN4 EVALUATION")
print("=" * 60)

r2_all = []

for i in range(3):

    r2 = r2_score(truth[:, i], pred[:, i])

    rmse = np.sqrt(
        mean_squared_error(
            truth[:, i],
            pred[:, i]
        )
    )

    mae = mean_absolute_error(
        truth[:, i],
        pred[:, i]
    )

    nrmse = rmse / np.std(truth[:, i])

    r2_all.append(r2)

    print(f"\n{names[i]}")
    print(f"R²     = {r2:.6f}")
    print(f"RMSE   = {rmse:.6f}")
    print(f"MAE    = {mae:.6f}")
    print(f"NRMSE  = {nrmse:.6f}")

print("\n" + "=" * 60)
print(f"Mean R² = {np.mean(r2_all):.6f}")
print("=" * 60)

# ============================================================
# PARITY PLOTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i in range(3):

    x = truth[:, i]
    y = pred[:, i]

    idx = np.random.choice(
        len(x),
        size=50000,
        replace=False
    )

    axes[i].scatter(
        x[idx],
        y[idx],
        s=2,
        alpha=0.25,
        rasterized=True
    )

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())

    axes[i].plot(
        [lo, hi],
        [lo, hi],
        "r--",
        linewidth=2
    )

    axes[i].set_title(
        f"{names[i]}\n$R^2$ = {r2_all[i]:.4f}",
        fontsize=13
    )

    axes[i].set_xlabel("True")
    axes[i].set_ylabel("Predicted")
    axes[i].grid(alpha=0.3)

plt.tight_layout()

# ============================================================
# RESIDUAL HISTOGRAMS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i in range(3):

    residual = pred[:, i] - truth[:, i]

    axes[i].hist(
        residual,
        bins=100
    )

    axes[i].set_title(
        f"Residuals {names[i]}"
    )

    axes[i].set_xlabel("Prediction − Truth")
    axes[i].set_ylabel("Count")

plt.tight_layout()

# ============================================================
# RESIDUAL VS TRUE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i in range(3):

    x = truth[:, i]
    residual = pred[:, i] - truth[:, i]

    idx = np.random.choice(
        len(x),
        size=50000,
        replace=False
    )

    axes[i].scatter(
        x[idx],
        residual[idx],
        s=2,
        alpha=0.25,
        rasterized=True
    )

    axes[i].axhline(
        0,
        color="red",
        linestyle="--"
    )

    axes[i].set_title(
        f"Residual vs True\n{names[i]}"
    )

    axes[i].set_xlabel("True")
    axes[i].set_ylabel("Residual")
    axes[i].grid(alpha=0.3)

plt.tight_layout()

plt.show()
