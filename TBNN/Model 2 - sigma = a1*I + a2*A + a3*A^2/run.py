"""
===============================================================
TBNN4_COEFFICIENT_ANALYSIS.py

Standalone analysis script for TBNN4.

This script

1. Loads learned coefficients.
2. Reconstructs I2, I3, phi and epsilon.
3. Produces publication-quality coefficient plots.
4. Computes power-law slopes.
5. Produces binned averages.

===============================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic

# ===============================================================
# PATHS
# ===============================================================

ROOT = "/Users/kavyanshrajsingh/Desktop/Data"

COEFF_FILE = ROOT + "/TBNN/coeffs.pt"

FABRIC_FILE = ROOT + "/fabric_temporal_cgs/fabric_temporal_cg.pt"

PHI_FILE = ROOT + "/volume_fraction_cgs/phi_smooth_timeseries.pt"

D_FILE = ROOT + "/D_tensor.pt"

SAVE_DIR = ROOT + "/TBNN/TBNN4_Coefficient_Plots"

# ===============================================================
# LOAD DATA
# ===============================================================

print("Loading coefficient tensor...")

coeffs = torch.load(
    COEFF_FILE,
    weights_only=False
)

coeffs = np.asarray(coeffs)

print(coeffs.shape)

print()

print("Loading fabric tensor...")

A = torch.load(
    FABRIC_FILE,
    weights_only=False
).numpy()

print(A.shape)

print()

print("Loading volume fraction...")

phi = torch.load(
    PHI_FILE,
    weights_only=False
).numpy()

print(phi.shape)

print()

print("Loading D tensor...")

D = torch.load(
    D_FILE,
    weights_only=False
).numpy()

print(D.shape)

# ===============================================================
# REFORMAT DATA
# ===============================================================

print()

print("Preparing tensors...")

# Fabric:
# (1524,50,77,1,3,3)
# ->
# (1524,50,77,3,3)

A = np.squeeze(A, axis=3)

# Volume fraction:
# (20001,50,77,1,1)
# ->
# first 1524 frames only

phi = phi[:1524]

phi = np.squeeze(phi, axis=-1)

phi = np.squeeze(phi, axis=-1)

# epsilon exactly as training script

epsilon = D[...,0,0] + D[...,1,1]

# ===============================================================
# FLATTEN EVERYTHING
# ===============================================================

A = A.reshape(-1,3,3)

phi = phi.reshape(-1)

epsilon = epsilon.reshape(-1)

print()

print("Flattened shapes")

print("Fabric :",A.shape)

print("phi :",phi.shape)

print("epsilon :",epsilon.shape)

# ===============================================================
# COMPUTE INVARIANTS
# ===============================================================

print()

print("Computing invariants...")

traceA = np.einsum(
    "nii->n",
    A
)

traceAA = np.einsum(
    "nij,nij->n",
    A,
    A
)

I2 = 0.5 * (
    traceA**2
    -
    traceAA
)

I3 = np.linalg.det(A)

print()

print("I2:",I2.shape)

print("I3:",I3.shape)

# ===============================================================
# COEFFICIENTS
# ===============================================================

a1 = coeffs[:,0]

a2 = coeffs[:,1]

a3 = coeffs[:,2]

print()

print("Coefficient shapes")

print(a1.shape)

print(a2.shape)

print(a3.shape)

# ===============================================================
# RANDOM SUBSAMPLING
# ===============================================================

N = len(a1)

Nplot = 50000

rng = np.random.default_rng(42)

idx = rng.choice(
    N,
    size=Nplot,
    replace=False
)

print()

print("Using",Nplot,"random points for scatter plots.")

# ===============================================================
# PLOTTING FUNCTION
# ===============================================================

def plot_coefficient(
    coeff,
    coeff_name,
    variables,
    variable_names,
    n_bins=60
):
    """
    Creates

    Row 1:
        scatter

    Row 2:
        log-log scatter

    Row 3:
        binned averages

    Returns
    -------
    slopes
    """

    ncols = len(variables)

    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(6*ncols,13),
        dpi=150
    )

    slopes = []

    for j,(x_phys,name) in enumerate(zip(variables,variable_names)):

        ###########################################################
        # Scatter
        ###########################################################

        x = x_phys[idx]

        y = coeff[idx]

        ax = axes[0,j]

        ax.scatter(
            x,
            y,
            s=2,
            alpha=0.25,
            rasterized=True
        )

        ax.set_xlabel(name)

        ax.set_ylabel(coeff_name)

        ax.grid(alpha=0.3)

        ax.set_title(
            f"{coeff_name} vs {name}"
        )

        ###########################################################
        # LOG-LOG
        ###########################################################

        ax = axes[1, j]

        xabs = np.abs(x)
        yabs = np.abs(y)

        mask = (
            np.isfinite(xabs)
            &
            np.isfinite(yabs)
            &
            (xabs > 1e-20)
            &
            (yabs > 1e-20)
        )

        xp = xabs[mask]
        yp = yabs[mask]

        if len(xp) > 50:

            ax.scatter(
                xp,
                yp,
                s=2,
                alpha=0.25,
                rasterized=True
            )

            lx = np.log10(xp)
            ly = np.log10(yp)

            slope, intercept = np.polyfit(
                lx,
                ly,
                1
            )

            xx = np.linspace(
                lx.min(),
                lx.max(),
                200
            )

            ax.plot(
                10**xx,
                10**(slope*xx + intercept),
                "k--",
                linewidth=2,
                label=f"slope={slope:.2f}"
            )

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.legend()

        else:

            slope = np.nan

            ax.text(
                0.5,
                0.5,
                "No positive data\nfor log-log plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12
            )

        slopes.append(slope)

        ax.set_xlabel("|" + name + "|")
        ax.set_ylabel("|" + coeff_name + "|")
        ax.grid(alpha=0.3)

        ###########################################################
        # BINNED MEAN
        ###########################################################

        ax = axes[2,j]

        finite = (
            np.isfinite(x_phys)
            &
            np.isfinite(coeff)
        )

        xall = x_phys[finite]

        yall = coeff[finite]

        means,edges,_ = binned_statistic(
            xall,
            yall,
            statistic="mean",
            bins=n_bins
        )

        centres = 0.5*(
            edges[:-1]
            +
            edges[1:]
        )

        ax.plot(
            centres,
            means,
            linewidth=2
        )

        ax.scatter(
            centres,
            means,
            s=20
        )

        ax.set_xlabel(name)

        ax.set_ylabel(
            "Mean " + coeff_name
        )

        ax.grid(alpha=0.3)

    fig.suptitle(
        coeff_name,
        fontsize=18
    )

    fig.tight_layout(
        rect=[0,0,1,0.97]
    )

    return fig,slopes
    
    # ===============================================================
# CREATE OUTPUT DIRECTORY
# ===============================================================

import os

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# ===============================================================
# VARIABLES TO ANALYSE
# ===============================================================

variables = [
    I2,
    I3,
    phi,
    epsilon
]

variable_names = [
    "I2",
    "I3",
    "phi",
    "epsilon"
]

coefficients = [
    a1,
    a2,
    a3
]

coefficient_names = [
    "a1",
    "a2",
    "a3"
]

# ===============================================================
# GENERATE ALL COEFFICIENT PLOTS
# ===============================================================

all_slopes = {}

for coeff, coeff_name in zip(
    coefficients,
    coefficient_names
):

    print()
    print("="*60)
    print(f"Processing {coeff_name}")
    print("="*60)

    fig, slopes = plot_coefficient(
        coeff,
        coeff_name,
        variables,
        variable_names,
        n_bins=60
    )

    all_slopes[coeff_name] = slopes

    outfile = os.path.join(
        SAVE_DIR,
        f"{coeff_name}_analysis.png"
    )

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved -> {outfile}")

    plt.close(fig)

# ===============================================================
# PRINT POWER LAW SLOPES
# ===============================================================

print()
print("="*70)
print("POWER LAW SLOPES")
print("="*70)

for coeff_name in coefficient_names:

    print()

    print(coeff_name)

    slopes = all_slopes[coeff_name]

    for var_name, slope in zip(
        variable_names,
        slopes
    ):

        print(
            f"{var_name:10s} : {slope:10.5f}"
        )

print()
print("="*70)

# ===============================================================
# COEFFICIENT HISTOGRAMS
# ===============================================================

print()
print("="*70)
print("COEFFICIENT DISTRIBUTIONS")
print("="*70)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18,5),
    dpi=150
)

for ax, coeff, name in zip(
    axes,
    coefficients,
    coefficient_names
):

    ax.hist(
        coeff,
        bins=100,
        density=True,
        alpha=0.75
    )

    ax.axvline(
        coeff.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"mean={coeff.mean():.3e}"
    )

    ax.set_title(name)

    ax.set_xlabel("Coefficient")

    ax.set_ylabel("Probability density")

    ax.grid(alpha=0.3)

    ax.legend()

hist_file = os.path.join(
    SAVE_DIR,
    "coefficient_histograms.png"
)

fig.tight_layout()

fig.savefig(
    hist_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved ->",hist_file)

# ===============================================================
# BOXPLOTS
# ===============================================================

fig = plt.figure(
    figsize=(8,6),
    dpi=150
)

plt.boxplot(
    coefficients,
    tick_labels=coefficient_names,
    showfliers=False
)

plt.ylabel("Coefficient value")

plt.grid(alpha=0.3)

box_file = os.path.join(
    SAVE_DIR,
    "coefficient_boxplots.png"
)

plt.tight_layout()

plt.savefig(
    box_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved ->",box_file)

# ===============================================================
# SUMMARY STATISTICS
# ===============================================================

print()
print("="*70)
print("SUMMARY STATISTICS")
print("="*70)

for coeff,name in zip(
    coefficients,
    coefficient_names
):

    print()

    print(name)

    print("-"*40)

    print(f"Mean      : {coeff.mean(): .6e}")

    print(f"Std       : {coeff.std(): .6e}")

    print(f"Minimum   : {coeff.min(): .6e}")

    print(f"Maximum   : {coeff.max(): .6e}")

    print(f"Median    : {np.median(coeff): .6e}")

    print(f"25 %ile   : {np.percentile(coeff,25): .6e}")

    print(f"75 %ile   : {np.percentile(coeff,75): .6e}")

# ===============================================================
# CORRELATION MATRIX
# ===============================================================

print()
print("="*70)
print("COMPUTING CORRELATION MATRIX")
print("="*70)

corr_data = np.vstack(
    [
        I2,
        I3,
        phi,
        epsilon,
        a1,
        a2,
        a3
    ]
)

corr = np.corrcoef(corr_data)

labels = [
    "I2",
    "I3",
    "phi",
    "eps",
    "a1",
    "a2",
    "a3"
]

fig = plt.figure(
    figsize=(8,7),
    dpi=150
)

plt.imshow(
    corr,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    np.arange(len(labels)),
    labels,
    rotation=45
)

plt.yticks(
    np.arange(len(labels)),
    labels
)

for i in range(len(labels)):
    for j in range(len(labels)):

        plt.text(
            j,
            i,
            f"{corr[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(corr[i,j])>0.5 else "black"
        )

plt.tight_layout()

corr_file = os.path.join(
    SAVE_DIR,
    "correlation_matrix.png"
)

plt.savefig(
    corr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved ->",corr_file)

# ===============================================================
# SAVE SUMMARY TABLE
# ===============================================================

import csv

summary_file = os.path.join(
    SAVE_DIR,
    "coefficient_summary.csv"
)

with open(summary_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Coefficient",
        "Mean",
        "Std",
        "Min",
        "Max",
        "Median",
        "25%",
        "75%"
    ])

    for coeff, name in zip(
        coefficients,
        coefficient_names
    ):

        writer.writerow([
            name,
            coeff.mean(),
            coeff.std(),
            coeff.min(),
            coeff.max(),
            np.median(coeff),
            np.percentile(coeff,25),
            np.percentile(coeff,75)
        ])

print()
print("Saved ->",summary_file)

# ===============================================================
# SAVE POWER LAW SLOPES
# ===============================================================

slope_file = os.path.join(
    SAVE_DIR,
    "power_law_slopes.csv"
)

with open(slope_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Coefficient",
        "Variable",
        "Slope"
    ])

    for coeff_name in coefficient_names:

        slopes = all_slopes[coeff_name]

        for var_name, slope in zip(
            variable_names,
            slopes
        ):

            writer.writerow([
                coeff_name,
                var_name,
                slope
            ])

print("Saved ->",slope_file)

# ===============================================================
# PRINT FINAL REPORT
# ===============================================================

print()
print("="*80)
print("TBNN4 COEFFICIENT ANALYSIS COMPLETE")
print("="*80)

print()

print(f"Total samples          : {N:,}")

print(f"Scatter samples plotted: {Nplot:,}")

print()

print("Input variables")

for name in variable_names:

    print("   •",name)

print()

print("Learned coefficients")

for name in coefficient_names:

    print("   •",name)

print()

print("Generated figures")

print("   ✓ a1_analysis.png")
print("   ✓ a2_analysis.png")
print("   ✓ a3_analysis.png")
print("   ✓ coefficient_histograms.png")
print("   ✓ coefficient_boxplots.png")
print("   ✓ correlation_matrix.png")

print()

print("Generated tables")

print("   ✓ coefficient_summary.csv")
print("   ✓ power_law_slopes.csv")

print()

print("Output directory")

print("   ",SAVE_DIR)

print()

print("="*80)
print("Finished.")
print("="*80)