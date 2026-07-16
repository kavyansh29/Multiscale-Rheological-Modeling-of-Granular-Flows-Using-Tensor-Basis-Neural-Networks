# Multiscale Rheological Modeling of Granular Flows Using Tensor Basis Neural Networks

> A research repository for coarse-grained continuum modeling of dense granular flows using Tensor Basis Neural Networks (TBNNs), combining LAMMPS simulations, spatial-temporal coarse graining, tensorial feature extraction, and invariant-based machine learning.

---

# Abstract

Understanding the constitutive behavior of dense granular materials remains one of the central challenges in continuum mechanics. Classical constitutive laws often fail to capture the complex anisotropic and history-dependent behavior emerging from particle-scale interactions.

This repository presents a complete computational pipeline for learning tensorial constitutive relations directly from Discrete Element Method (DEM) simulations using **Tensor Basis Neural Networks (TBNNs)**.

The workflow consists of:

- Large-scale LAMMPS simulations of dense granular shear flow.
- Spatial coarse graining of particle quantities onto an Eulerian grid.
- Temporal coarse graining using Gaussian strain-window averaging.
- Construction of physically meaningful tensor bases.
- Learning coefficient functions using invariant-based neural networks.
- Evaluation of multiple constitutive formulations.

The repository is intended for researchers working in

- Granular Mechanics
- Continuum Mechanics
- Machine Learning for Physics
- Constitutive Modeling
- Computational Rheology
- Tensor Representation Learning

---

# Repository Structure

```
.
├── LAMMPS Simulation
|
├── Coarse Graining
│
│   ├── Stress
│   ├── Strain
│   ├── Velocity and Deformation Tensors
│   └── Volume Fraction
│
└── TBNN
    ├── Model 1
    ├── Model 2
    ├── Model 3
    └── Model 4
```

Each directory corresponds to one stage of the computational workflow.

---

# Overall Workflow

```
LAMMPS DEM Simulation
          │
          ▼
Particle Dumps
          │
          ▼
Spatial Coarse Graining
          │
          ▼
Eulerian Tensor Fields
          │
          ▼
Temporal Coarse Graining
          │
          ▼
Stress
Velocity
Strain
Volume Fraction
          │
          ▼
Tensor Construction
(D, W, A, Ḋ, Dᵒ)
          │
          ▼
Tensor Basis Generation
          │
          ▼
Tensor Basis Neural Network
          │
          ▼
Constitutive Model
```

---

# Scientific Motivation

Traditional constitutive equations are generally derived from phenomenological assumptions.

Tensor Basis Neural Networks instead assume that the stress tensor can be expressed as

$$\boldsymbol{\sigma}=\sum_{i=1}^{N}a_i(\mathcal{I})\,\mathbf{T}_i$$

where

- $$(T_i)$$ are physically meaningful tensor basis functions

and

- $$(a_i(\mathcal I)\)$$ are scalar coefficient functions learned from simulation data.

This architecture preserves objectivity and rotational invariance while allowing nonlinear constitutive behavior.

---

# Repository Contents

---

## 1. LAMMPS Simulation

This directory contains the DEM simulation used for generating the particle dataset.

Simulation outputs include

- particle positions
- particle velocities
- contact information
- restart files
- simulation logs

The simulation provides the raw microscopic data used throughout the remainder of the workflow.

---

## 2. Coarse Graining

Particle quantities are converted into continuum fields.

The implementation follows Gaussian coarse graining over a structured Eulerian grid.

Generated continuum quantities include

- Stress tensor
- Velocity field
- Volume fraction
- Strain field

Each quantity is spatially coarse grained before temporal averaging.

---

### Stress

Computes

$$
\boldsymbol{\sigma}(\mathbf{x},t)
$$

from DEM particle data.

Outputs

```
stress_tensor.pt
```

---

### Velocity

Computes

$$
\mathbf{u}(\mathbf{x},t)
$$

Outputs

```
velocity_tensor_cg_*.pt
```

These velocity tensors are later temporally averaged.

---

### Strain

Computes cumulative strain fields and temporally coarse-grained strain.

Outputs

```
strain_temporal_cg.pt
```

---

### Volume Fraction

Computes

$$
\phi(\mathbf{x},t)
$$
used by later constitutive models.

---

# Temporal Coarse Graining

Rather than averaging over fixed time intervals, temporal averaging is performed using Gaussian windows defined in accumulated strain.

Advantages include

- frame-rate independence
- smoother constitutive signals
- improved machine learning targets

---

# Velocity Gradient and Deformation Tensors

The velocity field is differentiated to compute

$$
\nabla\mathbf{u}
$$

which is decomposed into

Symmetric part

$$
\mathbf{D}
=
\frac{1}{2}
\left(
\nabla\mathbf{u}
+
(\nabla\mathbf{u})^{T}
\right)
$$

and antisymmetric part

$$
\mathbf{W}
=
\frac{1}{2}
\left(
\nabla\mathbf{u}
-
(\nabla\mathbf{u})^{T}
\right)
$$

Additional tensors are computed as

Fabric Anisotropy Tensor

$$
A_{ij}
=
\frac{1}{N_c}
\sum_{c=1}^{N_c}
n_i^{(c)}
n_j^{(c)}
$$

Material derivative

$$
\dot{\mathbf{D}}
=
\frac{\partial\mathbf{D}}{\partial t}
+
\mathbf{u}\cdot\nabla\mathbf{D}
$$

and the Jaumann objective derivative

$$
\mathbf{D}^{\circ}
=
\dot{\mathbf{D}}
-
\mathbf{W}\mathbf{D}
+
\mathbf{D}\mathbf{W}
$$

These tensors form the basis for Model 3.

---

# Tensor Basis Neural Networks

Four different constitutive formulations have been investigated.

---

## Model 1

Constitutive equation

$$
\boldsymbol{\sigma}
=
a_0\mathbf{I}
+
a_1\mathbf{D}
+
a_2\mathbf{D}^{2}
$$

Purpose

Baseline deformation-rate model.

Features

- Invariant-based coefficients
- Tensor representation theorem
- Spatially varying coefficients

---

## Model 2

Constitutive equation

$$
\boldsymbol{\sigma}
=
a_1\mathbf{I}
+
a_2\mathbf{A}
+
a_3\mathbf{A}^{2}
$$


where

$$
\mathbf{A}
$$

is the coarse-grained fabric anisotropy tensor.


Purpose

Investigates the predictive capability of microstructural anisotropy.

---

## Model 3

Constitutive equation

$$
\boldsymbol{\sigma}
=
-a_1\mathbf{I}
+
a_2\mathbf{D}
+
a_3\mathbf{D}^{2}
+
a_4\mathbf{D}^{\circ}
$$

This formulation follows the constitutive proposal of Nott.

Additional tensors

- D
- W
- Material derivative
- Jaumann derivative

are computed prior to training.

---

## Model 4

Constitutive equation

$$
\boldsymbol{\sigma}
=
a_1\mathbf{I}
+
a_2\mathbf{A}
+
a_3\mathbf{A}^{2}
+
a_4\mathbf{D}
+
a_5\mathbf{D}^{2}
$$

This combines kinematic information with microstructural anisotropy.

Among the investigated models, this architecture provided the strongest predictive performance.

---

# Machine Learning Workflow

Each model follows the same pipeline.

```
Tensor Basis
        │
        ▼
Scalar Invariants
        │
        ▼
Symlog Normalization
        │
        ▼
MLP
        │
        ▼
Coefficient Functions
        │
        ▼
Tensor Reconstruction
        │
        ▼
Stress Prediction
```

The neural network never predicts stress directly.

Instead, it predicts only the scalar coefficients multiplying each tensor basis.

This guarantees tensorial consistency and preserves the representation theorem.

---

# Evaluation

Models are evaluated using

- RMSE
- MAE
- Coefficient statistics
- Residual distributions
- Prediction vs Ground Truth parity plots

The repository contains scripts for extracting

- learned coefficients
- parity plots
- coefficient histograms
- residual distributions

---

# Generated Figures

The repository includes

- coefficient distributions
- parity plots
- residual plots
- coefficient correlations
- tensor analysis figures

These figures are generated directly from the trained models and can be reproduced using the accompanying scripts.

---

# Software Requirements

Python 3.11+

Recommended packages

```
numpy
scipy
torch
matplotlib
pandas
scikit-learn
```

---

# Execution Pipeline

The recommended execution order is

```
LAMMPS Simulation

↓

Stress Coarse Graining

↓

Velocity Coarse Graining

↓

Temporal Velocity Averaging

↓

compute_D_tensor.py

↓

compute_W_tensor.py

↓

compute_Ddot_tensor.py

↓

compute_Do_tensor.py

↓

Strain Coarse Graining

↓

Volume Fraction

↓

Train TBNN Models
```

---

# Repository Philosophy

The repository intentionally separates

- simulation
- coarse graining
- tensor construction
- machine learning

into modular stages.

This allows future constitutive models to reuse the generated continuum tensor fields without rerunning the DEM simulation.

---

# Reproducibility

The repository includes pretrained model checkpoints and intermediate `.pt` files required to reproduce the reported results without repeating computationally intensive preprocessing.

Researchers interested in extending the work may directly use the stored continuum tensors or retrain the models using the provided scripts.

---

# Future Work

Potential extensions include

- Additional objective derivatives
- Three-dimensional granular flows
- Graph Neural Networks
- Physics-informed TBNNs
- Alternative tensor bases
- Generalization across loading paths
- Experimental validation
- Rate-independent constitutive formulations

---

# Citation

If you use this repository in academic work, please cite the associated publication (to be added upon publication).

---

# License

The code is intended for academic research and educational use.

A permissive open-source license such as the MIT License is recommended unless institutional or publication policies require otherwise.

---

# Contact

For questions regarding the implementation, methodology, or research, please open a GitHub Issue.

---

## Acknowledgements

This work was developed as part of research on constitutive modeling of dense granular materials using Tensor Basis Neural Networks and coarse-grained Discrete Element Method simulations.

The repository integrates continuum mechanics, granular physics, and machine learning into a unified computational framework intended to facilitate reproducible research and future methodological extensions.
