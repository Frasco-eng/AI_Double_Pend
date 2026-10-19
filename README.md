# Learning Complex Physical Systems: the Double Pendulum

Project for the **AI in Industry** course, University of Bologna.

This repository studies how machine-learning and deep-learning models can learn the dynamics of a non-linear, chaotic physical system — the double pendulum — from simulated data. It compares fully black-box predictors, Neural ODEs, Universal Differential Equations (UDEs), Lagrangian Neural Networks (LNNs) and a probabilistic predictor, with attention to state prediction, energy conservation and identification of physical parameters.

**Presentation:** [presentation/main.pdf](presentation/main.pdf)

---

## Table of Contents

0. [Background](#0-background)
1. [Repository Structure](#1-repository-structure)
2. [Installation](#2-installation)
3. [Pipeline](#3-pipeline)
4. [Branch 1 – Integrators, UDEs and Parameter Identification](#4-branch-1--integrators-udes-and-parameter-identification)
5. [Branch 2 – Black-Box, Lagrangian and Probabilistic Models](#5-branch-2--black-box-lagrangian-and-probabilistic-models)
6. [Main Findings](#6-main-findings)
7. [References](#7-references)
8. [Authors](#8-authors)
9. [License](#9-license)

---

# 0. Background

## Motivation

- Measuring the parameters of a complex physical system can be very costly, because it requires high-end instrumentation.
- In production this is unsustainable: not every product can carry expensive internal sensors.
- A model that can determine the dynamical parameters starting only from the observed state can therefore save a lot of money.
- More generally, real systems are chaotic and highly non-linear, and analytical models fail when system properties change over time (e.g. wear and tear). Data-driven approaches can adapt to that.

The double pendulum is a well-known benchmark with relatively simple equations but chaotic behaviour. Results transfer to multi-articulated systems in **robotics** and **aerospace** (motion control, energy optimisation, short-term prediction).

## Double Pendulum Modeling

The double pendulum consists of two masses $m_1$ and $m_2$ connected by rigid, massless rods of lengths $l_1$ and $l_2$. The first mass $m_1$ is attached to a fixed pivot, while $m_2$ is connected to $m_1$ via a second rod. Gravity acts downward with acceleration $g$. Viscous friction at the joints is modelled by coefficients $b_1, b_2$.

- **Generalized coordinates:**
  - $\theta_1$: angle of the first rod from the vertical
  - $\theta_2$: angle of the second rod relative to the first

- **State vector:** $\mathbf{x} = [q_1, q_2, \dot q_1, \dot q_2] = [\theta_1, \theta_2, \dot\theta_1, \dot\theta_2] \in \mathbb{R}^4$

- **Positions:**
  - $p_{1} = [l_1 \sin(\theta_1),\ l_1 \cos(\theta_1)]$
  - $p_{2} = [l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2),\ l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2)]$

- **Velocities:**
  - $v_{1} = [l_1 \cos(\theta_1)\, \dot{\theta}_1,\ -l_1 \sin(\theta_1)\, \dot{\theta}_1]$
  - $v_{2} = [l_1 \cos(\theta_1)\, \dot{\theta}_1 + l_2 \cos(\theta_1 + \theta_2)(\dot{\theta}_1 + \dot{\theta}_2),\ -l_1 \sin(\theta_1)\, \dot{\theta}_1 - l_2 \sin(\theta_1 + \theta_2)(\dot{\theta}_1 + \dot{\theta}_2)]$

- **Euler–Lagrange equations** with Lagrangian $\mathcal{L} = T - V$ and Rayleigh dissipation function $\mathcal{R}$:

$$\frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot q_i}\right) - \frac{\partial \mathcal{L}}{\partial q_i} + \frac{\partial \mathcal{R}}{\partial \dot q_i} = \tau_i, \qquad i = 1, 2$$

- **Matrix (manipulator) form:**

$$M(\mathbf{q})\,\ddot{\mathbf{q}} + \mathbf{c}(\mathbf{q}, \dot{\mathbf{q}}) + D\,\dot{\mathbf{q}} + G(\mathbf{q}) = \boldsymbol{\tau}$$

where $M$ is the mass matrix, $\mathbf{c}$ the Coriolis/centrifugal term, $D = \mathrm{diag}(b_1, b_2)$ the friction matrix and $G$ the gravity term. The model is implemented in [`models.py`](models.py) (`DoublePendulum`).

## Data Generation

- Integration with Forward Euler or Runge–Kutta 4 ([`simulation.py`](simulation.py)).
- Duration $T = 10\,\text{s}$, step $\Delta t = 0.01\,\text{s}$.
- Initial state $q_0 \sim \mathcal{U}[-\pi, \pi]$, $\dot q_0 \sim \mathcal{U}[-2, 2]$.
- Three datasets:

| Dataset    | Friction            | Notes                                              |
|------------|---------------------|----------------------------------------------------|
| `nofrict`  | $b_1 = b_2 = 0$     | Energy-conserving, chaotic                         |
| `lowfrict` | $b_1 = b_2 = 0.3$   | Slow dissipation                                   |
| `highfrict`| $b_1 = b_2 = 3.0$   | Reaches equilibrium in ~2 s; mostly excluded       |

---

# 1. Repository Structure

```
AI_Double_Pend/
├── models.py              # Analytical DoublePendulum model (M, C, D, G, energies)
├── simulation.py          # Simulator (Euler / RK4 integration, animation)
├── gen.py                 # Dataset generation -> data/raw/
├── data_inspection.py     # Heatmaps, histograms, energy & variance statistics
├── preprocessing.py       # Energy cut (KL-optimised threshold), white noise, angle wrapping
├── splitter.py            # Train / validation / test split by trajectory
├── metrics.py             # Evaluation metrics
├── test_model.py          # Sanity checks on the analytical model
├── considerations.md      # Experiment notes and observations
│
├── data/                  # raw/, processed*/, training/ CSV datasets
├── plots/                 # Data-inspection and run plots
├── figures/               # Architecture diagrams (RNN, UDE)
├── lightning_logs/        # Training logs
│
├── branch1/               # Integrator-based models, UDEs, parameter identification
│   ├── main/              # onestep.py, rnn.py, ude.py (entry points)
│   ├── utils/             # architecture.py, training.py, plots.py
│   ├── scripts/           # .sh / .bat experiment launchers
│   ├── data_enhanced/     # Datasets used by branch 1
│   └── models/            # Saved checkpoints, histories, plots, training recipes
│
├── branch2/               # Black-box FNNs, Neural ODE, LNNs, probabilistic model
│   ├── NNmodels.py        # BaselineFNN, DeepFNN, LagrangianNN, StructuredLNN, ...
│   ├── solver.py          # Euler / RK4 rollout solver
│   ├── train.py           # Main training script
│   ├── train_probabilistic.py
│   ├── test.py, test_longrun.py, test_control.py
│   ├── utils.py, metrics.py, models.py, simulation.py
│   └── runs/              # Saved runs
│
└── presentation/          # LaTeX Beamer slides (main.tex -> main.pdf)
```

---

# 2. Installation

Python 3.11+ is recommended.

```bash
git clone <repo-url>
cd AI_Double_Pend
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn tqdm torch torchdiffeq
```

`pina` (PINA) is only needed by a few experimental modules.

All commands below are meant to be run **from the repository root**, since scripts use paths relative to it.

---

# 3. Pipeline

1. **Generate data** — simulate many trajectories and save them to `data/raw/`:
   ```bash
   python gen.py
   ```
   Physical parameters and friction level are set inside `gen.py`.

2. **Inspect data** — heatmaps, distributions, energy and variance statistics (saved in `plots/`):
   ```bash
   python data_inspection.py
   ```

3. **Preprocess** — optional energy cut (`EC`) of the non-informative tail of dissipative trajectories (threshold chosen by minimising a Kullback–Leibler criterion), white noise (`WN`) and angle wrapping (`RP`). The output folder gets the matching suffix (e.g. `processed_EC_WN`):
   ```bash
   python preprocessing.py
   ```

4. **Split** — train / validation / test split by trajectory id:
   ```bash
   python splitter.py
   ```

5. **Train and evaluate** — see Branch 1 and Branch 2 below.

---

# 4. Branch 1 – Integrators, UDEs and Parameter Identification

Branch 1 embeds a learnable dynamics layer inside a differentiable integrator (`EulerModel`, `RK4Model`) and trains it on multi-step rollouts with a curriculum over the prediction horizon.

Available dynamics layers (`--dynlayer`):

| Option              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `ode`               | Analytical model with learnable physical parameters ($m_i, l_i, b_i$)       |
| `nn`                | Fully black-box neural network (`DynNN`)                                    |
| `ude`               | Universal Differential Equation: physics + neural correction (`DynUDE`)     |
| `informed_ude`      | UDE with more physical structure (`DynUDE_Informed`)                        |
| `more_informed_ude` | UDE that also learns the mass/friction matrices (`UDE_More_Informed`)       |
| `ude_d_constant`    | UDE with constant learnable friction matrix (`UDE_D_constant`)              |

Entry points:

```bash
# Multi-step integrator training (curriculum over horizon)
python -m branch1.main.rnn --dynlayer ude --integrator rk4 --training --verbose \
    --curriculum_steps 4 8 16 32 --epochs_per_stage 20 20 20 20 --lrs 1e-2 1e-3 1e-3 1e-4

# One-step-ahead training
python -m branch1.main.onestep --dynlayer ude_d_constant --training

# Direct prediction of the dynamics matrices entries
python -m branch1.main.ude --training
```

Useful flags: `--train_path/--val_path/--test_path`, `--sin_cos_transform`, `--normalization`, `--fourier_features`, `--sample_weights`, `--activation`, `--batch_size`, `--optimizer_type`, `--scheduler_type`, `--loss_type`, `--patience`, `--wd`. Ready-made launchers are in [`branch1/scripts/`](branch1/scripts).

Each run is saved in `branch1/models/<mode>/<dynlayer>/<version>/` with the checkpoint, loss history, plots and a `training_recipe.txt` with the full configuration.

---

# 5. Branch 2 – Black-Box, Lagrangian and Probabilistic Models

Branch 2 learns the vector field $\dot{\mathbf{x}} = f_\theta(\mathbf{x})$ and integrates it with a solver.

| Model                      | Type        | Description                                                                     |
|----------------------------|-------------|---------------------------------------------------------------------------------|
| `BaselineFNN`, `DeepFNN`   | Black box   | Feed-forward (residual) networks, optional trigonometric input layer            |
| `ClassicODE`               | Solver      | One-step-ahead training with Euler / RK4                                        |
| `NeuralODE`                | Solver      | Trajectory training via `torchdiffeq`                                           |
| `LagrangianNN`             | Structured  | Learns $\mathcal{L}(q,\dot q)$; dynamics from Euler–Lagrange via autograd       |
| `StructuredLNN`            | Grey box    | Learns $M(q)$ (positive definite, Cholesky + softplus), $V(q)$ and friction     |
| `ProbabilisticBaselineFNN` | Probabilistic | Predicts mean and variance; Monte Carlo rollouts for uncertainty              |
| `NormalizationWrapper`     | Utility     | Input/output normalisation around any model                                     |

Configuration is set through the constants at the top of [`branch2/train.py`](branch2/train.py) (`SOLVER_NAME`, `FUNC_NAME`, `ACTIVATION`, `FRICT_TYPE`, `LOSS`, `OPTIMIZER_NAME`, epochs, batch size, learning rate):

```bash
python branch2/train.py                 # deterministic models
python branch2/train_probabilistic.py   # probabilistic model + Monte Carlo
python branch2/test_longrun.py          # long-horizon evaluation of saved runs
python branch2/test_control.py          # comparison against the analytical simulator
```

Runs (checkpoints, configuration JSON, loss history, state and energy evolution plots) are stored in `branch2/runs/`.

---

# 6. Main Findings

- **Activations:** the ReLU family outperforms `tanh`; **SiLU** works best thanks to its smoothness.
- **Black-box FNNs** fit the one-step dynamics well, but energy drifts over long rollouts and they fit even a physically *wrong* model without complaint.
- **Neural ODE** is powerful but slow to train; state loss is good while energy evolution is worse.
- **Lagrangian NN** improves long-term behaviour and energy conservation, but its Hessian is not guaranteed positive definite (unstable), it cannot handle friction and it trains slowly with small batches.
- **Structured (grey-box) LNN** enforces a positive-definite mass matrix, is stable with SiLU, can learn friction with a dedicated module, and fails to converge on physically wrong data — useful for **anomaly detection**. The learned energy is defined only up to a gauge.
- **UDEs (Branch 1)** can identify physical parameters (e.g. $l_2$, $m_2$, $b_1$, $b_2$) from state observations alone.
- **Datasets:** `nofrict` and `lowfrict` are adequately sized; `highfrict` is too small after the energy cut and leads to overfitting.
- **Chaos:** predicting a chaotic system after many steps is intrinsically hard; metrics such as **parameter identification** and **power spectral density** are more meaningful than raw state error.

Future work: a real anomaly detector, a merged identification + energy-conservation pipeline, symplectic integrators (RK4 is not symplectic), and systems such as a charged particle in a magnetic field. More notes in [`considerations.md`](considerations.md).

---

# 7. References

- S. Greydanus, M. Dzamba, J. Yosinski — *Hamiltonian Neural Networks*, 2019. arXiv:1906.01563
- M. Cranmer, S. Greydanus, S. Hoyer, P. Battaglia, D. Spergel, S. Ho — *Lagrangian Neural Networks*, 2020. arXiv:2003.04630
- R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud — *Neural Ordinary Differential Equations*, 2018. arXiv:1806.07366
- M. Lutter, C. Ritter, J. Peters — *Deep Lagrangian Networks*, 2019. arXiv:1907.04490

---

# 8. Authors

- **Francesco Cenni** — [@Frasco-eng](https://github.com/Frasco-eng)
- **Francesco Giordani** — [@KNGLJordan](https://github.com/KNGLJordan)
- **Giulio Mazzaroppi** — [@Giulio-Mazzar](https://github.com/Giulio-Mazzar)
- **Giacomo Pezzolati** — [@giacomo2k2](https://github.com/giacomo2k2)

University of Bologna — AI in Industry

---

# 9. License

© 2025 Francesco Cenni, Francesco Giordani, Giulio Mazzaroppi, Giacomo Pezzolati. **All rights reserved.**

No part of this repository (source code, data, figures or documentation) may be reproduced, distributed, modified or used for commercial purposes without the prior written permission of the authors.
