
# Multi-Agent Reinforcement Learning with Weak Ties

[![Python 3.10+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Course Project**: Algorithm Design and Analysis  
> **Authors**: Tian Qiu, Jiangyue Chen, Yunchen Xu

This repository contains a reproduction implementation of the paper:

**"Multi-agent reinforcement learning with weak ties"**  
*Huan Wang, Xiaotian Hao, Yi Wu, et al.*  
Information Fusion, Volume 118, 2025  
DOI: [10.1016/j.inffus.2025.102942](https://doi.org/10.1016/j.inffus.2025.102942)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Algorithm Details](#algorithm-details)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project implements the **Weak Ties** framework for multi-agent reinforcement learning (MARL), which addresses information redundancy in cooperative tasks by:

1. **Dynamic Graph Construction**: Building agent relationship graphs based on spatial proximity
2. **Tie Strength Classification**: Distinguishing between weak ties (diverse information) and strong ties (redundant information)
3. **Selective Information Fusion**: Prioritizing weak-tie observations to reduce redundancy and improve learning efficiency

We validate our implementation on the **StarCraft Multi-Agent Challenge (SMAC)** benchmark.

---

## Key Features

- **WeakTieGraph Module**: Dynamic computation of tie strengths using shortest path length (Eq. 8 in paper)
- **WeakTieFusionLayer**: Selective observation fusion with strong-tie filtering (Algorithm 1)
- **PPO-based Training**: Proximal Policy Optimization with counterfactual baseline (Eq. 14)
- **SMAC Integration**: Custom environment wrapper for position extraction and graph updates
- **Checkpoint Management**: Auto-save/resume with best model tracking
- **Multi-map Support**: Pre-configured for 1c3s5z, 50m, 10m_vs_11m, 8m, MMM, 2s3z

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- StarCraft II (version 4.10)

### Step 1: Clone Repository

```bash
git clone https://github.com/ChristianOrsted/Algorithm_Project_for_Win.git
cd Algorithm_Project_for_Win
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv pysc2_env
pysc2_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Step 4: Install StarCraft II

**Windows**:  
Download [StarCraft II](https://github.com/Blizzard/s2client-proto#downloads) and extract to `D:\Program Files (x86)\StarCraft II`

### Step 5: Install SMAC Maps

```bash
python install_map.py
```

---

## Quick Start

### Train on 1c3s5z Map

```bash
python train_smac.py --map 1c3s5z --seed 42
```

### Resume from Checkpoint

```bash
python train_smac.py --map 1c3s5z --resume
```

### Evaluate Trained Model

```bash
python replay_model.py --map 1c3s5z --checkpoint checkpoints/1c3s5z/best_model.pth
```

---

## Project Structure

```
Algorithm_Project_for_Win/
├── train_smac.py           # Main training script
├── weak_tie_agent.py       # PPO agent with counterfactual baseline
├── weak_tie_module.py      # WeakTieGraph & WeakTieFusionLayer
├── weak_tie_env.py         # SMAC environment wrapper
├── replay_model.py         # Model evaluation script
├── install_map.py          # SMAC map installer
├── replay_model.py         # Replay game with best model
├── test_1000.py            # Evaluate model quality with 1000 Episodes
├── requirements.txt        # Python dependencies
├── eval_modual.py          # Model Evaluation
├── checkpoints/            # Saved models
├── logs/                   # Training logs
├── md/
├── README.md              # Documentation
└── algorithm.pdf          # Original paper

```

---

## Algorithm Details

### 1. Weak Tie Graph Construction

**Definition** (Eq. 8):  
Tie strength between agents $i$ and $j$:

$$
\text{TieStrength}(i, j) = \frac{1}{\text{ShortestPath}(i, j)}
$$

- **Strong ties**: Nearby agents (high redundancy)
- **Weak ties**: Distant agents (diverse information)
- **Dominant agent**: Highest total tie strength

### 2. Information Fusion

**WeakTieFusionLayer** (Eq. 11-13):

$$
\tilde{o}_i^t = \text{Concat}\left(o_i^t, \sum_{j \in \mathcal{W}_i} \alpha_{ij} \cdot o_j^t, o_{\text{dom}}^t\right)
$$

Where:
- $\mathcal{W}_i$: Weak-tie neighbors of agent $i$
- $\alpha_{ij}$: Attention weights
- $o_{\text{dom}}$: Dominant agent's observation

### 3. PPO Training

**Advantage Function** (Eq. 14):

$$
A_i^t = \sum_{t'=t}^{T} \gamma^{t'-t} r^{t'} + \gamma^{T-t+1} V(s^T) - V(s^t) - b_i^t
$$

Where $b_i^t$ is the counterfactual baseline.

---

## Training

### Basic Training

```bash
python train_smac.py \
  --map 1c3s5z \
  --seed 42 \
  --max-timesteps 2000000 \
  --eval-interval 10000
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--lam` | 0.95 | GAE lambda |
| `--clip-ratio` | 0.2 | PPO clip ratio |
| `--alpha` | 0.3 | Weak tie threshold (α-quantile) |
| `--ent-coef` | 0.01 | Entropy coefficient |
| `--batch-size` | 32 | Mini-batch size |

### Multi-map Training

```bash
# Train on all maps
for map in 1c3s5z 50m 10m_vs_11m 8m MMM 2s3z; do
  python train_smac.py --map $map --seed 42
done
```

---

## Evaluation

### Evaluate Single Model

```bash
python replay_model.py \
  --map 1c3s5z \
  --checkpoint checkpoints/1c3s5z/best_model.pth \
  --episodes 50
```

### Batch Evaluation

```bash
# Evaluate all checkpoints
python eval_module.py --map 1c3s5z --checkpoints-dir checkpoints/1c3s5z
```
---

## Citation

If you use this code in your research, please cite both the original paper and this implementation:

**Original Paper**:
```bibtex
@article{wang2025weak,
  title={Multi-agent reinforcement learning with weak ties},
  author={Wang, Huan and Hao, Xiaotian and Wu, Yi and others},
  journal={Information Fusion},
  volume={118},
  pages={102942},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.inffus.2025.102942}
}
```

**This Implementation**:
```bibtex
@misc{qiu2024weakties,
  title={Weak Ties MARL: A Reproduction Study},
  author={Qiu, Tian and Chen, Jiangyue and Xu, Yunchen},
  year={2024},
  note={Course Project: Algorithm Design and Analysis},
  howpublished={\url{https://github.com/yourusername/weak-ties-marl}}
}
```

---

## Acknowledgments

- **Original Authors**: Huan Wang, Xiaotian Hao, Yi Wu, et al. for the Weak Ties framework
- **SMAC**: [WhiRL Lab](https://github.com/oxwhirl/smac) for the StarCraft Multi-Agent Challenge
- **PySC2**: [DeepMind](https://github.com/deepmind/pysc2) for the StarCraft II Learning Environment
- **Course Instructor**: [Instructor Name] for guidance on algorithm reproduction

---

**Star this repository if you find it helpful!**
