# World Model: MiniGrid Project

A research project exploring **world models** in the MiniGrid environment, comparing prediction-based (JEPA-style) and reconstruction-based approaches for hierarchical representation learning.

## Project Overview

This repository implements the infrastructure for training and evaluating world models that learn hierarchical representations of MiniGrid environments (agent position, rooms, key/door states, etc.).

**Key Components:**
- **JEPA-style model**: Prediction-only world model using latent dynamics
- **Reconstruction-based model**: Traditional autoencoder approach with pixel reconstruction
- **Hierarchy probes**: Tools to analyze learned representations
- **Planning**: Model Predictive Control (MPC) for navigation tasks

---

## Repository Structure
```
World_Model/
├── Part_1/
│   ├── collect_minigrid_data.py   # Dataset generation script
│   └── read_me.md                 # Setup & data collection guide
├── Part_2/
│   ├── minigrid_dataset.py        # PyTorch Dataset + DataLoader
│   └── read_me.md                 # DataLoader usage instructions
└── README.md                      # This file
```

---

## Current Progress

### Part 1: Data Collection (MiniGrid → .npz)

**Goal:** Create an offline dataset of MiniGrid transitions for world model training.

**Implementation** (`Part_1/collect_minigrid_data.py`):
- Environment: `MiniGrid-Empty-8x8-v0` with RGB observations (64×64×3)
- Policy: Random exploration
- Output: Compressed NumPy file `data/minigrid_empty8x8_random_200eps.npz`

**Dataset Contents:**
| Field | Shape | Description |
|-------|-------|-------------|
| `obs` | `(N, 64, 64, 3)` | Current observations |
| `next_obs` | `(N, 64, 64, 3)` | Next observations |
| `actions` | `(N,)` | Discrete actions taken |
| `rewards` | `(N,)` | Rewards received |
| `dones` | `(N,)` | Episode termination flags |

**See [Part_1/read_me.md](Part_1/read_me.md) for detailed setup instructions**

---

### Part 2 – PyTorch Dataset + DataLoader

**Goal:** Provide a clean PyTorch interface for loading the dataset.

**Implementation** (`Part_2/minigrid_dataset.py`):

**Classes:**
- `MiniGridTransitionDataset(npz_path)`: Custom PyTorch Dataset
  - Loads .npz file
  - Converts images to tensors (3, 64, 64) with values in [0, 1]
  - Returns dict with keys: `obs`, `next_obs`, `action`, `reward`, `done`

- `make_dataloader(npz_path, batch_size=64, ...)`: DataLoader factory function

**Sanity Check:**
```bash
cd Part_2
python3 minigrid_dataset.py
```

**Expected Output:**
```
obs shape      : torch.Size([32, 3, 64, 64])
next_obs shape : torch.Size([32, 3, 64, 64])
actions shape  : torch.Size([32])
rewards shape  : torch.Size([32])
dones shape    : torch.Size([32])
```

**See [Part_2/read_me.md](Part_2/read_me.md) for PyTorch setup**

---

## Roadmap

### Part 3 – JEPA-style World Model
- [ ] CNN encoder → latent state `z_t`
- [ ] Action-conditioned latent dynamics predictor
- [ ] Target encoder with JEPA/VICReg-style loss
- [ ] No pixel reconstruction (prediction-only)

### Part 4 – Reconstruction-based World Model
- [ ] Encoder + dynamics network
- [ ] Decoder for pixel reconstruction
- [ ] Pixel-space reconstruction loss
- [ ] Performance comparison with JEPA

### Part 5 – Planning & Hierarchy Analysis
- [ ] Model Predictive Control (MPC) implementation
- [ ] Navigation task evaluation
- [ ] Hierarchical representation probes
  - Agent position encoding
  - Room structure understanding
  - Key/door state tracking

---

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/World_Model.git
   cd World_Model
```

2. **Set up Part 1 (Data Collection)**
```bash
   cd Part_1
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install "gymnasium[classic-control]" minigrid numpy
```

3. **Generate dataset**
```bash
   python3 collect_minigrid_data.py
```

4. **Set up Part 2 (PyTorch)**
```bash
   cd ../Part_2
   pip install torch
   python3 minigrid_dataset.py  # Run sanity check
```

---

## Current Capabilities

**Implemented:**
- Reproducible data pipeline from MiniGrid → .npz
- Clean PyTorch DataLoader with proper tensor formatting
- Modular structure for easy model integration

**Coming Soon:**
- JEPA and reconstruction-based world models
- Planning algorithms
- Hierarchical representation analysis

---

## Documentation

For detailed instructions on each component:
- **[Part 1: Data Collection Guide](Part_1/read_me.md)**
- **[Part 2: PyTorch DataLoader Guide](Part_2/read_me.md)**

---
