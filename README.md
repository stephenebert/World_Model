# World Model – MiniGrid Project

This repo is the first phase of a project exploring **world models** in the
MiniGrid environment, with the eventual goal of comparing:

- A **JEPA-style** (prediction-only) world model, and  
- A **reconstruction-based** world model

in terms of planning performance and how well their latent representations
capture the **hierarchy of MiniGrid** (agent position, rooms, key/door state, etc).

Right now the focus is on **infrastructure**:

1. Setting up MiniGrid and collecting an offline dataset of transitions  
2. Loading that dataset into PyTorch via a clean `Dataset` + `DataLoader`

---

## Repository Structure

```text
World_Model/
├── Part_1/
│   ├── collect_minigrid_data.py   # Script to generate the .npz dataset
│   └── read_me.md                 # Step-by-step setup & data collection guide
├── Part_2/
│   ├── minigrid_dataset.py        # PyTorch Dataset + DataLoader for the .npz file
│   └── read_me.md                 # Instructions for using the loader
└── README.md                      # High-level overview (this file)
```
