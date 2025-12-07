# World Model Setup Guide

## Step 1: Create a folder and navigate to it

Create a folder, then open terminal and navigate to that folder:
```bash
cd "/Users/steph/Desktop/World Model"
```

## Step 2: Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of the terminal prompt.

## Step 3: Install the required packages
```bash
pip install "gymnasium[classic-control]" minigrid numpy
```

## Step 4: Run the script
```bash
python3 collect_minigrid_data.py
```

## Optional
Run ```view_npz_sample.py``` to visualize obs and next_obs from the saved dataset.
