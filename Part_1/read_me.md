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

## Optional: Visualize a sample transition

Run the script below to visualize obs and next_obs from the saved dataset:

```python3 view_npz_sample.py```

Figure 1: Example MiniGrid transition

![Figure 1: Example MiniGrid transition](fig_1.png)


The left panel (obs) shows the observation at time step t.

The right panel (next_obs) shows the observation at time step t + 1 after taking the stored action.

Color / object meanings:

Red triangle – the agent. The triangle’s orientation indicates the direction the agent is facing.

Green square – the goal cell the agent is trying to reach.

Dark gray grid cells – free floor tiles the agent can move through.

Light gray border – walls or boundary tiles that cannot be crossed.

Black region – area outside the agent’s current field of view (unobserved space).

This confirms that the dataset really consists of valid environment transitions:
each (obs, action, next_obs) triplet corresponds to the agent moving in the
MiniGrid world.
