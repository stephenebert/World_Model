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


Left image: obs (time step t)

The red triangle is your little robot/agent.

It’s pointing to the right, so that’s the direction it’s facing.

The green square is the goal tile it wants to reach.

The dark grey grid is the floor it can walk on.

The light grey border is the wall / outside boundary.

The big black area at the bottom is “I can’t see this yet” – the environment is partially observable, so the agent only sees a window around itself.

So: at time t the agent is in the upper-left corridor, can see some of the grid, and the goal is down in the bottom-right corner of its view.


Right image: next_obs (time step t + 1)

This is what the agent sees one step later, after taking whatever action is stored with this transition.

The red triangle has moved and is now pointing down in the top-middle of the grid.

The green goal is still in the bottom-right.

The black region is gone here: from this new position the agent’s field of view now covers the whole central area, so more of the world is visible.


At each data point in your dataset is basically:

“At time t I saw obs, took action a, and at time t + 1 I saw next_obs.”

Your world model’s job later will be: given obs and a, predict something about next_obs (e.g., its latent representation).


Color / object meanings:

Red triangle – the agent. The triangle’s orientation indicates the direction the agent is facing.

Green square – the goal cell the agent is trying to reach.

Dark gray grid cells – free floor tiles the agent can move through.

Light gray border – walls or boundary tiles that cannot be crossed.

Black region – area outside the agent’s current field of view (unobserved space).

This confirms that the dataset really consists of valid environment transitions:
each (obs, action, next_obs) triplet corresponds to the agent moving in the
MiniGrid world.
