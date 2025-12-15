# Part 3 – JEPA-style World Model

This part adds a first **JEPA-style world model** on top of the MiniGrid
dataset from Part 1 and the PyTorch DataLoader from Part 2.

The model:

- Encodes observations into a latent vector `z_t`
- Uses an action-conditioned latent dynamics model to predict `z_{t+1}`
- Uses a **target encoder** (EMA copy of the online encoder)
- Minimizes MSE between the normalized predicted latent and the target latent


To run in terminal:

# 1. Go to your project folder
cd "/Users/steph/Desktop/World Model"

# 2. Activate your virtual environment
source .venv/bin/activate

# 3. pip install minigrid
pip install torch gymnasium[minigrid] minigrid numpy

# 4. Run the JEPA training script
python3 train_jepa.py



![Example MiniGrid loss](loss.png)
