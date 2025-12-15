# Part 3 – JEPA-style World Model

This part adds a first **JEPA-style world model** on top of the MiniGrid
dataset from Part 1 and the PyTorch DataLoader from Part 2.

The model:

- Encodes observations into a latent vector `z_t`
- Uses an action-conditioned latent dynamics model to predict `z_{t+1}`
- Uses a **target encoder** (EMA copy of the online encoder)
- Minimizes MSE between the normalized predicted latent and the target latent
