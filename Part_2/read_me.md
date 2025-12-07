# PyTorch Dataset + DataLoader

## Step 1: Install PyTorch

In the same World Model folder with the virtual environment activated:
```bash
pip install torch
```

## Step 2: Create `minigrid_dataset.py`

In `/Users/steph/Desktop/World Model/` create a new file called `minigrid_dataset.py`

## Step 3: Run the sanity check

In terminal:
```bash
cd "/Users/steph/Desktop/World Model"
python3 minigrid_dataset.py
```

We should see something like:

1. `obs shape : torch.Size([32, 3, 64, 64])`
2. `next_obs shape : torch.Size([32, 3, 64, 64])`
3. `actions shape : torch.Size([32])`

Once this works, we are ready for the next step: define the JEPA-style world model that takes these batches and learns latent dynamics (this is where we start really "modeling the hierarchy" in the representation).
