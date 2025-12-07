import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MiniGridTransitionDataset(Dataset):
    """
    Simple dataset of 1-step transitions:
    (obs_t, action_t, obs_{t+1}, reward_t, done_t)
    loaded from the .npz file we just created.
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.obs = data["obs"]          # (N, H, W, 3), uint8
        self.next_obs = data["next_obs"]
        self.actions = data["actions"]  # (N,)
        self.rewards = data["rewards"]  # (N,)
        self.dones = data["dones"]      # (N,)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        # Get raw arrays
        obs = self.obs[idx]        # (H, W, 3), uint8
        next_obs = self.next_obs[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        done = self.dones[idx]

        # Convert to torch tensors
        # Move channels to first dimension and normalize to [0, 1]
        obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0   # (3, H, W)
        next_obs = torch.from_numpy(next_obs).permute(2, 0, 1).float() / 255.0

        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)

        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": reward,
            "done": done,
        }


def make_dataloader(
    npz_path,
    batch_size=64,
    shuffle=True,
    num_workers=0,
):
    dataset = MiniGridTransitionDataset(npz_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


if __name__ == "__main__":
    # Quick sanity check
    npz_path = "data/minigrid_empty8x8_random_200eps.npz"
    loader = make_dataloader(npz_path, batch_size=32)

    batch = next(iter(loader))
    print("obs shape      :", batch["obs"].shape)       # (B, 3, 64, 64)
    print("next_obs shape :", batch["next_obs"].shape)
    print("actions shape  :", batch["action"].shape)
    print("rewards shape  :", batch["reward"].shape)
    print("dones shape    :", batch["done"].shape)
