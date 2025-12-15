import os
import sys
import torch

from world_model_jepa import JEPAWorldModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Load model architecture ---
    device = get_device()
    print("Using device:", device)

    model = JEPAWorldModel(in_channels=3, latent_dim=128, num_actions=7)
    model.to(device)

    # --- Load checkpoint (.pt file) ---
    ckpt_path = os.path.join(current_dir, "jepa_world_model.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    # --- Import DataLoader to get a batch of data ---
    sys.path.append(current_dir)
    from minigrid_dataset import make_dataloader

    npz_path = os.path.join(current_dir, "data", "minigrid_empty8x8_random_200eps.npz")
    dataloader = make_dataloader(npz_path, batch_size=16, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    obs = batch["obs"].to(device)
    next_obs = batch["next_obs"].to(device)
    actions = batch["action"].to(device)

    # --- Run a forward pass to inspect latents ---
    with torch.no_grad():
        loss, z_pred, z_target = model(obs, next_obs, actions)

    print("Batch size:", obs.shape[0])
    print("z_pred shape:", z_pred.shape)
    print("z_target shape:", z_target.shape)
    print("Example z_pred[0][:10]:", z_pred[0, :10].cpu().numpy())
    print("Current batch loss:", loss.item())

    # --- Peek at some weights (e.g., first conv layer) ---
    first_conv = model.encoder.conv_net[0]  # Conv2d
    w = first_conv.weight.data
    print("First conv weight shape:", w.shape)
    print("First conv weight mean:", w.mean().item())
    print("First conv weight std:", w.std().item())


if __name__ == "__main__":
    main()
