import os
import sys
import torch
import torch.optim as optim

from world_model_jepa import JEPAWorldModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    # Figure out project root (one level above this Part_3 folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Make sure we can import the DataLoader from Part_2
    sys.path.append(os.path.join(project_root, "Part_2"))
    try:
        from minigrid_dataset import make_dataloader
    except ImportError as e:
        raise ImportError(
            "Could not import make_dataloader from Part_2/minigrid_dataset.py. "
            "Make sure Part_2 exists and contains that file."
        ) from e

    # Path to the dataset generated in Part_1
    npz_path = os.path.join(project_root, "data", "minigrid_empty8x8_random_200eps.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Could not find dataset at {npz_path}. "
            "Make sure you've run Part_1/collect_minigrid_data.py from the project root."
        )

    # Build DataLoader
    dataloader = make_dataloader(npz_path, batch_size=64, shuffle=True, num_workers=0)

    device = get_device()
    print(f"Using device: {device}")

    model = JEPAWorldModel(in_channels=3, latent_dim=128, num_actions=7)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            obs = batch["obs"].to(device)          # (B, 3, 64, 64)
            next_obs = batch["next_obs"].to(device)
            actions = batch["action"].to(device)

            loss, z_pred, z_target = model(obs, next_obs, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target encoder with EMA
            model.update_target_encoder(momentum=0.99)

            running_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100.0
                print(f"Epoch {epoch+1} | Step {batch_idx+1} | Loss = {avg_loss:.4f}")
                running_loss = 0.0

        # Print leftover loss if dataset smaller than 100 batches
        if running_loss > 0.0:
            num_batches = (batch_idx + 1) % 100 or (batch_idx + 1)
            avg_loss = running_loss / num_batches
            print(f"Epoch {epoch+1} | End of epoch | Avg Loss = {avg_loss:.4f}")

    # Save model weights
    save_path = os.path.join(current_dir, "jepa_world_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved JEPA world model to {save_path}")


if __name__ == "__main__":
    main()
