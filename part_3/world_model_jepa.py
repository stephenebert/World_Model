import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """
    Simple CNN encoder: (B, 3, 64, 64) -> (B, latent_dim)
    """

    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # 16 -> 8
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class LatentDynamics(nn.Module):
    """
    One-step latent dynamics: z_t, a_t -> z_{t+1}^pred
    Uses an action embedding and an MLP.
    """

    def __init__(self, latent_dim=128, num_actions=7, action_dim=16):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z, actions):
        # z: (B, latent_dim), actions: (B,)
        a = self.action_emb(actions)           # (B, action_dim)
        x = torch.cat([z, a], dim=-1)         # (B, latent_dim + action_dim)
        z_next = self.mlp(x)                  # (B, latent_dim)
        return z_next


class JEPAWorldModel(nn.Module):
    """
    Minimal JEPA-style world model:
      - online encoder: E
      - target encoder: E_target (EMA of E)
      - latent dynamics: f_theta

    Loss: MSE between normalized predicted latent and target latent.
    """

    def __init__(self, in_channels=3, latent_dim=128, num_actions=7):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.target_encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.dynamics = LatentDynamics(latent_dim=latent_dim, num_actions=num_actions)
        self._init_target_encoder()

    def _init_target_encoder(self):
        # Copy weights and freeze target encoder params
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.99):
        """
        Exponential moving average (EMA) update:
        theta_target <- m * theta_target + (1 - m) * theta_online
        """
        for p_target, p_online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_online.data * (1.0 - momentum))

    def forward(self, obs, next_obs, actions):
        """
        obs, next_obs: (B, C, H, W)
        actions:      (B,)

        Returns:
          loss, z_pred, z_target
        """
        # Encode current obs with online encoder
        z = self.encoder(obs)  # (B, D)

        # Encode next_obs with target encoder (no grad)
        with torch.no_grad():
            z_target = self.target_encoder(next_obs)  # (B, D)

        # Predict next latent from z and action
        z_pred = self.dynamics(z, actions)            # (B, D)

        # Normalize for stability (BYOL-style)
        z_pred = F.normalize(z_pred, dim=-1)
        z_target = F.normalize(z_target, dim=-1)

        # JEPA-style regression loss
        loss = torch.mean((z_pred - z_target) ** 2)

        return loss, z_pred, z_target
