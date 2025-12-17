import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
The model expects inputs in the shape:
[batch_size, sequence_len, channels, height, width]
    batch_size: Number of sequences processed together.
    sequence_len: Number of consecutive frames in each sequence.
    channels: RGB channels (3 for color images).
    height, width: Image dimensions (e.g., 32x32, 64x64, 128x128).

JEPA Architecture Overview:
Images -> Basic Encoder -> Latents ---------
                                        |
                                Transformer Context -> context
                                        |
                          Predictor (context -> predicted latent)
                                        |
                             Compare with target latent (cosine loss)
'''

# -----------------------------
# 1. CNN Encoder for Images
# -----------------------------
'''
Purpose:
Maps raw images [C,H,W] to a latent vector representation.

IMPORTANT:
No convolution at all.
Just flattens the image and projects to latent space.

JEPA role:
Encoder gives us a compact representation of each frame.
The model predicts *future latent representations*, not pixels.
'''
class BasicEncoder(nn.Module):
    def __init__(self, input_channels=3, img_size=32, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_channels * img_size * img_size, latent_dim)
        
    def forward(self, x):
        # x: [batch, channels, height, width]
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)


# -----------------------------
# 2. Transformer-based Context Network
# -----------------------------
'''
Purpose:
Aggregates information from a sequence of latent vectors
to produce a context vector.

Input:
z_seq shape → [batch, sequence_len, latent_dim]

Transformer details:
- Captures temporal dependencies between frames
- Uses self-attention over latent sequence

Output:
Last token of transformer output → context vector

JEPA role:
Context represents "what the world knows so far"
and is used to predict the next latent state.
'''
class TransformerContext(nn.Module):
    def __init__(self, latent_dim=64, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, z_seq):
        # z_seq: [batch, sequence_len, latent_dim]
        context_seq = self.transformer(z_seq)

        # Use last token as context summary
        context = context_seq[:, -1, :]
        return context


# -----------------------------
# 3. Predictor
# -----------------------------
'''
Purpose:
Maps context vector → predicted latent of next frame.

Design:
Simple linear projection.
Context already contains temporal structure.

JEPA role:
Learns to predict future *latent states*,
not future observations.
'''
class Predictor(nn.Module):
    def __init__(self, context_dim=64, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(context_dim, latent_dim)
    
    def forward(self, context):
        return self.fc(context)


# -----------------------------
# 4. JEPA World Model
# -----------------------------
'''
Purpose:
Full JEPA world model that:
- Encodes image sequences
- Builds temporal context
- Predicts future latent representations
- Optimizes with JEPA cosine loss

This class wraps:
    Encoder
    Context Network
    Predictor
    Optimizer
'''
class JEPAWorldModel(nn.Module):
    def __init__(
        self,
        input_channels=3,
        img_size=32,
        latent_dim=64,
        nhead=4,
        num_layers=2,
        lr=1e-3
    ):
        super().__init__()

        # Core components
        self.encoder = BasicEncoder(
            input_channels=input_channels,
            img_size=img_size,
            latent_dim=latent_dim
        )

        self.context_net = TransformerContext(
            latent_dim=latent_dim,
            nhead=nhead,
            num_layers=num_layers
        )

        self.predictor = Predictor(
            context_dim=latent_dim,
            latent_dim=latent_dim
        )

        # Optimizer updates all JEPA components jointly
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    # -----------------------------
    # Forward Pass (JEPA logic)
    # -----------------------------
    '''
    Input:
        x_seq: [batch, seq_len, C, H, W]

    Steps:
        1. Encode each frame into a latent
        2. Build context from past frames
        3. Predict latent for next frame

    Output:
        z_pred   : predicted latent
        z_target : true latent (stop-gradient)
    '''
    def forward(self, x_seq):
        batch_size, seq_len, C, H, W = x_seq.shape

        # Encode all frames
        z_seq = torch.stack(
            [self.encoder(x_seq[:, t]) for t in range(seq_len)],
            dim=1
        )  # [batch, seq_len, latent_dim]

        # Context from first (seq_len - 1) frames
        context = self.context_net(z_seq[:, :-1])

        # Target latent (stop gradient for JEPA stability)
        z_target = z_seq[:, -1].detach()

        # Predict next latent
        z_pred = self.predictor(context)

        return z_pred, z_target

    # -----------------------------
    # JEPA Cosine Loss
    # -----------------------------
    '''
    Encourages alignment between predicted and true latent.
    No reconstruction loss.
    No contrastive negatives.
    '''
    @staticmethod
    def loss_fn(z_pred, z_target):
        z_pred = F.normalize(z_pred, dim=-1)
        z_target = F.normalize(z_target, dim=-1)
        return 2 - 2 * (z_pred * z_target).sum(dim=-1).mean()

    # -----------------------------
    # Training Step
    # -----------------------------
    '''
    Performs one JEPA training step:
        - Forward pass
        - Loss computation
        - Backpropagation
        - Parameter update
    '''
    def train_step(self, x_seq):
        self.train()

        z_pred, z_target = self.forward(x_seq)
        loss = self.loss_fn(z_pred, z_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# -----------------------------
# 5. Dummy Training Loop
# -----------------------------
'''
Generates dummy image sequences:
[batch, seq_len, 3, 32, 32]

In real settings:
- Replace with video frames
- Or environment observations
'''
if __name__ == "__main__":
    batch_size = 16
    sequence_len = 5
    img_size = 32

    model = JEPAWorldModel(
        input_channels=3,
        img_size=img_size,
        latent_dim=64
    )

    for step in range(200):
        x_seq = torch.randn(
            batch_size,
            sequence_len,
            3,
            img_size,
            img_size
        )

        loss = model.train_step(x_seq)

        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
