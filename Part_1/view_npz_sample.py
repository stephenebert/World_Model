import numpy as np
import matplotlib.pyplot as plt

path = "data/minigrid_empty8x8_random_200eps.npz"
data = np.load(path)

print("obs shape:", data["obs"].shape)

# Pick an index to visualize
i = 10  # or any index
print("action:", data["actions"][i])

obs = data["obs"][i]        # (64, 64, 3)
next_obs = data["next_obs"][i]

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.title("obs")
plt.imshow(obs)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("next_obs")
plt.imshow(next_obs)
plt.axis("off")

plt.tight_layout()
plt.show()
