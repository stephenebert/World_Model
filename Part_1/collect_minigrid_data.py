import os
import numpy as np
import gymnasium as gym
import minigrid  # needed so that env IDs register
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper


def make_env(env_id="MiniGrid-Empty-8x8-v0", seed=0):
    """
    Create a MiniGrid env whose observations are RGB images (H, W, 3).
    """
    env = gym.make(env_id, render_mode="rgb_array")
    # Get full grid as an RGB image
    env = RGBImgObsWrapper(env)
    # Drop the dict, keep only the image
    env = ImgObsWrapper(env)
    # Seed the RNG
    env.reset(seed=seed)
    return env


def collect_trajectories(
    env_id="MiniGrid-Empty-8x8-v0",
    num_episodes=100,
    max_steps_per_episode=100,
    seed=0,
):
    """
    Collect random-policy trajectories and return them as numpy arrays:
    obs, next_obs: uint8 images, shape (N, H, W, 3)
    actions: int64, shape (N,)
    rewards: float32, shape (N,)
    dones: bool, shape (N,)
    """
    env = make_env(env_id=env_id, seed=seed)

    obs_list = []
    next_obs_list = []
    action_list = []
    reward_list = []
    done_list = []

    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=seed + episode_idx)
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            # Random policy for now
            action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)

            obs = next_obs
            step += 1

    # Stack into arrays
    obs_array = np.stack(obs_list).astype(np.uint8)
    next_obs_array = np.stack(next_obs_list).astype(np.uint8)
    actions_array = np.array(action_list, dtype=np.int64)
    rewards_array = np.array(reward_list, dtype=np.float32)
    dones_array = np.array(done_list, dtype=bool)

    return {
        "obs": obs_array,
        "next_obs": next_obs_array,
        "actions": actions_array,
        "rewards": rewards_array,
        "dones": dones_array,
    }


def save_dataset(dataset, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=dataset["obs"],
        next_obs=dataset["next_obs"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        dones=dataset["dones"],
    )
    print(f"Saved dataset to {out_path}")
    print("Shapes:")
    print("  obs      :", dataset["obs"].shape)
    print("  next_obs :", dataset["next_obs"].shape)
    print("  actions  :", dataset["actions"].shape)
    print("  rewards  :", dataset["rewards"].shape)
    print("  dones    :", dataset["dones"].shape)


if __name__ == "__main__":
    # You can tweak these numbers later
    env_id = "MiniGrid-Empty-8x8-v0"
    num_episodes = 200
    max_steps = 100
    seed = 0

    dataset = collect_trajectories(
        env_id=env_id,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        seed=seed,
    )

    out_path = "data/minigrid_empty8x8_random_200eps.npz"
    save_dataset(dataset, out_path)
