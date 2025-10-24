import os
import sys
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

print(f"[DEBUG] Project root added to sys.path: {ROOT_DIR}")

from  Breakout.envs.Breakout.applyWrapper import applyAllWrappers  # make sure this path is correct




def test_model(model_type="DQN", render_mode="human", persona="TestRun"):
    """
    Load a trained DQN or PPO model and play one episode of Breakout.
    
    Args:
        model_type (str): "DQN" or "PPO"
        render_mode (str): "human" or None
        persona (str): Optional label for logging wrapper
    """
    model_type = model_type.upper()
    if model_type not in ["DQN", "PPO"]:
        raise ValueError("model_type must be 'DQN' or 'PPO'.")

    # --- Create environment ---
    env_fn = lambda: applyAllWrappers(
        gym.make("ALE/Breakout-v5", render_mode=render_mode),
        persona=persona
    )
    env = DummyVecEnv([env_fn])

    # --- Load the latest model for the chosen algorithm ---
    model_dir = "model"
    model_path = os.path.join(model_dir, f"{model_type}_agent_final.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {os.path.abspath(model_path)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "DQN":
        model = DQN.load(model_path, env=env, device=device)
    else:
        model = PPO.load(model_path, env=env, device=device)


    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        done = terminated
        total_reward += reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward

    print(f"{model_type} test run finished. Total reward: {total_reward}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 eval.py <algorithm>")
        print("algorithm: dqn or ppo")
        sys.exit(1)

    algo = sys.argv[1].lower()
    test_model(model_type=algo)
