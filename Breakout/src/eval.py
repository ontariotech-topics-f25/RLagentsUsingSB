import os
import sys
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

print(f"[DEBUG] Project root added to sys.path: {ROOT_DIR}")

from Breakout.envs.Breakout.applyWrapper import applyAllWrappers  # ensure this import path is valid


def test_model(model_type="DQN", render_mode="rgb_array", persona="TestRun", record_video=True):
    """
    Load a trained DQN or PPO model and play one episode of Breakout, optionally recording video.
    """
    model_type = model_type.upper()
    if model_type not in ["DQN", "PPO"]:
        raise ValueError("model_type must be 'DQN' or 'PPO'.")

    # --- Create environment factory ---
    env_fn = lambda: applyAllWrappers(
        gym.make("ALE/Breakout-v5", render_mode="rgb_array" if record_video else render_mode),
        persona=persona
    )
    env = DummyVecEnv([env_fn])

    # --- Optionally wrap for video recording ---
    if record_video:
        video_dir = os.path.join(ROOT_DIR, "Breakout", "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_name = f"{model_type}_{persona}.mp4"
        env = VecVideoRecorder(
            env,
            video_dir,
            record_video_trigger=lambda step: step == 0,
            video_length=10000,  # frames to record
            name_prefix=f"{model_type}_{persona}"
        )
        print(f"[INFO] Recording video to: {os.path.join(video_dir, video_name)}")

    # --- Load model ---
    model_dir = os.path.join(ROOT_DIR, "Breakout", "model")
    model_path = os.path.join(model_dir, f"{model_type}_agent_final.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {os.path.abspath(model_path)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = DQN if model_type == "DQN" else PPO
    model = model_cls.load(model_path, env=env, device=device)

    # --- Run evaluation episode ---
    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done and step < 10000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
        step += 1

    print(f"[RESULT] {model_type} test run finished. Total reward: {total_reward}")

    # --- Save and close video ---
    if record_video:
        env.close()
        print(f"[INFO] Video saved to: {video_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 eval.py <algorithm>")
        print("algorithm: dqn or ppo")
        sys.exit(1)

    algo = sys.argv[1].lower()
    test_model(model_type=algo)
