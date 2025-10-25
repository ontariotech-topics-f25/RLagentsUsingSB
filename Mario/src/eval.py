import os
import sys
import torch
import argparse
import cv2
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# -----------------------------
# Headless / safe display setup
# -----------------------------
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGLET_HEADLESS"] = "1"

# -----------------------------
# Paths setup
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from Mario.envs.mario_env import make_mario_env
from Mario.src.utils import load_config

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Evaluate trained Mario agent and save video")
parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
parser.add_argument("--persona", type=str, default="collector")
parser.add_argument("--config", type=str, default="dqn_config.yaml")
parser.add_argument("--model-path", type=str,
                    default="Mario/models/checkpoints/collector_ppo_checkpoint_500000_steps.zip")
parser.add_argument("--video-dir", type=str, default="videos")
parser.add_argument("--video-name", type=str, default="mario_eval")
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

# -----------------------------
# Load config
# -----------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
config = load_config(CONFIG_PATH)
print(f"[Config] Loaded from {CONFIG_PATH}")

# -----------------------------
# Load model
# -----------------------------
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"No model found at {args.model_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
ModelClass = PPO if args.algo.lower() == "ppo" else DQN
model = ModelClass.load(args.model_path, device=device)
print(f"[Model] Loaded {args.algo.upper()} for persona {args.persona} on {device}")

# -----------------------------
# Create evaluation environment
# -----------------------------
eval_env_raw = make_mario_env(config=config, persona=args.persona)
eval_env = DummyVecEnv([lambda: eval_env_raw])  # wrap in SB3 VecEnv

# Get observation shape for video
sample_obs = eval_env.reset()
frame_shape = sample_obs.shape[2:]  # (channels, H, W)
frame_height, frame_width = frame_shape[1], frame_shape[2]

# Prepare video writer
os.makedirs(args.video_dir, exist_ok=True)
video_path = os.path.join(args.video_dir, f"{args.video_name}.mp4")
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                               (frame_width, frame_height))

# -----------------------------
# Evaluation loop
# -----------------------------
obs = eval_env.reset()
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated[0] if isinstance(terminated, (list, tuple)) else terminated
    total_reward += reward[0] if isinstance(reward, (list, tuple)) else reward

    # Convert observation to frame
    # obs: (1, channels, H, W) for DummyVecEnv
    frame = obs[0].transpose(1, 2, 0)  # C,H,W -> H,W,C
    if frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    video_writer.write(frame)

# -----------------------------
# Cleanup
# -----------------------------
video_writer.release()
eval_env.close()
print(f"Finished! Total reward: {float(total_reward):.2f}")
print(f"Video saved at {video_path}")
