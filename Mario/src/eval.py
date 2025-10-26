import os
import sys
import cv2
import torch
import time
import argparse
import zipfile
import tempfile
import numpy as np
from stable_baselines3 import PPO, DQN

os.environ.setdefault("SDL_VIDEODRIVER", "x11")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from Mario.envs.mario_env import make_mario_env
from Mario.src.utils import load_config


def find_latest_model(model_dir, persona, algo, extension=".zip"):
    if not os.path.exists(model_dir):
        return None
    files = [f for f in os.listdir(model_dir)
             if f.startswith(persona) and algo in f and f.endswith(extension)]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
    return os.path.join(model_dir, files[0])


def safe_load_legacy_model(model_path, env, algo="ppo"):
    ModelClass = PPO if algo == "ppo" else DQN
    try:
        return ModelClass.load(model_path, env=env, device="auto")
    except Exception as e:
        print(f"[WARN] Normal load failed: {e}")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(model_path, "r") as archive:
                archive.extractall(tmpdir)
            return ModelClass.load(model_path, env=env, device="cpu", print_system_info=False)


def unwrap_all(env):
    """
    Drill through SB3 VecEnv and Gym wrappers until we reach the base NES env.
    """
    inner = env
    visited = set()
    while True:
        if id(inner) in visited:
            break
        visited.add(id(inner))
        if hasattr(inner, "envs"):      # DummyVecEnv / VecMonitor
            inner = inner.envs[0]
            continue
        if hasattr(inner, "venv"):      # VecMonitor has .venv
            inner = inner.venv
            continue
        if hasattr(inner, "env"):       # normal Gym wrappers
            inner = inner.env
            continue
        break
    return inner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mario agent and record video")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--persona", type=str, default="collector")
    parser.add_argument("--config", type=str, default="MarioEvalConfig.yaml")
    parser.add_argument("--video", action="store_true", help="Record gameplay video")
    args = parser.parse_args()

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    config = load_config(CONFIG_PATH)
    print(f"[Config] Loaded from {CONFIG_PATH}")

    model_dir = os.path.join(ROOT_DIR, "Mario", "models")
    model_path = find_latest_model(model_dir, args.persona, args.algo)
    if not model_path:
        raise FileNotFoundError(f"No model found for {args.persona}-{args.algo}")
    print(f"[INFO] Found model: {model_path}")

    env = make_mario_env(config=config, persona=args.persona)
    raw_env = unwrap_all(env)
    print(f"[INFO] Deep-unwrapped down to {type(raw_env)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ModelClass = PPO if args.algo == "ppo" else DQN
    try:
        model = ModelClass.load(model_path, device=device)
        print(f"[INFO] Loaded {args.algo.upper()} model on {device}")
    except Exception:
        model = safe_load_legacy_model(model_path, env, algo=args.algo)
        print(f"[INFO] Loaded legacy model on CPU")

    video_writer = None
    frame_count = 0
    if args.video:
        video_dir = os.path.join(ROOT_DIR, "Mario", "videos")
        os.makedirs(video_dir, exist_ok=True)
        output_path = os.path.join(video_dir, f"{args.persona}_{args.algo}_eval.mp4")
        print(f"[INFO] Recording → {output_path}")

    skip = config.get("num_skip", 4)
    fps = max(1, 60 / skip)
    frame_delay = 1.0 / fps

    obs = env.reset()
    done = False
    total_reward = 0.0
    print(f"[INFO] Running at {fps:.1f} FPS — press Q to quit")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Pull frame directly from the NES-Py core
        frame = None
        if hasattr(raw_env, "screen"):
            frame = np.array(raw_env.screen)
        elif hasattr(raw_env, "get_screen"):
            frame = raw_env.get_screen()
        elif hasattr(raw_env, "ale"):
            frame = raw_env.ale.getScreenRGB()
        elif hasattr(raw_env, "render"):
            frame = raw_env.render()

        if frame is not None:
            if args.video:
                if video_writer is None:
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    print(f"[INFO] VideoWriter initialized ({w}×{h}@{fps:.1f}fps)")
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1

            cv2.imshow("Mario Eval", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(frame_delay)

    print(f"\n[INFO] Episode finished | Total reward: {float(total_reward):.2f}")
    env.close()
    if video_writer:
        video_writer.release()
        print(f"[INFO] Video saved → {output_path} ({frame_count} frames)")
    else:
        print("[WARN] No video frames captured.")
    cv2.destroyAllWindows()
