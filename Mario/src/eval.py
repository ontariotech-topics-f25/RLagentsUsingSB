import os
import sys
import cv2
import torch
import time
import argparse
import tempfile
import zipfile
import numpy as np
from stable_baselines3 import PPO, DQN

# --- Ensure X11 rendering works (fallback for Wayland/headless) ---
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

# --- Path setup ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

# --- Internal imports ---
from Mario.envs.mario_env import make_mario_env
from Mario.src.utils import load_config, find_latest_model, unwrap_all


# --- Legacy load support (for older checkpoints) ---
def safe_load_legacy_model(model_path, env, algo="ppo"):
    """
    Load SB3 models trained under older Python/torch versions.
    """
    ModelClass = PPO if algo == "ppo" else DQN
    try:
        model = ModelClass.load(model_path, env=env, device="auto")
        print(f"[INFO] Loaded model normally from {model_path}")
        return model
    except Exception as e:
        print(f"[WARN] Normal load failed: {e}")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(model_path, "r") as archive:
                archive.extractall(tmpdir)
            model = ModelClass.load(model_path, env=env, device="cpu", print_system_info=False)
        print("[INFO] Loaded using legacy fallback mode (CPU).")
        return model


# --- Main evaluation script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mario agent and record video")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--persona", type=str, default="collector")
    parser.add_argument("--config", type=str, default="MarioEvalConfig.yaml")
    parser.add_argument("--video", action="store_true", help="Record gameplay video")
    args = parser.parse_args()

    # --- Load configuration ---
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    config = load_config(CONFIG_PATH)
    print(f"[Config] Loaded from {CONFIG_PATH}")

    # --- Locate latest model ---
    model_dir = os.path.join(ROOT_DIR, "Mario", "models")
    model_path = find_latest_model(model_dir, args.persona, args.algo)
    if not model_path:
        raise FileNotFoundError(f"No model found for persona={args.persona}, algo={args.algo}")
    print(f"[INFO] Found latest model: {os.path.basename(model_path)}")

    # --- Create environment ---
    env = make_mario_env(config=config, persona=args.persona)
    raw_env = unwrap_all(env)
    print(f"[INFO] Deep-unwrapped down to {type(raw_env)}")

    # --- Load model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ModelClass = PPO if args.algo == "ppo" else DQN
    try:
        model = ModelClass.load(model_path, device=device)
        print(f"[INFO] Loaded {args.algo.upper()} model on {device}")
    except Exception:
        model = safe_load_legacy_model(model_path, env, algo=args.algo)
        print(f"[INFO] Loaded legacy model on CPU")

    # --- Video setup ---
    video_writer = None
    frame_count = 0
    if args.video:
        video_dir = os.path.join(ROOT_DIR, "Mario", "videos")
        os.makedirs(video_dir, exist_ok=True)
        output_path = os.path.join(video_dir, f"{args.persona}_{args.algo}_eval.mp4")
        print(f"[INFO] Recording → {output_path}")

    # --- Timing ---
    skip = config.get("num_skip", 4)
    fps = max(1, 60 / skip)
    frame_delay = 1.0 / fps

    obs = env.reset()
    done = False
    total_reward = 0.0
    print(f"[INFO] Running at {fps:.1f} FPS — press Q to quit")

    # --- Evaluation loop ---
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Capture frame directly from NES-Py environment
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
            # Initialize video writer
            if args.video:
                if video_writer is None:
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    print(f"[INFO] VideoWriter initialized ({w}x{h}@{fps:.1f}fps)")
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1

            # Display live window
            cv2.imshow("Mario Eval", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(frame_delay)

    # --- Wrap-up ---
    print(f"\n[INFO] Episode finished | Total reward: {float(total_reward):.2f}")
    env.close()
    if video_writer:
        video_writer.release()
        print(f"[INFO] Video saved → {output_path} ({frame_count} frames)")
    else:
        print("[WARN] No video frames captured.")
    cv2.destroyAllWindows()
