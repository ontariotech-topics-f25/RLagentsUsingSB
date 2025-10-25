import os
import sys
import cv2
import torch
import time
import argparse
import zipfile
import tempfile
from stable_baselines3 import PPO, DQN

# --- Display compatibility (Wayland/Xorg safe) ---
os.environ.setdefault("SDL_VIDEODRIVER", "wayland")

# --- Path setup ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from Mario.envs.mario_env import make_mario_env
from Mario.src.utils import load_config


def find_latest_model(model_dir, persona, algo, extension=".zip"):
    """Find the most recent model file matching persona and algo."""
    if not os.path.exists(model_dir):
        return None
    files = [
        f for f in os.listdir(model_dir)
        if f.startswith(persona) and algo in f and f.endswith(extension)
    ]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
    return os.path.join(model_dir, files[0])


def safe_load_legacy_model(model_path, env, algo="ppo"):
    """
    Load SB3 models trained under older Python/torch versions.
    Tries normal load first, then a legacy fallback mode.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mario agent (normal speed + optional video)")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--persona", type=str, default="collector")
    parser.add_argument("--config", type=str, default="MarioEvalConfig.yaml")
    parser.add_argument("--video", action="store_true", help="Record gameplay video")
    args = parser.parse_args()

    # --- Load config ---
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    config = load_config(CONFIG_PATH)
    print(f"[Config] Loaded from {CONFIG_PATH}")

    # --- Locate model ---
    model_dir = os.path.join(ROOT_DIR, "Mario", "models")
    model_path = find_latest_model(model_dir, args.persona, args.algo)
    if not model_path:
        raise FileNotFoundError(f"No model found for {args.persona}-{args.algo}")
    print(f"Found model: {model_path}")

    # --- Create environment ---
    config["render_mode"] = "rgb_array"
    env = make_mario_env(config=config, persona=args.persona)
    base_env = env.envs[0]

    # --- Load model (new or legacy) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ModelClass = PPO if args.algo == "ppo" else DQN
    try:
        model = ModelClass.load(model_path, device=device)
        print(f"[INFO] Loaded {args.algo.upper()} model on {device}")
    except Exception:
        model = safe_load_legacy_model(model_path, env, algo=args.algo)

    # --- Setup optional video recording ---
    video_writer = None
    if args.video:
        video_dir = os.path.join(ROOT_DIR, "Mario", "videos")
        os.makedirs(video_dir, exist_ok=True)
        output_path = os.path.join(video_dir, f"{args.persona}_{args.algo}_eval.mp4")
        print(f"[INFO] Recording video to: {output_path}")

    # --- Determine frame rate from FrameSkipping wrapper ---
    skip = config.get("num_skip", 4)
    fps = max(1, 60 / skip)  # ~15 FPS for skip=4
    frame_delay = 1.0 / fps

    obs = env.reset()
    done = False
    total_reward = 0
    print(f"[INFO] Running at approx {fps:.1f} FPS (skip={skip}) — press 'q' to quit.")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        frame = base_env.render()
        if frame is not None:
            if args.video:
                if video_writer is None:
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            cv2.imshow("Mario", frame)

        # --- Maintain normal playback speed ---
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(frame_delay)

    print(f"\n[INFO] Finished! Total reward: {float(total_reward):.2f}")

    env.close()
    if video_writer:
        video_writer.release()
        print(f"[INFO] Video saved → {output_path}")
    cv2.destroyAllWindows()
