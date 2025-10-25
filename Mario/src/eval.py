import os
import sys
import cv2
import torch
import argparse
from stable_baselines3 import PPO, DQN

# --- Display compatibility (Wayland/Xorg safe) ---
os.environ["SDL_VIDEODRIVER"] = "wayland"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Mario agent")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--persona", type=str, default="speedrunner")
    parser.add_argument("--config", type=str, default="MarioEvalConfig.yaml")
    args = parser.parse_args()

    # --- Load config ---
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    config = load_config(CONFIG_PATH)
    print(f"[Config] Loaded from {CONFIG_PATH}")

    # --- Load latest model ---
    model_dir = os.path.join(ROOT_DIR, "Mario", "models")
    model_path = find_latest_model(model_dir, args.persona, args.algo)
    if not model_path:
        raise FileNotFoundError(f"No model found for {args.persona}-{args.algo}")
    print(f"Found model: {model_path}")

    # --- Load model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ModelClass = PPO if args.algo == "ppo" else DQN
    model = ModelClass.load(model_path, device=device)
    print(f"Loaded {args.algo.upper()} model on {device}")

    # --- Create environment in rgb_array mode ---
    config["render_mode"] = "rgb_array"  # <---- key change
    env = make_mario_env(config=config, persona=args.persona)
    base_env = env.envs[0]  # unwrap VecEnv

    print(" Playing... Press 'q' to quit.")

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # --- Get frame safely from the inner env ---
        frame = base_env.render()
        if frame is not None:
            cv2.imshow("Mario", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f" Finished! Total reward: {float(total_reward):.2f}")
    env.close()
    cv2.destroyAllWindows()
