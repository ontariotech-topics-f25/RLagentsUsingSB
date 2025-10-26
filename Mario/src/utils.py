import os
import sys
import yaml
import numpy as np
import torch
import datetime
import csv


# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#Configuration loading utility
def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from: {config_path}")
    return config



# Seed initialization for reproducibility
def initialize_seed(config_seed: int = 0, cli_seed_offset: int = 0) -> int:
    """
    Combine the base YAML seed with CLI offset.
    Sets all random seeds (NumPy, Torch, CUDA) for reproducibility.
    """
    run_seed = config_seed + cli_seed_offset
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    print(f"Base seed {config_seed} + offset {cli_seed_offset} → using run seed {run_seed}")
    return run_seed



#Model directory preparation
def prepare_model_directory(app, persona, algo, run_seed, config):
    """
    Creates a consistent model directory path inside the app folder (e.g. Mario/models).
    Includes persona, algorithm, and seed in the filename for easy comparison.
    """
    # Determine app root (two levels up from /src)
    app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Create models directory directly under the app folder
    model_dir = os.path.join(app_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate filename with persona, algo, and seed
    label = config.get("run_label") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{persona}_{algo}_seed{run_seed}_{label}.zip"
    
    # Final model path
    path_to_save = os.path.join(model_dir, filename)
    print(f"Model path prepared: {path_to_save}")
    
    return path_to_save


#Level generation utility
def get_mario_levels(config: dict) -> list:
    """
    Generate a list of environment IDs across multiple worlds and levels.
    Example:
        world 1–3, 4 levels each
        → ['SuperMarioBros-1-1-v0', ..., 'SuperMarioBros-3-4-v0']
    """
    start_world = config.get("start_world", 1)
    end_world = config.get("end_world", start_world)
    num_levels = config.get("num_levels", 4)

    levels = [
        f"SuperMarioBros-{world}-{stage}-v0"
        for world in range(start_world, end_world + 1)
        for stage in range(1, num_levels + 1)
    ]

    print(f"Generated level sequence: {levels}")
    return levels


#For episode-level training metrics logging
class TrainingMetricsLogger:
    """
    Logs episode-level metrics such as average reward, loss, epsilon, etc.
    """
    def __init__(self, algo="ppo", persona="default", app="Mario"):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.log_dir = os.path.join(project_root, "data", "training_metrics")
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{persona}_{algo}_{timestamp}_metrics.csv")

        self.fieldnames = ["episode", "total_reward", "avg_reward", "loss", "epsilon", "timesteps"]
        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

        print(f"Training metrics will be logged to {self.log_file}")

    def log(self, episode, total_reward, avg_reward, loss=None, epsilon=None, timesteps=None):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                "episode": episode,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
                "loss": loss,
                "epsilon": epsilon,
                "timesteps": timesteps
            })

#Stable Baselines3 callback for logging metrics during training
from stable_baselines3.common.callbacks import BaseCallback
class MetricsLoggingCallback(BaseCallback):
    def __init__(self, metrics_logger, verbose=0):
        super().__init__(verbose)
        self.metrics_logger = metrics_logger
        self.episode_count = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                self.episode_count += 1
                total_reward = info["episode"]["r"]
                timesteps = info["episode"]["l"]
                self.metrics_logger.log(
                    episode=self.episode_count,
                    total_reward=total_reward,
                    avg_reward=(total_reward / timesteps) if timesteps else 0.0,
                    loss=None,
                    epsilon=None,
                    timesteps=timesteps
                )
        return True
    
import glob

# ---------------------------------------------------------
# MODEL AND ENVIRONMENT UTILITIES FOR EVALUATION
# ---------------------------------------------------------

def find_latest_model(model_dir, persona, algo):
    """
    Find the most recent model checkpoint that matches the algorithm name
    and (optionally) the persona keyword.

    Supports filenames like:
        collector_ppo_checkpoint_500000_steps.zip
        coin_greedy_ppo_seed67_20251023_110455.zip
        speedrunner_dqn_checkpoint_300000_steps.pth

    Works for both .zip (SB3) and .pth (PyTorch) models.
    """
    if not os.path.exists(model_dir):
        return None

    # Gather all potential model files
    candidates = glob.glob(os.path.join(model_dir, "*.zip")) + glob.glob(os.path.join(model_dir, "*.pth"))
    if not candidates:
        return None

    # Filter by algorithm
    filtered = [f for f in candidates if algo.lower() in os.path.basename(f).lower()]
    if not filtered:
        print(f"[WARN] No models matched algorithm '{algo}' — using all files instead.")
        filtered = candidates

    # Prefer files that include persona keyword
    persona_lower = persona.lower()
    persona_matches = [f for f in filtered if persona_lower in os.path.basename(f).lower()]
    final_list = persona_matches if persona_matches else filtered

    # Return newest file by modification time
    latest = max(final_list, key=os.path.getmtime)
    return latest


def unwrap_all(env):
    """
    Drill through Stable-Baselines3 VecEnv and Gym wrappers until reaching
    the base NES-Py environment (SuperMarioBrosEnv).

    Example:
        VecMonitor → DummyVecEnv → SeedSafeEnv → LoggingWrapper → ... → SuperMarioBrosEnv
    """
    inner = env
    visited = set()
    while True:
        if id(inner) in visited:
            break
        visited.add(id(inner))

        # VecMonitor or DummyVecEnv
        if hasattr(inner, "envs"):
            inner = inner.envs[0]
            continue
        # Nested venv
        if hasattr(inner, "venv"):
            inner = inner.venv
            continue
        # Standard Gym wrappers
        if hasattr(inner, "env"):
            inner = inner.env
            continue
        break
    return inner
