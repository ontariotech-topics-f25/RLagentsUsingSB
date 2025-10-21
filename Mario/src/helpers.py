import os
import numpy as np
import torch
import yaml
import sys



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.SuperMario.Wrappers import SequentialMarioEnv, applyAllWrappers

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def initialize_seed(config_seed=0, cli_seed_offset=0):
    """
    Combine the base YAML seed with the CLI offset, 
    set all global RNG seeds, and return the effective run seed.
    """
    run_seed = config_seed + cli_seed_offset
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    print(f"Base seed: {config_seed}, offset: {cli_seed_offset} → using run seed {run_seed}")
    return run_seed

def prepare_model_directory(app, persona, algo, run_seed,config):

    #differentiate models by run label if provided
    label = config.get("run_label", "")
    label_suffix = f"_{label}" if label else ""

    #Add persona name to model path
    persona_dir = os.path.join("models", app, persona)
    os.makedirs(persona_dir, exist_ok=True)
    model_filename = f"{persona}_seed{run_seed}_{algo}_{label}.pth"
    model_path = os.path.join(persona_dir, model_filename)
    print(f"Model path prepared: {model_path}")

    return model_path


def get_mario_levels(config):
    """
    Dynamically generate environment IDs across multiple worlds and levels.
    Example: world 1–3, 4 levels each → ['SuperMarioBros-1-1-v0', ..., 'SuperMarioBros-3-4-v0']
    """
    start_world = config.get("start_world", 1)
    end_world = config.get("end_world", start_world)
    num_levels = config.get("num_levels", 4)

    levels = [
        f"SuperMarioBros-{world}-{stage}-v0"
        for world in range(start_world, end_world + 1)
        for stage in range(1, num_levels + 1)
    ]
    return levels

def create_mario_env(config, persona):
    """
    Creates the full Mario environment pipeline:
    Sequential levels → FrameSkip → Resize → Gray → Stack → PersonaReward → Logging
    """
    levels = get_mario_levels(config)
    print(f"Training on levels: {levels}")

    env = SequentialMarioEnv(levels, render_mode=config.get("render_mode", "human"), action_set=config.get("action_set", "SIMPLE_MOVEMENT"))
    env = applyAllWrappers(
        env,
        resize=config.get("resize", 84),
        num_stack=config.get("num_stack", 4),
        num_skip=config.get("num_skip", 4),
        persona=persona,
        config=config
    )
    return env