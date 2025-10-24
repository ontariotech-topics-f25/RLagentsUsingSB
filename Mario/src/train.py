import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# This file is located at <root>/Mario/src/train.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

print(f"[DEBUG] Project root added to sys.path: {ROOT_DIR}")

# --- Internal imports ---
from Mario.envs.mario_env import make_mario_env
from utils import load_config, initialize_seed, prepare_model_directory,TrainingMetricsLogger, MetricsLoggingCallback
from stable_baselines3 import PPO, DQN
from Agents.AgentDDQ import AgentDDQ


#  Helper to find the latest model
def find_latest_model(model_dir, persona, algo, extension=".zip"):
    """
    Find the most recent model file matching persona and algo in a directory.
    Returns full path or None.
    """
    if not os.path.exists(model_dir):
        return None
    candidates = [
        f for f in os.listdir(model_dir)
        if f.startswith(persona) and f.endswith(f"{algo}{extension}")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_path = os.path.join(model_dir, candidates[0])
    print(f"[AutoLoad] Found latest model for {persona}-{algo}: {latest_path}")
    return latest_path


def train(agent, env, config, episodes=1000, max_steps=10000, metrics_logger=None):
    total_rewards = []
    for episode in range(episodes):
        state, info = env.reset()
        terminated = truncated = False
        episode_reward = 0

        for step in range(max_steps):
            action = agent.next_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            agent.store_in_memory(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{episodes} — Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        if metrics_logger is not None:
            avg_reward = episode_reward / (step + 1)
            metrics_logger.log(
                episode=episode + 1,
                total_reward=episode_reward,
                avg_reward=avg_reward,
                loss=getattr(agent, "last_loss", None),
                epsilon=getattr(agent, "epsilon", None),
                timesteps=step + 1
            )
    env.close()
    return total_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mario RL agent")
    parser.add_argument("--algo", type=str, default="ddqn", help="RL algorithm (ddqn, ppo, etc.)")
    parser.add_argument("--app", type=str, default="SuperMario")
    parser.add_argument("--persona", type=str, default="speedrunner")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_model", action="store_true", help="Load pre-trained model if available")
    parser.add_argument("--config", type=str, default="SuperMarioConfig.yaml", help="Config file path under configs/")
    parser.add_argument("--run_label", type=str, default=None, help="Unique label for this training run")
    args = parser.parse_args()

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    config = load_config(CONFIG_PATH)

    run_seed = initialize_seed(config.get("seed", 0), args.seed)
    config["run_seed"] = run_seed
    print(f"Run seed set to {run_seed}")

    config["path_to_save_model"] = prepare_model_directory(
        args.app, args.persona, args.algo, run_seed, config
    )
    print(f"Model will be saved to: {config['path_to_save_model']}")

    env = make_mario_env(
        levels=config.get("levels", None),
        persona=args.persona,
        resize=config.get("resize", 84),
        num_stack=config.get("num_stack", 4),
        num_skip=config.get("num_skip", 4),
        render_mode=config.get("render_mode", None),
        action_set=config.get("action_set", "SIMPLE_MOVEMENT"),
        config=config
    )

    metrics_logger = TrainingMetricsLogger(algo=args.algo, persona=args.persona, app=args.app)
    algo = args.algo.lower()

    if algo == "ddqn":
        print("Using custom DDQN agent")
        agent = AgentDDQ(**config)
        if args.load_model:
            latest_model = find_latest_model(os.path.dirname(config["path_to_save_model"]), args.persona, args.algo, extension=".pth")
            if latest_model:
                print(f"Loading DDQN weights from {latest_model}")
                agent.load_model(load_path=latest_model)
            elif os.path.exists(config["path_to_save_model"]):
                print(f"Loading DDQN weights from {config['path_to_save_model']}")
                agent.load_model(load_path=config["path_to_save_model"])

        rewards = train(agent, env, config, episodes=config.get("max_episodes", 10), metrics_logger=metrics_logger)
        agent.save_model()
        print("Training complete. Model saved successfully.")

    elif algo == "ppo":
        print("Using PPO (Stable Baselines 3)")
        model = PPO("CnnPolicy", env, verbose=1, seed=run_seed, learning_rate=config.get("learning_rate", 2.5e-4))
        if args.load_model:
            latest_model = find_latest_model(os.path.dirname(config["path_to_save_model"]), args.persona, args.algo)
            if latest_model:
                model = PPO.load(latest_model, env=env)
            elif os.path.exists(config["path_to_save_model"]):
                model = PPO.load(config["path_to_save_model"], env=env)

        callback = MetricsLoggingCallback(metrics_logger)
        model.learn(total_timesteps=config.get("total_timesteps", 100_000), callback=callback)
        model.save(config["path_to_save_model"])
        print("Training complete. Model saved successfully.")

    elif algo == "dqn":
        print("Using DQN (Stable Baselines 3)")
        model = DQN(
            "CnnPolicy",
            env,
            verbose=1,
            seed=run_seed,
            learning_rate=config.get("learning_rate", 1e-4),
            buffer_size=config.get("buffer_size", 50_000),
            batch_size=config.get("batch_size", 32),
        )

        if args.load_model:
            latest_model = find_latest_model(os.path.dirname(config["path_to_save_model"]), args.persona, args.algo)
            if latest_model:
                model = DQN.load(latest_model, env=env)
            elif os.path.exists(config["path_to_save_model"]):
                model = DQN.load(config["path_to_save_model"], env=env)

        callback = MetricsLoggingCallback(metrics_logger)
        model.learn(total_timesteps=config.get("total_timesteps", 100_000), callback=callback)
        model.save(config["path_to_save_model"])
        print("Training complete. Model saved successfully.")

    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")
