import sys, os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.SuperMario.AgentDDQ import AgentDDQ
from helpers import create_mario_env,initialize_seed,prepare_model_directory,load_config

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="ddqn")
parser.add_argument("--app", type=str, default="SuperMario")
parser.add_argument("--persona", type=str, default="speedrunner")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--load_model", action="store_true", help="Load pre-trained model if available")
parser.add_argument("--config", type=str, default="SuperMarioConfig.yaml",
                    help="YAML config file for training")
parser.add_argument("--run_label", type=str, default=None, help="Unique label for the model/config variant")

args = parser.parse_args()



def train(agent, config, episodes=1000, max_steps=10000):
    env = create_mario_env(config, persona=args.persona)
    total_rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        terminated = truncated = False
        episode_reward = 0

        for step in range(max_steps):
            # (1) ε-greedy action
            action = agent.next_action(state)

            # (2) Step through env
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # (3) Store transition
            agent.store_in_memory(state, action, reward, next_state, done)

            # (4) Learn (if enough samples)
            agent.learn()

            # (5) Move to next state
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{episodes} — Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return total_rewards


if __name__ == "__main__":



    # load config if you have one
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    config = load_config(CONFIG_PATH)
    print(f"Configuration loaded from {args.config}")

    #seed for reproducibility
    run_seed = initialize_seed(config.get("seed", 0), args.seed)
    config["run_seed"] = run_seed
    print(f"Run seed set to {run_seed}")

    #model path
    config["path_to_save_model"] = prepare_model_directory(
    args.app, args.persona, args.algo, run_seed, config
    )
    print(f"Model will be saved to: {config['path_to_save_model']}")

    #create agent
    agent = AgentDDQ(**config)

    #load agent if it exists
    if args.load_model and os.path.exists(config["path_to_save_model"]):
        print(f"Loading existing model from {config['path_to_save_model']}")
        agent.load_model(load_path=config["path_to_save_model"])
    else:
        print("Starting new model training...")

    #train agent
    rewards = train(agent,config, episodes=config.get("max_episodes", 10))
    
    #save agent
    agent.save_model()
    print("Training complete. Model saved.")
