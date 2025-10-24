import os
import yaml
import torch
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from .applyWrapper import applyAllWrappers

class RLAgent:
    def __init__(self, config_path):
        config_path = os.path.join(
         os.path.dirname(__file__),  # src/Breakout
        "../../configs/BreakPoint/BreakPoint.yaml")
        config_path = os.path.abspath(config_path)


        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() and self.config.get("use_cuda", True) else "cpu"
        self.model_type = self.config.get("model_type", "DQN").upper()
        
        env_fn = lambda: applyAllWrappers(
            gym.make("ALE/Breakout-v5", render_mode=self.config.get("render_mode", None))
        )
        self.env = DummyVecEnv([env_fn])
        self.model = None

    def build_model(self):
        if self.model_type == "DQN":
            self.model = DQN(
                policy="CnnPolicy",
                env=self.env,
                learning_rate=self.config.get("learning_rate", 1e-4),
                buffer_size=self.config.get("buffer_size", 100_000),
                learning_starts=self.config.get("learning_starts", 50_000),
                batch_size=self.config.get("batch_size", 32),
                gamma=self.config.get("gamma", 0.99),
                target_update_interval=self.config.get("target_update_interval", 10_000),
                train_freq=self.config.get("train_freq", 4),
                exploration_fraction=self.config.get("exploration_fraction", 0.1),
                exploration_final_eps=self.config.get("exploration_final_eps", 0.01),
                verbose=1,
                device=self.device
            )
        elif self.model_type == "PPO":
            self.model = PPO(
                policy="CnnPolicy",
                env=self.env,
                learning_rate=self.config.get("learning_rate", 2.5e-4),
                n_steps=self.config.get("n_steps", 128),
                batch_size=self.config.get("batch_size", 64),
                gamma=self.config.get("gamma", 0.99),
                gae_lambda=self.config.get("gae_lambda", 0.95),
                verbose=1,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown model_type {self.model_type}. Supported: DQN, PPO")
        print(f"{self.model_type} agent created on device: {self.device}")

  

    def train(self):
        save_path = os.path.abspath("models/BreakPoint")
        os.makedirs(save_path, exist_ok=True)

        # --- Load existing model if available ---
        model_file = os.path.join(save_path, f"{self.model_type}_checkpoint_1000000_steps.zip")
        if os.path.exists(model_file):
            print(f"Resuming from {model_file}")
            self.model = self.model.load(model_file, env=self.env, device=self.device)

        # --- Create a checkpoint callback to save every 100,000 steps ---
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000, 
            save_path=save_path,
            name_prefix=f"{self.model_type}_checkpoint"
        )

        # --- Train model ---
        total_timesteps = self.config.get("total_timesteps", 1_000_000)
        print(f"Training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

        # --- Final save ---
        final_model_path = os.path.join(save_path, f"{self.model_type}_agent_final.zip")
        self.model.save(final_model_path)
        print(f"Training complete. Final model saved to {final_model_path}")
