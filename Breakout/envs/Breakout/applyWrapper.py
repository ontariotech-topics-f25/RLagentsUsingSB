import os
import csv
import datetime
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


import pygame

class FrameSkipping(Wrapper):
    """Custom frame skipping wrapper."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward +=  reward[0] if isinstance(reward, (list, np.ndarray)) else reward

            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class LazyFrameToNumpy(gym.ObservationWrapper):
    def observation(self, observation):
        if hasattr(observation, "__array__"):
            observation = np.array(observation)
        return observation

class LoggingWrapper(Wrapper):
    """Logs episode reward, lives, frame number, etc."""
    def __init__(self, env, log_dir="data/Breakout/logs", config=None, persona=None):
        super().__init__(env)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.episode_data = []
        self.episode_counter = 0
        self.config = config or {}
        self.persona = persona
        self.run_label = self.config.get("run_label", "")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label_suffix = f"_{self.run_label}" if self.run_label else ""
        self.log_file = os.path.join(
            self.log_dir, f"{self.persona}{label_suffix}_{timestamp}_log.csv"
        )
        print(f"Logging to {self.log_file}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        lives = info.get("lives", None)

        log_entry = {
            "episode": self.episode_counter,
            "reward": reward,
            "lives": lives,
            "frame_number": info.get("frame_number", None),
        }
        self.episode_data.append(log_entry)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.episode_data:
            keys = self.episode_data[0].keys()
            file_exists = os.path.isfile(self.log_file)
            with open(self.log_file, "a", newline="") as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                if not file_exists:
                    dict_writer.writeheader()
                dict_writer.writerows(self.episode_data)
            print(f"Episode {self.episode_counter} logged to {self.log_file}")

        self.episode_data = []
        self.episode_counter += 1
        return self.env.reset(**kwargs)


def applyAllWrappers(env, resize=84, num_stack=4, num_skip=4, persona=None, config=None):
    """Applies preprocessing and logging wrappers to the Breakout environment."""
    env = FrameSkipping(env, skip=num_skip)
    env = ResizeObservation(env, (resize, resize))
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=num_stack, lz4_compress=True)
    env = LazyFrameToNumpy(env)  
    env = LoggingWrapper(env, persona=persona, config=config)
    return env


def create_breakout_env(render_mode=None, persona="Standard", config=None):
    """Creates the Breakout environment with all wrappers applied."""
    try:
        pygame.display.quit()
    except Exception:
        pass

    env = gym.make("Breakout-v5", render_mode=render_mode)
    env = applyAllWrappers(env, persona=persona, config=config)
    return env
