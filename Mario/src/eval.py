import os
import sys
import argparse
import warnings

# --- Add project root to path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
print(f"[DEBUG] Project root added to sys.path: {ROOT_DIR}")

# --- Imports ---
try:
    import gymnasium as gym
except ImportError:
    import gym
from stable_baselines3 import PPO
from Agents.AgentDDQ import AgentDDQ

try:
    from utils import load_config
except ImportError:
    warnings.warn("utils.load_config not found. Using default parameters.")

import gym_super_mario_bros
from nes_py.nes_env import NESEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import actions as mario_actions

# --- NES-Py 5-tuple patch ---
if not hasattr(NESEnv, "_patched_step"):
    old_step = NESEnv.step
    def patched_step(self, action):
        result = old_step(self, action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result
    NESEnv.step = patched_step
    NESEnv._patched_step = True
    print("[PATCH] NES-Py environments now return 5-tuple (obs, reward, terminated, truncated, info)")

# --- Compatibility wrapper for safety ---
class CompatibilityWrapper(gym.Wrapper):
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            return result
        return result, {}

# --- Sequential levels wrapper ---
class SequentialLevelsWrapper:
    def __init__(self, levels, action_set="RIGHT_ONLY", render_mode="human"):
        self.levels = levels
        self.idx = 0
        self.render_mode = render_mode
        self.action_set = getattr(mario_actions, action_set)
        self.env = self.make_env(self.levels[self.idx])

    def make_env(self, level_id):
        env = gym_super_mario_bros.make(level_id, render_mode=self.render_mode)
        env = JoypadSpace(env, self.action_set)
        env = CompatibilityWrapper(env)
        return env

    def reset(self):
        if hasattr(self.env, 'close'):
            self.env.close()
        self.env = self.make_env(self.levels[self.idx])
        obs, info = self.env.reset()
        self.idx = (self.idx + 1) % len(self.levels)
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

# --- Make Mario environment factory ---
def make_mario_env(levels=None, action_set="RIGHT_ONLY", render_mode=None):
    if isinstance(levels, list):
        return SequentialLevelsWrapper(levels, action_set, render_mode=render_mode)
    return CompatibilityWrapper(JoypadSpace(gym_super_mario_bros.make(levels, render_mode=render_mode), getattr(mario_actions, action_set)))

# --- Command-line arguments ---
parser = argparse.ArgumentParser(description="Evaluate Mario RL agent")
parser.add_argument("--algo", type=str, default="ppo", help="RL algorithm (ppo, ddqn, dqn)")
parser.add_argument("--persona", type=str, default="collector")
parser.add_argument("--config", type=str, default='Mario/configs/ppo_config.yaml', help="Path to YAML config")
parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
parser.add_argument("--render_mode", type=str, default="human", help="human or rgb_array")
parser.add_argument("--levels", type=str, nargs='+', default=None, help="Levels (single or multiple)")
args = parser.parse_args()

ALGO = args.algo.lower()
PERSONA = args.persona
MODEL_PATH = args.model_path
LEVELS = args.levels if args.levels else "SuperMarioBros-1-1-v0"

# --- Load config ---
config = {}
if os.path.exists(args.config):
    config = load_config(os.path.abspath(args.config))
    print(f"Loaded config from: {args.config}")
else:
    warnings.warn(f"Config file {args.config} not found. Using defaults.")

# --- Default environment parameters ---
resize = config.get("resize", 84)
num_stack = config.get("num_stack", 4)
num_skip = config.get("num_skip", 4)
action_set = config.get("action_set", "RIGHT_ONLY")

# --- Model path ---
if MODEL_PATH is None:
    MODEL_DIR = os.path.join(ROOT_DIR, "Mario/models/checkpoints")
    MODEL_PATH = os.path.join(MODEL_DIR, f"{PERSONA}_{ALGO}_checkpoint_500000_steps.zip")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# --- Create environment ---
env = make_mario_env(levels=LEVELS, action_set=action_set, render_mode=args.render_mode)

# --- Load agent/model ---
if ALGO == "ppo":
    model = PPO.load(MODEL_PATH, env=env)
elif ALGO in ["ddqn", "dqn"]:
    agent = AgentDDQ(epsilon=0.0)
    agent.load_model(load_path=MODEL_PATH)
else:
    raise ValueError(f"Unsupported algorithm: {ALGO}")

# --- Evaluation loop ---
state, info = env.reset()
terminated = truncated = False
total_reward = 0
step_count = 0

while not terminated:
    if ALGO == "ppo":
        action, _ = model.predict(state, deterministic=True)
    else:
        action = agent.next_action(state, deterministic=True)

    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    total_reward += reward
    step_count += 1

print(f"Episode finished — Total Reward: {total_reward}, Steps: {step_count}")
env.close()
