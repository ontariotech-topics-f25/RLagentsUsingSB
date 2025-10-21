from nes_py.nes_env import NESEnv

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


import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import actions as mario_actions
from stable_baselines3.common.vec_env import DummyVecEnv,VecMonitor

from .wrappers import FrameSkipping, PersonaRewardWrapper, LoggingWrapper, SequentialMarioEnv, SeedSafeEnv

from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

from gym import Wrapper

class CompatibilityWrapper(Wrapper):
    """Force all envs to return 5-tuple (obs, reward, terminated, truncated, info)"""
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


def make_mario_env(
    levels=None,
    persona="speedrunner",
    resize=84,
    num_stack=4,
    num_skip=4,
    render_mode=None,
    action_set="SIMPLE_MOVEMENT",
    config=None,
):
    """
    Factory function to create a Mario environment ready for SB3 training.
    Automatically wraps with preprocessing, reward shaping, and logging.
    """

    # --- Base environment ---
    if levels is None:
        # Single-level mode
        base_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        base_env = JoypadSpace(base_env, getattr(mario_actions, action_set))
        base_env = CompatibilityWrapper(base_env)  # <--- ensures 5-tuple API
    else:
        # Sequential multi-level mode
        base_env = SequentialMarioEnv(levels, render_mode=render_mode, action_set=action_set)
        base_env = CompatibilityWrapper(base_env)  # <--- ensures 5-tuple API

    # --- Apply wrappers (preprocessing) ---
    env = FrameSkipping(base_env, skip=num_skip)
    env = ResizeObservation(env, (resize, resize))
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=num_stack, lz4_compress=True)
    env = PersonaRewardWrapper(env, persona=persona)
    env = LoggingWrapper(env, persona=persona, algo=config.get("algo", "unknown"), config=config)

    # --- Seed safety wrapper ---
    env = SeedSafeEnv(env)

    # --- Wrap for SB3 compatibility ---
    env = DummyVecEnv([lambda: env])

    # --- Monitor wrapper for logging ---
    env = VecMonitor(env)

    return env
