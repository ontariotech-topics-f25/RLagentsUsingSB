import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import actions as mario_actions
import pygame
from gym import Wrapper


class CompatibilityWrapper(Wrapper):
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
            return obs, reward, terminated, truncated, info
        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            return result
        return result, {}


class SequentialMarioEnv(Wrapper):
    def __init__(self, levels, render_mode=None, action_set="SIMPLE_MOVEMENT"):
        self.levels = levels
        self.current_level_index = 0
        self._render_mode = render_mode
        self._action_sets = {
            "RIGHT_ONLY": mario_actions.RIGHT_ONLY,
            "SIMPLE_MOVEMENT": mario_actions.SIMPLE_MOVEMENT,
            "COMPLEX_MOVEMENT": mario_actions.COMPLEX_MOVEMENT,
        }
        if action_set not in self._action_sets:
            raise ValueError(f"Unknown action_set '{action_set}'.")
        self._actions = self._action_sets[action_set]

        env = gym_super_mario_bros.make(levels[0])
        env = JoypadSpace(env, self._actions)
        env = CompatibilityWrapper(env)
        super().__init__(env)
        self._load_env()

    def _load_env(self):
        env_id = self.levels[self.current_level_index]
        print(f"\n Loading level: {env_id}")
        try:
            pygame.display.quit()
        except Exception:
            pass

        self.env = gym_super_mario_bros.make(env_id)
        self.env = JoypadSpace(self.env, self._actions)
        self.env = CompatibilityWrapper(self.env)

        reset_output = self.env.reset()
        if isinstance(reset_output, tuple):
            self.observation, self.info = reset_output
        else:
            self.observation, self.info = reset_output, {}

    def reset(self, **kwargs):
        self.current_level_index = 0
        self._load_env()
        return self.observation, self.info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result

        info["current_level"] = self.current_level_index + 1
        info["level_id"] = self.levels[self.current_level_index]

        if info.get("flag_get", False):
            self.current_level_index += 1
            if self.current_level_index < len(self.levels):
                print(f"Completed {self.levels[self.current_level_index - 1]} → Next level...")
                self.env.close()
                self._load_env()
                return self.observation, reward, False, False, info
            else:
                print(f"All {len(self.levels)} levels completed!")
                return obs, reward, True, truncated, info

        if terminated or truncated:
            print(f"Mario died in {self.levels[self.current_level_index]}")
            return obs, reward, True, truncated, info

        self.observation, self.info = obs, info
        return obs, reward, False, False, info

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
