import numpy as np;
from gym import Wrapper;
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation;
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import actions as mario_actions
import os
import csv
import datetime
import pygame


class FrameSkipping(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = done or terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info


class PersonaRewardWrapper(Wrapper):
    '''
    Custom reward wrapper to modify rewards based on different personas on top of base reward from environment.
    Personas:
    - speedrunner: rewards for moving right quickly
    - coin_greedy: rewards for collecting coins, penalizes time spent
    - highscore_greedy: rewards based on total score
    '''
    def __init__(self, env, persona="speedrunner"):
        super().__init__(env)
        self.persona = persona
        self.prev_info = None

    def step(self, action):

        obs, base_reward, terminated, truncated, info = self.env.step(action)
        # Base reward from environment
        persona_reward = base_reward

        '''
        Information from the environment to shape rewards
        info dictionary contains:
        "x_pos": current x position of Mario
        "y_pos": current y position of Mario
        "coins": total coins collected
        "score": total score
        "time": time left in the level
        "life: current lives left
        "flag_get": 1 if flag at end of level is reached, else 0
        "status": "small" / "tall" / "fire"
        "world": current world
        
        '''

        
        dx, dcoins, dscore = 0, 0, 0
        if self.prev_info is not None:
            dx = info.get("x_pos", 0) - self.prev_info.get("x_pos", 0)
            dcoins = info.get("coins", 0) - self.prev_info.get("coins", 0)
            dscore = info.get("score", 0) - self.prev_info.get("score", 0)

        if self.persona == "speedrunner":
            # reward for moving right faster
            persona_reward += dx* 0.1
            # time penalty per action to encourage finishing level quickly
            persona_reward -=  0.005

        elif self.persona == "coin_greedy":
            # reward coins 
            persona_reward += dcoins * 0.01
        elif self.persona == "highscore_greedy":
            # reward total score directly
            persona_reward += dscore * 0.005

        if info.get("flag_get", 0) == 1:
            # big bonus for finishing level
            persona_reward += 50.0

        self.prev_info = info
        

        return obs, persona_reward, terminated, truncated, info
    def reset(self, **kwargs):
        self.prev_info = None
        return self.env.reset(**kwargs)

class LoggingWrapper(Wrapper):
    def __init__(self, env, persona="speedrunner", log_dir="data/SuperMario/logs",config=None):
        super().__init__(env)
        self.persona = persona
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.episode_data = []
        self.episode_counter = 0
        self.total_deaths = 0
        self.prev_life = None
        self.config = config or {}

        self.run_label = self.config.get("run_label", "")

        #Timestamp to prevent overwriting logs with same config
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label_suffix = f"_{self.run_label}" if self.run_label else ""
        self.log_file = os.path.join(
            self.log_dir, f"{self.persona}{label_suffix}_{timestamp}_log.csv"
        )
        print(f"Logging to {self.log_file}")
        


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track deaths or failure from timeout
        if terminated or truncated:
            self.total_deaths += 1


        # Track level reached
        level_reached = None
        # Log relevant info
        log_entry = {
            "episode": self.episode_counter,
            "reward": reward,
            "x_pos": info.get("x_pos", 0),
            "y_pos": info.get("y_pos", 0),
            "coins": info.get("coins", 0),
            "score": info.get("score", 0),
            "time": info.get("time", 0),
            "life": info.get("life", 0),
            "flag_get": info.get("flag_get", 0),
            "status": info.get("status", ""),
            "world": info.get("world", ""),
            "total_deaths": self.total_deaths
        }
        self.episode_data.append(log_entry)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.episode_data:
            # Save log to CSV
            keys = self.episode_data[0].keys()
            file_exists = os.path.isfile(self.log_file)
            with open(self.log_file, 'a', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                if not file_exists:
                    dict_writer.writeheader()
                dict_writer.writerows(self.episode_data)
            print(f"Episode {self.episode_counter} completed data logged to {self.log_file}") 
        self.episode_data = []
        self.prev_life = None
        self.episode_counter += 1

        return self.env.reset(**kwargs)


class SequentialMarioEnv:
    """
    Wrapper that takes a list of Super Mario Bros levels and presents them sequentially.
    When Mario reaches the flag at the end of a level, it automatically loads the next level.
    If Mario dies or time runs out or the levels finish the run ends.
    """
    def __init__(self, levels, render_mode=None,action_set="SIMPLE_MOVEMENT"):
        self.levels = levels
        self.current_level_index = 0
        self.render_mode = render_mode
        self._action_sets = {
            "RIGHT_ONLY": mario_actions.RIGHT_ONLY,
            "SIMPLE_MOVEMENT": mario_actions.SIMPLE_MOVEMENT,
            "COMPLEX_MOVEMENT": mario_actions.COMPLEX_MOVEMENT,
        }

        if action_set not in self._action_sets:
            raise ValueError(f"Unknown action_set '{action_set}'. "
                             f"Choose one of {list(self._action_sets.keys())}.")
        self._actions = self._action_sets[action_set]
        self._load_env()
        

    def _load_env(self):
        env_id = self.levels[self.current_level_index]
        print(f"\n Loading level: {env_id}")

        # Quit any existing pygame instances to avoid conflicts
        try:
            pygame.display.quit()
        except Exception:
            pass

        self.env = gym_super_mario_bros.make(env_id, render_mode=self.render_mode, apply_api_compatibility=True)
        self.env = JoypadSpace(self.env, self._actions)
        self.observation, self.info = self.env.reset()

    def reset(self):
        self.current_level_index = 0
        self._load_env()
        return self.observation, self.info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Move to next level if Mario reaches the flag
        if info.get("flag_get", False):
            self.current_level_index += 1
            if self.current_level_index < len(self.levels):
                self.env.close()
                self._load_env()
                print("Level completed — moving to next level!")
                return self.observation, reward, False, False, info
            else:
                print("All levels completed! Episode finished.")
                return obs, reward, True, truncated, info

        # End run if Mario dies or time runs out
        if terminated or truncated:
            print("Mario died or time ran out.")
            return obs, reward, True, truncated, info

        self.observation, self.info = obs, info
        return obs, reward, False, False, info

    def close(self):
        self.env.close()
    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space     


def applyAllWrappers(env, resize, num_stack,num_skip,persona = None,config=None):
    
    #custom wrapper, it help us to avoid similar frames , can skip more frames as wekll
    env = FrameSkipping(env, skip=num_skip)

    #resizxe to make the picture smaller to avoid memory usuage
    env=ResizeObservation(env, (resize,resize))

    #greyscale reduce the amount of memory used as it convert  3 colors channels to 1
    env=GrayScaleObservation(env)

    # lz_compress uses lz4 to compress the frames in memory, which can save a lot of memory when using a large number of frames, 
    # we can stack any number of frames, but 4 is a good number to capture motion
    env=FrameStack(env, num_stack= num_stack, lz4_compress=True)

    #persona reward wrapper, it modify the reward based on the persona we want to use
    env=PersonaRewardWrapper(env, persona=persona)
    env = LoggingWrapper(env, persona=persona,config=config)

    return env