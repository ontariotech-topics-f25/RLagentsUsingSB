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