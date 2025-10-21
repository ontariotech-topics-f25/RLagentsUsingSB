from gym import Wrapper

class FrameSkipping(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            result = self.env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
                terminated, truncated = done, False
            else:
                obs, reward, terminated, truncated, info = result
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
