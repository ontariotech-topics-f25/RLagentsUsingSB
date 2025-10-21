from gym import Wrapper

class SeedSafeEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        return obs, reward, terminated, truncated, info
