from gym import Wrapper

class PersonaRewardWrapper(Wrapper):
    """
    Custom reward shaping based on different personas.
    Personas:
    - speedrunner: rewards moving right quickly.
    - coin_greedy: rewards collecting coins.
    - highscore_greedy: rewards increasing total score.
    """
    def __init__(self, env, persona="speedrunner"):
        super().__init__(env)
        self.persona = persona
        self.prev_info = None

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        persona_reward = base_reward

        dx, dcoins, dscore = 0, 0, 0
        if self.prev_info is not None:
            # Reset deltas if world changes
            if self.prev_info.get("world", "") != info.get("world", ""):
                self.prev_info = info
            dx = max(0, info.get("x_pos", 0) - self.prev_info.get("x_pos", 0))
            dcoins = info.get("coins", 0) - self.prev_info.get("coins", 0)
            dscore = info.get("score", 0) - self.prev_info.get("score", 0)

        # Persona-based logic
        if self.persona == "speedrunner":
            persona_reward += dx * 0.1
            persona_reward -= 0.005
        elif self.persona == "coin_greedy":
            persona_reward += dcoins * 0.1
        elif self.persona == "highscore_greedy":
            persona_reward += dscore * 0.01

        if info.get("flag_get", 0) == 1:
            persona_reward += 50.0

        self.prev_info = info
        return obs, persona_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_info = None
        return self.env.reset(**kwargs)
