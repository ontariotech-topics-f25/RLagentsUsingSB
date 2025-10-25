from gym import Wrapper
import numpy as np

class PersonaRewardWrapper(Wrapper):
    """
    Reward shaping for Super Mario Bros.
    Two personas:
    - "speedrunner": Rewards fast, efficient progress.
    - "collector": Rewards coins, exploration, and survival.
    """
    def __init__(self, env, persona="speedrunner"):
        super().__init__(env)
        if persona not in ["speedrunner", "collector"]:
            raise ValueError("Persona must be either 'speedrunner' or 'collector'")
        self.persona = persona
        self.prev_info = None
        self.max_reward = 15.0
        self.min_reward = -10.0
        self.time_penalty_threshold = 250 if persona == "speedrunner" else 150
        self.last_max_x = 0
        self.checkpoints = {100: False, 200: False, 300: False}
        self.stage_completion_bonus = 100.0
        self.current_stage = None
        self.consecutive_coins = 0
        self.last_coin_x = 0

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        persona_reward = 0.0

        dx, dcoins, dscore = 0, 0, 0
        if self.prev_info is not None:
            # Reset deltas if world changes
            if self.prev_info.get("world", "") != info.get("world", ""):
                self.prev_info = info
                self.last_max_x = 0

            # Compute progress
            dx = max(0, info.get("x_pos", 0) - self.prev_info.get("x_pos", 0))
            dcoins = info.get("coins", 0) - self.prev_info.get("coins", 0)
            dscore = info.get("score", 0) - self.prev_info.get("score", 0)
            current_x = info.get("x_pos", 0)

            # New territory reward
            if current_x > self.last_max_x:
                self.last_max_x = current_x
                persona_reward += 1.0

                # Checkpoint bonuses
                for checkpoint in sorted(self.checkpoints.keys()):
                    if not self.checkpoints[checkpoint] and current_x >= checkpoint:
                        self.checkpoints[checkpoint] = True
                        persona_reward += 5.0

            # Stage completion check
            current_stage = info.get("world", "")
            if current_stage != self.current_stage and self.current_stage is not None:
                persona_reward += self.stage_completion_bonus
                self.checkpoints = {100: False, 200: False, 300: False}
            self.current_stage = current_stage

        # Persona-specific reward schemes
        if self.persona == "speedrunner":
            distance_reward = dx * 0.6
            if dx > 2:
                distance_reward *= 1.5
            if dx > 5:
                distance_reward *= 1.8

            coin_reward = dcoins * 0.5

        else:  # collector
            distance_reward = dx * 0.4
            if dx > 2:
                distance_reward *= 1.2

            if dcoins > 0:
                self.consecutive_coins += dcoins
                coin_reward = dcoins * 2.0
                if abs(info.get("x_pos", 0) - self.last_coin_x) > 50:
                    coin_reward *= 1.5
                if self.consecutive_coins > 2:
                    coin_reward *= (1 + min(self.consecutive_coins * 0.1, 0.5))
                self.last_coin_x = info.get("x_pos", 0)
            else:
                self.consecutive_coins = 0
                coin_reward = 0.0

        # Time efficiency
        current_time = info.get("time", 400)
        prev_time = self.prev_info.get("time", current_time) if self.prev_info else current_time
        time_spent = prev_time - current_time

        if dx > 0 and time_spent > 0:
            efficiency = dx / time_spent
            if efficiency < 0.3:
                time_bonus = -0.5
            elif efficiency < 1.0:
                time_bonus = 0.2
            else:
                time_bonus = 0.5
        else:
            time_bonus = -0.5

        if current_time > self.time_penalty_threshold:
            time_bonus -= 0.001 * (current_time - self.time_penalty_threshold)

        persona_reward += distance_reward + coin_reward + time_bonus

        # Survival
        if terminated:
            persona_reward -= 5.0
        elif truncated:
            persona_reward -= 2.0

        # Flag reward
        if info.get("flag_get", 0) == 1:
            persona_reward += 50.0
            persona_reward += min(15.0, info.get("time", 0) * 0.05)
            persona_reward += min(10.0, info.get("coins", 0) * 0.3)

        persona_reward = np.clip(persona_reward, self.min_reward, self.max_reward)

        self.prev_info = info
        return obs, persona_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_info = None
        self.consecutive_coins = 0
        self.last_coin_x = 0
        return self.env.reset(**kwargs)
