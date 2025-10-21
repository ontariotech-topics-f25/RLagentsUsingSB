from gym import Wrapper
import os, csv, datetime

class LoggingWrapper(Wrapper):
    def __init__(self, env, persona="speedrunner", algo="ppo", log_dir="data/logs", config=None):
        super().__init__(env)
        self.persona = persona
        self.algo = algo
        self.config = config or {}
        self.episode_data = []
        self.episode_counter = 0
        self.total_deaths = 0
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.log_dir = os.path.join(project_root, "Mario", log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label_suffix = f"_{self.config.get('run_label', '')}" if self.config.get("run_label") else ""
        self.log_file = os.path.join(self.log_dir, f"{self.persona}_{self.algo}{label_suffix}_{timestamp}_log.csv")
        print(f"Logging to {self.log_file}")

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        if terminated or truncated:
            self.total_deaths += 1
        self.episode_data.append({
            "episode": self.episode_counter,
            "reward": reward,
            "x_pos": info.get("x_pos", 0),
            "coins": info.get("coins", 0),
            "score": info.get("score", 0),
            "time": info.get("time", 0),
            "life": info.get("life", 0),
            "flag_get": info.get("flag_get", 0),
            "world": info.get("world", ""),
            "total_deaths": self.total_deaths
        })
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.episode_data:
            keys = self.episode_data[0].keys()
            file_exists = os.path.isfile(self.log_file)
            with open(self.log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.episode_data)
            print(f"Episode {self.episode_counter} logged → {self.log_file}")
        self.episode_data = []
        self.episode_counter += 1
        return self.env.reset(**kwargs)
