from stable_baselines3 import PPO, DQN

def train_agent(env, algo="ppo", total_timesteps=1_000_000, save_path=None, log_dir=None):
    """
    Trains a given algorithm (PPO or DQN) on the provided environment.
    """
    if algo.lower() == "ppo":
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)
    elif algo.lower() == "dqn":
        model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)
    else:
        raise ValueError(f"Unsupported algo '{algo}'. Choose 'ppo' or 'dqn'.")

    model.learn(total_timesteps=total_timesteps)
    if save_path:
        model.save(save_path)
        print(f"✅ Model saved at {save_path}")
    env.close()
    return model