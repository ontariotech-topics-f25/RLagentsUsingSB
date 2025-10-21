from envs.Breakout.agent import RLAgent

# Path to your config file
config_path = "configs/breakout_config.yaml"

# Initialize the agent
agent = RLAgent(config_path)

# Build the model (DQN or PPO depending on config)
agent.build_model()

# Start training
agent.train()
