import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from envs.Breakout.agent import RLAgent
# Path to your config file
config_path = "../../configs/BreakPoint/BreakPoint.yaml"

# Initialize the agent
agent = RLAgent(config_path)

# Build the model (DQN or PPO depending on config)
agent.build_model()

# Start training
agent.train()
