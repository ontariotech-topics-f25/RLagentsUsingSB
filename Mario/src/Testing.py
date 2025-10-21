import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

# Create the environment
env = gym_super_mario_bros.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human",
    apply_api_compatibility=True
)

# Limit actions to RIGHT_ONLY (move right, jump, etc.)
env = JoypadSpace(env, RIGHT_ONLY)

# Reset the environment (Gymnasium returns observation, info)
observation, info = env.reset()
terminated = truncated = False

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Number of actions:", env.action_space.n)
print("Example action index:", env.action_space.sample())
print("Starting test run...")

while not (terminated or truncated):
    # Always take the "right" action (index 1 in RIGHT_ONLY)
    action = RIGHT_ONLY.index(["right"])
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.2f}, X position: {info.get('x_pos', 'N/A')}")

env.close()
print("✅ Environment closed successfully.")
