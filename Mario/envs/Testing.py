import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human', apply_api_compatibility=True)

#JoypadSpace limits the actions that can be taken. Here, we limit to only moving right., it's a wrapper for the env. wrapper are something with inteact with the env 
# to modify its behavior., like it can ask enviroment to rlimti actiopns or ,modify reward or get screen in a certain way
env = JoypadSpace(env, RIGHT_ONLY)


observation=env.reset()# Gymnasium returns 2 values now
done=False
env.reset()

while not done:
    action = RIGHT_ONLY.index(['right'])  # Move right
    observation, reward, done, terminated, info = env.step(action)
    print("Observation space:", env.observation_space)
    print("Observation space shape:", env.observation_space.shape)
    print("Observation space type:", env.observation_space.dtype)
    print("Action space:", env.action_space)
    print("Number of actions:", env.action_space.n)
    print("Example action:", env.action_space.sample())
    print("Reward received:", reward)



env.close()
