import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
csv_path = "/home/amandeepsingh/Sem7/Projects/TopicsOfCS1/Assignment1/TestAssignment1/RLagentsUsingSB/Mario/data/logs/speedrunner_dqn_20251025_040700_661460_log.csv"
df = pd.read_csv(csv_path)

# Make sure there's a frame_number column
df["frame_number"] = df.get("frame_number", df.index)

# Rolling average window (number of frames)
window = 200  # increase for smoother curve

# Compute rolling average of reward
df["reward_smoothed"] = df["reward"].rolling(window=window, min_periods=1).mean()

# Plot
plt.figure(figsize=(12,6))
plt.plot(df["frame_number"], df["reward_smoothed"], color='blue')
plt.title(f"Smoothed Reward (rolling window={window} frames)")
plt.xlabel("Frame Number")
plt.ylabel("Reward (smoothed)")
plt.grid(True)
plt.show()
