#!/bin/bash
# === Automated overnight training script for PPO & DQN on Mario ===
# Total runtime target: ~4.5 hours
# Each run gets ~1h07m (100k timesteps)

source venv/bin/activate

LOG_DIR="Mario/data/run_logs"
mkdir -p $LOG_DIR

declare -a ALGOS=("ppo" "dqn")
declare -a PERSONAS=("speedrunner" "coin_greedy")

for algo in "${ALGOS[@]}"; do
  for persona in "${PERSONAS[@]}"; do
    CONFIG_PATH="Mario/configs/${algo}_config.yaml"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    echo "========================================"
    echo "🚀 Starting training: $algo — $persona ($TIMESTAMP)"
    echo "========================================"

    python Mario/src/train.py \
      --algo "$algo" \
      --persona "$persona" \
      --config "$(basename $CONFIG_PATH)" \
      --run_label "$persona"_"$algo"_"$TIMESTAMP" \
      > "$LOG_DIR/${persona}_${algo}_${TIMESTAMP}.out" 2>&1

    echo "✅ Completed: $algo — $persona ($TIMESTAMP)"
    echo "----------------------------------------"
    echo
  done
done

echo "🎯 All training runs completed!"
