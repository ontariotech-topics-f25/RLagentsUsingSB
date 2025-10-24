#!/bin/bash
# Automated overnight training script for PPO & DQN on Mario
# Runs two personas per algorithm in parallel

source venv/bin/activate

LOG_DIR="Mario/data/run_logs"
mkdir -p $LOG_DIR

declare -a ALGOS=("ppo" "dqn")
declare -a PERSONAS=("speedrunner" "coin_greedy")

for algo in "${ALGOS[@]}"; do
  echo "========================================"
  echo "Starting parallel runs for: $algo"
  echo "========================================"

  for persona in "${PERSONAS[@]}"; do
    CONFIG_PATH="Mario/configs/${algo}_config.yaml"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    echo "Launching training: $algo — $persona ($TIMESTAMP)"

    python Mario/src/train.py \
      --algo "$algo" \
      --persona "$persona" \
      --config "$(basename $CONFIG_PATH)" \
      --run_label "${persona}_${algo}_${TIMESTAMP}" \
      --load_model \
      > "$LOG_DIR/${persona}_${algo}_${TIMESTAMP}.out" 2>&1 &
  done

  # Wait for both personas of this algo to finish before next algo
  wait
  echo "Completed all runs for: $algo"
  echo "----------------------------------------"
done

echo "All training runs completed!"
