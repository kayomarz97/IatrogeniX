#!/bin/bash
# IatrogeniX — scripts/start_background_ablation.sh
# Sets up a tmux session for the 4-way ablation study.

SESSION_NAME="iatrogenix-ablation"

# 1. Create session if it doesn't exist
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  echo "Creating new tmux session: $SESSION_NAME"
  
  # Start session detached
  tmux new-session -d -s $SESSION_NAME
  
  # Pane 1: Run the Ablation Runner (Modes B, C, D)
  tmux send-keys -t $SESSION_NAME "python3 scripts/safety_ablation.py" C-m
  
  # Split window for the Dashboard
  tmux split-window -h -t $SESSION_NAME
  tmux send-keys -t $SESSION_NAME "python3 scripts/ablation_dashboard.py" C-m
  
  echo "Ablation Runner started in background."
else
  echo "Session $SESSION_NAME already exists."
fi

echo "----------------------------------------------------"
echo "To monitor the study, run:"
echo "  tmux attach -t $SESSION_NAME"
echo "----------------------------------------------------"
