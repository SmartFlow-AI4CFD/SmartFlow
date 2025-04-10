#!/bin/bash
while true; do
  wandb sync --sync-all
  echo "Synced at $(date)"
  sleep 10  # Sync every hour
done

# ./sync_wandb.sh > sync_wandb.log 2>&1 & disown