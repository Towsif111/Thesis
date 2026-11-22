#!/usr/bin/env bash
# Push the GUIDE EDA notebook to Kaggle and run it remotely.
# Usage: ./push.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

KERNEL_ID="$(python -c "import json; print(json.load(open('kernel-metadata.json'))['id'])")"

echo "Pushing notebook to Kaggle as $KERNEL_ID ..."
kaggle kernels push -p .

echo "Waiting for the run to start..."
sleep 10

# Poll status until it's no longer queued/running.
while true; do
  STATUS_OUTPUT="$(kaggle kernels status "$KERNEL_ID" 2>&1 || true)"
  echo "$STATUS_OUTPUT"

  if echo "$STATUS_OUTPUT" | grep -qiE 'complete'; then
    echo "Run finished successfully."
    break
  fi
  if echo "$STATUS_OUTPUT" | grep -qiE 'error|cancelled|failed'; then
    echo "Run finished with an error. Check the kernel logs on Kaggle."
    break
  fi

  sleep 15
done

echo "Pulling output notebook and logs..."
kaggle kernels output "$KERNEL_ID" -p ./output

echo "Done. Output saved to ./output"
