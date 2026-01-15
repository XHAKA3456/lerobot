#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG="$SCRIPT_DIR/scripts/bi_so101_record_policy.yaml"
if [ $# -eq 0 ]; then
  lerobot-record --config "$CONFIG" --policy.path=xhaka3456/bi_so101_screw_the_cap_on3 --policy.n_action_steps=50
else
  lerobot-record --config "$CONFIG" "$@"
fi
