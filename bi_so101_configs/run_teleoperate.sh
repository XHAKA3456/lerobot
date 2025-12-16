#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lerobot-teleoperate --config "$SCRIPT_DIR/scripts/bi_so101_teleoperate.yaml"
