#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lerobot-record --config "$SCRIPT_DIR/scripts/bi_so101_record.yaml"
