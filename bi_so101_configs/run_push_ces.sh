#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TMPDIR="$SCRIPT_DIR/tmp"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
DATA_ROOT="$SCRIPT_DIR/datasets"
REPO_ID="xhaka3456/ces"
source /home/stream/miniconda3/bin/activate xlerobot >/dev/null
TMPDIR="$TMPDIR" python - <<PYTHON_EOF
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root = Path("$DATA_ROOT")
repo = "$REPO_ID"
ds = LeRobotDataset(repo, root=root)
ds.push_to_hub(private=False)
PYTHON_EOF
