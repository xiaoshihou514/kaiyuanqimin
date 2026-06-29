#!/usr/bin/env bash
set -euo pipefail

configs=()
for f in configs/*.toml; do
    configs+=("$f")
done

echo "=== 共 ${#configs[@]} 个训练任务 ==="

for config in "${configs[@]}"; do
    name=$(basename "$config" .toml)
    echo "=== [short] $name ==="
    uv run python -m kyqm --config "$config" --pipeline short --model all
    echo "=== [long]  $name ==="
    uv run python -m kyqm --config "$config" --pipeline long --model all
done

echo "=== 全部训练完成 ==="
