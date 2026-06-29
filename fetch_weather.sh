#!/usr/bin/env bash
set -euo pipefail

for config in qixiang/configs/*.toml; do
    province=$(basename "$config" .toml)
    echo "=== 获取天气：$province ==="
    uv run python -m qixiang --config "$config"
done

echo "=== 所有省份天气已获取完成 ==="
