# Kaiyuanqimin

## Weather retrieval (`qixiang`)

Use `uv` to run weather retrieval with TOML defaults:

```bash
uv run python -m qixiang
```

Defaults come from `qixiang/config.toml`:
- `start_date = 2022-01-01`
- `end_date = 2026-04-30`
- `output_path = data/weather.csv`

Override at runtime:

```bash
uv run python -m qixiang --start-date 2024-01-01 --end-date 2024-01-31 --output-path data/weather_jan_2024.csv
```

Or run as a module:

```bash
uv run python -m qixiang --config qixiang/config.toml
```
