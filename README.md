# Kaiyuanqimin

## Weather retrieval (`qixiang`)

Use `uv` to run weather retrieval with TOML defaults:

```bash
uv run python -m qixiang
```

Defaults come from `qixiang/config.toml`:
- `start_date = 2020-01-01`
- `end_date = 2026-05-21`
- `output_path = data/weather.csv`
- includes a `tqdm` progress bar over configured cities.

Override at runtime:

```bash
uv run python -m qixiang --start-date 2024-01-01 --end-date 2024-01-31 --output-path data/weather_jan_2024.csv
```

Or run as a module:

```bash
uv run python -m qixiang --config qixiang/config.toml
```

## SWYB market price retrieval (`swyb`)

Use `uv` to fetch 商务预报 market prices with TOML defaults:

```bash
uv run python -m swyb
```

Defaults come from `swyb/config.toml`:
- `start_date = 2020-01-01`
- `end_date = 2026-05-21`
- `cate_id = 170130` (黄瓜, explicit id mode)
- `output_path = data/market_prices.csv`
- `failure_policy = partial_with_report`
- includes a `tqdm` progress bar over crawl dates.

Category selection precedence:
1. `cate_id` (explicit, highest priority)
2. `category_path` (e.g. `蔬菜-黄瓜`)
3. `category_group + category_name` (e.g. `蔬菜` + `茄子`)

Examples:

```bash
# Explicit cate_id
uv run python -m swyb --cate-id 170130 --start-date 2024-01-01 --end-date 2024-01-07 --output-path data/market_prices_huanggua.csv

# Category path
uv run python -m swyb --category-path 蔬菜-茄子 --start-date 2024-01-01 --end-date 2024-01-07 --output-path data/market_prices_qiezi.csv
uv run python -m swyb --category-path 粮油-面粉 --start-date 2024-01-01 --end-date 2024-01-07 --output-path data/market_prices_mianfen.csv

# Group + name
uv run python -m swyb --category-group 蔬菜 --category-name 黄瓜 --start-date 2024-01-01 --end-date 2024-01-07 --output-path data/market_prices_huanggua.csv
```

Fetch **all daily categories** with multithreading and save into separate CSV files:

```bash
uv run python -m swyb --all --start-date 2024-01-01 --end-date 2024-01-07 --output-path data/swyb_all
```

- `--all` writes one CSV per category (e.g. `...__shucai__黄瓜__170130.csv`).
- Runs category jobs concurrently (threaded) for faster full-category retrieval.

Override at runtime:

```bash
uv run python -m swyb --start-date 2020-01-01 --end-date 2026-05-21 --output-path data/market_prices.csv
```

Run with a custom config file:

```bash
uv run python -m swyb --config swyb/config.toml
```

## Multi-model training (`kyqm`)

`kyqm` now supports multiple forecasting models with shared feature engineering:
- LightGBM (point + quantile P10/P90)
- GRU + attention (point/median + quantile P10/P90)
- Prophet baseline (point forecast)

Run all models:

```bash
uv run python -m kyqm --config kyqm/config.toml --model all
```

Run a single model:

```bash
uv run python -m kyqm --model lgbm
uv run python -m kyqm --model gru
uv run python -m kyqm --model prophet
```

Helpful overrides (mainly for GRU smoke/debug runs):

```bash
uv run python -m kyqm --model gru --epochs 5 --batch-size 16 --learning-rate 0.001 --device cpu
```

Main outputs:
- `data/cleaned_data.csv`
- `data/feature_data.csv`
- `data/predictions/*.csv` (per-model predictions)
- `data/model_summary.json` (cross-model metrics summary)
- `data/models/lgbm/*`, `data/models/gru/*`, `data/models/prophet/*`
