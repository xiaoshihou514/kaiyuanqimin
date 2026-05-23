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
- cached requests are stored under `.cache/swyb`
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
- LSTM + attention (point forecast)
- Prophet baseline (point forecast)

Province and city filtering for market data now use `cpca` parsing from both `market` and `county_name`, with no hardcoded `area_code` province table. `city_name` is optional in `kyqm/config.toml`; when set, only rows with a confidently resolved matching city are retained. `新疆生产建设兵团` rows are still merged into `新疆` totals.

`kyqm` now trains on an explicit **1-day-ahead** target by default (`forecast_horizon = 1`). The model summary also includes a `naive_last_price` baseline for comparison.

The prepared daily table can still include companion-product, nearby-province, holiday, and 4-city-average weather signals, plus the newer **Beijing cucumber lead** and refined weather-event candidate features. The default LightGBM path is still intentionally conservative: it learns a **residual over the naive last-price baseline** from the local autoregressive features (`lag_*`, `roll_*`, `price_diff_*`). That remained the most stable setup on the current small sample, so the extra Beijing/weather feature groups are available behind config toggles instead of enabled by default.

Optional LightGBM experiment toggles in `kyqm/config.toml`:

```toml
[lgbm]
use_beijing_lead_features = true
use_refined_weather_features = true
```

Run all models:

```bash
uv run python -m kyqm --config kyqm/config.toml --model all
```

Run a single model:

```bash
uv run python -m kyqm --model lgbm
uv run python -m kyqm --model gru
uv run python -m kyqm --model lstm
uv run python -m kyqm --model prophet
```

Optional city filter in `kyqm/config.toml`:

```toml
[data]
province_name = "山东省"
city_name = "淄博"
product_name = "黄瓜"
forecast_horizon = 1
```

Helpful overrides (for GRU/LSTM smoke or debug runs):

```bash
uv run python -m kyqm --model gru --epochs 5 --batch-size 16 --learning-rate 0.001 --device cpu
uv run python -m kyqm --model lstm --epochs 5 --batch-size 16 --learning-rate 0.001 --device cpu
```

Main outputs:
- `data/cleaned_data.csv`
- `data/feature_data.csv`
- `data/predictions/*.csv` (per-model predictions)
- `data/model_summary.json` (cross-model metrics summary, including `naive_last_price`)
- `data/models/lgbm/*`, `data/models/gru/*`, `data/models/lstm/*`, `data/models/prophet/*`

### Long-horizon comparison path

Use the separate long-horizon pipeline to compare `1d / 7d / 30d / 90d` behavior:

```bash
uv run python -m kyqm --pipeline long --model all
```

This path keeps the existing 1-day builder intact and adds:
- weekly anchor samples with a 90-day history window
- training-only `±3 day` anchor jitter augmentation
- long-horizon LightGBM + Ridge comparisons
- horizon-aware naive baselines (`current_price` for 7d, `seasonal_last_year` for 30d/90d)

Main long-horizon outputs:
- `data/feature_data_long.csv`
- `data/model_summary_long.json`
- `data/model_comparison_long.csv`
- `data/predictions/long/**`
- `data/models/long/lgbm/**`
- `data/models/long/ridge/**`

Launch the Streamlit long-horizon app after generating those artifacts:

```bash
uv run streamlit run app.py
```

The Streamlit app is the primary visualization path for long-horizon analysis and now focuses only on:
- **实际值 vs 预测值**：深色模式下展示 `7d / 30d / 90d` 三个周期在完整评估集（验证集 + 测试集）上的预测曲线，包含 LightGBM 与 Ridge

Current validated long-horizon outcome on the repo data:
- `1d`: LightGBM is still effectively tied with / slightly worse than naive
- `7d`: models still do not beat the current-price naive baseline
- `30d`: LightGBM and Ridge both beat the seasonal naive baseline, with LightGBM stronger
- `90d`: both still beat the seasonal naive baseline, again with LightGBM stronger
