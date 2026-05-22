# Copilot Instructions

## Running modules

All commands use `uv run`:

```bash
# Fetch weather data (Shandong 4-city archive) with defaults from qixiang/config.toml
uv run python -m qixiang

# Override dates or output path at runtime
uv run python -m qixiang --start-date 2024-01-01 --end-date 2024-12-31 --output-path data/weather.csv

# Use a different config file
uv run python -m qixiang --config path/to/config.toml
```

No linter, build, or test commands are configured yet.

## High-level architecture
The project is organized as a data pipeline for agricultural price forecasting, with top-level modules mapped to pipeline stages:

- `swyb/`: scrape Ministry of Commerce price bulletin data.
- `ncpscxx/`: scrape national agricultural market platform data.
- `qixiang/`: fetch weather data.
- `kyqm/`: data fusion, feature engineering, model training, and prediction.

Cross-module flow expected by current project docs:

1. Collect market/national/weather data.
2. Align by date and merge into a province+product daily table.
3. Build time-series features and train an LSTM+attention model.
4. Serve short-horizon forecasts (1-7 days) and expose attention/policy-signal views.

## Key conventions in this repository
- `qixiang/config.toml` controls default date range (`start_date`, `end_date`) and output path (`output_path`); override at runtime with CLI flags.
- All data artifacts live under `data/` (gitignored). Cache from Open-Meteo responses lives under `.cache/` (also gitignored).
- Keep each top-level directory focused on one stage of the pipeline (crawler modules vs. model module); avoid mixing scraping and modeling logic in the same package.
- Preserve the canonical merged schema used across stages: `date`, `local_price`, `national_price`, `temp_avg`, `precip`, and reserved policy/sentiment feature (`sentiment_score` / `policy_impact` placeholder).
- Use time-ordered dataset splits (train/val/test) to prevent future-data leakage; do not random-shuffle time-series samples.
- Keep data artifacts and naming consistent with the current project plan (`market_prices.csv`, `national_price.csv`, `weather.csv`, `cleaned_data.csv`, `model.pth`) unless a repository-wide rename is applied.
