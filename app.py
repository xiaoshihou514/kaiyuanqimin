from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kyqm.config import load_config


CONFIG_PATH = Path("kyqm/config.toml")
MODEL_LABELS = {"lightgbm": "LightGBM", "ridge": "Ridge"}
MODEL_COLORS = {"lightgbm": "#60a5fa", "ridge": "#f59e0b"}
FIGURE_BG = "#0f172a"
AXIS_BG = "#111827"
GRID_COLOR = "#334155"
TEXT_COLOR = "#e5e7eb"


@st.cache_data(show_spinner=False)
def _load_long_predictions(
    config_path: Path,
) -> tuple[dict[int, dict[str, pd.DataFrame]], list[str]]:
    cfg = load_config(config_path)
    predictions: dict[int, dict[str, pd.DataFrame]] = {}
    missing_prediction_paths: list[str] = []
    for horizon_days in cfg.long.horizons:
        horizon_predictions: dict[str, pd.DataFrame] = {}
        for model_name, filename_prefix in (
            ("lightgbm", "lgbm"),
            ("ridge", "ridge"),
        ):
            prediction_path = (
                cfg.long.prediction_output_dir
                / f"h{horizon_days}"
                / f"{filename_prefix}_{horizon_days}d_predictions.csv"
            )
            if not prediction_path.exists():
                missing_prediction_paths.append(str(prediction_path))
                continue
            prediction_frame = pd.read_csv(
                prediction_path,
                parse_dates=["date"],
            ).sort_values("date")
            prediction_frame["horizon_days"] = horizon_days
            prediction_frame["model"] = model_name
            horizon_predictions[model_name] = prediction_frame
        if horizon_predictions:
            predictions[horizon_days] = horizon_predictions
    return predictions, missing_prediction_paths


def _render_missing_artifacts() -> None:
    st.error("Long-horizon prediction artifacts are missing.")
    st.code("uv run python -m kyqm --pipeline long --model all", language="bash")
    st.stop()


def _apply_dark_mode() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #020617;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background-color: #0f172a;
        }
        [data-testid="stMarkdownContainer"] {
            color: #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _prediction_chart(
    predictions: dict[int, dict[str, pd.DataFrame]],
    *,
    selected_horizons: list[int],
    selected_models: list[str],
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(selected_horizons),
        1,
        figsize=(10, max(3.6 * len(selected_horizons), 4)),
        sharex=False,
    )
    if len(selected_horizons) == 1:
        axes = [axes]
    fig.patch.set_facecolor(FIGURE_BG)
    for ax, horizon_days in zip(axes, selected_horizons, strict=True):
        ax.set_facecolor(AXIS_BG)
        horizon_predictions = predictions.get(horizon_days, {})
        actual_frame = next(iter(horizon_predictions.values()), None)
        if actual_frame is not None:
            ax.plot(
                actual_frame["date"],
                actual_frame["y_true"],
                color="#f8fafc",
                linewidth=2,
                label="Actual",
            )
        plotted = actual_frame is not None
        for model_name in selected_models:
            frame = horizon_predictions.get(model_name)
            if frame is None:
                continue
            ax.plot(
                frame["date"],
                frame["y_pred"],
                linewidth=2,
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
            if {"y_pred_p10", "y_pred_p90"}.issubset(frame.columns):
                ax.fill_between(
                    frame["date"],
                    frame["y_pred_p10"],
                    frame["y_pred_p90"],
                    color=MODEL_COLORS[model_name],
                    alpha=0.18,
                    label=f"{MODEL_LABELS[model_name]} P10-P90",
                )
            plotted = True
        if not plotted:
            ax.text(
                0.5,
                0.5,
                f"No prediction files available for {horizon_days}d",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=TEXT_COLOR,
            )
        ax.set_title(f"Actual vs prediction ({horizon_days}d horizon)", color=TEXT_COLOR)
        ax.set_ylabel("Price", color=TEXT_COLOR)
        ax.grid(alpha=0.35, color=GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        legend = ax.legend(facecolor=AXIS_BG, edgecolor=GRID_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
    axes[-1].set_xlabel("Target date", color=TEXT_COLOR)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Kaiyuanqimin Long-Horizon Predictions",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_dark_mode()
    st.title("Kaiyuanqimin long-horizon predictions")
    st.caption(
        "Dark-mode view of actual vs predicted prices across multiple held-out long-horizon evaluation samples."
    )

    config_path = CONFIG_PATH
    if not config_path.exists():
        st.error(f"Config file not found: {config_path}")
        st.stop()

    cfg = load_config(config_path)
    if not cfg.long.prediction_output_dir.exists():
        _render_missing_artifacts()

    predictions, missing_prediction_paths = _load_long_predictions(config_path)
    if not predictions:
        _render_missing_artifacts()

    st.sidebar.header("Artifact source")
    st.sidebar.write(f"Predictions: `{cfg.long.prediction_output_dir}`")
    st.sidebar.code("uv run python -m kyqm --pipeline long --model all", language="bash")
    if missing_prediction_paths:
        st.sidebar.caption(
            f"Missing prediction files: {len(missing_prediction_paths)} expected path(s) were not found."
        )

    horizon_choices = [horizon for horizon in cfg.long.horizons if horizon in predictions]
    selected_horizons = st.sidebar.multiselect(
        "Show horizons",
        options=horizon_choices,
        default=horizon_choices,
        format_func=lambda value: f"{value}d",
    )
    available_prediction_models = [
        model_name
        for model_name in ("lightgbm", "ridge")
        if any(model_name in horizon_predictions for horizon_predictions in predictions.values())
    ]
    selected_prediction_models = st.sidebar.multiselect(
        "Prediction traces",
        options=available_prediction_models,
        default=available_prediction_models,
        format_func=lambda value: MODEL_LABELS[value],
    )

    st.subheader("Actual vs prediction on the long-horizon evaluation set")
    if not selected_horizons:
        st.info("Select at least one horizon in the sidebar.")
    elif not selected_prediction_models:
        st.info("Select at least one prediction trace in the sidebar.")
    else:
        st.pyplot(
            _prediction_chart(
                predictions,
                selected_horizons=selected_horizons,
                selected_models=selected_prediction_models,
            ),
            width="stretch",
        )
        st.markdown(
            "Each subplot shows multiple held-out test samples for one long horizon; the shaded band is the LightGBM `P10-P90` interval when available."
        )

    st.sidebar.caption("This app only shows long-term prediction vs actual traces.")


if __name__ == "__main__":
    main()
