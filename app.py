from __future__ import annotations

from pathlib import Path
import subprocess

import matplotlib.font_manager as fm
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
CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "Source Han Serif CN",
    "SimSun",
    "Droid Sans Fallback",
]


def _prediction_artifact_paths(config_path: Path) -> list[Path]:
    cfg = load_config(config_path)
    paths: list[Path] = []
    for horizon_days in cfg.long.horizons:
        for filename_prefix in ("lgbm", "ridge"):
            paths.append(
                cfg.long.prediction_output_dir
                / f"h{horizon_days}"
                / f"{filename_prefix}_{horizon_days}d_predictions.csv"
            )
    return paths


@st.cache_data(show_spinner=False)
def _load_long_predictions(
    config_path: Path,
    _data_version: tuple[tuple[str, float | None], ...],
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
    st.error("缺少长期预测结果文件。")
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


def _resolve_font_path(font_name: str) -> Path | None:
    try:
        output = subprocess.check_output(
            ["fc-match", "-f", "%{file}\n", font_name],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if not output:
        return None
    path = Path(output)
    return path if path.exists() else None


@st.cache_resource
def _configure_matplotlib_fonts() -> str | None:
    for font_name in CJK_FONT_CANDIDATES:
        font_path = _resolve_font_path(font_name)
        if font_path is None:
            continue
        fm.fontManager.addfont(font_path)
        resolved_font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [resolved_font_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return resolved_font_name
    return None


def _prediction_chart(
    predictions: dict[int, dict[str, pd.DataFrame]],
    *,
    selected_horizons: list[int],
    selected_models: list[str],
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(selected_horizons),
        1,
        figsize=(10, max(3.8 * len(selected_horizons), 4)),
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
                linewidth=2.2,
                label="实际值",
            )
            test_rows = actual_frame[actual_frame["split"] == "test"]
            if not test_rows.empty:
                boundary_date = test_rows["date"].min()
                ax.axvline(
                    boundary_date,
                    color="#94a3b8",
                    linestyle="--",
                    linewidth=1.2,
                    label="验证/测试分界",
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
                f"{horizon_days}天预测缺少结果文件",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=TEXT_COLOR,
            )

        ax.set_title(f"{horizon_days}天预测：验证集 + 测试集", color=TEXT_COLOR)
        ax.set_ylabel("价格", color=TEXT_COLOR)
        ax.grid(alpha=0.35, color=GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        legend = ax.legend(facecolor=AXIS_BG, edgecolor=GRID_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

    axes[-1].set_xlabel("目标日期", color=TEXT_COLOR)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="开源启民长期预测",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_dark_mode()
    selected_font = _configure_matplotlib_fonts()
    st.title("开源启民长期预测结果")
    st.caption("展示 7/30/90 天长期预测在完整评估集（验证集 + 测试集）上的实际值与预测值。")

    config_path = CONFIG_PATH
    if not config_path.exists():
        st.error(f"未找到配置文件：{config_path}")
        st.stop()

    cfg = load_config(config_path)
    if not cfg.long.prediction_output_dir.exists():
        _render_missing_artifacts()

    artifact_paths = _prediction_artifact_paths(config_path)
    data_version = tuple(
        (str(path), path.stat().st_mtime if path.exists() else None) for path in artifact_paths
    )
    predictions, missing_prediction_paths = _load_long_predictions(
        config_path,
        data_version,
    )
    if not predictions:
        _render_missing_artifacts()

    st.sidebar.header("结果文件")
    st.sidebar.write(f"预测目录：`{cfg.long.prediction_output_dir}`")
    if selected_font is not None:
        st.sidebar.write(f"绘图字体：`{selected_font}`")
    st.sidebar.code("uv run python -m kyqm --pipeline long --model all", language="bash")
    if missing_prediction_paths:
        st.sidebar.caption(
            f"缺少 {len(missing_prediction_paths)} 个预测文件，请重新生成长期预测结果。"
        )

    horizon_choices = [horizon for horizon in cfg.long.horizons if horizon in predictions]
    selected_horizons = st.sidebar.multiselect(
        "显示预测周期",
        options=horizon_choices,
        default=horizon_choices,
        format_func=lambda value: f"{value}天",
    )
    available_prediction_models = [
        model_name
        for model_name in ("lightgbm", "ridge")
        if any(model_name in horizon_predictions for horizon_predictions in predictions.values())
    ]
    selected_prediction_models = st.sidebar.multiselect(
        "显示模型",
        options=available_prediction_models,
        default=available_prediction_models,
        format_func=lambda value: MODEL_LABELS[value],
    )

    summary_cols = st.columns(len(horizon_choices))
    for col, horizon_days in zip(summary_cols, horizon_choices, strict=True):
        horizon_frame = next(iter(predictions[horizon_days].values()))
        split_counts = horizon_frame["split"].value_counts()
        val_count = int(split_counts.get("val", 0))
        test_count = int(split_counts.get("test", 0))
        col.metric(
            label=f"{horizon_days}天样本数",
            value=len(horizon_frame),
            delta=f"验证 {val_count} / 测试 {test_count}",
        )

    st.subheader("长期预测：实际值 vs 预测值")
    if not selected_horizons:
        st.info("请在侧边栏中至少选择一个预测周期。")
    elif not selected_prediction_models:
        st.info("请在侧边栏中至少选择一个模型。")
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
            "每个子图对应一个预测周期，覆盖完整评估集；虚线表示**验证集**与**测试集**的分界，阴影带表示 LightGBM 的 `P10-P90` 区间。"
        )

    st.sidebar.caption("当前页面仅展示长期预测在完整评估集上的结果。")


if __name__ == "__main__":
    main()
