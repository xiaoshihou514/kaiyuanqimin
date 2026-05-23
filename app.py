from __future__ import annotations

from pathlib import Path
import json
import subprocess

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kyqm.config import load_config


CONFIG_PATH = Path("kyqm/config.toml")
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


def _artifact_paths(config_path: Path) -> list[Path]:
    cfg = load_config(config_path)
    paths = [
        cfg.long.summary_output_path,
        cfg.long.comparison_output_path,
        cfg.long.comparison_output_path.with_name("model_candidates_long.csv"),
    ]
    for horizon_days in cfg.long.horizons:
        paths.append(
            cfg.long.prediction_output_dir
            / f"h{horizon_days}"
            / f"selected_{horizon_days}d_predictions.csv"
        )
    return paths


@st.cache_data(show_spinner=False)
def _load_long_visual_artifacts(
    config_path: Path,
    _data_version: tuple[tuple[str, float | None], ...],
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict, list[str]]:
    cfg = load_config(config_path)
    selected_predictions: dict[int, pd.DataFrame] = {}
    missing_paths: list[str] = []
    for horizon_days in cfg.long.horizons:
        path = (
            cfg.long.prediction_output_dir
            / f"h{horizon_days}"
            / f"selected_{horizon_days}d_predictions.csv"
        )
        if not path.exists():
            missing_paths.append(str(path))
            continue
        frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        frame["horizon_days"] = horizon_days
        selected_predictions[horizon_days] = frame

    candidate_path = cfg.long.comparison_output_path.with_name("model_candidates_long.csv")
    candidates = (
        pd.read_csv(candidate_path).sort_values(["horizon_days", "val_mae", "candidate"])
        if candidate_path.exists()
        else pd.DataFrame()
    )
    comparison = pd.read_csv(cfg.long.comparison_output_path) if cfg.long.comparison_output_path.exists() else pd.DataFrame()
    summary = (
        json.loads(cfg.long.summary_output_path.read_text(encoding="utf-8"))
        if cfg.long.summary_output_path.exists()
        else {}
    )
    return selected_predictions, candidates, comparison, summary, missing_paths


def _render_missing_artifacts() -> None:
    st.error("缺少长期优化后的可视化结果文件。")
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
        [data-testid="stMetric"] {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 0.75rem;
            padding: 0.75rem;
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


def _selected_prediction_chart(
    selected_predictions: dict[int, pd.DataFrame],
    *,
    selected_horizons: list[int],
    selection_map: dict[int, str],
) -> plt.Figure:
    _configure_matplotlib_fonts()
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
        frame = selected_predictions[horizon_days]
        ax.plot(
            frame["date"],
            frame["y_true"],
            color="#f8fafc",
            linewidth=2.4,
            label="实际值",
        )
        ax.plot(
            frame["date"],
            frame["y_pred"],
            color="#60a5fa",
            linewidth=2.0,
            label=f"最终模型：{selection_map.get(horizon_days, 'selected')}",
        )
        if {"y_pred_p10", "y_pred_p90"}.issubset(frame.columns):
            ax.fill_between(
                frame["date"],
                frame["y_pred_p10"],
                frame["y_pred_p90"],
                color="#60a5fa",
                alpha=0.18,
                label="预测区间 P10-P90",
            )
        if "high_risk_flag" in frame.columns:
            high_risk_rows = frame[frame["high_risk_flag"] == 1]
            if not high_risk_rows.empty:
                ax.scatter(
                    high_risk_rows["date"],
                    high_risk_rows["y_pred"],
                    color="#f59e0b",
                    s=28,
                    label="高风险期",
                    zorder=4,
                )
        if "sell_alert_flag" in frame.columns:
            sell_rows = frame[frame["sell_alert_flag"] == 1]
            if not sell_rows.empty:
                ax.scatter(
                    sell_rows["date"],
                    sell_rows["y_pred_p90"] if "y_pred_p90" in sell_rows.columns else sell_rows["y_pred"],
                    color="#22c55e",
                    s=24,
                    marker="^",
                    label="卖出预警",
                    zorder=5,
                )
        if "hedge_alert_flag" in frame.columns:
            hedge_rows = frame[frame["hedge_alert_flag"] == 1]
            if not hedge_rows.empty:
                ax.scatter(
                    hedge_rows["date"],
                    hedge_rows["y_pred_p10"] if "y_pred_p10" in hedge_rows.columns else hedge_rows["y_pred"],
                    color="#ef4444",
                    s=24,
                    marker="v",
                    label="避险预警",
                    zorder=5,
                )

        test_rows = frame[frame["split"] == "test"]
        if not test_rows.empty:
            boundary_date = test_rows["date"].min()
            ax.axvline(
                boundary_date,
                color="#94a3b8",
                linestyle="--",
                linewidth=1.2,
                label="验证/测试分界",
            )

        ax.set_title(f"{horizon_days}天：最终选中方案", color=TEXT_COLOR)
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


def _format_candidate_table(candidates: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    subset = candidates[candidates["horizon_days"] == horizon_days].copy()
    if subset.empty:
        return subset
    subset["候选方案"] = subset["candidate"]
    subset["验证MAE"] = subset["val_mae"].round(4)
    subset["测试MAE"] = subset["test_mae"].round(4)
    subset["验证RMSE"] = subset["val_rmse"].round(4)
    subset["测试RMSE"] = subset["test_rmse"].round(4)
    subset["已选中"] = subset["selected"].astype(int).map({1: "是", 0: ""})
    keep = ["候选方案", "验证MAE", "测试MAE", "验证RMSE", "测试RMSE", "已选中"]
    optional = [
        ("best_shift_days", "平移天数"),
        ("lightgbm_weight", "LightGBM权重"),
        ("weight_scheme", "加权方案"),
        ("bias", "偏差校准"),
        ("low_vol_weight", "低波动权重"),
        ("mid_vol_weight", "中波动权重"),
        ("high_vol_weight", "高波动权重"),
        ("blend_source", "混合主模型"),
        ("residual_threshold", "残差阈值"),
        ("residual_correction_rate", "残差触发占比"),
        ("selection_penalty", "选择惩罚"),
        ("test_picp", "测试覆盖率"),
        ("test_interval_width", "测试区间宽度"),
    ]
    for source, label in optional:
        if source in subset.columns and subset[source].notna().any():
            subset[label] = subset[source] if subset[source].dtype == object else subset[source].round(4)
            keep.append(label)
    return subset[keep]


def _risk_summary_table(
    selected_predictions: dict[int, pd.DataFrame],
    selection_map: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for horizon_days, frame in selected_predictions.items():
        if "high_risk_flag" not in frame.columns:
            continue
        test_frame = frame[frame["split"] == "test"].copy()
        if test_frame.empty:
            continue
        interval_ratio = (
            float(test_frame["interval_width_ratio"].mean())
            if "interval_width_ratio" in test_frame.columns
            else float("nan")
        )
        rows.append(
            {
                "预测周期": f"{horizon_days}天",
                "最终方案": selection_map.get(horizon_days, "selected"),
                "高风险占比": float(test_frame["high_risk_flag"].mean()),
                "卖出预警占比": float(
                    (test_frame["sell_alert_flag"] if "sell_alert_flag" in test_frame.columns else pd.Series(0, index=test_frame.index)).mean()
                ),
                "避险预警占比": float(
                    (test_frame["hedge_alert_flag"] if "hedge_alert_flag" in test_frame.columns else pd.Series(0, index=test_frame.index)).mean()
                ),
                "平均区间宽度比": interval_ratio,
            }
        )
    return pd.DataFrame(rows)


def _risk_event_table(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"high_risk_flag", "sell_alert_flag", "hedge_alert_flag"}.intersection(frame.columns):
        return pd.DataFrame()
    event_mask = pd.Series(False, index=frame.index)
    for column in ("high_risk_flag", "sell_alert_flag", "hedge_alert_flag"):
        if column in frame.columns:
            event_mask = event_mask | (frame[column].astype(int) == 1)
    events = frame.loc[event_mask].copy()
    if events.empty:
        return events
    events["日期"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")
    events["集合"] = events["split"].map({"val": "验证", "test": "测试"}).fillna(events["split"])
    events["实际值"] = events["y_true"].round(4)
    events["预测值"] = events["y_pred"].round(4)
    if "y_pred_p10" in events.columns:
        events["P10"] = events["y_pred_p10"].round(4)
    if "y_pred_p90" in events.columns:
        events["P90"] = events["y_pred_p90"].round(4)
    if "interval_width_ratio" in events.columns:
        events["区间宽度比"] = events["interval_width_ratio"].round(3)
    for source, label in (
        ("high_risk_flag", "高风险"),
        ("sell_alert_flag", "卖出预警"),
        ("hedge_alert_flag", "避险预警"),
    ):
        values = events[source] if source in events.columns else pd.Series(0, index=events.index)
        events[label] = values.astype(int).map({1: "是", 0: ""})
    keep = ["日期", "集合", "实际值", "预测值"]
    for column in ("P10", "P90", "区间宽度比"):
        if column in events.columns:
            keep.append(column)
    keep.extend(["高风险", "卖出预警", "避险预警"])
    return events[keep]


def main() -> None:
    st.set_page_config(
        page_title="开源启民长期优化可视化",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_dark_mode()
    selected_font = _configure_matplotlib_fonts()
    st.title("开源启民长期优化结果")
    st.caption("基于 `LONG_PERF_3` 重训后的最终长期预测结果、候选排行榜、风险区间与预警视图。")

    config_path = CONFIG_PATH
    if not config_path.exists():
        st.error(f"未找到配置文件：{config_path}")
        st.stop()

    artifact_paths = _artifact_paths(config_path)
    data_version = tuple(
        (str(path), path.stat().st_mtime if path.exists() else None) for path in artifact_paths
    )
    selected_predictions, candidates, comparison, summary, missing_paths = _load_long_visual_artifacts(
        config_path,
        data_version,
    )
    if not selected_predictions:
        _render_missing_artifacts()

    selection_map = {
        int(horizon.replace("d", "")): values["selected_candidate"]
        for horizon, values in summary.get("long_selection", {}).items()
    }
    horizon_choices = sorted(selected_predictions)

    st.sidebar.header("结果文件")
    st.sidebar.write("预测目录：`data/predictions/long/**`")
    st.sidebar.write("候选榜单：`data/model_candidates_long.csv`")
    st.sidebar.write("最终对比：`data/model_comparison_long.csv`")
    if selected_font is not None:
        st.sidebar.write(f"绘图字体：`{selected_font}`")
    st.sidebar.code("uv run python -m kyqm --pipeline long --model all", language="bash")
    if missing_paths:
        st.sidebar.caption(f"仍缺少 {len(missing_paths)} 个 selected 预测文件。")

    selected_horizons = st.sidebar.multiselect(
        "显示预测周期",
        options=horizon_choices,
        default=horizon_choices,
        format_func=lambda value: f"{value}天",
    )

    summary_cols = st.columns(len(horizon_choices))
    comparison_selected = comparison[comparison["model"].str.startswith("selected_")].copy()
    for col, horizon_days in zip(summary_cols, horizon_choices, strict=True):
        frame = selected_predictions[horizon_days]
        split_counts = frame["split"].value_counts()
        row = comparison_selected[comparison_selected["horizon_days"] == horizon_days]
        mae_ratio = float(row["mae_ratio"].iloc[0]) if not row.empty else float("nan")
        col.metric(
            label=f"{horizon_days}天最终方案",
            value=selection_map.get(horizon_days, "selected"),
            delta=f"MAE比值 {mae_ratio:.3f} | 验证 {int(split_counts.get('val', 0))} / 测试 {int(split_counts.get('test', 0))}",
        )

    st.subheader("最终选中方案：实际值 vs 预测值")
    if not selected_horizons:
        st.info("请在侧边栏中至少选择一个预测周期。")
    else:
        st.pyplot(
            _selected_prediction_chart(
                selected_predictions,
                selected_horizons=selected_horizons,
                selection_map=selection_map,
            ),
            width="stretch",
        )
        st.markdown(
            "每个子图展示当前 **最终选中方案** 在完整评估集上的表现；虚线表示验证集与测试集的分界，橙/绿/红标记分别表示高风险、卖出预警、避险预警。"
        )

    st.subheader("风险预警汇总")
    risk_summary = _risk_summary_table(selected_predictions, selection_map)
    if risk_summary.empty:
        st.info("当前 selected 预测文件还没有风险列，请先重跑长期 pipeline。")
    else:
        formatted_risk_summary = risk_summary.copy()
        for column in ("高风险占比", "卖出预警占比", "避险预警占比", "平均区间宽度比"):
            formatted_risk_summary[column] = formatted_risk_summary[column].round(3)
        st.dataframe(formatted_risk_summary, width="stretch")

    st.subheader("候选方案排行榜")
    if candidates.empty:
        st.info("当前没有候选排行榜文件。")
    else:
        tabs = st.tabs([f"{h}天" for h in horizon_choices])
        for tab, horizon_days in zip(tabs, horizon_choices, strict=True):
            with tab:
                st.dataframe(_format_candidate_table(candidates, horizon_days), width="stretch")

    st.subheader("风险事件明细")
    risk_tabs = st.tabs([f"{h}天" for h in horizon_choices])
    for tab, horizon_days in zip(risk_tabs, horizon_choices, strict=True):
        with tab:
            risk_events = _risk_event_table(selected_predictions[horizon_days])
            if risk_events.empty:
                st.info("当前周期没有高风险或预警事件。")
            else:
                st.dataframe(risk_events, width="stretch")

    st.subheader("最终方案 vs 基线")
    if comparison_selected.empty:
        st.info("当前没有最终对比结果。")
    else:
        final_table = comparison_selected.copy()
        final_table["预测周期"] = final_table["horizon_days"].map(lambda value: f"{value}天")
        final_table["测试MAE"] = final_table["test_mae"].round(4)
        final_table["基线MAE"] = final_table["baseline_mae"].round(4)
        final_table["MAE比值"] = final_table["mae_ratio"].round(3)
        final_table["最终方案"] = final_table["horizon_days"].map(
            lambda value: selection_map.get(int(value), "selected")
        )
        st.dataframe(
            final_table[["预测周期", "最终方案", "baseline_name", "测试MAE", "基线MAE", "MAE比值"]],
            width="stretch",
        )

    st.sidebar.caption("当前页面已同步 LONG_PERF_3 的候选搜索、风险区间与最终选择结果。")


if __name__ == "__main__":
    main()
