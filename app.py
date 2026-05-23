from __future__ import annotations

from pathlib import Path
import json
import subprocess

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

from kyqm.config import load_config


CONFIG_PATH = Path("kyqm/config.toml")
REPORT_ASSET_DIR = Path("devdocs/assets/tech_report")
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
        frame = (
            pd.read_csv(path, parse_dates=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        frame["horizon_days"] = horizon_days
        selected_predictions[horizon_days] = frame

    candidate_path = cfg.long.comparison_output_path.with_name(
        "model_candidates_long.csv"
    )
    candidates = (
        pd.read_csv(candidate_path).sort_values(
            ["horizon_days", "val_mae", "candidate"]
        )
        if candidate_path.exists()
        else pd.DataFrame()
    )
    comparison = (
        pd.read_csv(cfg.long.comparison_output_path)
        if cfg.long.comparison_output_path.exists()
        else pd.DataFrame()
    )
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
    return None


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
                    sell_rows["y_pred_p90"]
                    if "y_pred_p90" in sell_rows.columns
                    else sell_rows["y_pred"],
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
                    hedge_rows["y_pred_p10"]
                    if "y_pred_p10" in hedge_rows.columns
                    else hedge_rows["y_pred"],
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


def _format_candidate_table(
    candidates: pd.DataFrame, horizon_days: int
) -> pd.DataFrame:
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
            subset[label] = (
                subset[source]
                if subset[source].dtype == object
                else subset[source].round(4)
            )
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
                "高风险占比": float(test_frame["high_risk_flag"].mean()),
                "卖出预警占比": float(
                    (
                        test_frame["sell_alert_flag"]
                        if "sell_alert_flag" in test_frame.columns
                        else pd.Series(0, index=test_frame.index)
                    ).mean()
                ),
                "避险预警占比": float(
                    (
                        test_frame["hedge_alert_flag"]
                        if "hedge_alert_flag" in test_frame.columns
                        else pd.Series(0, index=test_frame.index)
                    ).mean()
                ),
                "平均区间宽度比": interval_ratio,
            }
        )
    return pd.DataFrame(rows)


def _risk_event_table(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"high_risk_flag", "sell_alert_flag", "hedge_alert_flag"}.intersection(
        frame.columns
    ):
        return pd.DataFrame()
    event_mask = pd.Series(False, index=frame.index)
    for column in ("high_risk_flag", "sell_alert_flag", "hedge_alert_flag"):
        if column in frame.columns:
            event_mask = event_mask | (frame[column].astype(int) == 1)
    events = frame.loc[event_mask].copy()
    if events.empty:
        return events
    events["日期"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")
    events["集合"] = (
        events["split"].map({"val": "验证", "test": "测试"}).fillna(events["split"])
    )
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
        values = (
            events[source]
            if source in events.columns
            else pd.Series(0, index=events.index)
        )
        events[label] = values.astype(int).map({1: "是", 0: ""})
    keep = ["日期", "集合", "实际值", "预测值"]
    for column in ("P10", "P90", "区间宽度比"):
        if column in events.columns:
            keep.append(column)
    keep.extend(["高风险", "卖出预警", "避险预警"])
    return events[keep]


def _render_markdown_card(
    title: str, body: str, *, kicker: str | None = None, hero: bool = False
) -> None:
    with st.container(border=True):
        if kicker:
            st.caption(kicker)
        if hero:
            st.subheader(title)
        else:
            st.markdown(f"**{title}**")
        st.markdown(body)


def _report_asset_path(name: str) -> Path:
    return REPORT_ASSET_DIR / name


@st.cache_data(show_spinner=False)
def _load_dark_presentation_image(path: str, mtime: float) -> Image.Image:
    image = Image.open(path)
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = rgba.convert("RGB")
    rgb = ImageOps.invert(rgb)
    rgb = ImageEnhance.Brightness(rgb).enhance(0.88)
    rgb = ImageEnhance.Contrast(rgb).enhance(0.96)
    if has_alpha:
        dark_image = rgb.convert("RGBA")
        dark_image.putalpha(alpha)
        return dark_image
    return rgb


def _render_presentation_image(path: Path, caption: str) -> None:
    if not path.exists():
        return
    st.image(
        _load_dark_presentation_image(str(path), path.stat().st_mtime),
        caption=caption,
        width="stretch",
    )


def _final_model_table(
    comparison: pd.DataFrame,
    selection_map: dict[int, str],
) -> pd.DataFrame:
    comparison_selected = comparison[
        comparison["model"].str.startswith("selected_")
    ].copy()
    if comparison_selected.empty:
        return pd.DataFrame()
    use_cases = {
        7: "日常价格监测、短期采购决策",
        30: "月度库存规划、中期定价策略",
        90: "季度合同定价、长期风险对冲",
    }
    design_notes = {
        7: "Ridge + LightGBM 等权平均，压低短期波动带来的方差。",
        30: "按市场波动状态切换混合权重，并在高波动时引入分位数响应。",
        90: "沿用动态混合，再叠加区间与风险信号，为长期不确定性管理服务。",
    }
    rows: list[dict[str, str | float]] = []
    for _, row in comparison_selected.sort_values("horizon_days").iterrows():
        horizon_days = int(row["horizon_days"])
        baseline_improvement = (1.0 - float(row["mae_ratio"])) * 100.0
        rows.append(
            {
                "预测窗口": f"{horizon_days}天",
                "设计思路": design_notes.get(horizon_days, ""),
            }
        )
    return pd.DataFrame(rows)


def _value_story_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "对象": "政府部门",
                "可用价值": "监测供需变化、识别价格异常、辅助储备与调运决策",
            },
            {
                "对象": "企业与合作社",
                "可用价值": "支持采购、库存、合同报价和销售节奏安排",
            },
            {
                "对象": "科研与教育",
                "可用价值": "沉淀高频农业经济数据，支撑研究、训练和实验教学",
            },
            {
                "对象": "乡村发展",
                "可用价值": "把零散市场价格数字化，形成可被资本、产业和平台利用的基础设施",
            },
        ]
    )


def _render_final_evaluation_dashboard(
    selected_predictions: dict[int, pd.DataFrame],
    candidates: pd.DataFrame,
    comparison: pd.DataFrame,
    selection_map: dict[int, str],
    *,
    selected_horizons: list[int],
    heading_prefix: str | None = None,
) -> None:
    horizon_choices = sorted(selected_predictions)
    comparison_selected = comparison[
        comparison["model"].str.startswith("selected_")
    ].copy()

    if heading_prefix:
        st.markdown(f"## {heading_prefix}最终模型评估")

    summary_cols = st.columns(len(horizon_choices))
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
        st.info("请至少选择一个预测周期。")
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
                st.dataframe(
                    _format_candidate_table(candidates, horizon_days), width="stretch"
                )

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
        final_table["预测周期"] = final_table["horizon_days"].map(
            lambda value: f"{value}天"
        )
        final_table["测试MAE"] = final_table["test_mae"].round(4)
        final_table["基线MAE"] = final_table["baseline_mae"].round(4)
        final_table["MAE比值"] = final_table["mae_ratio"].round(3)
        final_table["最终方案"] = final_table["horizon_days"].map(
            lambda value: selection_map.get(int(value), "selected")
        )
        st.dataframe(
            final_table[
                [
                    "预测周期",
                    "最终方案",
                    "baseline_name",
                    "测试MAE",
                    "基线MAE",
                    "MAE比值",
                ]
            ],
            width="stretch",
        )


def _render_hackathon_page(
    selected_predictions: dict[int, pd.DataFrame],
    candidates: pd.DataFrame,
    comparison: pd.DataFrame,
    selection_map: dict[int, str],
) -> None:
    st.title("开源齐民")

    _render_markdown_card(
        "把分散的农产品价格，变成可预测、可决策、可投资的数据基础设施",
        """
我们以山东黄瓜为例，整合 **商务预报、全国批发价和天气数据**，输出 **7天、30天、90天** 三个预测窗口，既服务价格判断，也服务风险预警。

- 数据源：商务部商业预报系统
- 对象：政府、企业、合作社、研究机构
- 形式：预测 + 区间 + 预警
        """,
        kicker="第一部分 · 项目定位",
        hero=True,
    )
    overview_cols = st.columns(3)
    overview_cols[0].metric("7天", "短期监测", "采购与价格巡检")
    overview_cols[1].metric("30天", "中期规划", "采收时间推荐")
    overview_cols[2].metric("90天", "长期风控", "农户生产决策")
    timeline_path = _report_asset_path("timeline.png")
    if timeline_path.exists():
        _render_presentation_image(
            timeline_path, "项目从初始方案到最终方案的演进时间线"
        )

    _render_markdown_card(
        "技术方案：三层结构，先把数据做好，再做多窗口预测",
        """
**第一层：数据数字化。** 把各个市场的农产品价格、全国行情和天气统一到日频数据表。  
**第二层：多窗口建模。** 短期与中长期使用不同策略，不强求一个模型包打天下。  
**第三层：业务输出。** 除了点预测，还给出区间、高风险、卖出和避险提示。
        """,
        kicker="第二部分 · 技术设计",
        hero=True,
    )
    architecture_col, note_col = st.columns([1.35, 1])
    with architecture_col:
        architecture_path = _report_asset_path("architecture.png")
        if architecture_path.exists():
            _render_presentation_image(
                architecture_path, "从数据接入到业务输出的整体架构"
            )
    with note_col:
        _render_markdown_card(
            "最终三种模型策略",
            """
            - **7天：** `mean_blend`，用简单平均换取稳定性
            - **30天：** `dynamic_blend`，按波动状态动态配比
            - **90天：** `dynamic_blend` + 区间与风险信号，更适合长期判断
            """,
        )
        final_models = _final_model_table(comparison, selection_map)
        if not final_models.empty:
            st.dataframe(final_models, width="stretch", hide_index=True)

    _render_markdown_card(
        "技术效果",
        """
        """,
        kicker="第三部分 · 最终模型评估",
        hero=True,
    )
    _render_final_evaluation_dashboard(
        selected_predictions,
        candidates,
        comparison,
        selection_map,
        selected_horizons=sorted(selected_predictions),
    )

    _render_markdown_card(
        "预测服务的价值",
        """
**对合作社：** 利用卖出与避险提示安排采收、销售和保价策略。

**对政府：** 看到价格风险何时抬升，辅助调运、监测和预警。  

**对企业：** 看 30 天库存节奏、看 90 天合同和长期采购风险。  
        """,
        kicker="第四部分 · 服务能力",
        hero=True,
    )
    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        forecast_path = _report_asset_path("selected_forecasts.png")
        if forecast_path.exists():
            _render_presentation_image(
                forecast_path, "最终选中模型的预测曲线、区间与风险标记"
            )
    with right_col:
        risk_summary = _risk_summary_table(selected_predictions, selection_map)
        if not risk_summary.empty:
            display_risk = risk_summary.copy()
            for column in (
                "高风险占比",
                "卖出预警占比",
                "避险预警占比",
                "平均区间宽度比",
            ):
                display_risk[column] = display_risk[column].round(3)
            st.dataframe(display_risk, width="stretch", hide_index=True)
        risk_path = _report_asset_path("risk_summary.png")
        if risk_path.exists():
            _render_presentation_image(risk_path, "高风险、卖出、避险信号概览")

    _render_markdown_card(
        "更大的价值：乡村资源数字化，能够把数据变成发展能力",
        """
农产品价格原本分散在各个市场里，难以被系统使用。数字化之后，它们就成为了 **可计算、可研究、可投资** 的基础设施。

这类基础设施能够吸引过剩资本和其他数字平台进入乡村，进而支持就业、消费和本地开发。

当城市存在过剩资本与过剩劳动力时，乡村数字基础设施可以成为一个缓冲带，吸纳大量的各界人才。
        """,
        kicker="第五部分 · 社会价值",
        hero=True,
    )
    value_col, table_col = st.columns([1.1, 1])
    with value_col:
        _render_markdown_card(
            "为什么这件事值得做",
            """
            - 把乡村价格信息从“碎片化消息”升级为“标准化资产”
            - 帮助政府和企业更早做判断，减少盲目波动带来的损失
            - 为研究、训练、教学和行业服务提供长期可积累的数据底座
            - 增强乡村经济活力与消费力，形成更可持续的本地发展循环
            """,
        )
    with table_col:
        st.dataframe(_value_story_table(), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="开源齐民路演与分析看板",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_dark_mode()
    selected_font = _configure_matplotlib_fonts()

    config_path = CONFIG_PATH
    if not config_path.exists():
        st.error(f"未找到配置文件：{config_path}")
        st.stop()

    artifact_paths = _artifact_paths(config_path)
    data_version = tuple(
        (str(path), path.stat().st_mtime if path.exists() else None)
        for path in artifact_paths
    )
    selected_predictions, candidates, comparison, summary, missing_paths = (
        _load_long_visual_artifacts(
            config_path,
            data_version,
        )
    )
    if not selected_predictions:
        _render_missing_artifacts()

    selection_map = {
        int(horizon.replace("d", "")): values["selected_candidate"]
        for horizon, values in summary.get("long_selection", {}).items()
    }
    horizon_choices = sorted(selected_predictions)

    st.sidebar.header("页面模式")
    page_mode = st.sidebar.radio(
        "选择展示方式",
        options=["路演幻灯片", "分析看板"],
        index=0,
    )

    if page_mode == "路演幻灯片":
        _render_hackathon_page(
            selected_predictions, candidates, comparison, selection_map
        )
        st.sidebar.caption("当前为黑客松代表演示模式。")
        st.sidebar.write(
            "建议顺序：项目定位 → 技术方案 → 完整评估 → 服务能力 → 社会价值"
        )
        st.sidebar.write("当前为单页连续展示，可直接滚动路演。")
        return

    st.title("开源齐民长期优化结果")
    st.caption(
        "基于 `LONG_PERF_3` 重训后的最终长期预测结果、候选排行榜、风险区间与预警视图。"
    )

    st.sidebar.header("结果文件")
    st.sidebar.write("预测目录：`data/predictions/long/**`")
    st.sidebar.write("候选榜单：`data/model_candidates_long.csv`")
    st.sidebar.write("最终对比：`data/model_comparison_long.csv`")
    if selected_font is not None:
        st.sidebar.write(f"绘图字体：`{selected_font}`")
    st.sidebar.code(
        "uv run python -m kyqm --pipeline long --model all", language="bash"
    )
    if missing_paths:
        st.sidebar.caption(f"仍缺少 {len(missing_paths)} 个 selected 预测文件。")

    selected_horizons = st.sidebar.multiselect(
        "显示预测周期",
        options=horizon_choices,
        default=horizon_choices,
        format_func=lambda value: f"{value}天",
    )

    _render_final_evaluation_dashboard(
        selected_predictions,
        candidates,
        comparison,
        selection_map,
        selected_horizons=selected_horizons,
    )

    st.sidebar.caption(
        "当前页面已同步 LONG_PERF_3 的候选搜索、风险区间与最终选择结果。"
    )


if __name__ == "__main__":
    main()
