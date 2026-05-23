from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "devdocs" / "assets" / "tech_report"
SUMMARY_LONG_PATH = ROOT / "data" / "model_summary_long.json"
COMPARISON_LONG_PATH = ROOT / "data" / "model_comparison_long.csv"
CANDIDATES_LONG_PATH = ROOT / "data" / "model_candidates_long.csv"
SELECTED_30D_PATH = ROOT / "data" / "predictions" / "long" / "h30" / "selected_30d_predictions.csv"
SELECTED_90D_PATH = ROOT / "data" / "predictions" / "long" / "h90" / "selected_90d_predictions.csv"

CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Microsoft YaHei",
    "Source Han Sans CN",
    "SimSun",
]


def _resolve_font_path(font_name: str) -> Path | None:
    try:
        output = subprocess.check_output(
            ["fc-match", "-f", "%{file}\n", font_name],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    path = Path(output)
    return path if output and path.exists() else None


def configure_matplotlib() -> None:
    matplotlib.use("Agg")
    for font_name in CJK_FONT_CANDIDATES:
        font_path = _resolve_font_path(font_name)
        if font_path is None:
            continue
        fm.fontManager.addfont(font_path)
        resolved = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [resolved, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#cbd5e1",
            "text.color": "#0f172a",
        }
    )


def _save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_timeline() -> None:
    phases = [
        ("初始方案", "多源数据 +\nLSTM 注意力"),
        ("天气接入", "Open-Meteo\n四城天气"),
        ("多模型扩展", "LightGBM / GRU /\nLSTM / Prophet"),
        ("短期优化", "1日特征增强\n与基线审计"),
        ("长周期追加", "7/30/90天\n周锚点样本"),
        ("LONG_PERF", "时移修正 /\n残差堆叠"),
        ("LONG_PERF_2", "Stacking / 动态混合 /\n自适应区间"),
        ("LONG_PERF_3", "冲击特征 / 尾部加权 /\n风险预警"),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    x = np.arange(len(phases))
    ax.plot(x, np.zeros_like(x), color="#2563eb", linewidth=2.0)
    ax.scatter(x, np.zeros_like(x), color="#2563eb", s=90, zorder=3)
    for idx, (title, subtitle) in enumerate(phases):
        y = 0.25 if idx % 2 == 0 else -0.42
        ax.text(
            idx,
            y,
            f"{idx + 1}. {title}\n{subtitle}",
            ha="center",
            va="center",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eff6ff", "edgecolor": "#93c5fd"},
        )
    ax.set_title("项目开发演进时间线", fontsize=14, pad=14)
    ax.set_xlim(-0.5, len(phases) - 0.5)
    ax.set_ylim(-0.8, 0.6)
    ax.axis("off")
    _save(fig, "timeline.png")


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.axis("off")
    boxes = [
        ((0.05, 0.64, 0.2, 0.2), "数据源", "SWYB / 全国批发价 /\n天气 / 预留舆情"),
        ((0.3, 0.64, 0.22, 0.2), "数据处理", "省市解析、对齐、缺失处理\n统一宽表与长周期样本"),
        ((0.58, 0.64, 0.22, 0.2), "特征工程", "滞后 / 滚动 / 节假日 /\n天气 / 冲击特征"),
        ((0.05, 0.22, 0.2, 0.2), "短周期模型", "LSTM+Attention\nLightGBM / GRU / Prophet"),
        ((0.3, 0.22, 0.22, 0.2), "长周期候选", "Ridge / LGBM /\n动态混合 / 区间校准"),
        ((0.58, 0.22, 0.22, 0.2), "输出层", "预测 CSV / 模型摘要 /\n风险标记 / Streamlit / PDF"),
    ]
    for x, y, w, h in [b[0] for b in boxes]:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5)
        )
    for (x, y, w, h), title, body in boxes:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.25, 0.74), (0.3, 0.74)),
        ((0.52, 0.74), (0.58, 0.74)),
        ((0.15, 0.64), (0.15, 0.42)),
        ((0.41, 0.64), (0.41, 0.42)),
        ((0.69, 0.64), (0.69, 0.42)),
        ((0.25, 0.32), (0.3, 0.32)),
        ((0.52, 0.32), (0.58, 0.32)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#2563eb", "lw": 1.8})
    ax.set_title("系统架构与交付链路（白底报告版）", fontsize=14, pad=12)
    _save(fig, "architecture.png")


def plot_long_ratio_chart() -> None:
    comparison = pd.read_csv(COMPARISON_LONG_PATH)
    selected = comparison[comparison["model"].str.startswith(("selected_", "lightgbm_1d"))].copy()
    selected["label"] = selected["horizon_days"].map({1: "1天", 7: "7天", 30: "30天", 90: "90天"})
    selected["ratio"] = selected["mae_ratio"].astype(float)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = ["#64748b" if ratio >= 1.0 else "#2563eb" for ratio in selected["ratio"]]
    bars = ax.bar(selected["label"], selected["ratio"], color=colors, edgecolor="#1e293b")
    ax.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.2, label="基线水平")
    for bar, ratio in zip(bars, selected["ratio"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, ratio + 0.02, f"{ratio:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("各预测周期相对基线的 MAE 比值", fontsize=14)
    ax.set_ylabel("模型 MAE / 基线 MAE")
    ax.grid(axis="y", alpha=0.5)
    ax.legend(frameon=False)
    _save(fig, "long_mae_ratio.png")


def plot_leaderboard() -> None:
    candidates = pd.read_csv(CANDIDATES_LONG_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, horizon in zip(axes, [30, 90], strict=True):
        subset = (
            candidates[candidates["horizon_days"] == horizon]
            .sort_values("val_mae")
            .head(6)
            .sort_values("val_mae", ascending=True)
        )
        colors = ["#2563eb" if selected == 1 else "#94a3b8" for selected in subset["selected"]]
        ax.barh(subset["candidate"], subset["val_mae"], color=colors, edgecolor="#1e293b")
        for value, name in zip(subset["val_mae"], subset["candidate"], strict=True):
            ax.text(value + 0.01, name, f"{value:.3f}", va="center", fontsize=9)
        ax.set_title(f"{horizon}天候选排行榜（按验证 MAE）", fontsize=12)
        ax.set_xlabel("验证 MAE")
        ax.grid(axis="x", alpha=0.5)
    _save(fig, "long_leaderboard.png")


def plot_risk_summary() -> None:
    summary = json.loads(SUMMARY_LONG_PATH.read_text(encoding="utf-8"))
    horizons = ["30天", "90天"]
    high_risk = [summary["selected_30d"]["test_high_risk_rate"], summary["selected_90d"]["test_high_risk_rate"]]
    sell = [summary["selected_30d"]["test_sell_alert_rate"], summary["selected_90d"]["test_sell_alert_rate"]]
    hedge = [summary["selected_30d"]["test_hedge_alert_rate"], summary["selected_90d"]["test_hedge_alert_rate"]]
    x = np.arange(len(horizons))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width, high_risk, width, label="高风险占比", color="#f59e0b")
    ax.bar(x, sell, width, label="卖出预警占比", color="#22c55e")
    ax.bar(x + width, hedge, width, label="避险预警占比", color="#ef4444")
    ax.set_xticks(x, horizons)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("占比")
    ax.set_title("LONG_PERF_3 风险预警输出概览", fontsize=14)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(frameon=False, ncols=3, loc="upper center")
    _save(fig, "risk_summary.png")


def plot_selected_forecasts() -> None:
    frames = {
        30: pd.read_csv(SELECTED_30D_PATH, parse_dates=["date"]),
        90: pd.read_csv(SELECTED_90D_PATH, parse_dates=["date"]),
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=False)
    for ax, (horizon, frame) in zip(axes, frames.items(), strict=True):
        ax.plot(frame["date"], frame["y_true"], color="#0f172a", linewidth=2.2, label="实际值")
        ax.plot(frame["date"], frame["y_pred"], color="#2563eb", linewidth=2.0, label="最终预测")
        if {"y_pred_p10", "y_pred_p90"}.issubset(frame.columns):
            ax.fill_between(frame["date"], frame["y_pred_p10"], frame["y_pred_p90"], color="#bfdbfe", alpha=0.8, label="P10-P90")
        if "high_risk_flag" in frame.columns:
            risk_rows = frame[frame["high_risk_flag"] == 1]
            if not risk_rows.empty:
                ax.scatter(risk_rows["date"], risk_rows["y_pred"], color="#f59e0b", s=18, label="高风险")
        test_rows = frame[frame["split"] == "test"]
        if not test_rows.empty:
            ax.axvline(test_rows["date"].min(), color="#64748b", linestyle="--", linewidth=1.2, label="验证/测试分界")
        ax.set_title(f"{horizon}天最终预测与风险区间", fontsize=12)
        ax.set_ylabel("价格")
        ax.grid(alpha=0.45)
        ax.legend(frameon=False, ncols=5, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("目标日期")
    _save(fig, "selected_forecasts.png")


def main() -> None:
    configure_matplotlib()
    plot_timeline()
    plot_architecture()
    plot_long_ratio_chart()
    plot_leaderboard()
    plot_risk_summary()
    plot_selected_forecasts()


if __name__ == "__main__":
    main()
