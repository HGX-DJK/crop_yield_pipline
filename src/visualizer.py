"""
农作物产量估算与 PPI 统计推断可视化模块。
用于生成出版级图表：
1. 模型交叉验证散点图（实测单产 vs 预测单产 1:1 对比）
2. PPI 95% 置信区间收窄对比柱状图（Survey-Only vs Map-Only vs PTD）
3. 全域农田地块单产空间分布直方图
4. 关键物候与气象特征重要性贡献度排名
"""

import os
import matplotlib
# 使用 Agg 后端，确保在无图形界面（命令行/服务器）环境下稳定运行
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 自动配置跨平台中文字体支持
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',    # Windows 首选微软雅黑
    'SimHei',             # Windows 黑体
    'SimSun',             # Windows 宋体
    'WenQuanYi Micro Hei',# Linux 文泉驿
    'Noto Sans CJK SC',   # Linux Noto
    'PingFang SC',        # macOS
    'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.get("paths", {}).get("output_dir", "output")
        self.dpi = config.get("visualization", {}).get("plot_dpi", 300)
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_actual_vs_predicted(self, cv_predictions, metrics, filename="actual_vs_predicted.png"):
        """绘制交叉验证中实际单产与预测单产的 1:1 对比散点图。"""
        fig, ax = plt.subplots(figsize=(7, 6))
        y_true = cv_predictions["y_true"]
        y_pred = cv_predictions["y_pred"]

        ax.scatter(y_true, y_pred, alpha=0.7, color="#2b5c8f", edgecolors="black", s=50, label="实测样本地块 (Survey Parcels)")
        
        # 绘制 1:1 理想参考线
        min_val = min(y_true.min(), y_pred.min()) * 0.95
        max_val = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.8, label="1:1 理想参考线 (1:1 Perfect Line)")

        ax.set_xlabel("地面实测单产 (kg/ha)", fontsize=11, fontweight="bold")
        ax.set_ylabel("模型预测单产 (kg/ha)", fontsize=11, fontweight="bold")
        ax.set_title("交叉验证：实测单产 vs 预测单产", fontsize=13, fontweight="bold", pad=12)

        # 标注评估指标
        stats_text = f"R² = {metrics.get('r2', 0):.3f}\nRMSE = {metrics.get('rmse', 0):.1f} kg/ha\nMAPE = {metrics.get('mape_pct', 0):.2f}%"
        ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc"))

        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path

    def plot_ppi_confidence_intervals(self, ppi_results, filename="ppi_confidence_intervals.png"):
        """
        绘制三种统计方法的 95% 置信区间对比柱状图：
        仅实地调查 (Survey-Only) vs 纯卫星预测 (Map-Only) vs 预测后去偏 (PTD)
        """
        estimates = ppi_results["estimates"]
        methods = [e["method"] for e in estimates]
        points = [e["point_estimate_kg_ha"] for e in estimates]
        errors = [
            [e["point_estimate_kg_ha"] - e["ci_95_lower"] for e in estimates],
            [e["ci_95_upper"] - e["point_estimate_kg_ha"] for e in estimates]
        ]
        
        colors = ["#7f8c8d", "#e74c3c", "#27ae60"]  # 灰色（实地）、红色（纯卫星）、绿色（去偏PTD）

        fig, ax = plt.subplots(figsize=(8, 5.5))
        bars = ax.bar(methods, points, yerr=errors, capsize=8, color=colors, alpha=0.85, edgecolor="black", width=0.5)

        ax.set_ylabel("估计平均单产 (kg/ha)", fontsize=11, fontweight="bold")
        ax.set_title("联合国手册 PPI：三种估产方法的 95% 置信区间对比", fontsize=13, fontweight="bold", pad=12)
        
        # 标注估计值与误差范围
        for i, e in enumerate(estimates):
            val = e["point_estimate_kg_ha"]
            ci_w = e["ci_width"]
            ax.text(i, val + errors[1][i] + 60, f"{val:.0f} kg/ha\n(±{ci_w/2:.0f})", 
                    ha="center", va="bottom", fontsize=9, fontweight="semibold")

        var_red = ppi_results.get("variance_reduction_pct", 0)
        subtitle = f"Predict-Then-Debias 相比仅实地调查将不确定性区间收窄了 {var_red:.1f}%"
        ax.text(0.5, -0.15, subtitle, transform=ax.transAxes, ha="center", fontsize=10, style="italic", color="#2c3e50")

        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path

    def plot_yield_distribution(self, df_preds, filename="yield_distribution.png"):
        """绘制全域农田地块的单产空间分布直方图。"""
        fig, ax = plt.subplots(figsize=(8, 5))
        yields = df_preds["map_pred_yield_kg_ha"]

        ax.hist(yields, bins=15, color="#3498db", edgecolor="black", alpha=0.75, density=False)
        ax.axvline(yields.mean(), color="#e74c3c", linestyle="--", linewidth=2, label=f"均值: {yields.mean():.1f} kg/ha")
        ax.axvline(yields.median(), color="#f39c12", linestyle=":", linewidth=2, label=f"中位数: {yields.median():.1f} kg/ha")

        ax.set_xlabel("预测单产 (kg/ha)", fontsize=11, fontweight="bold")
        ax.set_ylabel("农田地块数量", fontsize=11, fontweight="bold")
        ax.set_title("全域农田地块单产分布直方图", fontsize=13, fontweight="bold", pad=12)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path

    def plot_feature_importance(self, feature_importances, top_n=10, filename="feature_importance.png"):
        """绘制前 N 个关键特征重要性贡献度排名。"""
        if not feature_importances:
            return None
        
        items = list(feature_importances.items())[:top_n]
        features = [k for k, v in items][::-1]
        values = [v for k, v in items][::-1]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(features, values, color="#16a085", edgecolor="black", alpha=0.85)
        ax.set_xlabel("相对特征重要性得分", fontsize=11, fontweight="bold")
        ax.set_title(f"对产量影响最大的前 {len(features)} 个物候/气象特征", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path
