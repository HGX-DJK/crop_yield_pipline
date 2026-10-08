"""
农作物产量预测与提取系统主运行脚本。
执行流程：
1. 多源农业数据读取与校验（卫星遥感 + 气象 + 地面实测调查）
2. 农业气象有效积温（GDD）与热量时间物候特征工程
3. 机器学习单产预测建模与全域地块推断
4. 《联合国农业统计遥感手册》第 26 章预测增强推断（PPI）去偏校准
5. 出版级统计报告与可视化图表导出
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import argparse
import pandas as pd

from src.data_loader import DataLoader, load_config
from src.feature_engineering import FeatureEngineer
from src.models import YieldPredictor
from src.ppi_evaluator import PPIEvaluator
from src.visualizer import Visualizer


def run_pipeline(config_path="config.yaml"):
    print("=" * 70)
    print("🌾 联合国手册标准：农作物产量提取与官方统计推断流水线")
    print("=" * 70)

    # 1. 加载参数配置
    config = load_config(config_path)
    output_dir = config.get("paths", {}).get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[初始化] 已成功加载配置，目标监测作物: {config.get('project', {}).get('crop', 'Crop')}")

    # 2. 读取多源农业数据
    loader = DataLoader(config)
    print("\n[步骤 1/5] 读取多源农业观测数据...")
    df_rs = loader.load_remote_sensing_data()
    df_weather = loader.load_weather_data()
    df_survey = loader.load_ground_survey_data()
    print(f"  -> 已加载 {len(df_rs)} 条卫星遥感时序观测记录，覆盖 {df_rs['parcel_id'].nunique()} 个地块。")
    print(f"  -> 已加载 {len(df_weather)} 天的逐日农业气象要素观测记录。")
    print(f"  -> 已加载 {len(df_survey)} 个地块的实地抽样测产校准真值。")

    # 3. 特征工程处理
    print("\n[步骤 2/5] 提取热量时间（GDD 积温）与物候生长特征...")
    engineer = FeatureEngineer(config)
    df_features = engineer.extract_parcel_features(df_rs, df_weather)
    print(f"  -> 每个地块已提炼 {df_features.shape[1] - 1} 个关键农学特征（GDD 积温节点、AUC 积分、峰值绿度）。")

    # 4. 机器学习模型训练与预测
    print("\n[步骤 3/5] 训练机器学习单产回归预测模型...")
    predictor = YieldPredictor(config)
    train_results = predictor.train_and_evaluate(df_features, df_survey)
    df_all_preds = predictor.predict_all_parcels(df_features)
    print(f"  -> 已为全域所有 {len(df_all_preds)} 个地块生成初步卫星单产预测值。")

    # 5. 联合国 PPI 预测增强推断统计去偏
    print("\n[步骤 4/5] 执行联合国手册第 26 章预测增强推断（PPI）统计去偏...")
    evaluator = PPIEvaluator(config)
    ppi_results = evaluator.evaluate_yield(df_all_preds, df_survey)

    # 计算 PTD 偏置修正量，用于校准地块级别的单产图
    omega = ppi_results["tuning_parameter_omega"]
    mean_survey = ppi_results["estimates"][0]["point_estimate_kg_ha"]
    mean_map_all = ppi_results["estimates"][1]["point_estimate_kg_ha"]
    ptd_estimate = ppi_results["estimates"][2]["point_estimate_kg_ha"]

    # 新增校准后单产列：修正系统偏差，使其均值符合 PTD 理论无偏期望
    bias_shift = ptd_estimate - mean_map_all
    df_all_preds["ptd_calibrated_yield_kg_ha"] = (
        df_all_preds["map_pred_yield_kg_ha"] + bias_shift
    ).round(2)

    # 6. 保存预测结果与报告图表
    print("\n[步骤 5/5] 导出成果数据表、分析报告与可视化图表...")
    preds_out_path = os.path.join(output_dir, "parcel_yield_predictions.csv")
    df_all_preds.to_csv(preds_out_path, index=False)
    print(f"  -> 已保存全域地块单产预测清单至: {preds_out_path}")

    # 保存 JSON 摘要报告
    report = {
        "project": config.get("project", {}),
        "model_performance": train_results["metrics"],
        "top_features": dict(list(train_results["feature_importances"].items())[:8]),
        "ppi_statistical_inference": ppi_results,
    }
    report_out_path = os.path.join(output_dir, "summary_report.json")
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  -> 已保存综合统计报告至: {report_out_path}")

    # 生成可视化图表
    if config.get("visualization", {}).get("generate_plots", True):
        viz = Visualizer(config)
        p1 = viz.plot_actual_vs_predicted(train_results["cv_predictions"], train_results["metrics"])
        p2 = viz.plot_ppi_confidence_intervals(ppi_results)
        p3 = viz.plot_yield_distribution(df_all_preds)
        p4 = viz.plot_feature_importance(train_results["feature_importances"])
        print(f"  -> 已在 '{output_dir}/' 目录下生成高清可视化图表:")
        print(f"     * 1:1 交叉验证散点图: {os.path.basename(p1)}")
        print(f"     * 95% 置信区间对比图: {os.path.basename(p2)}")
        print(f"     * 单产空间分布直方图: {os.path.basename(p3)}")
        if p4:
            print(f"     * 关键特征重要性图: {os.path.basename(p4)}")

    # 7. 打印控制台权威统计汇总表
    print("\n" + "=" * 75)
    print("📊 联合国统计司 / 粮农组织（FAO）官方估产统计汇总表")
    print("=" * 75)
    print(f"{'估产技术路线':<26} | {'点估计值 (kg/ha)':<16} | {'95% 置信区间':<22} | {'统计是否无偏？'}")
    print("-" * 75)
    for est in ppi_results["estimates"]:
        ci_str = f"[{est['ci_95_lower']:.1f}, {est['ci_95_upper']:.1f}]"
        unbiased_str = "✓ 是 (数学保证无偏)" if est['unbiased'] else "✗ 否 (存在系统偏差)"
        print(f"{est['method']:<26} | {est['point_estimate_kg_ha']:<16.1f} | {ci_str:<22} | {unbiased_str}")
    print("-" * 75)
    print(f"🌟 联合国 PTD 去偏算法相比纯实地调查将不确定性区间收窄了 {ppi_results['variance_reduction_pct']}%！")
    print("=" * 75)
    print("🎉 产量预测流水线全部运行完毕！\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="联合国手册农作物产量预测流水线")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    run_pipeline(args.config)
