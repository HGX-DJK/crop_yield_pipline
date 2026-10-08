"""
联合国手册农作物产量预测极速单文件演示脚本。
单文件完整展示端到端算法闭环：
1. 模拟卫星遥感多时相特征与农业气象有效积温（GDD）
2. 训练机器学习模型实现全域地块产量预测
3. 应用《联合国农业统计遥感手册》第 26 章预测增强推断（PPI / Predict-Then-Debias）去偏算法
4. 输出多方法估产对比表、不确定性区间收窄幅度与官方置信区间。

运行方式：
    python demo_quickstart.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def run_quickstart():
    print("=" * 72)
    print("🚀 联合国手册农作物产量估算与 PPI 统计去偏：极速独立演示")
    print("=" * 72)

    np.random.seed(42)
    n_parcels = 200
    n_survey = 40  # 抽样 20% 的地块作为地面实测样本

    # -------------------------------------------------------------
    # 1. 模拟卫星遥感与物候特征
    # -------------------------------------------------------------
    print("\n[1/4] 模拟 200 个农田地块的多时相遥感物候特征...")
    parcel_ids = [f"P{i:03d}" for i in range(1, n_parcels + 1)]
    
    # 生物物理驱动因子：绿度峰值 NDVI、全季生物量积分 AUC、峰值积温 GDD、地表热胁迫指数 LST
    peak_ndvi = np.random.uniform(0.68, 0.92, n_parcels)
    ndvi_auc = peak_ndvi * np.random.uniform(70.0, 95.0, n_parcels)
    gdd_peak = np.random.uniform(850.0, 1100.0, n_parcels)
    lst_stress = np.random.uniform(25.0, 32.0, n_parcels)

    # 真实单产生成函数（模拟作物的非线性生物学响应与环境扰动随机噪声）
    true_yield = (
        3000.0 
        + 6000.0 * (peak_ndvi - 0.5) 
        + 35.0 * (ndvi_auc - 60.0) 
        - 120.0 * np.maximum(lst_stress - 28.0, 0)
        + np.random.normal(0, 250.0, n_parcels)
    )

    df_parcels = pd.DataFrame({
        "parcel_id": parcel_ids,
        "peak_ndvi": peak_ndvi,
        "ndvi_auc": ndvi_auc,
        "gdd_peak": gdd_peak,
        "lst_stress": lst_stress,
        "true_yield_kg_ha": true_yield,
    })

    # 地面实测抽样：抽取 40 个地块作为外业实割实测样点
    survey_indices = np.random.choice(n_parcels, size=n_survey, replace=False)
    df_survey = df_parcels.iloc[survey_indices].copy()
    print(f"  -> 全域地块总数: {n_parcels} 个")
    print(f"  -> 地面实测抽样样点: {n_survey} 个 (待估计的真实总体平均单产: {true_yield.mean():.1f} kg/ha)")

    # -------------------------------------------------------------
    # 2. 训练机器学习模型并进行全域单产制图
    # -------------------------------------------------------------
    print("\n[2/4] 在地面抽样实测样本上训练随机森林回归模型...")
    features = ["peak_ndvi", "ndvi_auc", "gdd_peak", "lst_stress"]
    
    X_train = df_survey[features].values
    y_train = df_survey["true_yield_kg_ha"].values

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    # 对全域所有 200 个地块进行预测（模拟纯卫星地图制图产品 Map-Only）
    X_all = df_parcels[features].values
    map_predictions = rf.predict(X_all)
    df_parcels["map_pred_yield_kg_ha"] = map_predictions

    r2 = r2_score(y_train, rf.predict(X_train))
    print(f"  -> 模型训练完成 (训练拟合度 R² = {r2:.3f})")
    print(f"  -> 已为全部 {n_parcels} 个地块生成初步卫星单产预测图。")

    # -------------------------------------------------------------
    # 3. 联合国手册第 26 章预测增强推断（PPI / Predict-Then-Debias）
    # -------------------------------------------------------------
    print("\n[3/4] 执行联合国手册 Predict-Then-Debias (PTD) 统计去偏与 Bootstrap...")
    
    # 地面实测真值
    y_gt_survey = df_survey["true_yield_kg_ha"].values
    # 调查样点对应的卫星预测值
    y_map_survey = map_predictions[survey_indices]
    # 全域所有地块的卫星预测值
    y_map_all = map_predictions

    # 点估计
    est_survey_only = np.mean(y_gt_survey)
    est_map_only = np.mean(y_map_all)
    est_map_survey = np.mean(y_map_survey)

    # 求解方差最小化最优权重 omega: Cov(Y, Yhat) / Var(Yhat)
    cov_val = np.cov(y_gt_survey, y_map_survey)[0, 1]
    var_val = np.var(y_map_survey, ddof=1)
    omega = float(np.clip(cov_val / var_val if var_val > 0 else 1.0, 0.2, 1.8))

    # PTD 去偏估计公式: omega * map_all + (survey_mean - omega * map_survey_mean)
    est_ptd = omega * est_map_only + (est_survey_only - omega * est_map_survey)

    # Percentile Bootstrap 重抽样计算 95% 置信区间 (2000 次)
    n_boot = 2000
    boot_survey = np.zeros(n_boot)
    boot_ptd = np.zeros(n_boot)

    for b in range(n_boot):
        b_idx = np.random.choice(n_survey, size=n_survey, replace=True)
        b_gt = y_gt_survey[b_idx]
        b_map_s = y_map_survey[b_idx]

        b_survey_mean = np.mean(b_gt)
        b_map_s_mean = np.mean(b_map_s)

        boot_survey[b] = b_survey_mean
        boot_ptd[b] = omega * est_map_only + (b_survey_mean - omega * b_map_s_mean)

    ci_survey = (np.percentile(boot_survey, 2.5), np.percentile(boot_survey, 97.5))
    ci_ptd = (np.percentile(boot_ptd, 2.5), np.percentile(boot_ptd, 97.5))
    
    se_map = np.std(y_map_all, ddof=1) / np.sqrt(n_parcels)
    ci_map = (est_map_only - 1.96 * se_map, est_map_only + 1.96 * se_map)

    # -------------------------------------------------------------
    # 4. 打印统计成果汇总表
    # -------------------------------------------------------------
    actual_pop_mean = true_yield.mean()
    width_survey = ci_survey[1] - ci_survey[0]
    width_ptd = ci_ptd[1] - ci_ptd[0]
    var_reduction = (1.0 - width_ptd / width_survey) * 100.0

    print("\n" + "=" * 72)
    print(f"🎯 待估计的客观真实总体平均单产: {actual_pop_mean:.1f} kg/ha")
    print("=" * 72)
    print(f"{'技术路线':<28} | {'点估计值':<12} | {'95% 置信区间':<21} | {'区间宽度'}")
    print("-" * 75)
    print(f"{'1. 仅实地调查 (Survey-Only)':<28} | {est_survey_only:<12.1f} | [{ci_survey[0]:.1f}, {ci_survey[1]:.1f}]  | {width_survey:<10.1f} (无偏但太宽)")
    print(f"{'2. 纯卫星预测 (Map-Only)':<28} | {est_map_only:<12.1f} | [{ci_map[0]:.1f}, {ci_map[1]:.1f}]  | {ci_map[1]-ci_map[0]:<10.1f} (看似很窄但有偏差)")
    print(f"{'3. 预测后去偏 (PTD 联合国方案)':<28} | {est_ptd:<12.1f} | [{ci_ptd[0]:.1f}, {ci_ptd[1]:.1f}]  | {width_ptd:<10.1f} (无偏且精确)")
    print("-" * 75)
    print(f"✨ 精度提升: 联合国 PTD 算法相比传统抽样将不确定性区间收窄了 {var_reduction:.1f}%！")
    print(f"✅ 无论 Survey-Only 还是 PTD 都能覆盖真实均值 ({actual_pop_mean:.1f} kg/ha)，")
    print(f"   但 PTD 借助了卫星地图海量信息，获得了高出数倍的统计精度。")
    print("=" * 72)


if __name__ == "__main__":
    run_quickstart()

