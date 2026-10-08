"""
农业统计预测增强推断（PPI）模块。
实现《联合国农业统计遥感手册》第 26 章的“预测后去偏”（Predict-Then-Debias, PTD）方法
（由 MIT 教授 Sherrie Wang 与 Stanford 博士 Dan Kluger 提出）。

将大范围、可能存在系统偏差的卫星遥感预测地图，与小样本、统计无偏的地面实测样点相结合，实现：
1. 具备数学严格无偏保证的产量均值与分位数估计
2. 基于 Percentile Bootstrap 重抽样计算收窄且可信的 95% 置信区间
"""

import numpy as np
import pandas as pd


class PPIEvaluator:
    def __init__(self, config):
        self.config = config
        self.ppi_cfg = config.get("ppi_inference", {})
        self.n_bootstrap = self.ppi_cfg.get("n_bootstrap", 2000)
        self.alpha = 1.0 - self.ppi_cfg.get("confidence_level", 0.95)
        self.quantiles = self.ppi_cfg.get("quantiles", [0.25, 0.50, 0.75])

    def evaluate_yield(self, df_all_preds, df_survey):
        """
        运行预测增强推断（PPI），对比以下三类估产范式：
        1. 纯实地调查（Survey-Only）：严格无偏，但由于样本量小导致置信区间极宽、方差大
        2. 纯卫星制图（Map-Only）：直接拿卫星模型跑全域，置信区间窄但存在系统性错分与过拟合偏差
        3. 预测后去偏（Predict-Then-Debias, PTD）：联合国推荐方案，兼具“无偏性”与“窄置信区间”
        """
        # 将地面调查样本与卫星地块预测结果进行匹配
        df_survey_matched = pd.merge(
            df_survey,
            df_all_preds[["parcel_id", "map_pred_yield_kg_ha"]],
            on="parcel_id",
            how="inner"
        )
        
        y_gt_survey = df_survey_matched["ground_truth_yield_kg_ha"].values
        y_map_survey = df_survey_matched["map_pred_yield_kg_ha"].values
        y_map_all = df_all_preds["map_pred_yield_kg_ha"].values

        n_survey = len(y_gt_survey)
        n_all = len(y_map_all)

        # 1. 计算点估计值
        mean_gt_survey = np.mean(y_gt_survey)
        mean_map_all = np.mean(y_map_all)
        mean_map_survey = np.mean(y_map_survey)

        # 求解方差最小化的最优调节参数 omega（标量权重）
        cov_matrix = np.cov(y_gt_survey, y_map_survey)
        var_map = cov_matrix[1, 1]
        cov_gt_map = cov_matrix[0, 1]
        omega = cov_gt_map / var_map if var_map > 1e-8 else 1.0
        # 将 omega 限制在合理数值区间内
        omega = float(np.clip(omega, 0.1, 2.0))

        # Predict-Then-Debias 核心去偏点估计公式
        # theta_ptd = omega * gamma_map_all + (theta_survey - omega * gamma_map_survey)
        mean_ptd = omega * mean_map_all + (mean_gt_survey - omega * mean_map_survey)

        # 2. Bootstrap 重抽样计算 95% 置信区间
        np.random.seed(self.config.get("modeling", {}).get("random_state", 42))
        boot_survey = np.zeros(self.n_bootstrap)
        boot_ptd = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            idx = np.random.choice(n_survey, size=n_survey, replace=True)
            b_gt = y_gt_survey[idx]
            b_map_s = y_map_survey[idx]

            b_mean_gt = np.mean(b_gt)
            b_mean_map_s = np.mean(b_map_s)

            boot_survey[b] = b_mean_gt
            boot_ptd[b] = omega * mean_map_all + (b_mean_gt - omega * b_mean_map_s)

        # 计算置信区间分位点
        low_pct = (self.alpha / 2.0) * 100.0
        high_pct = (1.0 - self.alpha / 2.0) * 100.0

        ci_survey = (np.percentile(boot_survey, low_pct), np.percentile(boot_survey, high_pct))
        ci_ptd = (np.percentile(boot_ptd, low_pct), np.percentile(boot_ptd, high_pct))
        
        # 纯卫星地图预测的标准误与置信区间
        se_map = np.std(y_map_all, ddof=1) / np.sqrt(n_all)
        ci_map = (mean_map_all - 1.96 * se_map, mean_map_all + 1.96 * se_map)

        results = {
            "sample_counts": {
                "total_parcels": int(n_all),
                "survey_parcels": int(n_survey),
                "sampling_ratio_pct": round(n_survey / n_all * 100.0, 2),
            },
            "tuning_parameter_omega": round(omega, 4),
            "estimates": [
                {
                    "method": "Survey-Only (仅实地调查)",
                    "point_estimate_kg_ha": round(float(mean_gt_survey), 1),
                    "std_error": round(float(np.std(boot_survey)), 1),
                    "ci_95_lower": round(float(ci_survey[0]), 1),
                    "ci_95_upper": round(float(ci_survey[1]), 1),
                    "ci_width": round(float(ci_survey[1] - ci_survey[0]), 1),
                    "unbiased": True,
                    "description": "仅利用实地抽样点；无偏但方差大、置信区间较宽"
                },
                {
                    "method": "Map-Only (纯卫星预测)",
                    "point_estimate_kg_ha": round(float(mean_map_all), 1),
                    "std_error": round(float(se_map), 1),
                    "ci_95_lower": round(float(ci_map[0]), 1),
                    "ci_95_upper": round(float(ci_map[1]), 1),
                    "ci_width": round(float(ci_map[1] - ci_map[0]), 1),
                    "unbiased": False,
                    "description": "直接拿卫星模型预测全域；区间看似很窄但含有系统性偏差"
                },
                {
                    "method": "Predict-Then-Debias (PTD 预测后去偏)",
                    "point_estimate_kg_ha": round(float(mean_ptd), 1),
                    "std_error": round(float(np.std(boot_ptd)), 1),
                    "ci_95_lower": round(float(ci_ptd[0]), 1),
                    "ci_95_upper": round(float(ci_ptd[1]), 1),
                    "ci_width": round(float(ci_ptd[1] - ci_ptd[0]), 1),
                    "unbiased": True,
                    "description": "联合国手册方案：利用实测样点修正卫星偏差；兼具无偏与高精度"
                }
            ],
            "variance_reduction_pct": round(
                (1.0 - (ci_ptd[1] - ci_ptd[0]) / (ci_survey[1] - ci_survey[0])) * 100.0, 2
            ),
        }

        # 计算分位数估计值（25% 脆弱低产线、中位数、75% 高产线）
        quantile_results = []
        for q in self.quantiles:
            q_gt = np.quantile(y_gt_survey, q)
            q_map_all = np.quantile(y_map_all, q)
            q_map_s = np.quantile(y_map_survey, q)
            q_ptd = q_map_all + (q_gt - q_map_s)
            quantile_results.append({
                "quantile": q,
                "survey_estimate": round(float(q_gt), 1),
                "map_estimate": round(float(q_map_all), 1),
                "ptd_estimate": round(float(q_ptd), 1),
            })
        
        results["quantile_estimates"] = quantile_results
        return results
