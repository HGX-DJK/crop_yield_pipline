"""
农作物产量估算特征工程模块。
实现功能：
1. 生长度日（GDD，有效积温）计算与热量时间重采样（对应手册第 21 章）
2. 物候与生物量特征提取（NDVI/EVI 峰值、全周期积分 AUC、返青拔节速率等）
3. 多时相遥感与气象特征矩阵构建
"""

import numpy as np
import pandas as pd


def _trapezoid_integral(y, x):
    """梯形数值积分兼容函数（自适应兼容 NumPy 1.x 的 np.trapz 与 NumPy 2.0+ 的 np.trapezoid）。"""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    elif hasattr(np, "trapz"):
        return np.trapz(y, x)
    else:
        from scipy.integrate import trapezoid
        return trapezoid(y, x)


class FeatureEngineer:

    def __init__(self, config):
        self.config = config
        self.pheno_cfg = config.get("phenology_and_gdd", {})
        self.base_temp = self.pheno_cfg.get("base_temperature", 10.0)
        self.checkpoints = self.pheno_cfg.get("gdd_checkpoints", [300, 600, 900, 1200])

    def compute_growing_degree_days(self, df_weather):
        """
        基于气象数据计算每日生长度日（GDD）与累积有效积温。
        计算公式：GDD_d = max(((T_max + T_min)/2 - T_base), 0)
        """
        df = df_weather.copy()
        if "daily_gdd" not in df.columns:
            mean_temp = (df["t_max"] + df["t_min"]) / 2.0
            daily_gdd = np.maximum(mean_temp - self.base_temp, 0.0)
            df["daily_gdd"] = daily_gdd
        
        if "cum_gdd" not in df.columns:
            df["cum_gdd"] = df["daily_gdd"].cumsum()
        
        return df

    def extract_parcel_features(self, df_rs, df_weather):
        """
        提取每个地块的全季物候与生物量特征。
        融合气象累积积温（GDD），提取关键生物物理学描述因子。
        """
        # 确保已计算累积积温 GDD
        weather_gdd = self.compute_growing_degree_days(df_weather)
        
        # 将累积积温与天气要素按日历日（DOY）合并到遥感观测数据中
        weather_subset = weather_gdd[["doy", "cum_gdd", "precip_mm", "solar_rad_mj"]]
        df_merged = pd.merge(df_rs, weather_subset, on="doy", how="left")
        
        # 向前/向后填补可能缺失的积温数值
        df_merged["cum_gdd"] = df_merged["cum_gdd"].ffill().bfill()
        
        # 汇总全生长季气象特征
        total_precip = weather_gdd["precip_mm"].sum()
        mean_solar = weather_gdd["solar_rad_mj"].mean()
        total_cum_gdd = weather_gdd["cum_gdd"].iloc[-1]

        feature_records = []
        parcels = df_merged["parcel_id"].unique()

        for pid in parcels:
            pdf = df_merged[df_merged["parcel_id"] == pid].sort_values("doy")
            
            # 1. 植被指数极值与全季均值
            ndvi_max = pdf["ndvi"].max()
            ndvi_mean = pdf["ndvi"].mean()
            evi_max = pdf["evi"].max()
            evi_mean = pdf["evi"].mean()
            ndre_max = pdf["ndre"].max() if "ndre" in pdf.columns else np.nan
            
            # 2. 绿度峰值出现的日期与对应积温节点
            peak_row = pdf.loc[pdf["ndvi"].idxmax()]
            doy_peak = peak_row["doy"]
            gdd_at_peak = peak_row["cum_gdd"]
            
            # 3. 植被指数时序积分 AUC（反映全生长季光合作用干物质积累总量）
            # 采用梯形数值积分法沿日历日（DOY）进行积分
            ndvi_auc = _trapezoid_integral(pdf["ndvi"].values, pdf["doy"].values)
            evi_auc = _trapezoid_integral(pdf["evi"].values, pdf["doy"].values)

            
            # 4. 返青拔节与衰老成熟阶段的变化速率
            early_row = pdf.iloc[0]
            late_row = pdf.iloc[-1]
            green_up_rate = (ndvi_max - early_row["ndvi"]) / max(doy_peak - early_row["doy"], 1)
            senescence_rate = (ndvi_max - late_row["ndvi"]) / max(late_row["doy"] - doy_peak, 1)
            
            # 5. 生殖生长阶段的热量与干旱胁迫（围绕作物峰值灌浆期的地表温度 LST）
            mid_season = pdf[(pdf["doy"] >= doy_peak - 20) & (pdf["doy"] <= doy_peak + 20)]
            lst_peak_season = mid_season["lst_celsius"].mean() if not mid_season.empty else pdf["lst_celsius"].mean()
            
            # 6. 热量时间关键积温节点插值（GDD 300, 600, 900, 1200 节点）
            checkpoint_features = {}
            for gdd_cp in self.checkpoints:
                # 沿累积积温轴进行线性插值，对齐各年份的植被指数
                if pdf["cum_gdd"].min() <= gdd_cp <= pdf["cum_gdd"].max():
                    interp_ndvi = np.interp(gdd_cp, pdf["cum_gdd"], pdf["ndvi"])
                else:
                    interp_ndvi = ndvi_mean
                checkpoint_features[f"ndvi_gdd_{gdd_cp}"] = round(interp_ndvi, 4)

            record = {
                "parcel_id": pid,
                "ndvi_max": round(ndvi_max, 4),
                "ndvi_mean": round(ndvi_mean, 4),
                "ndvi_auc": round(ndvi_auc, 2),
                "evi_max": round(evi_max, 4),
                "evi_mean": round(evi_mean, 4),
                "evi_auc": round(evi_auc, 2),
                "ndre_max": round(ndre_max, 4),
                "doy_peak": int(doy_peak),
                "gdd_at_peak": round(gdd_at_peak, 1),
                "green_up_rate": round(green_up_rate, 4),
                "senescence_rate": round(senescence_rate, 4),
                "lst_peak_season": round(lst_peak_season, 2),
                "total_precip": round(total_precip, 1),
                "mean_solar_rad": round(mean_solar, 2),
                "total_cum_gdd": round(total_cum_gdd, 1),
            }
            record.update(checkpoint_features)
            feature_records.append(record)

        df_features = pd.DataFrame(feature_records)
        return df_features
