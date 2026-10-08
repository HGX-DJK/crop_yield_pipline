"""
农作物产量估算数据加载模块。
用于读取并校验多源农业数据集：
- 卫星遥感时间序列（NDVI、EVI、NDRE、LST 等）
- 逐日农业气象要素（气温、降水、太阳辐射）
- 地面抽样调查实测单产记录（实割实测数据）
"""

import os
import yaml
import pandas as pd
import numpy as np


def load_config(config_path="config.yaml"):
    """加载 YAML 参数配置文件。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.paths = config.get("paths", {})

    def load_remote_sensing_data(self, file_path=None):
        """读取并校验多时相卫星遥感观测数据。"""
        path = file_path or self.paths.get("remote_sensing_file", "data/sample_remote_sensing.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到遥感数据文件: {path}")
        
        df_rs = pd.read_csv(path)
        required_cols = {"parcel_id", "doy"}
        if not required_cols.issubset(df_rs.columns):
            raise ValueError(f"遥感数据必须包含以下列: {required_cols}")
        
        df_rs["parcel_id"] = df_rs["parcel_id"].astype(str)
        df_rs["doy"] = df_rs["doy"].astype(int)
        # 按地块编号和日历日排序
        df_rs = df_rs.sort_values(["parcel_id", "doy"]).reset_index(drop=True)
        return df_rs

    def load_weather_data(self, file_path=None):
        """读取逐日气象站/气候再分析数据。"""
        path = file_path or self.paths.get("weather_file", "data/sample_weather.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到气象数据文件: {path}")
        
        df_weather = pd.read_csv(path)
        if "doy" not in df_weather.columns:
            raise ValueError("气象数据必须包含 'doy' (日历日) 列。")
        
        df_weather["doy"] = df_weather["doy"].astype(int)
        df_weather = df_weather.sort_values("doy").reset_index(drop=True)
        return df_weather

    def load_ground_survey_data(self, file_path=None):
        """读取实地抽样校准数据（地面实测真实单产）。"""
        path = file_path or self.paths.get("survey_file", "data/sample_ground_survey.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到地面调查数据文件: {path}")
        
        df_survey = pd.read_csv(path)
        required_cols = {"parcel_id", "ground_truth_yield_kg_ha"}
        if not required_cols.issubset(df_survey.columns):
            raise ValueError(f"调查数据必须包含以下列: {required_cols}")
        
        df_survey["parcel_id"] = df_survey["parcel_id"].astype(str)
        df_survey["ground_truth_yield_kg_ha"] = df_survey["ground_truth_yield_kg_ha"].astype(float)
        return df_survey
