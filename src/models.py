"""
机器学习作物产量预测建模模块。
实现基于随机森林（Random Forest）、XGBoost 和 Ridge 岭回归的单产预测估算，
支持 K 折交叉验证、特征重要性评估与全域地块预测。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 优雅导入 XGBoost，若未安装则自动回退
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class YieldPredictor:
    def __init__(self, config):
        self.config = config
        self.model_cfg = config.get("modeling", {})
        self.model_type = self.model_cfg.get("model_type", "random_forest").lower()
        self.random_state = self.model_cfg.get("random_state", 42)
        self.model = self._init_model()
        self.feature_columns = None
        self.metrics = {}

    def _init_model(self):
        rf_params = self.model_cfg.get("rf_params", {})
        xgb_params = self.model_cfg.get("xgb_params", {})

        if self.model_type == "xgboost":
            if HAS_XGBOOST:
                return XGBRegressor(
                    n_estimators=xgb_params.get("n_estimators", 100),
                    max_depth=xgb_params.get("max_depth", 4),
                    learning_rate=xgb_params.get("learning_rate", 0.05),
                    random_state=self.random_state,
                )
            else:
                print("[提示] 未安装 XGBoost，自动平滑回退至 GradientBoostingRegressor。")
                return GradientBoostingRegressor(
                    n_estimators=xgb_params.get("n_estimators", 100),
                    max_depth=xgb_params.get("max_depth", 4),
                    learning_rate=xgb_params.get("learning_rate", 0.05),
                    random_state=self.random_state,
                )
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0)
        else:  # 默认使用随机森林 random_forest
            return RandomForestRegressor(
                n_estimators=rf_params.get("n_estimators", 100),
                max_depth=rf_params.get("max_depth", 8),
                min_samples_leaf=rf_params.get("min_samples_leaf", 2),
                random_state=self.random_state,
            )

    def train_and_evaluate(self, df_features, df_survey):
        """
        在具有地面实测真值调查记录的地块上训练回归模型。
        采用 5 折交叉验证评估预测精度。
        """
        # 将特征矩阵与地面调查真值按地块编号匹配
        train_df = pd.merge(df_features, df_survey[["parcel_id", "ground_truth_yield_kg_ha"]], on="parcel_id")
        
        if len(train_df) < 5:
            raise ValueError(f"用于训练模型的地面调查样本不足（仅 {len(train_df)} 个）。")

        # 排除地块 ID 列，提取自变量特征列
        feature_cols = [c for c in df_features.columns if c != "parcel_id"]
        self.feature_columns = feature_cols

        X = train_df[feature_cols].values
        y = train_df["ground_truth_yield_kg_ha"].values

        # K 折交叉验证
        cv_folds = self.model_cfg.get("cv_folds", 5)
        kf = KFold(n_splits=min(cv_folds, len(train_df)), shuffle=True, random_state=self.random_state)
        
        y_cv_pred = np.zeros_like(y)
        for train_idx, val_idx in kf.split(X):
            fold_model = self._init_model()
            fold_model.fit(X[train_idx], y[train_idx])
            y_cv_pred[val_idx] = fold_model.predict(X[val_idx])

        # 计算交叉验证精度指标
        r2 = r2_score(y, y_cv_pred)
        rmse = np.sqrt(mean_squared_error(y, y_cv_pred))
        mae = mean_absolute_error(y, y_cv_pred)
        mape = np.mean(np.abs((y - y_cv_pred) / y)) * 100.0

        self.metrics = {
            "r2": round(float(r2), 4),
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
            "mape_pct": round(float(mape), 2),
            "n_samples": int(len(train_df)),
        }

        # 在全部地面调查样本上训练最终模型
        self.model.fit(X, y)
        print(f"[模型] 已基于 {len(train_df)} 个实测地块完成 {self.model_type.upper()} 模型训练。")
        print(f"[指标] 交叉验证 R²: {r2:.3f}, RMSE: {rmse:.1f} kg/ha, MAPE: {mape:.2f}%")

        # 计算特征重要性
        importances = {}
        if hasattr(self.model, "feature_importances_"):
            for col, imp in zip(feature_cols, self.model.feature_importances_):
                importances[col] = round(float(imp), 4)
            importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

        return {
            "metrics": self.metrics,
            "feature_importances": importances,
            "cv_predictions": pd.DataFrame({
                "parcel_id": train_df["parcel_id"],
                "y_true": y,
                "y_pred": y_cv_pred,
            }),
        }

    def predict_all_parcels(self, df_features):
        """
        为全域所有地块生成初步卫星单产预测值（hat_gamma_map）。
        """
        if self.feature_columns is None:
            raise ValueError("模型必须先训练后方可执行预测。")

        X_all = df_features[self.feature_columns].values
        predictions = self.model.predict(X_all)

        df_preds = pd.DataFrame({
            "parcel_id": df_features["parcel_id"],
            "map_pred_yield_kg_ha": np.round(predictions, 2),
        })
        return df_preds
