# 🌾 农作物产量提取与统计推断系统 (Crop Yield Estimation Pipeline)

本项目基于**联合国粮农组织 (FAO) 与联合国统计司 (UNSD)** 联合发布的《农业统计遥感手册》（*UN Handbook on Remote Sensing for Agricultural Statistics*）构建，是一个端到端、模块化且开箱即用的农作物产量提取与去偏统计推断系统。

---

## 📖 理论渊源与核心技术

传统遥感估产往往直接通过“卫星特征拟合产量模型”并对全域像元求平均。这种“仅依赖地图（Map-Only）”的估产方式存在严重系统性偏差（分类误差、云雨遮挡、尺度效应）。而传统“地面调查（Survey-Only）”虽然统计无偏，但抽样率低、置信区间极宽、成本昂贵。

本项目融合了手册中的两大标杆方案：
1. **农业气象与生长度日（GDD）对齐（手册第 21 章 - 波兰统计局）**：
   将日历时间转化为“有效积温热量时间”，计算植被指数时序积分（AUC）、绿度峰值与生长度日相关性，消除年际物候错位。
2. **预测增强推断（PPI / Predict-Then-Debias，手册第 26 章 - MIT & Stanford）**：
   利用大量卫星预测作为代理变量（Proxy），配合少量实地测产抽样样点（Ground Truth），在数学上严格消除地图偏差：
   $$\hat{\theta}_{\text{PTD}} = \omega \hat{\gamma}_{\text{map,all}} + (\bar{\hat{\theta}}_{\text{GT,survey}} - \omega \bar{\hat{\gamma}}_{\text{map,survey}})$$
   通过 Bootstrap 重抽样输出具备法律和统计合规性的 95% 置信区间，并将不确定性范围收窄 40%~60%。

---

## 📁 目录结构

```
crop_yield_pipeline/
├── README.md                      # 本说明文档
├── requirements.txt               # 运行依赖清单
├── config.yaml                    # 配置文件（作物基温、模型超参数、数据路径等）
├── demo_quickstart.py             # 零门槛极简单文件演示脚本
├── main.py                        # 一键全流程主入口
├── data/                          # 数据目录（内置真实模拟的基准测试集）
│   ├── sample_remote_sensing.csv  # 遥感多时相地块观测（NDVI, EVI, NDRE, LST）
│   ├── sample_weather.csv         # 逐日气象要素（Tmin, Tmax, 降水, GDD）
│   └── sample_ground_survey.csv   # 地面小样本抽样实测单产（用于模型校验与 PPI 去偏）
├── src/                           # 核心算法模块
│   ├── __init__.py
│   ├── data_loader.py             # 多源农业数据加载与验证
│   ├── feature_engineering.py     # GDD 积温计算、热量时间对齐与物候特征提取
│   ├── models.py                  # 产量回归预测模型（Random Forest / XGBoost / Ridge）
│   ├── ppi_evaluator.py           # 联合国第 26 章 Predict-Then-Debias 统计去偏与 Bootstrap 引擎
│   └── visualizer.py              # 成果制图模块（散点图、直方图、置信区间对比图）
└── output/                        # 运行生成目录（预测结果 CSV、评估报告 JSON、高清图表）
```

---

## 🚀 快速开始

### 1. 安装依赖

推荐在 Python 3.8+ 环境下运行：

```bash
cd crop_yield_pipeline
pip install -r requirements.txt
```

> **注意**：如果未安装 `xgboost` 或 `matplotlib`，程序具备完备的容错机制（自动平滑回退到 `scikit-learn` 的梯度提升/随机森林，或以纯文本/无图模式运行）。

### 2. 方式一：快速单文件试跑 (Demo Quickstart)

无需配置任何路径，直接执行极简单文件：

```bash
python demo_quickstart.py
```

终端将立即展示：
* 200 个地块的多源特征生成；
* 随机森林模型的训练；
* Survey-Only、Map-Only、Predict-Then-Debias 三种估产方法的对比表及置信区间收窄幅度。

### 3. 方式二：运行完整生产流水线 (Full Pipeline)

```bash
python main.py --config config.yaml
```

执行后，系统将自动完成五大步骤并在 `output/` 目录下生成成果：
* **`output/parcel_yield_predictions.csv`**：所有地块的初步卫星预测产量与经过 PPI 校正后的最终单产。
* **`output/summary_report.json`**：包含模型交叉验证指标（$R^2$, RMSE, MAPE）、特征重要性排名、不同分位数产量估算。
* **高清成果图表**：
  * `actual_vs_predicted.png`：实测与预测 1:1 对比散点图；
  * `ppi_confidence_intervals.png`：三种估产方法的 95% 置信区间收窄对比图；
  * `yield_distribution.png`：全域农田地块单产空间分布直方图；
  * `feature_importance.png`：对产量影响最大的关键物候特征排名。

---

## ⚙️ 核心参数定制 (`config.yaml`)

您可以轻松调整 `config.yaml` 适配不同的农作物与地区：

```yaml
phenology_and_gdd:
  base_temperature: 10.0      # 作物生长度日基础温度（玉米/大豆为 10°C，小麦为 5°C）
  gdd_checkpoints: [300, 600, 900, 1200]  # 采样积温节点

modeling:
  model_type: "random_forest"  # 支持 'random_forest', 'xgboost', 'ridge'
  rf_params:
    n_estimators: 100
    max_depth: 8

ppi_inference:
  n_bootstrap: 2000            # Bootstrap 重抽样次数
  confidence_level: 0.95       # 置信水平（95% 置信区间）
  quantiles: [0.25, 0.50, 0.75] # 估算的单产分位数（评估低产脆弱户 vs 高产示范田）
```

---

## 🌾 如何接入您自己的实际生产数据？

替换 `data/` 目录中的数据文件即可：
1. **遥感数据 (`sample_remote_sensing.csv`)**：提供字段 `parcel_id, doy, ndvi, evi, lst_celsius`（可通过 Google Earth Engine、Sentinel-2 或 UAV 多光谱反演导出）。
2. **气象数据 (`sample_weather.csv`)**：提供字段 `date, doy, t_min, t_max, precip_mm`（可通过 ERA-5 气象重分析数据或当地农业气象站导出）。
3. **地面调查数据 (`sample_ground_survey.csv`)**：提供抽样实测的实割实测单产 `parcel_id, ground_truth_yield_kg_ha`（仅需数十个样点即可激活 PPI 算法去偏）。
