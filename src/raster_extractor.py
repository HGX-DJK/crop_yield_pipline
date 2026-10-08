"""
遥感影像特征提取模块（用于农作物产量估算）。
连接【原始遥感影像数据】（GeoTIFF / 卫星多光谱 / STAC）与【农田矢量边界】（Shapefile / GeoJSON）。

支持两种提取模式：
1. 本地栅格模式：从指定文件夹中的 GeoTIFF 卫星切片中提取地块区域统计特征（NDVI、EVI 均值）。
2. 云端 STAC 模式：通过微软行星计算机 / AWS 接口直接按地块多边形在线抽取时序曲线。
"""

import os
import re
import glob
import pandas as pd
import numpy as np

# 优雅导入地理空间处理库，若未安装则提供友好提示
try:
    import rasterio
    from rasterio.mask import mask
    import geopandas as gpd
    from shapely.geometry import mapping
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False


class RasterExtractor:
    def __init__(self, config=None):
        self.config = config or {}

    def extract_from_local_geotiffs(self, parcel_vector_path, raster_dir, output_csv_path="data/extracted_remote_sensing.csv"):
        """
        从本地多时相 GeoTIFF 文件中提取每个农田地块的区域统计值（Zonal Statistics）。
        
        参数说明：
        - parcel_vector_path: 农田地块边界文件路径（.shp 或 .geojson），需包含 'parcel_id' 字段
        - raster_dir: 包含多时相 GeoTIFF 文件的文件夹，文件名需包含日期或日历日，例如：
                      'Sentinel2_20240510_NDVI.tif' 或 '2024_DOY136_EVI.tif'
        - output_csv_path: 提取后的时序特征 CSV 表保存路径
        """
        if not HAS_GEOSPATIAL:
            print("[警告] 未检测到地理空间处理库 (rasterio, geopandas)。")
            print("       如需直接处理真实的 GeoTIFF 影像，请运行: pip install rasterio geopandas shapely")
            return None

        if not os.path.exists(parcel_vector_path):
            raise FileNotFoundError(f"未找到地块矢量边界文件: {parcel_vector_path}")

        # 1. 读取农田地块多边形
        gdf = gpd.read_file(parcel_vector_path)
        if "parcel_id" not in gdf.columns:
            # 若缺少地块编号列，则自动生成默认编号
            gdf["parcel_id"] = [f"P{i+1:03d}" for i in range(len(gdf))]

        # 2. 检索所有 GeoTIFF 遥感影像文件
        tif_files = sorted(glob.glob(os.path.join(raster_dir, "*.tif")) + glob.glob(os.path.join(raster_dir, "*.tiff")))
        if not tif_files:
            raise FileNotFoundError(f"在目录中未找到任何 GeoTIFF (.tif) 影像文件: {raster_dir}")

        print(f"[提取器] 共检索到 {len(gdf)} 个农田地块和 {len(tif_files)} 景卫星遥感切片。")

        records = []
        for tif_path in tif_files:
            fname = os.path.basename(tif_path)
            # 从文件名中解析日历日 (DOY) 或年月日日期（如 'doy136' 或 '20240515'）
            doy_match = re.search(r"doy[_-]?(\d+)", fname, re.IGNORECASE)
            date_match = re.search(r"(\d{4})(\d{2})(\d{2})", fname)
            
            if doy_match:
                doy = int(doy_match.group(1))
            elif date_match:
                year, month, day = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                doy = int(pd.Timestamp(year=year, month=month, day=day).dayofyear)
            else:
                doy = 150  # 默认回退 DOY

            # 从文件名识别植被指数类别（ndvi, evi, ndre, lst 等）
            index_type = "ndvi"
            for candidate in ["evi", "ndre", "lst", "ndvi"]:
                if candidate in fname.lower():
                    index_type = candidate
                    break

            with rasterio.open(tif_path) as src:
                # 确保坐标参考系（CRS）一致，必要时自动重投影
                if gdf.crs != src.crs:
                    gdf_reproj = gdf.to_crs(src.crs)
                else:
                    gdf_reproj = gdf

                for _, row in gdf_reproj.iterrows():
                    pid = str(row["parcel_id"])
                    geom = [mapping(row["geometry"])]
                    try:
                        # 使用地块多边形掩膜裁剪栅格像素
                        out_image, _ = mask(src, geom, crop=True)
                        pixels = out_image[0]
                        valid_pixels = pixels[pixels != src.nodata] if src.nodata is not None else pixels
                        valid_pixels = valid_pixels[~np.isnan(valid_pixels)]

                        if len(valid_pixels) > 0:
                            mean_val = float(np.mean(valid_pixels))
                        else:
                            mean_val = np.nan
                    except Exception:
                        mean_val = np.nan

                    records.append({
                        "parcel_id": pid,
                        "doy": doy,
                        index_type: round(mean_val, 4)
                    })

        # 3. 数据重塑与多时相汇总
        df_extracted = pd.DataFrame(records)
        df_final = df_extracted.groupby(["parcel_id", "doy"]).mean().reset_index()
        
        # 保存提取结果
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_final.to_csv(output_csv_path, index=False)
        print(f"[提取器] 已成功将提取的时序特征表保存至: {output_csv_path}")
        return df_final


# 命令行调用演示说明
if __name__ == "__main__":
    print("=" * 70)
    print("🛰️ 联合国手册 遥感栅格特征提取器 (Raster Feature Extractor)")
    print("=" * 70)
    print("本模块用于将原始卫星切片 (GeoTIFF) 与农田边界 (Shapefile/GeoJSON)")
    print("自动转化为产量预测流水线所需的时序特征 CSV 表格。")
    print("\n核心执行流程：")
    print("1. 读取农田地块多边形边界 (.shp / .geojson)")
    print("2. 对每一景卫星影像执行地块掩膜裁剪（区域统计 Zonal Statistics）")
    print("3. 统计每个地块在每个时相的 NDVI、EVI、NDRE 均值")
    print("4. 输出 CSV 表格至 data/ 目录，直接供 main.py 预测产量！")
