from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import folium
from folium.plugins import HeatMap
from pyproj import Transformer

# ✅ 移除原本全域執行（頂層）的路徑檢查邏輯，避免 import 時報錯
# 所有的路徑與模型物件都將透過 run_prediction_task 傳入

def run_prediction_task(booster, df_hourly, cent_path, output_csv_path, output_html_path=None):
    """
    由 main.py 呼叫的核心預測函式
    booster: 已載入的模型物件
    df_hourly: 歷史每小時資料 (DataFrame)
    cent_path: Centroid 檔案路徑 (Path)
    """
    print("🔮 開始執行進階預測任務...")
    df_hourly = df_hourly.copy()
    df_hourly["pickup_hour"] = pd.to_datetime(df_hourly["pickup_hour"], errors="coerce")
    df_hourly = df_hourly.dropna(subset=["pickup_hour"])

    # 1. 時間處理
    last_hour = df_hourly["pickup_hour"].max()
    next_hour = last_hour + pd.Timedelta(hours=1)
    
    # 2. 建立特徵 (邏輯保留)
    rows = []
    for loc_id, g in df_hourly.groupby("PULocationID"):
        g = g.sort_values("pickup_hour")
        g = g[g["pickup_hour"] <= last_hour]
        y = g["rides"].values
        if len(y) < 24: continue

        rows.append({
            "PULocationID": loc_id,
            "hour": next_hour.hour,
            "dow": next_hour.dayofweek,
            "is_weekend": 1 if next_hour.dayofweek >= 5 else 0,
            "lag_1": float(y[-1]),
            "lag_24": float(y[-24]),
            "ma_3": float(np.mean(y[-3:])),
            "ma_24": float(np.mean(y[-24:])),
            "predict_hour": next_hour
        })

    df_feat = pd.DataFrame(rows)
    if df_feat.empty:
        print("⚠️ 沒有足夠資料可預測")
        return pd.DataFrame()

    feature_cols = ["PULocationID", "hour", "dow", "is_weekend", "lag_1", "lag_24", "ma_3", "ma_24"]

    # 3. 預測
    dtest = xgb.DMatrix(df_feat[feature_cols])
    df_feat["pred_rides"] = booster.predict(dtest, validate_features=False)

    # 4. 輸出 CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 預測已存檔：{output_csv_path}")

    # 5. 地圖視覺化 (若有提供 html 路徑)
    if output_html_path:
        df_cent = pd.read_csv(cent_path)
        transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
        
        df_merged = df_feat.merge(
            df_cent[["LocationID", "lat", "lon"]],
            left_on="PULocationID",
            right_on="LocationID",
            how="left"
        )
        
        lons, lats = transformer.transform(df_merged["lon"].values, df_merged["lat"].values)
        df_merged["lat_wgs"], df_merged["lon_wgs"] = lats, lons
        
        center = [df_merged["lat_wgs"].mean(), df_merged["lon_wgs"].mean()]
        m = folium.Map(location=center, zoom_start=11)
        heat_data = df_merged[["lat_wgs", "lon_wgs", "pred_rides"]].dropna().values.tolist()
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
        
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_html_path))
        print(f"✅ 熱力圖已存檔：{output_html_path}")

    return df_feat