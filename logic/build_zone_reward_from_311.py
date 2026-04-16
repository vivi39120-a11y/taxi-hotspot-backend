from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import Transformer

# ===== Min-Max Norm（含全相等保護：全 0 → 全 0，符合「補 0」公平性）=====
def minmax_norm(s: pd.Series) -> pd.Series:
    s = s.fillna(0).astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

# ===== 用最近 centroid 把 311 點位配到 Taxi Zone =====
def nearest_zone_index(lat_arr, lon_arr, zones_lat, zones_lon):
    """
    回傳每個 311 點對應到 zones 的 index（最近 centroid）
    """
    lat_arr = np.asarray(lat_arr, dtype=float)
    lon_arr = np.asarray(lon_arr, dtype=float)
    zones_lat = np.asarray(zones_lat, dtype=float)
    zones_lon = np.asarray(zones_lon, dtype=float)

    cos_lat = np.cos(np.deg2rad(lat_arr))
    best = np.empty(len(lat_arr), dtype=int)

    batch = 8000
    for i in range(0, len(lat_arr), batch):
        sl = slice(i, min(i + batch, len(lat_arr)))
        dlat = lat_arr[sl, None] - zones_lat[None, :]
        dlon = (lon_arr[sl, None] - zones_lon[None, :]) * cos_lat[sl, None]
        dist2 = dlat * dlat + dlon * dlon
        best[sl] = dist2.argmin(axis=1)

    return best

# ✅ 核心函式化：由 main.py 傳入路徑，不再自己猜測路徑
def run_311_reward_analysis(path_311, path_cent, out_path):
    """
    執行 311 獎懲機制分析
    """
    print(f"⚖️ 正在分析 311 報案數據...")

    if not path_311.exists():
        raise FileNotFoundError(f"❌ 找不到 311 原始檔案：{path_311}")
    if not path_cent.exists():
        raise FileNotFoundError(f"❌ 找不到 Centroid 檔案：{path_cent}")

    # =========================
    # 讀檔與座標轉換
    # =========================
    df311 = pd.read_csv(path_311)
    dfcent = pd.read_csv(path_cent)

    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    lon_wgs, lat_wgs = transformer.transform(dfcent["lon"].values, dfcent["lat"].values)
    dfcent["lon_wgs"] = np.round(lon_wgs, 6)
    dfcent["lat_wgs"] = np.round(lat_wgs, 6)

    # =========================
    # 311 時間與分類處理 (邏輯保留)
    # =========================
    df311["created_date"] = pd.to_datetime(df311["created_date"], errors="coerce")
    df311 = df311.dropna(subset=["created_date", "latitude", "longitude"]).copy()
    df311["hour_ts"] = df311["created_date"].dt.floor("h")

    latest_hour = df311["hour_ts"].max()
    df311 = df311[df311["hour_ts"] == latest_hour].copy()

    ct = df311["complaint_type"].astype(str).str.lower()
    desc = df311["descriptor"].astype(str).str.lower()

    # 需求面 (D) 與 阻礙面 (C) 分類
    demand_mask = (ct.str.startswith("noise") | desc.str.contains(r"party|loud music|music|loud", regex=True))
    constraint_mask = (ct.str.contains(r"illegal parking|blocked driveway|street condition|traffic", regex=True) | 
                      desc.str.contains(r"blocked|no access|double parked|obstruction|road", regex=True))

    df311["DC"] = np.where(demand_mask, "D", np.where(constraint_mask, "C", ""))
    df311 = df311[df311["DC"] != ""].copy()

    # =========================
    # 配對 Taxi Zone 與計算 DRS
    # =========================
    zones_lat = dfcent["lat_wgs"].to_numpy()
    zones_lon = dfcent["lon_wgs"].to_numpy()

    idx = nearest_zone_index(df311["latitude"].to_numpy(), df311["longitude"].to_numpy(), zones_lat, zones_lon)
    df311["PULocationID"] = dfcent["LocationID"].iloc[idx].to_numpy()

    g = df311.groupby(["PULocationID", "DC"]).size().unstack(fill_value=0)
    if "D" not in g.columns: g["D"] = 0
    if "C" not in g.columns: g["C"] = 0
    g = g.reset_index()[["PULocationID", "D", "C"]]

    all_zones = pd.DataFrame({"PULocationID": dfcent["LocationID"].astype(int)})
    z = all_zones.merge(g, on="PULocationID", how="left").fillna(0)

    # 計算 DRS 分數 (邏輯保留)
    alpha, beta = 0.7, 0.3
    z["D_norm"] = minmax_norm(z["D"])
    z["C_norm"] = minmax_norm(z["C"])
    z["DRS"] = alpha * z["D_norm"] - beta * z["C_norm"]

    # 最終 0-100 評分
    z["final_score"] = (minmax_norm(z["DRS"]) * 100).round(2)
    z["used_hour"] = str(latest_hour)

    # =========================
    # 輸出 (確保目錄存在)
    # =========================
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 獎勵機制分析完成 -> {out_path} (時段: {latest_hour})")
    
    return z

# 為了讓原本單獨執行功能還在，保留這段
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent # 修正單獨跑時的路徑
    run_311_reward_analysis(
        path_311=BASE / "rewards" / "nyc_311_2025_07.csv",
        path_cent=BASE / "data" / "taxi_zone_centroids.csv",
        out_path=BASE / "outputs" / "zone_reward.csv"
    )