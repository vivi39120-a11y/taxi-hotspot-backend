import os
from pathlib import Path
from datetime import timedelta

import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
from pyproj import Transformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET", "taxi-artifacts")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

KEY_PARQUET = os.getenv("R2_KEY_PARQUET", "data/test_hourly.parquet")
KEY_MODEL = os.getenv("R2_KEY_MODEL", "model/xgb_demand_poisson.model")
KEY_CENT = os.getenv("R2_KEY_CENTROID", "meta/taxi_zone_centroids.csv")

PARQUET_PATH = DATA_DIR / "test_hourly.parquet"
MODEL_PATH = DATA_DIR / "xgb_demand_poisson.model"
CENT_PATH = DATA_DIR / "taxi_zone_centroids.csv"

ZONE_COL = "PULocationID"
RAW_TIME_COL = "pickup_hour"        # ✅ raw parquet 真的有的欄位
RAW_Y_COL = "rides"                 # ✅ raw parquet 真的有的欄位

PRED_TIME_COL = "predict_hour"      # ✅ 我們在後端產生
PRED_COL = "pred_rides"             # ✅ 我們在後端產生

FEATURE_COLS = [
    "PULocationID",
    "hour",
    "dow",
    "is_weekend",
    "lag_1",
    "lag_24",
    "ma_3",
    "ma_24",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {
    "booster": None,
    "cent": None,
    "hourly": None,        # raw df
    "latest_hour": None,   # latest pickup_hour
    "pred_df": None,       # df with your 10 columns
    "pred_hour": None,     # next hour timestamp
}

def s3_client():
    if not (R2_ENDPOINT and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        raise RuntimeError("Missing R2_ENDPOINT / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )

def download(key: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        return
    s3 = s3_client()
    dst.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(R2_BUCKET, key, str(dst))

def centroids_to_wgs84(df_cent: pd.DataFrame) -> pd.DataFrame:
    # centroid.csv: LocationID,Borough,Zone,lat,lon 其中 lat/lon 是 EPSG:2263 的 y/x
    t = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = t.transform(
        df_cent["lon"].astype(float).values,
        df_cent["lat"].astype(float).values
    )
    out = df_cent.copy()
    out["lon"] = lon_deg
    out["lat"] = lat_deg
    return out[["LocationID", "Borough", "Zone", "lat", "lon"]]

def build_features_for_hour(df_hourly: pd.DataFrame, target_hour: pd.Timestamp) -> pd.DataFrame:
    """
    target_hour: 你要預測的那個小時（predict_hour）
    使用 <= target_hour-1h 的歷史資料產生 lag/ma 特徵
    """
    if df_hourly is None or df_hourly.empty:
        return pd.DataFrame()

    base_hour = target_hour - pd.Timedelta(hours=1)

    rows = []
    for loc_id, g in df_hourly.groupby(ZONE_COL):
        g = g.sort_values(RAW_TIME_COL)
        g = g[g[RAW_TIME_COL] <= base_hour]
        y = g[RAW_Y_COL].to_numpy()

        if len(y) < 24:
            continue

        lag_1 = float(y[-1])
        lag_24 = float(y[-24])
        ma_3 = float(np.mean(y[-3:]))
        ma_24 = float(np.mean(y[-24:]))

        dow = int(target_hour.dayofweek)
        is_weekend = 1 if dow >= 5 else 0
        hour = int(target_hour.hour)

        rows.append({
            "PULocationID": int(loc_id),
            "hour": hour,
            "dow": dow,
            "is_weekend": is_weekend,
            "lag_1": lag_1,
            "lag_24": lag_24,
            "ma_3": ma_3,
            "ma_24": ma_24,
            "predict_hour": target_hour,
        })

    return pd.DataFrame(rows)

def predict_next_hour(df_hourly: pd.DataFrame, booster: xgb.Booster, target_hour: pd.Timestamp) -> pd.DataFrame:
    df_feat = build_features_for_hour(df_hourly, target_hour)
    if df_feat.empty:
        return df_feat

    dtest = xgb.DMatrix(df_feat[FEATURE_COLS])
    pred = booster.predict(dtest, validate_features=False)
    df_feat["pred_rides"] = pred
    return df_feat

@app.on_event("startup")
def startup():
    # 1) 下載檔案
    download(KEY_MODEL, MODEL_PATH)
    download(KEY_PARQUET, PARQUET_PATH)
    download(KEY_CENT, CENT_PATH)

    # 2) 載入模型
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    STATE["booster"] = booster

    # 3) centroid
    cent = pd.read_csv(CENT_PATH)
    STATE["cent"] = centroids_to_wgs84(cent)

    # 4) 讀 raw parquet（只需要 3 欄）
    df = pd.read_parquet(PARQUET_PATH, columns=[ZONE_COL, RAW_TIME_COL, RAW_Y_COL])
    df[RAW_TIME_COL] = pd.to_datetime(df[RAW_TIME_COL], errors="coerce")
    df = df.dropna(subset=[RAW_TIME_COL])
    STATE["hourly"] = df

    latest = df[RAW_TIME_COL].max()
    STATE["latest_hour"] = latest

    # 5) 預測下一小時
    if pd.isna(latest):
        STATE["pred_df"] = None
        STATE["pred_hour"] = None
        return

    next_hour = (latest + pd.Timedelta(hours=1)).floor("H")
    STATE["pred_hour"] = next_hour
    STATE["pred_df"] = predict_next_hour(df, booster, next_hour)

@app.get("/health")
def health():
    return {
        "ok": True,
        "latest_hour": None if STATE["latest_hour"] is None else str(STATE["latest_hour"]),
        "predict_hour": None if STATE["pred_hour"] is None else str(STATE["pred_hour"]),
        "pred_rows": 0 if STATE["pred_df"] is None else int(len(STATE["pred_df"])),
    }

@app.get("/api/hotspots")
def hotspots(n: int = 20, predict_hour: str | None = None):
    if STATE["cent"] is None or STATE["booster"] is None or STATE["hourly"] is None:
        raise HTTPException(status_code=500, detail="Server not ready")

    # 1) 決定要預測哪個小時
    if predict_hour is None:
        target = STATE["pred_hour"]
        if target is None:
            raise HTTPException(status_code=400, detail="predict_hour required")
    else:
        try:
            target = pd.to_datetime(predict_hour)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid predict_hour")

    # 2) 若 request 的 hour 不是 startup 的那個，就即時計算（資料量大就改成只允許 next hour）
    if STATE["pred_hour"] is not None and pd.Timestamp(target) == pd.Timestamp(STATE["pred_hour"]):
        dfp = STATE["pred_df"]
    else:
        dfp = predict_next_hour(STATE["hourly"], STATE["booster"], pd.Timestamp(target))

    if dfp is None or dfp.empty:
        raise HTTPException(status_code=404, detail=f"No prediction rows for {target}")

    out = (
        dfp.groupby(ZONE_COL, as_index=False)[PRED_COL].mean()
           .sort_values(PRED_COL, ascending=False)
           .head(int(n))
    )

    out = out.merge(STATE["cent"], left_on=ZONE_COL, right_on="LocationID", how="left")

    return {
        "predict_hour": str(pd.Timestamp(target)),
        "rows": out[["PULocationID", "pred_rides", "Borough", "Zone", "lat", "lon"]].to_dict(orient="records"),
    }
