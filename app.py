import os
from pathlib import Path

import boto3
import pandas as pd
import xgboost as xgb
from pyproj import Transformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

R2_ENDPOINT = os.getenv("R2_ENDPOINT")  # https://<accountid>.r2.cloudflarestorage.com
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
TIME_COL = "predict_hour"
PRED_COL = "pred_rides"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 跑通後改成你的前端 Render 網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"booster": None, "cent": None, "latest_hour": None}

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
    lon_deg, lat_deg = t.transform(df_cent["lon"].astype(float).values,
                                   df_cent["lat"].astype(float).values)
    out = df_cent.copy()
    out["lon"] = lon_deg
    out["lat"] = lat_deg
    return out[["LocationID", "Borough", "Zone", "lat", "lon"]]

@app.on_event("startup")
def startup():
    # 下載檔案（只在容器第一次啟動或檔案不存在時下載）
    download(KEY_MODEL, MODEL_PATH)
    download(KEY_PARQUET, PARQUET_PATH)
    download(KEY_CENT, CENT_PATH)

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    STATE["booster"] = booster

    cent = pd.read_csv(CENT_PATH)
    STATE["cent"] = centroids_to_wgs84(cent)

    dfh = pd.read_parquet(PARQUET_PATH, columns=[TIME_COL])
    STATE["latest_hour"] = dfh[TIME_COL].max()

@app.get("/health")
def health():
    return {"ok": True, "latest_hour": STATE["latest_hour"]}

@app.get("/api/hotspots")
def hotspots(n: int = 20, predict_hour: str | None = None):
    if STATE["cent"] is None:
        raise HTTPException(status_code=500, detail="Centroids not loaded")

    if predict_hour is None:
        predict_hour = STATE["latest_hour"]
        if predict_hour is None:
            raise HTTPException(status_code=400, detail="predict_hour required")

    df = pd.read_parquet(PARQUET_PATH, columns=[ZONE_COL, TIME_COL, PRED_COL])
    df = df[df[TIME_COL] == predict_hour]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No rows for {TIME_COL}={predict_hour}")

    out = df.groupby(ZONE_COL, as_index=False)[PRED_COL].mean()
    out = out.sort_values(PRED_COL, ascending=False).head(n)

    out = out.merge(STATE["cent"], left_on=ZONE_COL, right_on="LocationID", how="left")

    return {
        "predict_hour": predict_hour,
        "rows": out[["PULocationID", "pred_rides", "Borough", "Zone", "lat", "lon"]].to_dict(orient="records")
    }
