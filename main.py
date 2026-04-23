import os
import random
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel




def _get_r2_env():
    return {
        "R2_ENDPOINT": os.getenv("R2_ENDPOINT"),
        "R2_BUCKET": os.getenv("R2_BUCKET"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }


print("cwd =", Path.cwd())
print("env exists =", Path(".env").exists())

env = _get_r2_env()
print("R2_ENDPOINT =", env["R2_ENDPOINT"])
print("R2_BUCKET =", env["R2_BUCKET"])
print("AWS_ACCESS_KEY_ID exists =", bool(env["AWS_ACCESS_KEY_ID"]))
print("AWS_SECRET_ACCESS_KEY exists =", bool(env["AWS_SECRET_ACCESS_KEY"]))

# =========================
# Config
# =========================
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path(os.getenv("OUT_DIR", "./outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("./model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REWARDS_DIR = Path("./rewards")
REWARDS_DIR.mkdir(parents=True, exist_ok=True)


KEY_PARQUET = "data/test_hourly.parquet"
KEY_MODEL_XGB = "model/xgb_demand_poisson.model"
KEY_CENT = "meta/taxi_zone_centroids.csv"
KEY_NET = "meta/nyc.net.xml"
KEY_311 = "rewards/nyc_311_2025_07.csv"

PARQUET_PATH = DATA_DIR / "test_hourly.parquet"
MODEL_PATH_XGB = MODEL_DIR / "xgb_demand_poisson.model"
CENT_PATH = DATA_DIR / "taxi_zone_centroids.csv"
NET_PATH = DATA_DIR / "nyc.net.xml"
PATH_311 = REWARDS_DIR / "nyc_311_2025_07.csv"

KEY_OUT_PRED = "outputs/pred_next_hour_advanced.csv"
KEY_OUT_REWARD = "outputs/zone_reward.csv"

ZONE_COL = "PULocationID"
RAW_TIME_COL = "pickup_hour"
PRED_COL = "pred_rides"
MODEL_ENABLED = os.getenv("MODEL_ENABLED", "1").strip() not in ("0", "false", "False")

# =========================
# Hybrid dispatch config (ported from revised dispatch backend)
# =========================
W_DEMAND = 2.0
W_PRIORITY = 0.9
W_DISTANCE = 0.9
W_ZONE_SUPPLY = 0.45
W_LOCAL_SUPPLY = 0.20
MIN_GAIN = 0.20

LOCAL_RADIUS_KM = float(os.getenv("LOCAL_RADIUS_KM", "2.0"))
MAX_CANDIDATE_RADIUS_KM = float(os.getenv("MAX_CANDIDATE_RADIUS_KM", "8.0"))
MIN_NEAR = int(os.getenv("MIN_NEAR", "15"))
K_NEAREST = int(os.getenv("K_NEAREST", "80"))
TOP_K_RESULT = int(os.getenv("TOP_K_RESULT", "3"))

SYNTH_IDLE_COUNT = int(os.getenv("SYNTH_IDLE_COUNT", "2000"))
SYNTH_RANDOM_SEED = int(os.getenv("SYNTH_RANDOM_SEED", "20250801"))
AIRPORT_BIAS = float(os.getenv("AIRPORT_BIAS", "1.8"))
MIDTOWN_BIAS = float(os.getenv("MIDTOWN_BIAS", "1.5"))
MANHATTAN_CORE_BIAS = float(os.getenv("MANHATTAN_CORE_BIAS", "1.25"))

ACTIVE_ORDER_STATUSES = {
    "assigned", "accepted", "en_route", "enroute", "picked_up", "in_progress", "on_trip", "ongoing"
}
MIDTOWN_KEYWORDS = [
    "Midtown", "Times Sq", "Theatre District", "Penn Station", "Garment District",
    "Union Sq", "Flatiron", "Murray Hill", "Kips Bay", "Chelsea"
]
AIRPORT_KEYWORDS = ["Airport", "JFK", "LaGuardia", "LGA"]

# =========================
# FastAPI app & State
# =========================
app = FastAPI()

ALLOWED_ORIGINS = [
    "https://taxi-dispatch-frontend-2.onrender.com",
    "http://localhost:4173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

STATE: Dict[str, Any] = {
    "booster": None,
    "cent": None,
    "hourly": None,
    "latest_hour": None,
    "pred_df": None,
    "pred_hour": None,
    "pred_dispatch_df": None,
    "model_ready": False,
    "reward_df": None,
    "hotspots_df": None,
    "model_init_error": None,
    "dispatch_zones_loaded_at": None,
}
INIT_LOCK = threading.Lock()
PRELOAD_STARTED = False

def preload_model_background():
    global PRELOAD_STARTED
    if PRELOAD_STARTED:
        return
    PRELOAD_STARTED = True

    def _worker():
        try:
            # 稍微等 server 起來，避免卡住 Render port 偵測
            time.sleep(2)
            print("🚀 Background preload started...")
            ensure_model_ready()
            print("✅ Background preload finished.")
        except Exception as e:
            print(f"⚠️ Background preload failed: {e}")

    threading.Thread(target=_worker, daemon=True).start()
# =========================
# JSON Store (派遣系統資料)
# =========================
USERS_PATH = DATA_DIR / "users.json"
DRIVERS_PATH = DATA_DIR / "drivers.json"
ORDERS_PATH = DATA_DIR / "orders.json"
META_PATH = DATA_DIR / "meta.json"

STORE: Dict[str, Any] = {
    "users": [],
    "drivers": [],
    "orders": [],
    "meta": {"next_user_id": 1, "next_order_id": 1, "next_driver_id": 1},
}

# =========================
# Helpers
# =========================
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



def _r2_config_ok() -> bool:
    env = _get_r2_env()
    return all(env.values())
def ensure_model_ready():
    if STATE["model_ready"]:
        return

    if STATE.get("model_init_error"):
        print(f"ℹ️ Previous model_init_error: {STATE['model_init_error']}")

    try:
        init_model()
    except Exception as e:
        raise HTTPException(503, f"Model init failed: {e}")     

def save_store():
    for k, p in zip(
        ["users", "drivers", "orders", "meta"],
        [USERS_PATH, DRIVERS_PATH, ORDERS_PATH, META_PATH],
    ):
        p.write_text(json.dumps(STORE[k], ensure_ascii=False, indent=2), encoding="utf-8")


def load_store():
    for k, p in zip(
        ["users", "drivers", "orders", "meta"],
        [USERS_PATH, DRIVERS_PATH, ORDERS_PATH, META_PATH],
    ):
        if p.exists():
            STORE[k] = json.loads(p.read_text(encoding="utf-8"))


def _find_driver(driver_id: int):
    return next((d for d in STORE["drivers"] if int(d.get("id", 0)) == int(driver_id)), None)


def _find_driver_by_name(name: str):
    name = (name or "").strip()
    return next((d for d in STORE["drivers"] if str(d.get("name", "")).strip() == name), None)


def _find_user(username: str):
    username = (username or "").strip()
    return next((u for u in STORE["users"] if str(u.get("username", "")).strip() == username), None)


def _find_order(order_id: int):
    return next((o for o in STORE["orders"] if int(o.get("id", 0)) == int(order_id)), None)


def norm_status(status: str) -> str:
    return str(status or "").strip().lower()


def get_order_driver_id(order):
    return order.get("driverId") or order.get("assignedDriverId") or order.get("driver_id")


def is_driver_busy(driver_id: int) -> bool:
    for o in STORE["orders"]:
        if int(get_order_driver_id(o) or 0) != int(driver_id):
            continue
        if norm_status(o.get("status")) in ACTIVE_ORDER_STATUSES:
            return True
    return False


def s3_client():
    env = _get_r2_env()
    if not all(env.values()):
        raise RuntimeError("Missing R2 environment variables")

    import boto3

    return boto3.client(
        "s3",
        endpoint_url=env["R2_ENDPOINT"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )

def upload_to_r2(local_path: Path, r2_key: str):
    if not local_path.exists():
        return
    try:
        env = _get_r2_env()
        s3_client().upload_file(str(local_path), env["R2_BUCKET"], r2_key)
        print(f"⬆️ [Upload] {local_path.name} -> {r2_key} ✅")
    except Exception as e:
        print(f"❌ Upload error: {e}")


def download_sync(key: str, dst: Path, label: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        env = _get_r2_env()
        print(f"DOWNLOAD TRY: label={label}, bucket={env['R2_BUCKET']}, key={key}, dst={dst}")
        s3_client().download_file(env["R2_BUCKET"], key, str(dst))

        if not dst.exists():
            raise FileNotFoundError(f"{label} downloaded but file not found at {dst}")

        print(f"⬇️ [Download] {label} ✅ -> {dst}")
    except Exception as e:
        print(f"❌ {label} 下載失敗: {e}")
        raise

def centroids_to_wgs84(df_cent):
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = t.transform(
        df_cent["lon"].astype(float).values,
        df_cent["lat"].astype(float).values,
    )
    out = df_cent.copy()
    out["lon"], out["lat"] = lon_deg, lat_deg
    return out[["LocationID", "Borough", "Zone", "lat", "lon"]]


def build_hotspots_df(pred_df, cent_df, reward_df):
    import pandas as pd

    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    df = pred_df.copy()

    if ZONE_COL not in df.columns:
        if "LocationID" in df.columns:
            df[ZONE_COL] = df["LocationID"]
        else:
            return df

    cent = cent_df.rename(columns={"LocationID": ZONE_COL})
    df = df.merge(cent, on=ZONE_COL, how="left")

    if reward_df is not None and not reward_df.empty:
        r = reward_df.copy()
        if ZONE_COL not in r.columns:
            for cand in ["LocationID", "zone_id", "zone", "pulocationid"]:
                if cand in r.columns:
                    r = r.rename(columns={cand: ZONE_COL})
                    break

        keep_cols = [ZONE_COL, "D", "C", "D_norm", "C_norm", "DRS", "final_score", "used_hour"]
        keep_cols = [c for c in keep_cols if c in r.columns]
        r = r[keep_cols].drop_duplicates(subset=[ZONE_COL])
        df = df.merge(r, on=ZONE_COL, how="left")

    for c in ["D", "C", "D_norm", "C_norm", "DRS", "final_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "used_hour" in df.columns:
        df["used_hour"] = df["used_hour"].fillna("")

    if "lat" in df.columns and "lat_wgs" not in df.columns:
        df["lat_wgs"] = df["lat"]
    if "lon" in df.columns and "lon_wgs" not in df.columns:
        df["lon_wgs"] = df["lon"]

    return df


def get_prediction_df_for_dispatch():
    import pandas as pd

    pred_csv_path = OUT_DIR / "pred_next_hour_advanced.csv"
    if pred_csv_path.exists():
        df = pd.read_csv(pred_csv_path)
    elif STATE.get("pred_df") is not None:
        df = STATE["pred_df"].copy()
    else:
        raise FileNotFoundError(f"Prediction file not found: {pred_csv_path}")

    if PRED_COL not in df.columns:
        for cand in ["pred_next_hour", "pred_next_hour_advanced", "yhat", "pred"]:
            if cand in df.columns:
                df = df.rename(columns={cand: PRED_COL})
                break

    required = {ZONE_COL, PRED_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pred csv missing columns: {sorted(missing)}")

    return df.copy()


def minmax01(series):
    import pandas as pd

    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    mn = float(s.min()) if len(s) else 0.0
    mx = float(s.max()) if len(s) else 0.0
    if mx - mn < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index, dtype=float)
    return (s - mn) / (mx - mn)


def build_dispatch_zone_table():
    import pandas as pd

    if not CENT_PATH.exists():
        raise FileNotFoundError(f"Centroid file not found: {CENT_PATH}")

    df_pred = get_prediction_df_for_dispatch()
    df_cent = pd.read_csv(CENT_PATH)
    cent = centroids_to_wgs84(df_cent)
    df = df_pred.merge(cent, left_on=ZONE_COL, right_on="LocationID", how="left")
    df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["lat", "lon"]).copy()

    if "Borough" not in df.columns:
        df["Borough"] = ""
    if "Zone" not in df.columns:
        df["Zone"] = df[ZONE_COL].astype(str)

    df["priority"] = minmax01(df[PRED_COL])

    def zone_bias(row) -> float:
        zone = str(row.get("Zone", ""))
        borough = str(row.get("Borough", ""))
        bias = 1.0
        if borough == "Manhattan":
            bias *= MANHATTAN_CORE_BIAS
        if any(k.lower() in zone.lower() for k in MIDTOWN_KEYWORDS):
            bias *= MIDTOWN_BIAS
        if any(k.lower() in zone.lower() for k in AIRPORT_KEYWORDS):
            bias *= AIRPORT_BIAS
        return bias

    df["synthetic_bias"] = df.apply(zone_bias, axis=1)
    df["demand_weight_for_supply"] = (df[PRED_COL].clip(lower=0.0) + 1e-9) * df["synthetic_bias"]
    df["zone_id"] = df[ZONE_COL].astype(int)
    df["lat_wgs"] = df["lat"].astype(float)
    df["lon_wgs"] = df["lon"].astype(float)
    return df.reset_index(drop=True)


def refresh_dispatch_zones():
    df = build_dispatch_zone_table()
    STATE["pred_dispatch_df"] = df
    STATE["dispatch_zones_loaded_at"] = _now_iso()
    return df


def get_dispatch_zones(force: bool = False):
    if force or STATE.get("pred_dispatch_df") is None:
        return refresh_dispatch_zones()
    return STATE["pred_dispatch_df"]


def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, asin

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def synthetic_idle_supply(df) -> Dict[int, int]:
    rng = random.Random(SYNTH_RANDOM_SEED)
    zone_ids = df["zone_id"].tolist()
    weights = df["demand_weight_for_supply"].tolist()
    picks = rng.choices(zone_ids, weights=weights, k=max(0, SYNTH_IDLE_COUNT))
    supply = {int(z): 0 for z in zone_ids}
    for z in picks:
        supply[int(z)] += 1
    return supply


def nearest_zone_id(df, lat: float, lng: float) -> Optional[int]:
    best_z, best_d = None, float("inf")
    for _, r in df.iterrows():
        d = haversine_km(lat, lng, float(r["lat_wgs"]), float(r["lon_wgs"]))
        if d < best_d:
            best_d = d
            best_z = int(r["zone_id"])
    return best_z


def real_idle_supply(df) -> Dict[int, int]:
    supply = {int(z): 0 for z in df["zone_id"].tolist()}
    for d in STORE["drivers"]:
        did = d.get("id")
        lat = d.get("lat")
        lng = d.get("lng")
        if did is None or lat is None or lng is None:
            continue
        try:
            did = int(did)
            lat = float(lat)
            lng = float(lng)
        except Exception:
            continue
        if is_driver_busy(did):
            continue
        zid = nearest_zone_id(df, lat, lng)
        if zid is not None:
            supply[int(zid)] = supply.get(int(zid), 0) + 1
    return supply


def local_supply_for_zone(df, zone_supply_map: Dict[int, int], zone_id: int) -> int:
    row = df.loc[df["zone_id"] == zone_id]
    if row.empty:
        return 0
    src = row.iloc[0]
    src_lat, src_lon = float(src["lat_wgs"]), float(src["lon_wgs"])
    total = 0
    for _, r in df.iterrows():
        other_id = int(r["zone_id"])
        if other_id == zone_id:
            continue
        d = haversine_km(src_lat, src_lon, float(r["lat_wgs"]), float(r["lon_wgs"]))
        if d <= LOCAL_RADIUS_KM:
            total += int(zone_supply_map.get(other_id, 0))
    return total


def build_supply_maps(df) -> Dict[str, Dict[int, int]]:
    synth = synthetic_idle_supply(df)
    real = real_idle_supply(df)
    total = {int(z): int(synth.get(int(z), 0)) + int(real.get(int(z), 0)) for z in df["zone_id"].tolist()}
    local = {int(z): local_supply_for_zone(df, total, int(z)) for z in df["zone_id"].tolist()}
    return {"synthetic": synth, "real": real, "total": total, "local": local}


def normalized_candidate_frame(df, driver_lat: float, driver_lng: float):
    import pandas as pd

    base = []
    for _, r in df.iterrows():
        dkm = haversine_km(driver_lat, driver_lng, float(r["lat_wgs"]), float(r["lon_wgs"]))
        base.append({
            "zone_id": int(r["zone_id"]),
            "Zone": str(r.get("Zone", "")),
            "Borough": str(r.get("Borough", "")),
            "lat_wgs": float(r["lat_wgs"]),
            "lon_wgs": float(r["lon_wgs"]),
            PRED_COL: float(r[PRED_COL]),
            "priority": float(r["priority"]),
            "distance_km": float(dkm),
        })

    near = [x for x in base if x["distance_km"] <= MAX_CANDIDATE_RADIUS_KM]
    if len(near) < MIN_NEAR:
        near = sorted(base, key=lambda x: x["distance_km"])[:K_NEAREST]

    cand = pd.DataFrame(near)
    if cand.empty:
        return cand

    cand["DemandN"] = minmax01(cand[PRED_COL])
    cand["PriorityN"] = minmax01(cand["priority"])
    cand["DistanceN"] = minmax01(cand["distance_km"])
    return cand


def score_frame(cand, supply_maps: Dict[str, Dict[int, int]]):
    if cand.empty:
        return cand
    cand = cand.copy()
    cand["ZoneSupply"] = cand["zone_id"].map(lambda z: int(supply_maps["total"].get(int(z), 0)))
    cand["LocalSupply"] = cand["zone_id"].map(lambda z: int(supply_maps["local"].get(int(z), 0)))
    cand["ZoneSupplyN"] = minmax01(cand["ZoneSupply"])
    cand["LocalSupplyN"] = minmax01(cand["LocalSupply"])
    cand["Score"] = (
        W_DEMAND * cand["DemandN"]
        + W_PRIORITY * cand["PriorityN"]
        - W_DISTANCE * cand["DistanceN"]
        - W_ZONE_SUPPLY * cand["ZoneSupplyN"]
        - W_LOCAL_SUPPLY * cand["LocalSupplyN"]
    )
    return cand

# =========================
# Lazy model init
# =========================
def init_model():
    print("=== INIT_MODEL CALLED ===")
    print("DEBUG _get_r2_env =", _get_r2_env())

    if STATE["model_ready"]:
        return

    if not MODEL_ENABLED:
        STATE["model_init_error"] = "MODEL_ENABLED is false"
        return

    if not _r2_config_ok():
        STATE["model_init_error"] = "Missing R2 environment variables"
        raise RuntimeError(STATE["model_init_error"])

    import pandas as pd
    import xgboost as xgb
    from logic import run_prediction_task, generate_ranking_reports

    try:
        from logic.build_zone_reward_from_311 import run_311_reward_analysis as run_311_analysis
    except Exception:
        run_311_analysis = None
        print("⚠️ build_zone_reward_from_311 匯入失敗，已跳過 311 分析功能")

    try:
        STATE["model_init_error"] = None

        print("🔄 Initializing Models & Data...")

        print("STEP 1: download XGB")
        download_sync(KEY_MODEL_XGB, MODEL_PATH_XGB, "XGB")

        print("STEP 2: download Parquet")
        download_sync(KEY_PARQUET, PARQUET_PATH, "Parquet")

        print("STEP 3: download Centroids")
        download_sync(KEY_CENT, CENT_PATH, "Centroids")

        print("STEP 4: download 311 CSV")
        download_sync(KEY_311, PATH_311, "311_CSV")

        if NET_PATH.exists():
            print(f"NetXML exists, skip: {NET_PATH}")
        else:
            print("STEP 5: download NetXML")
            download_sync(KEY_NET, NET_PATH, "NetXML")

        print("STEP 6: load xgboost model")
        if not MODEL_PATH_XGB.exists():
            raise FileNotFoundError(f"XGB model file not found: {MODEL_PATH_XGB}")

        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH_XGB))
        STATE["booster"] = booster

        print("STEP 7: read centroids csv")
        if not CENT_PATH.exists():
            raise FileNotFoundError(f"Centroid file not found: {CENT_PATH}")
        cent = pd.read_csv(CENT_PATH)
        STATE["cent"] = centroids_to_wgs84(cent)

        print("STEP 8: read parquet")
        if not PARQUET_PATH.exists():
            raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")
        df = pd.read_parquet(PARQUET_PATH)
        df[RAW_TIME_COL] = pd.to_datetime(df[RAW_TIME_COL])
        STATE["hourly"] = df

        latest = df[RAW_TIME_COL].max()
        STATE["latest_hour"] = latest

        if not pd.isna(latest):
            print("🔮 Running Initial Prediction & Analysis...")

            df_pred = run_prediction_task(
                booster,
                df,
                CENT_PATH,
                OUT_DIR / "pred_next_hour_advanced.csv",
                OUT_DIR / "heatmap.html",
            )

            pred_csv_path = OUT_DIR / "pred_next_hour_advanced.csv"
            if pred_csv_path.exists():
                df_pred = pd.read_csv(pred_csv_path)

            STATE["pred_df"] = df_pred
            STATE["pred_hour"] = (
                df_pred["predict_hour"].iloc[0]
                if (df_pred is not None and not df_pred.empty and "predict_hour" in df_pred.columns)
                else (latest + pd.Timedelta(hours=1))
            )

            reward_path = OUT_DIR / "zone_reward.csv"
            try:
                if run_311_analysis is not None:
                    run_311_analysis(PATH_311, CENT_PATH, reward_path)
                    upload_to_r2(reward_path, KEY_OUT_REWARD)

                STATE["reward_df"] = pd.read_csv(reward_path) if reward_path.exists() else None
            except Exception as e:
                print(f"⚠️ Reward Analysis Fail: {e}")
                STATE["reward_df"] = None

            generate_ranking_reports(df_pred, CENT_PATH, OUT_DIR)

            try:
                STATE["hotspots_df"] = build_hotspots_df(
                    STATE["pred_df"],
                    STATE["cent"],
                    STATE["reward_df"],
                )
            except Exception as e:
                print(f"⚠️ build_hotspots_df failed: {e}")
                STATE["hotspots_df"] = None

            try:
                refresh_dispatch_zones()
            except Exception as e:
                print(f"⚠️ dispatch zone refresh failed during init: {e}")
                STATE["pred_dispatch_df"] = None

        STATE["model_ready"] = True
        print("✅ Model init complete.")

    except Exception as e:
        STATE["model_ready"] = False
        STATE["model_init_error"] = str(e)
        print(f"❌ Init failed: {e}")
        raise

# =========================
# Pydantic models
# =========================
class RegisterBody(BaseModel):
    username: str
    password: str
    role: str = "passenger"
    carType: Optional[str] = None


class LoginBody(BaseModel):
    username: str
    password: str


class DriverLoginBody(BaseModel):
    name: str
    carType: Optional[str] = None


class DriverLocationBody(BaseModel):
    lat: float
    lng: float


class LatLng(BaseModel):
    lat: float
    lng: float


class CreateOrderBody(BaseModel):
    customer: Optional[str] = None
    pickup: str
    dropoff: str
    pickupLocation: LatLng
    dropoffLocation: LatLng
    stops: Optional[List[Dict[str, Any]]] = None
    vehicleType: Optional[str] = None
    estimatedPrice: Optional[float] = None
    distanceKm: Optional[float] = None


class AcceptOrderBody(BaseModel):
    driverId: int

# =========================
# Startup
# =========================
@app.on_event("startup")
def startup_all():
    try:
        load_store()
        print("✅ Store loaded.")
    except Exception as e:
        print(f"⚠️ load_store failed: {e}")

    print("🚀 Server started. Model preload will run in background.")
    preload_model_background()
# =========================
# Health
# =========================
@app.get("/api/health")
def api_health():
    return {
        "ok": True,
        "model_ready": STATE["model_ready"],
        "model_init_error": STATE["model_init_error"],
        "drivers": len(STORE["drivers"]),
        "orders": len(STORE["orders"]),
        "dispatch_formula": "Score = 2.0*Demand + 0.9*Priority - 0.9*Distance - 0.45*ZoneSupply - 0.20*LocalSupply",
        "min_gain": MIN_GAIN,
        "dispatch_zones_loaded_at": STATE.get("dispatch_zones_loaded_at"),
    }

# =========================
# Hotspots
# =========================
@app.get("/api/hotspots")
def hotspots(n: int = 20, sort_by: str = "final_score"):
    ensure_model_ready()

    if not STATE["model_ready"]:
        raise HTTPException(503, "Model not ready")

    if STATE["pred_df"] is None:
        raise HTTPException(503, "Prediction not ready")

    df = STATE.get("hotspots_df")
    if df is None or len(df) == 0:
        df = STATE["pred_df"].copy()

    sort_by = (sort_by or "").strip()
    if sort_by not in df.columns:
        sort_by = PRED_COL if PRED_COL in df.columns else df.columns[0]

    try:
        n = int(n)
    except Exception:
        n = 20

    df_out = df.sort_values(sort_by, ascending=False).head(n)
    return {
        "predict_hour": str(STATE["pred_hour"]),
        "rows": df_out.to_dict(orient="records"),
    }


@app.get("/api/zone-hotspots")
def api_zone_hotspots():
    ensure_model_ready()

    try:
        df = get_dispatch_zones(force=False).copy()
    except Exception as e:
        raise HTTPException(503, f"Dispatch hotspots not ready: {e}")

    out = df[["zone_id", ZONE_COL, "Borough", "Zone", "lat_wgs", "lon_wgs", PRED_COL, "priority"]].copy()
    return {
        "rows": out.to_dict(orient="records"),
        "prediction_mode": "model_pipeline",
        "predict_hour": str(STATE.get("pred_hour")),
    }


@app.get("/api/dispatch-recommendations")
def api_dispatch_recommendations(driver_id: Optional[int] = None, lat: Optional[float] = None, lng: Optional[float] = None, top_k: int = TOP_K_RESULT):
    ensure_model_ready()

    try:
        df = get_dispatch_zones(force=False)
    except Exception as e:
        raise HTTPException(503, f"Dispatch zones not ready: {e}")

    driver_name = None

    if driver_id is not None:
        d = _find_driver(driver_id)
        if d is None:
            raise HTTPException(404, f"Driver {driver_id} not found")
        driver_name = d.get("name")
        if lat is None:
            lat = d.get("lat")
        if lng is None:
            lng = d.get("lng")

    if lat is None or lng is None:
        raise HTTPException(400, "Missing driver lat/lng")

    driver_lat = float(lat)
    driver_lng = float(lng)
    current_zone_id = nearest_zone_id(df, driver_lat, driver_lng)
    if current_zone_id is None:
        raise HTTPException(503, "Unable to infer current zone")

    cand = normalized_candidate_frame(df, driver_lat, driver_lng)
    if cand.empty:
        return {"rows": [], "current_zone_id": current_zone_id}

    supply_maps = build_supply_maps(df)
    cand = score_frame(cand, supply_maps)

    current_row = cand[cand["zone_id"] == current_zone_id]
    current_score = float(current_row["Score"].iloc[0]) if not current_row.empty else None
    cand["Gain"] = cand["Score"] - (current_score if current_score is not None else 0.0)
    cand["move_recommended"] = cand["Gain"] > MIN_GAIN
    cand = cand[cand["zone_id"] != current_zone_id].copy()
    cand.sort_values("Score", ascending=False, inplace=True)
    cand["road_km"] = cand["distance_km"]

    out = cand.head(max(1, int(top_k))).copy()
    rows = []
    for _, r in out.iterrows():
        rows.append({
            "zone_id": int(r["zone_id"]),
            "PULocationID": int(r["zone_id"]),
            "Borough": str(r["Borough"]),
            "Zone": str(r["Zone"]),
            "lat_wgs": float(r["lat_wgs"]),
            "lon_wgs": float(r["lon_wgs"]),
            "pred_rides": float(r[PRED_COL]),
            "priority": float(r["priority"]),
            "distance_km": float(r["distance_km"]),
            "road_km": float(r["road_km"]),
            "zone_supply": int(r["ZoneSupply"]),
            "local_supply": int(r["LocalSupply"]),
            "DemandN": float(r["DemandN"]),
            "PriorityN": float(r["PriorityN"]),
            "DistanceN": float(r["DistanceN"]),
            "ZoneSupplyN": float(r["ZoneSupplyN"]),
            "LocalSupplyN": float(r["LocalSupplyN"]),
            "score": float(r["Score"]),
            "gain": float(r["Gain"]),
            "move_recommended": bool(r["move_recommended"]),
        })

    current_zone = df.loc[df["zone_id"] == current_zone_id].iloc[0]
    return {
        "driver_id": driver_id,
        "driver_name": driver_name,
        "driver_lat": driver_lat,
        "driver_lng": driver_lng,
        "current_zone_id": int(current_zone_id),
        "current_zone": str(current_zone["Zone"]),
        "current_score": current_score,
        "min_gain": MIN_GAIN,
        "formula": {
            "W_DEMAND": W_DEMAND,
            "W_PRIORITY": W_PRIORITY,
            "W_DISTANCE": W_DISTANCE,
            "W_ZONE_SUPPLY": W_ZONE_SUPPLY,
            "W_LOCAL_SUPPLY": W_LOCAL_SUPPLY,
        },
        "rows": rows,
    }


@app.get("/api/driver-bias/{driver_id}")
def get_driver_reward_bias(driver_id: int):
    bias = 1.0
    try:
        try:
            from MOD.reward_mod import get_bias
        except Exception:
            get_bias = None

        if get_bias is not None:
            bias = float(get_bias(OUT_DIR, enable=True))
    except Exception:
        bias = 1.0
    return {"driver_id": driver_id, "bias": bias}

# =========================
# 派遣系統：Register / Login
# =========================
@app.post("/api/register")
def api_register(body: RegisterBody):
    if _find_user(body.username) is not None:
        raise HTTPException(409, "Exists")

    uid = STORE["meta"]["next_user_id"]
    STORE["meta"]["next_user_id"] += 1

    user = {
        "id": uid,
        "username": body.username,
        "password": body.password,
        "role": body.role,
        "createdAt": _now_iso(),
    }
    STORE["users"].append(user)

    if body.role == "driver":
        did = STORE["meta"]["next_driver_id"]
        STORE["meta"]["next_driver_id"] += 1
        STORE["drivers"].append(
            {
                "id": did,
                "name": body.username,
                "carType": body.carType,
                "lat": None,
                "lng": None,
                "updatedAt": _now_iso(),
            }
        )

    save_store()
    return {"ok": True, "user": user}


@app.post("/api/login")
def api_login(body: LoginBody):
    u = _find_user(body.username)
    if not u:
        raise HTTPException(404, detail={"errorCode": "NO_SUCH_ACCOUNT", "error": "Not found"})

    if str(u.get("password", "")) != str(body.password):
        raise HTTPException(401, detail={"errorCode": "BAD_PASSWORD", "error": "Wrong password"})

    return {"ok": True, "user": u}


@app.post("/api/driver-login")
def api_driver_login(body: DriverLoginBody):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(400, "Missing name")

    d = _find_driver_by_name(name)
    if d is None:
        did = STORE["meta"]["next_driver_id"]
        STORE["meta"]["next_driver_id"] += 1
        d = {
            "id": did,
            "name": name,
            "carType": body.carType,
            "lat": None,
            "lng": None,
            "updatedAt": _now_iso(),
        }
        STORE["drivers"].append(d)
        save_store()
        return d

    if body.carType is not None:
        d["carType"] = body.carType
    d["updatedAt"] = _now_iso()
    save_store()
    return d

# =========================
# 派遣系統：Orders / Drivers
# =========================
@app.post("/api/orders")
def api_create_order(body: CreateOrderBody):
    oid = STORE["meta"]["next_order_id"]
    STORE["meta"]["next_order_id"] += 1

    order = {
        "id": oid,
        "status": "pending",
        "customer": body.customer or "guest",
        "pickup": body.pickup,
        "dropoff": body.dropoff,
        "pickupLocation": body.pickupLocation.model_dump(),
        "dropoffLocation": body.dropoffLocation.model_dump(),
        "stops": body.stops or [],
        "vehicleType": body.vehicleType,
        "estimatedPrice": body.estimatedPrice,
        "distanceKm": body.distanceKm,
        "createdAt": _now_iso(),
    }
    STORE["orders"].append(order)
    save_store()
    return {"ok": True, "order": order}


@app.post("/api/orders/{order_id}/accept")
def api_accept_order(order_id: int, body: AcceptOrderBody):
    o = _find_order(order_id)
    d = _find_driver(body.driverId)
    if not o or not d:
        raise HTTPException(404, "Not found")

    o.update({"status": "assigned", "driverId": body.driverId, "updatedAt": _now_iso()})
    save_store()
    return {"ok": True, "order": o}


@app.post("/api/orders/{order_id}/complete")
def api_complete_order(order_id: int):
    o = _find_order(order_id)
    if not o:
        raise HTTPException(404, "Not found")

    now = _now_iso()
    o["status"] = "completed"
    o["completedAt"] = o.get("completedAt") or now
    o["updatedAt"] = now

    save_store()
    return {"ok": True, "order": o}


@app.patch("/api/drivers/{driver_id}/location")
def api_driver_location(driver_id: int, body: DriverLocationBody):
    d = _find_driver(driver_id)
    if not d:
        raise HTTPException(404, f"找不到 ID 為 {driver_id} 的司機")

    d["lat"] = float(body.lat)
    d["lng"] = float(body.lng)
    d["updatedAt"] = _now_iso()
    save_store()

    return {"ok": True, "updated_driver": d["id"], "name": d["name"]}


@app.get("/api/orders")
def api_get_orders():
    return {"rows": STORE["orders"]}


@app.get("/api/drivers")
def api_get_drivers():
    return {"rows": STORE["drivers"]}

# =========================
# Pipeline admin
# =========================
@app.post("/api/admin/run-pipeline")
def api_run_pipeline(background_tasks: BackgroundTasks):
    ensure_model_ready()

    if STATE["booster"] is None or STATE["hourly"] is None:
        raise HTTPException(503, "Model artifacts not ready")

    def task():
        import pandas as pd
        from logic import run_prediction_task, generate_ranking_reports

        try:
            try:
                from logic.build_zone_reward_from_311 import run_311_reward_analysis as run_311_analysis
            except Exception:
                run_311_analysis = None

            df_pred = run_prediction_task(
                STATE["booster"],
                STATE["hourly"],
                CENT_PATH,
                OUT_DIR / "pred_next_hour_advanced.csv",
            )

            pred_csv_path = OUT_DIR / "pred_next_hour_advanced.csv"
            if pred_csv_path.exists():
                df_pred = pd.read_csv(pred_csv_path)

            STATE["pred_df"] = df_pred
            upload_to_r2(OUT_DIR / "pred_next_hour_advanced.csv", KEY_OUT_PRED)

            try:
                reward_path = OUT_DIR / "zone_reward.csv"
                if run_311_analysis is not None:
                    run_311_analysis(PATH_311, CENT_PATH, reward_path)

                if reward_path.exists():
                    STATE["reward_df"] = pd.read_csv(reward_path)
                    upload_to_r2(reward_path, KEY_OUT_REWARD)
                else:
                    STATE["reward_df"] = None
            except Exception as e:
                print(f"⚠️ Reward update failed: {e}")
                STATE["reward_df"] = None

            generate_ranking_reports(df_pred, CENT_PATH, OUT_DIR)

            try:
                STATE["hotspots_df"] = build_hotspots_df(
                    STATE["pred_df"],
                    STATE["cent"],
                    STATE["reward_df"],
                )
            except Exception as e:
                print(f"⚠️ build_hotspots_df failed: {e}")
                STATE["hotspots_df"] = None

            try:
                refresh_dispatch_zones()
            except Exception as e:
                print(f"⚠️ dispatch zone refresh failed after pipeline: {e}")
                STATE["pred_dispatch_df"] = None

        except Exception as e:
            print(f"❌ Pipeline task failed: {e}")

    background_tasks.add_task(task)
    return {"ok": True, "message": "Pipeline started"}


@app.post("/api/admin/reload-dispatch-zones")
def api_reload_dispatch_zones():
    ensure_model_ready()

    try:
        df = get_dispatch_zones(force=True)
    except Exception as e:
        raise HTTPException(503, f"Reload dispatch zones failed: {e}")

    return {
        "ok": True,
        "zones_loaded_at": STATE.get("dispatch_zones_loaded_at"),
        "rows": len(df),
    }

# =========================
# Route
# =========================
@app.get("/api/route")
async def api_route(fromLat: float, fromLng: float, toLat: float, toLng: float):
    try:
        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{fromLng},{fromLat};{toLng},{toLat}"
            "?overview=full&geometries=geojson"
        )

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers={"User-Agent": "taxi-app"})
            r.raise_for_status()
            data = r.json()

        routes = data.get("routes") or []
        if not routes:
            return {"coords": [], "dist": None}

        route = routes[0]
        geometry = route.get("geometry", {})
        raw_coords = geometry.get("coordinates") or []

        coords = []
        for item in raw_coords:
            try:
                lng, lat = item
                coords.append([float(lat), float(lng)])
            except Exception:
                continue

        dist = route.get("distance")
        dist_km = float(dist) / 1000.0 if dist is not None else None

        return {
            "coords": coords,
            "dist": dist_km,
        }
    except Exception as e:
        print(f"⚠️ route api failed: {e}")
        return {"coords": [], "dist": None}
    
# =========================
# Geocode
# =========================
@app.get("/api/geocode")
async def api_geocode(q: str):
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "jsonv2", "limit": 5},
                headers={"User-Agent": "taxi-app"},
            )
            data = r.json()
            out = []
            for x in (data or []):
                try:
                    out.append({"label": x["display_name"], "lat": float(x["lat"]), "lng": float(x["lon"])})
                except Exception:
                    continue
            return out if out else [{"label": "Times Square", "lat": 40.758, "lng": -73.9855}]
        except Exception:
            return [{"label": "Times Square", "lat": 40.758, "lng": -73.9855}]
