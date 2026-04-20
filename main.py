# main.py
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

print("R2_ENDPOINT =", os.getenv("R2_ENDPOINT"))
print("R2_BUCKET =", os.getenv("R2_BUCKET"))
print("AWS_ACCESS_KEY_ID exists =", bool(os.getenv("AWS_ACCESS_KEY_ID")))
print("AWS_SECRET_ACCESS_KEY exists =", bool(os.getenv("AWS_SECRET_ACCESS_KEY")))
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

R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

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
# FastAPI app & State
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 正式上線再縮成前端網域
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, Any] = {
    "booster": None,
    "cent": None,
    "hourly": None,
    "latest_hour": None,
    "pred_df": None,
    "pred_hour": None,
    "model_ready": False,
    "reward_df": None,
    "hotspots_df": None,
    "model_init_error": None,
}

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
    return all([R2_ENDPOINT, R2_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY])

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

def s3_client():
    if not _r2_config_ok():
        raise RuntimeError("Missing R2 environment variables")

    import boto3

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto",
    )

def upload_to_r2(local_path: Path, r2_key: str):
    if not local_path.exists():
        return
    try:
        s3_client().upload_file(str(local_path), R2_BUCKET, r2_key)
        print(f"⬆️ [Upload] {local_path.name} -> {r2_key} ✅")
    except Exception as e:
        print(f"❌ Upload error: {e}")

def download_sync(key: str, dst: Path, label: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3_client().download_file(R2_BUCKET, key, str(dst))
        print(f"⬇️ [Download] {label} ✅")
    except Exception as e:
        print(f"⚠️ {label} 下載跳過: {e}")

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

# =========================
# Lazy model init
# =========================
def init_model():
    if STATE["model_ready"]:
        return

    if not MODEL_ENABLED:
        STATE["model_init_error"] = "MODEL_ENABLED is false"
        return

    if not _r2_config_ok():
        STATE["model_init_error"] = "Missing R2 environment variables"
        raise RuntimeError(STATE["model_init_error"])

    # 重套件與重邏輯全部延後到這裡
    import pandas as pd
    import xgboost as xgb
    from logic import run_prediction_task, generate_ranking_reports

    try:
        from logic.build_zone_reward_from_311 import run_311_reward_analysis as run_311_analysis
    except Exception:
        run_311_analysis = None
        print("⚠️ build_zone_reward_from_311 匯入失敗，已跳過 311 分析功能")

    print("🔄 Initializing Models & Data...")
    STATE["model_init_error"] = None

    download_sync(KEY_MODEL_XGB, MODEL_PATH_XGB, "XGB")
    download_sync(KEY_PARQUET, PARQUET_PATH, "Parquet")
    download_sync(KEY_CENT, CENT_PATH, "Centroids")
    download_sync(KEY_311, PATH_311, "311_CSV")

    if NET_PATH.exists():
        print(f"NetXML exists, skip: {NET_PATH}")
    else:
        download_sync(KEY_NET, NET_PATH, "NetXML")

    try:
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH_XGB))
        STATE["booster"] = booster

        cent = pd.read_csv(CENT_PATH)
        STATE["cent"] = centroids_to_wgs84(cent)

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
    }

# =========================
# Hotspots
# =========================
@app.get("/api/hotspots")
def hotspots(n: int = 20, sort_by: str = "final_score"):
    if not STATE["model_ready"]:
        try:
            init_model()
        except Exception as e:
            raise HTTPException(503, f"Model init failed: {e}")

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
    if not STATE["model_ready"]:
        try:
            init_model()
        except Exception as e:
            raise HTTPException(503, f"Model init failed: {e}")

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

        except Exception as e:
            print(f"❌ Pipeline task failed: {e}")

    background_tasks.add_task(task)
    return {"ok": True, "message": "Pipeline started"}

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
