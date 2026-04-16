import os
import boto3
from pathlib import Path

# 1. R2 設定 (保持不變)
R2_ENDPOINT = "https://10fdbc4ee28881b5403e531b6f547454.r2.cloudflarestorage.com"
R2_BUCKET = "taxi-artifacts"
AWS_ACCESS_KEY_ID = "cf2c89481fb139e09fe89c367ef518b3"
AWS_SECRET_ACCESS_KEY = "516718477cb23ba49cea380198590ea822e260cfdc0b80a08e63f9bb60d1ec52"

# 取得此腳本所在的絕對路徑，確保路徑偵測不會出錯
BASE_DIR = Path(__file__).resolve().parent

# 2. ✅ 修正後的清單 (確保本地路徑與 R2 路徑分開處理)
FILES_TO_UPLOAD = {
    # 本地路徑 (Local) : R2 雲端 Key (Remote)
    BASE_DIR / "data/test_hourly.parquet": "data/test_hourly.parquet",
    BASE_DIR / "data/taxi_zone_centroids.csv": "meta/taxi_zone_centroids.csv",
    
    # 3 個模型檔案
   BASE_DIR / "model/xgb_demand_poisson.model": "model/xgb_demand_poisson.model",
    BASE_DIR / "model/convlstm_resid_map_32x32_ms.best.h5": "model/convlstm_resid_map_32x32_ms.best.h5",
    BASE_DIR / "model/convlstm_resid_map_32x32_ms.h5": "model/convlstm_resid_map_32x32_ms.h5",
    
    # 獎懲機制 (建議確認資料夾名稱)
    BASE_DIR / "rewards/nyc_311_2025_07.csv": "rewards/nyc_311_2025_07.csv",
    
    # 分析產出 (需先跑過 main.py 才會出現)
    BASE_DIR / "outputs/pred_next_hour_advanced.csv": "outputs/pred_next_hour_advanced.csv",
    BASE_DIR / "outputs/zone_reward.csv": "outputs/zone_reward.csv",
    BASE_DIR / "outputs/next_hour_rank_top20.csv": "outputs/next_hour_rank_top20.csv",
    BASE_DIR / "outputs/heatmap.html": "outputs/heatmap.html",
}

def upload_files():
    print(f"🚀 開始同步本地架構至 R2... (根目錄: {BASE_DIR})")

    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto"
    )

    for local_path, r2_key in FILES_TO_UPLOAD.items():
        if not local_path.exists():
            # 如果是 outputs 檔案缺失，給予明確提示
            if "outputs" in str(local_path):
                print(f"💡 提示: 尚未產出 {local_path.name}，請先執行 main.py")
            else:
                print(f"❌ 找不到關鍵檔案: {local_path}")
            continue

        print(f"⬆️ 上傳中: {local_path.name} -> {r2_key}")
        
        try:
            extra_args = {}
            if local_path.suffix == ".html":
                extra_args['ContentType'] = 'text/html'
            elif local_path.suffix == ".csv":
                extra_args['ContentType'] = 'text/csv'

            s3.upload_file(str(local_path), R2_BUCKET, r2_key, ExtraArgs=extra_args)
            print("   ✅ 成功")
        except Exception as e:
            print(f"   ❌ 失敗: {e}")

    print("\n🎉 同步嘗試結束。")

if __name__ == "__main__":
    upload_files()