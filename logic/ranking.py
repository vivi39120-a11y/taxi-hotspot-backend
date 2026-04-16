from pathlib import Path
import pandas as pd

# ✅ 移除頂層的路徑設定與執行邏輯，改為由函式傳入
# 確保在 main.py 呼叫時不會因為路徑偏差而崩潰

def generate_ranking_reports(df_pred, cent_path, output_dir):
    """
    由 main.py 呼叫的排名報表產出函式
    df_pred: 預測產出的 DataFrame
    cent_path: Centroid 檔案路徑 (Path)
    output_dir: 報表輸出目錄 (Path)
    """
    print("📊 開始產生排名報表...")

    if df_pred.empty:
        print("⚠️ 預測資料為空，跳過報表產生")
        return

    # 1. 載入 Centroid 對照資料
    df_cent = pd.read_csv(cent_path)[["LocationID", "Borough", "Zone"]]

    # 2. 合併與排名邏輯 (完全保留你的原始邏輯)
    df = df_pred.merge(
        df_cent,
        left_on="PULocationID",
        right_on="LocationID",
        how="left"
    )

    df = df.sort_values("pred_rides", ascending=False).reset_index(drop=True)
    df["rank_overall"] = df.index + 1

    # 確保輸出目錄存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 輸出全部地區排名 (對接 main.py 預期檔名)
    all_rank_path = output_dir / "next_hour_rank_all.csv"
    df.to_csv(all_rank_path, index=False, encoding="utf-8-sig")
    print("✅ 全區排名儲存至：", all_rank_path)

    # 4. 輸出前 20 名 (完全保留你的原始邏輯)
    top_n = 20
    top_rank_path = output_dir / f"next_hour_rank_top{top_n}.csv"
    df.head(top_n).to_csv(top_rank_path, index=False, encoding="utf-8-sig")
    print(f"✅ 前 {top_n} 名儲存至：", top_rank_path)

    # 5. 各 Borough 前 5 名 (完全保留你的分群邏輯)
    df_borough_top5 = (
        df.sort_values(["Borough", "pred_rides"], ascending=[True, False])
          .groupby("Borough")
          .head(5)
          .reset_index(drop=True)
    )

    borough_rank_path = output_dir / "next_hour_rank_borough_top5.csv"
    df_borough_top5.to_csv(borough_rank_path, index=False, encoding="utf-8-sig")
    print("✅ 各 Borough 前 5 名儲存至：", borough_rank_path)

    return df