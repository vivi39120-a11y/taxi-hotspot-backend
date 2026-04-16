import pandas as pd
import sumolib
from pyproj import Transformer
from pathlib import Path

def get_color(score):
    """依照分數決定顏色 (R,G,B) - 原汁原味保留"""
    if score > 0.5:
        return "1,0,0"      # 紅色
    elif score > 0.2:
        return "1,0.5,0"    # 橘色
    elif score > 0.05:
        return "1,1,0"      # 黃色
    else:
        return "0,1,0"      # 綠色

def generate_visualization_xml(net_file, cent_file, pred_df, output_path):
    print(f"🎨 [Viz Gen] 讀取路網: {net_file}")
    try:
        net = sumolib.net.readNet(str(net_file))
    except Exception as e:
        print(f"❌ 讀取路網失敗: {e}")
        return

    print("📊 讀取並合併資料...")
    df_cent = pd.read_csv(cent_file)
    
    # 確保欄位名稱一致
    left_key = 'PULocationID' if 'PULocationID' in pred_df.columns else 'LocationID'
    df = pred_df.merge(df_cent, left_on=left_key, right_on="LocationID", how="inner")
    
    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

    print(f"💾 正在寫入: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<additional>\n')
        
        for idx, row in df.iterrows():
            try:
                # 1. 轉經緯度
                lon, lat = transformer.transform(row['lon'], row['lat'])
                # 2. 轉 SUMO 座標
                x, y = net.convertLonLat2XY(lon, lat)
                # 3. 取得分數
                score = row.get('pred_rides', 0)
                
                color = get_color(score)
                radius = 30 + (score * 50) # 原本邏輯
                
                f.write(f'    <poi id="zone_{int(row["LocationID"])}" '
                        f'type="taxi_zone" '
                        f'color="{color}" '
                        f'x="{x:.2f}" y="{y:.2f}" '
                        f'width="{radius:.2f}" height="{radius:.2f}" '
                        f'layer="100"/>\n')
            except Exception:
                continue
                
        f.write('</additional>\n')
    
    print("✅ Visualization XML 生成完畢")