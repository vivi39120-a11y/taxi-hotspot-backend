import pandas as pd
import sumolib
import random
from pyproj import Transformer

def generate_traffic_trips(net_file, cent_file, pred_df, output_path):
    print(f"🚗 [Traffic Gen] 讀取路網: {net_file}")
    try:
        net = sumolib.net.readNet(str(net_file))
    except Exception as e:
        print(f"❌ 讀取路網失敗: {e}")
        return

    all_edge_ids = [e.getID() for e in net.getEdges()]
    print(f"🛣️ 地圖共有 {len(all_edge_ids)} 條道路")

    df_cent = pd.read_csv(cent_file)
    
    left_key = 'PULocationID' if 'PULocationID' in pred_df.columns else 'LocationID'
    df = pred_df.merge(df_cent, left_on=left_key, right_on="LocationID", how="inner")
    
    vehicle_count = 0
    print(f"💾 正在寫入: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<routes>\n')
        f.write('    <vType id="taxi" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="50" color="1,0.8,0"/>\n')

        for idx, row in df.iterrows():
            try:
                demand = int(row.get('pred_rides', 0))
                if demand <= 0: continue

                # ⚠️ 注意：如果車太多電腦跑不動，可以打開下面這行除以 10
                # demand = max(1, demand // 10) 

                lon, lat = row['lon'], row['lat']
                x, y = net.convertLonLat2XY(lon, lat)
                
                # 搜尋半徑 200 公尺內的最近道路
                edges = net.getNeighboringEdges(x, y, 200)
                if len(edges) == 0: continue
                
                start_edge = edges[0][0].getID()

                for i in range(demand):
                    veh_id = f"veh_{int(row['LocationID'])}_{i}"
                    end_edge = random.choice(all_edge_ids)
                    depart_time = random.randint(0, 3600)
                    
                    f.write(f'    <trip id="{veh_id}" type="taxi" depart="{depart_time}" from="{start_edge}" to="{end_edge}"/>\n')
                    vehicle_count += 1
            except Exception:
                continue

        f.write('</routes>\n')

    print(f"✅ Trips XML 生成完畢 (共 {vehicle_count} 輛車)")