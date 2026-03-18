import socket
import time
import struct
import numpy as np
import mne
import os
from pathlib import Path

# ================= 配置区域 =================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_GDF = PROJECT_ROOT / "Data" / "BCICIV_2a_gdf"
SUBJECT_ID = 3

# 网络配置
HOST = '0.0.0.0' # 本地地址
PORT = 65432       # 通信端口

# 仿真参数
CHUNK_SIZE = 10    # 每次发送 10 个点 (40ms)5                                                                                                                                                                                                                                                                                                                                                   
FS = 250.0         # 采样率

# ================= 数据加载 =================
def load_data():
    gdf_path = os.path.join(DATA_DIR_GDF, f"A0{SUBJECT_ID}E.gdf")
    print(f"[Sender] Loading GDF file: {gdf_path}...")
    
    # 读取 GDF
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    
    # 确保加载 25 通道 (22 EEG + 3 EOG)
    try: 
        raw.pick_types(eeg=True, eog=True)
    except: 
        # 如果 pick 失败，尝试 pick all non-stim channels
        raw.pick(mne.pick_types(raw.info, eeg=True, eog=True, meg=False))
    
    # 获取数据 (单位: Volts)
    data = raw.get_data() 
    
    # 获取 Events (用于发送 Marker)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    
    print(f"[Sender] Data loaded successfully.")
    print(f"         Shape: {data.shape} (Channels x Samples)")
    print(f"         Duration: {data.shape[1]/FS/60:.1f} minutes")
    
    return data, events

# ================= 主逻辑 =================
def run_server():
    data, events = load_data()
    n_channels = data.shape[0]
    n_samples = data.shape[1]
    
    event_map = {ev[0]: ev[2] for ev in events}
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        s.bind((HOST, PORT))
        s.listen()
        
        print("\n" + "="*60)
        print(f"★ [Sender] SERVER STARTED on {HOST}:{PORT}")
        print(f"★ [Sender] STATUS: STANDBY (Waiting for Receiver...)")
        print("="*60 + "\n")
        
        conn, addr = s.accept()
        
        with conn:
            print(f"★ [Sender] RECEIVER DETECTED! Connected by {addr}")
            print(f"★ [Sender] Stream starting in 3 seconds...")
            time.sleep(1)
            print(f"           2...")
            time.sleep(1)
            print(f"           1...")
            time.sleep(1)
            print(f"★ [Sender] GO! Streaming real-time data...\n")
            
     
            conn.sendall(struct.pack('!I', n_channels))
            
            # --- 推流循环 ---
            current_ptr = 0
            
            try:
                while current_ptr < n_samples:
                    loop_start = time.time()
                    
                    chunk_end = min(current_ptr + CHUNK_SIZE, n_samples)
                    
         
                    if current_ptr in event_map:
                        eid = event_map[current_ptr]
               
                        msg = struct.pack('!cI', b'M', eid)
                        conn.sendall(msg)
                        print(f"   -> [Event] Sent Marker {eid} at sample {current_ptr}")

                 
                    chunk = data[:, current_ptr : chunk_end].astype(np.float32)
               
                    chunk_bytes = chunk.tobytes()
                    
           
                    header = struct.pack('!cI', b'D', len(chunk_bytes))
                    conn.sendall(header + chunk_bytes)
                    
     
                    current_ptr = chunk_end
                    

                    elapsed = time.time() - loop_start
                    target_interval = CHUNK_SIZE / FS # 10/250 = 0.04s
                    
                    sleep_time = target_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
            except BrokenPipeError:
                print("\n[Sender] Connection lost (Receiver disconnected).")
            except KeyboardInterrupt:
                print("\n[Sender] Stopped by user.")
            except Exception as e:
                print(f"\n[Sender] Error: {e}")
                
    print("[Sender] Session finished.")

if __name__ == "__main__":
    run_server()
