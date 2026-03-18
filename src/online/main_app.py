import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch_bootstrap import prepare_torch_dlls
prepare_torch_dlls()

# ！！！绝杀手段：在所有第三方库之前，强制优先唤醒 PyTorch 的底层 DLL！！！
import torch 

# 下面再放原本的其他导入
import threading
import time
import socket
import struct
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import mne
import pyxdf 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from receiver import BCIProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMELINE_FILE = PROJECT_ROOT / "outputs" / "timelines" / "cue_timeline0tt.txt"

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

class BCIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("脑控小车演示系统 v6.1")
        self.root.geometry("1150x950")
        
        self.is_running = False
        self.processor = None
        self.client_socket = None
        self.EXPECTED_CHANNELS = 25 
        
        self.OFFICIAL_ORDER = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 
                               'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
        self.plot_ch_indices = [7, 9, 11]
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        self.wave_buffer = np.zeros((self.EXPECTED_CHANNELS, 500)) 

        # --- UI 布局 ---
        mode_frame = tk.LabelFrame(root, text="运行模式", padx=10, pady=10)
        mode_frame.pack(fill="x", padx=15, pady=5)
        
        self.run_mode = tk.StringVar(value="offline")
        tk.Radiobutton(mode_frame, text="离线模式 (读取本地 EEG+TXT)", variable=self.run_mode, value="offline", 
                       command=self.update_ui_state, font=("微软雅黑", 10)).pack(side="left", padx=20)
        tk.Radiobutton(mode_frame, text="在线模式 (连接 Sender.py + 读取 TXT)", variable=self.run_mode, value="online", 
                       command=self.update_ui_state, font=("微软雅黑", 10), fg="#e67e22").pack(side="left", padx=20)

        config_frame = tk.LabelFrame(root, text="文件配置", padx=10, pady=10)
        config_frame.pack(fill="x", padx=15, pady=5)
        
        tk.Label(config_frame, text="脑电文件:").grid(row=0, column=0, sticky="e")
        self.data_path = tk.StringVar()
        self.data_entry = tk.Entry(config_frame, textvariable=self.data_path, width=40)
        self.data_entry.grid(row=0, column=1, padx=5)
        self.data_btn = tk.Button(config_frame, text="浏览", command=lambda: self.select_file(self.data_path))
        self.data_btn.grid(row=0, column=2)
        
        tk.Label(config_frame, text="标记TXT:").grid(row=1, column=0, sticky="e", pady=5)
        self.txt_path = tk.StringVar(value=str(DEFAULT_TIMELINE_FILE))
        self.txt_entry = tk.Entry(config_frame, textvariable=self.txt_path, width=40)
        self.txt_entry.grid(row=1, column=1, padx=5)
        self.txt_btn = tk.Button(config_frame, text="浏览", command=lambda: self.select_file(self.txt_path))
        self.txt_btn.grid(row=1, column=2)
        
        tk.Label(config_frame, text="原始频率:").grid(row=0, column=3, padx=15)
        self.fs_var = tk.StringVar(value="250")
        self.fs_combo = ttk.Combobox(config_frame, textvariable=self.fs_var, values=("250", "500", "1000"), width=6)
        self.fs_combo.grid(row=0, column=4)
        
        tk.Label(config_frame, text="离线倍速:").grid(row=1, column=3, padx=15)
        self.speed_var = tk.StringVar(value="1")
        self.speed_combo = ttk.Combobox(config_frame, textvariable=self.speed_var, values=("1", "2", "4", "8"), width=6)
        self.speed_combo.grid(row=1, column=4)
        
        self.start_btn = tk.Button(config_frame, text="启动系统", command=self.start_task, bg="#27ae60", fg="white", width=15, height=2)
        self.start_btn.grid(row=0, column=5, rowspan=2, padx=20)
        tk.Button(config_frame, text="强行停止", command=self.stop, bg="#c0392b", fg="white", width=15).grid(row=1, column=5, sticky="s")

        self.wave_frame = tk.LabelFrame(root, text="信号波形监控 (C3, Cz, C4)", padx=5, pady=5)
        self.wave_frame.pack(fill="both", expand=True, padx=15)
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 4), sharex=True)
        self.lines = [ax.plot(np.zeros(500), color=c, lw=1)[0] for ax, c in zip(self.axes, self.colors)]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.wave_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.tree = ttk.Treeview(root, columns=("time", "cue", "res", "prob"), show="headings", height=5)
        for col, head in zip(self.tree["columns"], ["时间", "标记索引/ID", "识别动作", "置信度"]): self.tree.heading(col, text=head)
        self.tree.pack(fill="x", padx=15, pady=5)
        self.log_text = tk.Text(root, height=4, state='disabled', bg="#f8f8f8"); self.log_text.pack(fill="x", padx=15, pady=5)

    def update_ui_state(self):
        """修复点 1：仅禁用 EEG 框，坚决保留 TXT 框"""
        if self.run_mode.get() == "online":
            self.data_entry.config(state='disabled')
            self.data_btn.config(state='disabled')
            self.txt_entry.config(state='normal')
            self.txt_btn.config(state='normal')
            self.log("🌐 在线模式已激活：请加载 TXT 标记文件，然后启动接收。")
        else:
            self.data_entry.config(state='normal')
            self.data_btn.config(state='normal')
            self.txt_entry.config(state='normal')
            self.txt_btn.config(state='normal')
            self.log("📂 离线模式已激活：请配置本地 EEG 文件和 TXT 标记。")

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert('end', f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see('end'); self.log_text.config(state='disabled')

    def select_file(self, var): var.set(filedialog.askopenfilename())

    def stop(self):
        self.is_running = False
        if self.client_socket:
            try: self.client_socket.close()
            except: pass
        self.log("⏹ 系统已停止。")

    def load_data(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.xdf':
            streams, _ = pyxdf.load_xdf(path)
            eeg = next(s for s in streams if "SAGA" in s['info']['name'][0] or s['info']['type'][0] == 'EEG')
            srate = float(eeg['info']['nominal_srate'][0])
            self.root.after(0, lambda: self.fs_var.set(str(int(srate))))
            raw_xdf = eeg['time_series'].T
            try:
                ch_nodes = eeg['info']['desc'][0]['channels'][0]['channel']
                file_labels = [str(ch['label'][0]).strip().upper() for ch in ch_nodes]
            except: file_labels = []

            data = np.zeros((self.EXPECTED_CHANNELS, raw_xdf.shape[1]))
            for i, target in enumerate(self.OFFICIAL_ORDER):
                if target.upper() in file_labels:
                    data[i, :] = raw_xdf[file_labels.index(target.upper()), :]
            return data, srate
        else:
            raw = mne.io.read_raw(path, preload=True, verbose=False)
            fs = raw.info['sfreq']
            self.root.after(0, lambda: self.fs_var.set(str(int(fs))))
            d = raw.get_data() * 1e6
            res = np.zeros((25, d.shape[1]))
            res[:min(25, d.shape[0]), :] = d[:min(25, d.shape[0]), :]
            return res, fs

    def start_task(self):
        mode = self.run_mode.get()
        if not self.txt_path.get():
            messagebox.showwarning("提示", "在线和离线模式都必须加载 TXT 标记文件！")
            return
        
        if mode == "offline" and not self.data_path.get():
            messagebox.showwarning("提示", "离线模式必须加载脑电文件！")
            return

        self.is_running = True
        self.tree.delete(*self.tree.get_children())
        self.wave_buffer = np.zeros((self.EXPECTED_CHANNELS, 500))

        if mode == "online":
            threading.Thread(target=self.run_online_client, daemon=True).start()
        else:
            threading.Thread(target=self.run_offline_analysis, daemon=True).start()

    def run_online_client(self):
        HOST, PORT = '127.0.0.1', 65432
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.log(f"🔌 正在尝试连接 Sender 服务端 {HOST}:{PORT} ...")
            self.client_socket.connect((HOST, PORT))
            self.log("✅ 成功连接到 Sender 服务端！")

            n_ch_bytes = recvall(self.client_socket, 4)
            if not n_ch_bytes: raise ConnectionError("未收到通道数头信息")
            n_channels = struct.unpack('!I', n_ch_bytes)[0]
            
            self.processor = BCIProcessor(n_channels)
            
            # --- 修复点 2：在在线模式中也把 TXT 读入并交给 Processor ---
            fs_actual = int(self.fs_var.get())
            raw_cues = np.loadtxt(self.txt_path.get(), dtype=int)
            self.processor.cue_schedule = (raw_cues * (250.0/fs_actual)).astype(int)
            self.log(f"📄 TXT 标记加载完成，准备基于 {fs_actual}Hz 触发推理。")

            loop_count = 0
            while self.is_running:
                header = recvall(self.client_socket, 5)
                if not header: break
                
                tag, val = struct.unpack('!cI', header)

                if tag == b'M':
                    self.log(f"🚩 收到 Marker 标记: {val}")

                elif tag == b'D':
                    data_bytes = recvall(self.client_socket, val)
                    if not data_bytes: break
                    
                    chunk = np.frombuffer(data_bytes, dtype=np.float32).reshape(n_channels, -1)
                    
                    if self.processor.process_chunk(chunk):
                        self.root.after(0, self.update_result, "Online", self.processor.last_result)
                    
                    self.wave_buffer = np.roll(self.wave_buffer, -chunk.shape[1], axis=1)
                    self.wave_buffer[:, -chunk.shape[1]:] = chunk * 1e6
                    
                    loop_count += 1
                    if loop_count % 5 == 0: 
                        self.root.after(0, self.update_wave, self.wave_buffer)

        except Exception as e:
            self.log(f"❌ 网络异常: {e}")
        finally:
            self.is_running = False
            if self.client_socket: self.client_socket.close()

    def run_offline_analysis(self):
        try:
            self.log("📂 启动离线读取...")
            raw_data, fs = self.load_data(self.data_path.get())
            fs_actual = int(self.fs_var.get())
            speed = int(self.speed_var.get())
            step = max(1, fs_actual // 250)
            
            data = raw_data[:, ::step]
            raw_cues = np.loadtxt(self.txt_path.get(), dtype=int)
            self.processor = BCIProcessor(data.shape[0])
            self.processor.cue_schedule = (raw_cues * (250.0/fs_actual)).astype(int)
            
            ptr = 0; chunk_size = 10; target_sleep = (chunk_size / 250.0) / speed
            
            while self.is_running and ptr < data.shape[1]:
                t_start = time.time()
                chunk = data[:, ptr : ptr + chunk_size]
                
                if self.processor.process_chunk(chunk / 1e6):
                    self.root.after(0, self.update_result, "Offline", self.processor.last_result)
                
                if ptr % (20 * speed) == 0:
                    vis = data[:, max(0, ptr-500):ptr]
                    self.root.after(0, self.update_wave, vis)
                
                ptr += chunk_size
                dt = time.time() - t_start
                if target_sleep > dt: time.sleep(target_sleep - dt)
            self.log("🏁 离线任务结束")
            self.is_running = False
        except Exception as e:
            self.log(f"离线错误: {e}")
            self.is_running = False

    def update_wave(self, vis):
        if vis.shape[1] < 2: return
        for i, idx in enumerate(self.plot_ch_indices):
            self.lines[i].set_ydata(vis[idx, :])
        self.canvas.draw_idle()

    def update_result(self, mode_str, r):
        self.tree.insert("", 0, values=(time.strftime("%H:%M:%S"), f"[{mode_str}]", r[0], f"{r[1]:.3f}"))

if __name__ == "__main__":
    root = tk.Tk(); app = BCIApp(root); root.mainloop()
