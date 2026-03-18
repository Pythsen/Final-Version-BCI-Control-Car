# control.py (modified)
# 集成：实时接收/处理/时间线可视化 + 双线程控制小车（持续到分类改变）

# --- Receiver / Processing packages ---
import socket
import struct
import os
import time
import queue
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

# --- Control packages ---
import serial


# ================= 配置 =================
ARTIFACTS_DIR = r"/home/openEuler/documents/onlinev50pro"
TIMELINE_FILE = r"/home/openEuler/documents/cue_timeline/cue_timeline8.txt"
HOST = '192.168.112.104'
PORT = 65432

WINDOW_SIZE = 1001
FS = 250.0
PLOT_WINDOW = 750   # 3秒波形窗
TOTAL_DURATION_MIN = 47
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {0: 'Left', 1: 'Right', 2: 'Foot', 3: 'Tongue'}
PLOT_CHANNELS = [7, 9, 11]  # C3, Cz, C4
PLOT_LABELS = ['C3 (uV)', 'Cz (uV)', 'C4 (uV)']
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

# ============== 小车控制参数（固定，无需参数启动） ==============
SERIAL_PORT = "/dev/ttyUSB2"   # 你要求固定使用 USB2
SERIAL_BAUD = 115200
# 一次性转向参数（先刹车再转）
TURN_BRAKE_TIME = 1   # 刹车后等待时间（秒），0.05~0.15 之间调
TURN_DURATION   = 0.7   # 转向持续时间（秒），用它标定约90°

CRUISE_SPEED = 300
TURN_SPEED = 400

# 超时安全：超过这个时间没有新分类结果 -> 自动刹车（建议保留，安全）
SAFE_TIMEOUT = 20
# ===============================================================


# ===================== 双线程：控制线程 =====================

def push_latest(q: "queue.Queue[str]", label: str):
    """只保留最新标签，避免队列堆积导致控制滞后"""
    while True:
        try:
            q.put(label, timeout=0.001)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                return


class SerialControlWorker(threading.Thread):
    """
    控制线程：持续到分类改变
    - 只有 label 变化时才发送新的串口指令
    - SAFE_TIMEOUT 超时自动刹车（可选）
    """
    def __init__(self, port: str, baud: int, cmd_queue: "queue.Queue[str]",
                 cruise_speed: int = 120, turn_speed: int = 160,
                 safe_timeout: float = 2.0):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.q = cmd_queue

        self.cruise_speed = cruise_speed
        self.turn_speed = turn_speed
        self.safe_timeout = safe_timeout

        self._stop_evt = threading.Event()
        self._ser = None
        self._last_label = None
        self._last_update_t = time.time()

        self._override_end_t = 0.0     # 一次性转向结束时间
        self._resume_label = None      # 转完恢复到哪个状态（Tongue/Foot）
        self._turning = False          # 是否正在执行一次性转向

    def stop(self):
        self._stop_evt.set()

    def _send(self, cmd: str):
        # 统一加 \r\n，提高兼容性（有些板子需要换行才解析）
        self._ser.write((cmd + "\r\n").encode("utf-8"))

    def _brake(self):
        self._send("$pwm:0,0,0,0#")
        time.sleep(0.02)

    def _label_to_cmd(self, label: str):
        if label == "Tongue":
            v = self.cruise_speed
            return f"$spd:{v},{v},{v},{v}#"
        if label == "Foot":
            return "$pwm:0,0,0,0#"
        if label == "Left":
            t = self.turn_speed
            return f"$spd:-{t},-{t},{t},{t}#"
        if label == "Right":
            t = self.turn_speed
            return f"$spd:{t},{t},-{t},-{t}#"
        return None

    def run(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        self._ser.dtr = False
        self._ser.rts = False

        # 上电先刹车一次
        self._brake()

        try:
            while not self._stop_evt.is_set():
                # 取最新标签
                try:
                    label = self.q.get(timeout=0.05)
                    self._last_update_t = time.time()
                except queue.Empty:
                    label = None
                now = time.time()
                
                # 0) 如果正在转向，到点就恢复之前状态（不依赖新分类）
                if self._turning and now >= self._override_end_t:
                    self._turning = False
                    if self._resume_label is not None:
                        cmd = self._label_to_cmd(self._resume_label)
                        if cmd is not None:
                            self._brake()
                            time.sleep(2)
                            print(f"[CTRL] turn end -> resume {self._resume_label} | SEND: {cmd}")
                            self._send(cmd)
                            try:
                                self._ser.flush()
                            except Exception:
                                pass
                            self._last_label = self._resume_label
                    self._resume_label = None

                # 超时安全刹车
                if self.safe_timeout is not None:
                    if (time.time() - self._last_update_t) > self.safe_timeout:
                        if self._last_label != "Foot":
                            print("[CTRL] timeout -> brake")
                            self._brake()
                            time.sleep(2)
                            self._last_label = "Foot"
                        continue

                if label is None:
                    continue
                if label is None:
                    continue

                # Foot：优先级最高，立刻刹车并取消转向
                if label == "Foot":
                    self._turning = False
                    self._resume_label = None
                    cmd = self._label_to_cmd("Foot")
                    if self._last_label != "Foot":
                        print(f"[CTRL] {self._last_label} -> Foot | SEND: {cmd}")
                        self._send(cmd)
                        try:
                            self._ser.flush()
                        except Exception:
                            pass
                        self._last_label = "Foot"
                    continue
                
                # Left/Right：一次性动作（先刹车再转），结束后恢复到“转向前状态”
                if label in ("Left", "Right"):
                    # 记录恢复目标：转向前若是 Tongue/Foot，就恢复它；否则默认恢复 Foot
                    self._resume_label = self._last_label if self._last_label in ("Tongue", "Foot") else "Foot"
                
                    # 先刹车再转（你要求的）
                    print(f"[CTRL] one-shot {label}: brake -> turn {TURN_DURATION}s -> resume {self._resume_label}")
                    self._brake()
                    time.sleep(TURN_BRAKE_TIME)
                
                    cmd = self._label_to_cmd(label)  # Left/Right 对应你的原地转向 spd
                    if cmd is not None:
                        print(f"[CTRL] SEND: {cmd}")
                        self._send(cmd)
                        try:
                            self._ser.flush()
                        except Exception:
                            pass
                
                        self._turning = True
                        self._override_end_t = time.time() + TURN_DURATION
                
                    # 注意：这里不把 _last_label 设为 Left/Right，因为它不是“持续状态”
                    continue
                
                # 其它（比如 Tongue）：持续到改变
                if label == self._last_label:
                    continue
                
                cmd = self._label_to_cmd(label)
                if cmd is None:
                    continue
                print(f"[CTRL] {self._last_label} -> {label} | SEND: {cmd}")
                self._send(cmd)
                try:
                    self._ser.flush()
                except Exception:
                    pass
                self._last_label = label



        finally:
            try:
                self._brake()
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass


# ================= 模型定义 (必须包含) =================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SmallTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                         batch_first=True, activation='gelu')
        self.trm = nn.TransformerEncoder(enc, num_layers)
        self.pe = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.pe(x)
        return self.trm(x)


class TabNetHeadPlaceholder(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.layers(x)


class EEGNetLight(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=4, csp_dim=6):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.GELU()
        )
        self.proj = nn.Linear(64, 128)
        self.trm = SmallTransformer(d_model=128, nhead=4, num_layers=2)
        self.csp_fc = nn.Sequential(nn.Linear(csp_dim, 64), nn.GELU(), nn.Dropout(0.4))
        input_dim = 128 + 64
        self.head = TabNetHeadPlaceholder(input_dim, n_classes)

    def forward(self, x, csp_feat):
        t = self.temporal(x)
        t = t.permute(0, 2, 1)
        t = self.proj(t)
        t = self.trm(t)
        tpool = t.mean(dim=1)
        csp_out = self.csp_fc(csp_feat)
        cat = torch.cat([tpool, csp_out], dim=1)
        return self.head(cat)


# ================= 处理器组件 =================

class OnlineFilter:
    def __init__(self, n_channels, fs=250.0):
        nyq = fs / 2
        self.b_bp, self.a_bp = butter(4, [0.5/nyq, 100.0/nyq], btype='bandpass')
        self.b_notch, self.a_notch = iirnotch(50.0, 30.0, fs)
        self.n_channels = n_channels
        self.zi_bp_unit = lfilter_zi(self.b_bp, self.a_bp)
        self.zi_notch_unit = lfilter_zi(self.b_notch, self.a_notch)
        self.initialized = False
        self.zi_bp = None
        self.zi_notch = None

    def init_state(self, first_sample):
        self.zi_bp = np.zeros((self.n_channels, len(self.zi_bp_unit)))
        self.zi_notch = np.zeros((self.n_channels, len(self.zi_notch_unit)))
        for i in range(self.n_channels):
            self.zi_bp[i] = self.zi_bp_unit * first_sample[i]
            self.zi_notch[i] = self.zi_notch_unit * first_sample[i]
        self.initialized = True

    def process(self, chunk):
        if not self.initialized:
            self.init_state(chunk[:, 0])
        n_ch = chunk.shape[0]
        filtered = np.zeros_like(chunk)
        for i in range(n_ch):
            out_bp, self.zi_bp[i] = lfilter(self.b_bp, self.a_bp, chunk[i], zi=self.zi_bp[i])
            out_notch, self.zi_notch[i] = lfilter(self.b_notch, self.a_notch, out_bp, zi=self.zi_notch[i])
            filtered[i] = out_notch
        return filtered


class BCIProcessor:
    def __init__(self, n_channels):
        print("[System] Loading Artifacts & Models...")
        self.csp = joblib.load(os.path.join(ARTIFACTS_DIR, "best_csp.pkl"))
        self.scalers = joblib.load(os.path.join(ARTIFACTS_DIR, "best_scalers.pkl"))

        self.model = EEGNetLight(n_channels=n_channels, n_times=WINDOW_SIZE, n_classes=4, csp_dim=6).to(DEVICE)
        self.model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "best_model.pth"), map_location=DEVICE))
        self.model.eval()

        self.filter = OnlineFilter(n_channels)
        self.buffer = np.zeros((n_channels, WINDOW_SIZE), dtype=np.float32)
        self.vis_buffer = np.zeros((n_channels, PLOT_WINDOW), dtype=np.float32)
        self.n_channels = n_channels
        self.total_samples = 0

        try:
            self.cue_schedule = np.loadtxt(TIMELINE_FILE, dtype=int)
            self.cue_pointer = 0
            print(f"[System] Loaded {len(self.cue_schedule)} cues from {TIMELINE_FILE}")
        except Exception:
            print("[Error] TIMELINE_FILE not found! Run generate_cue_timeline.py first.")
            self.cue_schedule = []

        self.predict_countdown = -1
        self.last_result = None
        self.last_triggered_cue_idx = -1

    def process_chunk(self, raw_chunk):
        filtered = self.filter.process(raw_chunk)
        n_new = filtered.shape[1]

        self.buffer = np.roll(self.buffer, -n_new, axis=1)
        self.buffer[:, -n_new:] = filtered

        self.vis_buffer = np.roll(self.vis_buffer, -n_new, axis=1)
        self.vis_buffer[:, -n_new:] = filtered

        current_pos = self.total_samples
        self.total_samples += n_new

        # cue 触发 -> 倒计时 -> 推理
        if self.cue_pointer < len(self.cue_schedule):
            next_cue = self.cue_schedule[self.cue_pointer]
            if current_pos <= next_cue < self.total_samples:
                print(f"\n>>> [PROTOCOL] Cue #{self.cue_pointer+1} triggered. Analyzing in 4s...")
                self.predict_countdown = WINDOW_SIZE
                self.last_triggered_cue_idx = self.cue_pointer
                self.cue_pointer += 1

        if self.predict_countdown > 0:
            self.predict_countdown -= n_new
            if self.predict_countdown <= 0:
                result = self.run_inference()
                self.predict_countdown = -1
                if result:
                    lbl, conf = result
                    self.last_result = (lbl, conf, self.last_triggered_cue_idx)
                    return True

        return False

    def run_inference(self):
        try:
            trial_data = self.buffer[np.newaxis, :, :]
            temp_input = self.buffer.copy()

            for ch in range(self.n_channels):
                s_idx = ch if ch < len(self.scalers) else -1
                temp_input[ch] = self.scalers[s_idx].transform(temp_input[ch].reshape(1, -1)).flatten()

            t_in = torch.from_numpy(temp_input[np.newaxis, :, :]).float().to(DEVICE)
            c_feat = self.csp.transform(trial_data)
            c_in = torch.from_numpy(c_feat).float().to(DEVICE)

            with torch.no_grad():
                logits = self.model(t_in, c_in)
                probs = F.softmax(logits, dim=1)

            p_idx = probs.argmax().item()
            conf = probs.max().item()
            return LABEL_MAP[p_idx], conf
        except Exception as e:
            print(f"Inference Error: {e}")
            return None


# ================= 界面循环 =================

def run_visual_receiver():
    # --- Matplotlib 初始化 ---
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 2, 2, 1])

    axes_wave = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    ax_prog = fig.add_subplot(gs[3])

    lines = []
    for i, ax in enumerate(axes_wave):
        line, = ax.plot(np.zeros(PLOT_WINDOW), color=PLOT_COLORS[i], lw=1.5)
        ax.set_ylabel(PLOT_LABELS[i])
        ax.set_ylim(-50, 50)
        ax.grid(True, alpha=0.3)
        lines.append(line)
    axes_wave[0].set_title("Real-time EEG Stream (Filtered)")

    total_samples_est = TOTAL_DURATION_MIN * 60 * FS
    ax_prog.set_xlim(0, total_samples_est)
    ax_prog.set_ylim(0, 1)
    ax_prog.set_yticks([])
    ax_prog.set_xlabel(f"Timeline (Total {TOTAL_DURATION_MIN} mins)")

    cue_lines = []
    cues = None
    if os.path.exists(TIMELINE_FILE):
        cues = np.loadtxt(TIMELINE_FILE, dtype=int)
        for c in cues:
            l = ax_prog.axvline(x=c, color='lightgray', linewidth=2, alpha=0.8)
            cue_lines.append(l)

    current_time_line = ax_prog.axvline(x=0, color='red', linewidth=2)
    status_text = ax_prog.text(0.01, 1.1, "Status: Waiting...",
                               transform=ax_prog.transAxes, fontsize=10, fontweight='bold')

    plt.tight_layout()

    # --- 控制线程启动（在 socket 之前也可以，这里放一起更好管） ---
    cmd_q = queue.Queue(maxsize=5)
    ctrl = SerialControlWorker(
        port=SERIAL_PORT,
        baud=SERIAL_BAUD,
        cmd_queue=cmd_q,
        cruise_speed=CRUISE_SPEED,
        turn_speed=TURN_SPEED,
        safe_timeout=SAFE_TIMEOUT
    )
    print(f"[CTRL] Starting control worker on {SERIAL_PORT} @ {SERIAL_BAUD}")
    ctrl.start()

    # --- Socket 连接 ---
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            print(f"[Receiver] Connecting to {HOST}:{PORT}...")
            s.connect((HOST, PORT))

            data = s.recv(4)
            n_channels = struct.unpack('!I', data)[0]
            print(f"[Receiver] Connected. Channels: {n_channels}")

            processor = BCIProcessor(n_channels)
            plot_counter = 0

            while True:
                header = s.recv(1)
                if not header:
                    break

                if header == b'D':
                    len_bytes = s.recv(4)
                    length = struct.unpack('!I', len_bytes)[0]

                    chunks = b''
                    while len(chunks) < length:
                        packet = s.recv(length - len(chunks))
                        if not packet:
                            break
                        chunks += packet

                    raw_chunk = np.frombuffer(chunks, dtype=np.float32).reshape(n_channels, -1)

                    # 处理
                    has_result = processor.process_chunk(raw_chunk)

                    # 有结果 -> 推送给控制线程（持续到分类改变）
                    if has_result:
                        lbl, conf, c_idx = processor.last_result
                        print(f"  ★ RESULT: {lbl} ({conf:.2f})")
                        push_latest(cmd_q, lbl)

                        if cues is not None and c_idx < len(cue_lines):
                            cue_lines[c_idx].set_color('green')
                            ax_prog.text(cues[c_idx], 0.5, lbl[0], color='black', fontsize=8, ha='center')

                    # --- 绘图刷新 ---
                    plot_counter += 1
                    if plot_counter >= 10:
                        vis_data = processor.vis_buffer * 1e6  # uV
                        for i, ch_idx in enumerate(PLOT_CHANNELS):
                            if ch_idx < n_channels:
                                lines[i].set_ydata(vis_data[ch_idx, :])

                        current_time_line.set_xdata([processor.total_samples])

                        curr_sec = processor.total_samples / FS
                        curr_min = int(curr_sec // 60)
                        curr_sec_rem = int(curr_sec % 60)
                        status_text.set_text(f"Time: {curr_min:02d}:{curr_sec_rem:02d} / {TOTAL_DURATION_MIN}:00")

                        plt.draw()
                        plt.pause(0.001)
                        plot_counter = 0

                elif header == b'M':
                    s.recv(4)

        except KeyboardInterrupt:
            print("\n[System] Ctrl+C received. Exiting...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 停止控制线程（会自动刹车并关闭串口）
            ctrl.stop()
            ctrl.join(timeout=1.0)


if __name__ == "__main__":
    if not os.path.exists(TIMELINE_FILE):
        print(f"Please run 'generate_cue_timeline.py' first!")
    else:
        run_visual_receiver()
