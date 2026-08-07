# stdlib 
import os, sys, csv, pickle, hashlib, threading, queue
import time, warnings, webbrowser, platform
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, END
from datetime import datetime
from collections import deque

# scientific 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ML 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# optional libs 
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Dropout, LSTM,
        Bidirectional, BatchNormalization)
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from scapy.all import sniff, IP, TCP, UDP as SUDP
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

warnings.filterwarnings("ignore")

#  constants
MODEL_DIR = "model"
LOGS_DIR  = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,  exist_ok=True)

LEAKAGE_COLS = [
    "Unnamed: 0", "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Timestamp"
]

CSV_FILES = [
    "DrDOS_DNS.csv", "DrDOS_LDAP.csv", "DrDOS_MSSQL.csv",
    "DrDOS_NTP.csv",  "DrDOS_NetBIOS.csv", "DrDOS_SNMP.csv",
    "DrDOS_SSDP.csv", "DrDOS_UDP.csv",     "Syn.csv", "UDP_LAG.csv"
]

# Severity base weights per attack type (0-100) 
ATTACK_WEIGHTS = {
    "BENIGN":        0,
    "DrDoS_DNS":    80,
    "DrDoS_LDAP":   85,
    "DrDoS_MSSQL":  75,
    "DrDoS_NTP":    90,
    "DrDoS_NetBIOS":70,
    "DrDoS_SNMP":   80,
    "DrDoS_SSDP":   75,
    "DrDoS_UDP":    85,
    "Syn":          95,
    "UDP-lag":      65,
}

# Colours for timeline chart 
ATTACK_COLORS = {
    "BENIGN":        "#52c41a",
    "DrDoS_DNS":     "#1890ff",
    "DrDoS_LDAP":    "#722ed1",
    "DrDoS_MSSQL":   "#eb2f96",
    "DrDoS_NTP":     "#fa8c16",
    "DrDoS_NetBIOS": "#13c2c2",
    "DrDoS_SNMP":    "#faad14",
    "DrDoS_SSDP":    "#a0d911",
    "DrDoS_UDP":     "#f5222d",
    "Syn":           "#ff4d4f",
    "UDP-lag":       "#d46b08",
    "UNKNOWN":       "#8c8c8c",
}

#  Global State
filename       = None
dataset        = None
labels         = []
X = Y          = None
X_train = X_test = y_train = y_test = None
scaler         = None
selector       = None
label_encoders = []
train_columns  = []
classifier     = None
model_results  = {}

accuracy  = []
precision = []
recall    = []
fscore    = []
metrics_dict = {}  

rt_running       = False
rt_queue         = queue.Queue()

recent_attack_count = 0          # counts attacks in last 10 sec window
last_attack_time    = 0.0

# Timeline buffers  (deque keeps last 60 seconds of data)
timeline_times   = deque(maxlen=60)
timeline_counts  = deque(maxlen=60)
timeline_labels  = deque(maxlen=60)   # most frequent label each second
_timeline_bucket = {}                  # label → count for current second
_timeline_lock   = threading.Lock()

# CSV log file for this session
_session_log_path = os.path.join(
    LOGS_DIR,
    f"attack_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
)
_log_file   = None
_log_writer = None

#  Session csv log
def _init_session_log():
    global _log_file, _log_writer
    _log_file = open(_session_log_path, "w", newline="")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow([
        "Timestamp", "Predicted_Class", "Severity",
        "Confidence_%", "Packet_Rate_pps", "Is_Attack"
    ])
    _log_file.flush()

def _write_log_row(label, severity, confidence_pct, pkt_rate):
    if _log_writer is None:
        return
    _log_writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        label,
        severity,
        f"{confidence_pct:.1f}",
        f"{pkt_rate:.0f}",
        "YES" if label != "BENIGN" else "NO"
    ])
    _log_file.flush()

# severity scoring
def compute_severity(label, confidence, packet_rate, attack_count):
    """
    Returns 0-100 integer severity score.
      label        : predicted class string
      confidence   : model probability for that class (0.0-1.0)
      packet_rate  : packets per second in current batch
      attack_count : number of consecutive attack batches seen recently
    """
    base         = ATTACK_WEIGHTS.get(label, 60)
    conf_boost   = (confidence - 0.5) * 30       # 0–15 pts
    rate_boost   = min(packet_rate / 1000 * 20, 20)   # 0–20 pts
    persist_boost= min(attack_count * 2, 15)      # 0–15 pts
    raw = base * 0.6 + conf_boost + rate_boost + persist_boost
    return int(min(100, max(0, round(raw))))

def severity_color(score):
    if score == 0:
        return "#52c41a"  
    elif score < 40:
        return "#a0d911" 
    elif score < 65:
        return "#faad14" 
    elif score < 85:
        return "#fa541c"  
    else:
        return "#f5222d"   

def severity_label(score):
    if score == 0:   return "SAFE"
    elif score < 40: return "LOW"
    elif score < 65: return "MEDIUM"
    elif score < 85: return "HIGH"
    else:            return "CRITICAL"

# Alert Sound
def _play_alert_sound(severity):
    """Non-blocking system beep. Frequency scales with severity."""
    freq = 800 + severity * 4   # 800Hz–1200Hz
    dur  = 300                  # ms
    try:
        if platform.system() == "Windows":
            import winsound
            threading.Thread(
                target=lambda: winsound.Beep(freq, dur), daemon=True
            ).start()
        elif platform.system() == "Darwin":
            threading.Thread(
                target=lambda: os.system("afplay /System/Library/Sounds/Funk.aiff"),
                daemon=True
            ).start()
        else:
            threading.Thread(
                target=lambda: os.system("paplay /usr/share/sounds/freedesktop/stereo/dialog-warning.oga 2>/dev/null || aplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null || printf '\a'"),
                daemon=True
            ).start()
    except Exception:
        pass   


class ToastPopup(tk.Toplevel):
    def __init__(self, parent, label, severity, auto_close_ms=4000):
        super().__init__(parent)
        self.overrideredirect(True)        
        self.attributes("-topmost", True)

        color  = severity_color(severity)
        slabel = severity_label(severity)

        outer = tk.Frame(self, bg=color, padx=2, pady=2)
        outer.pack(fill="both", expand=True)

        inner = tk.Frame(outer, bg="white", padx=14, pady=10)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text="⚠  ATTACK DETECTED",
                 bg="white", fg=color,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")

        tk.Label(inner, text=label,
                 bg="white", fg="#001529",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(2, 0))

        sev_frame = tk.Frame(inner, bg="white")
        sev_frame.pack(anchor="w", pady=(4, 0))
        tk.Label(sev_frame, text=f"Severity: ",
                 bg="white", fg="#595959",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(sev_frame, text=f"{severity}/100  [{slabel}]",
                 bg="white", fg=color,
                 font=("Segoe UI", 9, "bold")).pack(side="left")

        tk.Label(inner,
                 text=datetime.now().strftime("%H:%M:%S"),
                 bg="white", fg="#8c8c8c",
                 font=("Segoe UI", 8)).pack(anchor="e")

        tk.Button(inner, text="✕", command=self.destroy,
                  bg="white", relief="flat", fg="#8c8c8c",
                  font=("Segoe UI", 8), cursor="hand2").pack(
                  side="right", anchor="ne")

        self.update_idletasks()
        sw = self.winfo_screenwidth()
        w  = self.winfo_width()
        self.geometry(f"+{sw - w - 30}+30")

        self.after(auto_close_ms, self._safe_destroy)

    def _safe_destroy(self):
        try:
            self.destroy()
        except Exception:
            pass


_last_toast_time = 0.0

def show_alert_toast(label, severity):
    global _last_toast_time
    now = time.time()
    if now - _last_toast_time < 5.0:
        return
    _last_toast_time = now
    try:
        ToastPopup(root, label, severity)
    except Exception:
        pass

# Model Versioning
def _model_hash():
    if X_train is None:
        return "none"
    key = f"{X_train.shape}_{len(labels)}"
    return hashlib.md5(key.encode()).hexdigest()[:8]

def _save_model(obj, name):
    path = os.path.join(MODEL_DIR, f"{name}_{_model_hash()}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log(f"Model saved → {path}")

def _load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}_{_model_hash()}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

#  UI — Root Window
root = tk.Tk()
root.title("DDoS Intelligence Dashboard")
root.geometry("1550x940")
root.configure(bg="#f0f2f5")

# Sidebar 
sidebar = tk.Frame(root, bg="#001529", width=265)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

# Header 
header = tk.Frame(root, bg="white", height=58)
header.pack(side="top", fill="x")
tk.Label(header,
         text="Network Security Intelligence  |  DDoS Mitigation",
         bg="white", fg="#001529",
         font=("Segoe UI", 14, "bold")).pack(side="left", padx=22, pady=14)

# KPI bar 
kpi_bar = tk.Frame(root, bg="#f0f2f5")
kpi_bar.pack(fill="x", padx=16, pady=(6, 0))

def _kpi(parent, title, init, color):
    f = tk.Frame(parent, bg="white",
                 highlightbackground="#e0e0e0", highlightthickness=1)
    f.pack(side="left", padx=6, expand=True, fill="both")
    tk.Label(f, text=title, bg="white", fg="#8c8c8c",
             font=("Segoe UI", 8)).pack(pady=(6, 0))
    lbl = tk.Label(f, text=init, bg="white", fg=color,
                   font=("Segoe UI", 14, "bold"))
    lbl.pack(pady=(0, 6))
    return lbl

kpi_acc   = _kpi(kpi_bar, "Best Accuracy",    "—", "#1890ff")
kpi_f1    = _kpi(kpi_bar, "Best F1",          "—", "#52c41a")
kpi_recs  = _kpi(kpi_bar, "Records",          "—", "#722ed1")
kpi_feats = _kpi(kpi_bar, "Features",         "—", "#fa8c16")
kpi_cls   = _kpi(kpi_bar, "Classes",          "—", "#eb2f96")
kpi_sev   = _kpi(kpi_bar, "Last Severity",    "—", "#f5222d")   
kpi_logs  = _kpi(kpi_bar, "Events Logged",    "0", "#13c2c2")   

def _update_kpis():
    if model_results:
        kpi_acc.config(text=f"{max(model_results.values()):.2f}%")
    if dataset is not None:
        kpi_recs.config(text=f"{len(dataset):,}")
        kpi_cls.config(text=str(len(labels)))
    if X_train is not None:
        kpi_feats.config(text=str(X_train.shape[1]))
    if metrics_dict:
        best_f1 = max(v["f1"] for v in metrics_dict.values())
        kpi_f1.config(text=f"{best_f1:.2f}%")

# Notebook tabs 
content  = tk.Frame(root, bg="#f0f2f5")
content.pack(side="right", expand=True, fill="both")

notebook = ttk.Notebook(content)
notebook.pack(expand=True, fill="both", padx=8, pady=6)

tab_viz  = tk.Frame(notebook, bg="#f0f2f5")
tab_rt   = tk.Frame(notebook, bg="#f0f2f5")
tab_logs = tk.Frame(notebook, bg="white")  
tab_log  = tk.Frame(notebook, bg="white")

notebook.add(tab_viz,  text="  Charts & Metrics  ")
notebook.add(tab_rt,   text="  Real-Time Detection  ")
notebook.add(tab_logs, text="  Session History  ")    
notebook.add(tab_log,  text="  Execution Log  ")

# Chart frame
viz_frame = tk.Frame(tab_viz, bg="#f0f2f5")
viz_frame.pack(expand=True, fill="both")

# Real-Time Tab
rt_top = tk.Frame(tab_rt, bg="#f0f2f5")
rt_top.pack(fill="x", padx=14, pady=(10, 0))

rt_ctrl = tk.Frame(rt_top, bg="#f0f2f5")
rt_ctrl.pack(side="left", fill="y")

tk.Label(rt_ctrl, text="Live Network Analyser",
         bg="#f0f2f5", fg="#001529",
         font=("Segoe UI", 12, "bold")).pack(anchor="w")

rt_status = tk.Label(rt_ctrl, text="Status: Idle",
                     bg="#f0f2f5", fg="#595959",
                     font=("Segoe UI", 10))
rt_status.pack(anchor="w", pady=(2, 6))

iface_row = tk.Frame(rt_ctrl, bg="#f0f2f5")
iface_row.pack(anchor="w")
tk.Label(iface_row, text="Interface:", bg="#f0f2f5",
         font=("Segoe UI", 10)).pack(side="left")
rt_iface_var = tk.StringVar(value="eth0")
tk.Entry(iface_row, textvariable=rt_iface_var,
         font=("Segoe UI", 10), width=14).pack(side="left", padx=6)

btn_row = tk.Frame(rt_ctrl, bg="#f0f2f5")
btn_row.pack(anchor="w", pady=8)

def _rt_btn(p, txt, cmd, col):
    tk.Button(p, text=txt, command=cmd, bg=col, fg="white",
              font=("Segoe UI", 10), relief="flat",
              padx=12, pady=5).pack(side="left", padx=(0, 6))

_rt_btn(btn_row, "▶  Start Capture", lambda: start_rt_detection(), "#52c41a")
_rt_btn(btn_row, "■  Stop",          lambda: stop_rt_detection(),  "#ff4d4f")

tk.Label(rt_ctrl,
         text="Requires admin/root + pip install scapy\nTrain a model first.",
         bg="#f0f2f5", fg="#8c8c8c", font=("Segoe UI", 8),
         justify="left").pack(anchor="w")

# Severity Gauge
sev_frame = tk.Frame(rt_top, bg="white",
                     highlightbackground="#e0e0e0", highlightthickness=1)
sev_frame.pack(side="left", padx=(20, 0), fill="y")

tk.Label(sev_frame, text="Threat Severity",
         bg="white", fg="#595959",
         font=("Segoe UI", 9, "bold")).pack(pady=(8, 0))

sev_canvas = tk.Canvas(sev_frame, width=160, height=100,
                        bg="white", highlightthickness=0)
sev_canvas.pack(padx=10, pady=4)

sev_label_var  = tk.StringVar(value="SAFE")
sev_score_var  = tk.StringVar(value="0")
sev_attack_var = tk.StringVar(value="—")

_sev_score_lbl = tk.Label(sev_frame, textvariable=sev_score_var,
                         bg="white", fg="#52c41a",
                         font=("Segoe UI", 26, "bold"))
_sev_score_lbl.pack()

_sev_text_lbl = tk.Label(sev_frame, textvariable=sev_label_var,
                        bg="white", fg="#52c41a",
                        font=("Segoe UI", 10, "bold"))
_sev_text_lbl.pack()

_sev_atk_lbl = tk.Label(sev_frame, textvariable=sev_attack_var,
                       bg="white", fg="#595959",
                       font=("Segoe UI", 9))
_sev_atk_lbl.pack(pady=(0, 8))

def _draw_severity_gauge(score):
    sev_canvas.delete("all")
    cx, cy, r = 80, 80, 60
    # background arc
    sev_canvas.create_arc(cx-r, cy-r, cx+r, cy+r,
                          start=180, extent=180,
                          style="arc", outline="#e8e8e8", width=12)
    # value arc
    angle = int(score * 1.8)   # 0-100 → 0-180 degrees
    col = severity_color(score)
    if angle > 0:
        sev_canvas.create_arc(cx-r, cy-r, cx+r, cy+r,
                              start=180, extent=angle,
                              style="arc", outline=col, width=12)
    # labels
    sev_canvas.create_text(20, 78, text="0",   fill="#aaa", font=("Segoe UI", 8))
    sev_canvas.create_text(140, 78, text="100", fill="#aaa", font=("Segoe UI", 8))

def update_severity_ui(label, score):
    col = severity_color(score)
    _draw_severity_gauge(score)
    sev_score_var.set(str(score))
    sev_label_var.set(severity_label(score))
    sev_attack_var.set(label)
    _sev_score_lbl.config(fg=col)
    _sev_text_lbl.config(fg=col)
    kpi_sev.config(text=f"{score}  [{severity_label(score)}]", fg=col)

_draw_severity_gauge(0)   # initial empty gauge

# Live Attack TimeLine
timeline_frame = tk.Frame(tab_rt, bg="white",
                           highlightbackground="#e0e0e0", highlightthickness=1)
timeline_frame.pack(fill="x", padx=14, pady=(10, 0))

tk.Label(timeline_frame, text="Live Attack Timeline  (last 60 sec)",
         bg="white", fg="#595959",
         font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 0))

fig_tl, ax_tl = plt.subplots(figsize=(12, 2.2))
fig_tl.patch.set_facecolor("white")
ax_tl.set_facecolor("#fafafa")
ax_tl.set_xlim(0, 60)
ax_tl.set_ylim(0, 30)
ax_tl.set_xlabel("seconds ago", fontsize=8)
ax_tl.set_ylabel("events/sec", fontsize=8)
ax_tl.tick_params(labelsize=7)
plt.tight_layout(pad=0.6)

canvas_tl = FigureCanvasTkAgg(fig_tl, master=timeline_frame)
canvas_tl.get_tk_widget().pack(fill="x", padx=10, pady=(2, 8))

def _animate_timeline(frame_num):
    with _timeline_lock:
        counts = list(timeline_counts)
        tlabels = list(timeline_labels)

    ax_tl.cla()
    ax_tl.set_facecolor("#fafafa")
    ax_tl.set_xlim(0, 60)
    ax_tl.set_xlabel("seconds ago", fontsize=8)
    ax_tl.set_ylabel("events/sec", fontsize=8)
    ax_tl.tick_params(labelsize=7)

    n = len(counts)
    if n == 0:
        ax_tl.set_ylim(0, 10)
        return

    xs = list(range(n - 1, -1, -1)) 
    ys = counts

    bar_colors = [
        ATTACK_COLORS.get(tlabels[i], "#8c8c8c") for i in range(n)
    ]
    ax_tl.bar(xs, ys, color=bar_colors, width=0.8, edgecolor="none")
    ax_tl.set_ylim(0, max(max(ys) * 1.3, 10))

    if max(ys) > 0:
        ax_tl.axhline(max(ys), color="#ff4d4f", linewidth=0.6,
                      linestyle="--", alpha=0.5)
    canvas_tl.draw()

_tl_anim = animation.FuncAnimation(
    fig_tl, _animate_timeline, interval=1000, cache_frame_data=False
)

# Real-time text log
rt_log_frame = tk.Frame(tab_rt, bg="white",
                         highlightbackground="#e0e0e0", highlightthickness=1)
rt_log_frame.pack(fill="both", expand=True, padx=14, pady=(8, 10))

tk.Label(rt_log_frame, text="Detection Log",
         bg="white", fg="#595959",
         font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(4, 0))

rt_log_text = tk.Text(rt_log_frame, font=("Consolas", 9),
                       bg="white", relief="flat", height=8)
rt_log_text.pack(fill="both", expand=True, padx=8, pady=(0, 6))
rt_log_scroll = tk.Scrollbar(rt_log_frame, command=rt_log_text.yview)
rt_log_scroll.pack(side="right", fill="y")
rt_log_text.config(yscrollcommand=rt_log_scroll.set)

rt_log_text.tag_config("attack",  foreground="#f5222d")
rt_log_text.tag_config("benign",  foreground="#52c41a")
rt_log_text.tag_config("warning", foreground="#fa8c16")
rt_log_text.tag_config("info",    foreground="#1890ff")

def rt_log_msg(msg, tag="info"):
    rt_log_text.insert(END, f"[{time.strftime('%H:%M:%S')}] {msg}\n", tag)
    rt_log_text.see(END)

#  Session History Tab
tk.Label(tab_logs, text="Session History & CSV Log",
         bg="white", fg="#001529",
         font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(12, 4))

log_path_var = tk.StringVar(value=f"Log file: {_session_log_path}")
tk.Label(tab_logs, textvariable=log_path_var,
         bg="white", fg="#8c8c8c",
         font=("Segoe UI", 9)).pack(anchor="w", padx=14)

# Filter row
filter_frame = tk.Frame(tab_logs, bg="white")
filter_frame.pack(fill="x", padx=14, pady=6)

tk.Label(filter_frame, text="Filter:",
         bg="white", font=("Segoe UI", 9)).pack(side="left")
filter_var = tk.StringVar(value="ALL")
filter_combo = ttk.Combobox(filter_frame, textvariable=filter_var,
                             values=["ALL", "ATTACK", "BENIGN"],
                             state="readonly", width=10,
                             font=("Segoe UI", 9))
filter_combo.pack(side="left", padx=6)

events_count_var = tk.StringVar(value="Events: 0")
tk.Label(filter_frame, textvariable=events_count_var,
         bg="white", fg="#595959",
         font=("Segoe UI", 9)).pack(side="left", padx=10)

def _open_log_folder():
    path = os.path.abspath(LOGS_DIR)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        os.system(f"open '{path}'")
    else:
        os.system(f"xdg-open '{path}' 2>/dev/null")

tk.Button(filter_frame, text="📁 Open Logs Folder",
          command=_open_log_folder, font=("Segoe UI", 9),
          relief="flat", bg="#f0f2f5",
          padx=8, pady=3).pack(side="right")

# Treeview table
cols = ("Time", "Attack Class", "Severity", "Level",
        "Confidence%", "Pkt/s", "Is Attack")

tree_frame = tk.Frame(tab_logs, bg="white")
tree_frame.pack(fill="both", expand=True, padx=14, pady=(0, 12))

tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=18)
for c in cols:
    w = 160 if c == "Attack Class" else 90
    tree.heading(c, text=c)
    tree.column(c, width=w, anchor="center")

tree.tag_configure("attack_row", background="#fff1f0")
tree.tag_configure("benign_row", background="#f6ffed")

vsb = ttk.Scrollbar(tree_frame, orient="vertical",   command=tree.yview)
hsb = ttk.Scrollbar(tree_frame, orient="horizontal",  command=tree.xview)
tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
tree.grid(row=0, column=0, sticky="nsew")
vsb.grid(row=0, column=1, sticky="ns")
hsb.grid(row=1, column=0, sticky="ew")
tree_frame.grid_rowconfigure(0, weight=1)
tree_frame.grid_columnconfigure(0, weight=1)

_log_rows = []    

def _add_tree_row(label, severity, confidence_pct, pkt_rate):
    def _insert():
        row = (
            datetime.now().strftime("%H:%M:%S"),
            label,
            severity,
            severity_label(severity),
            f"{confidence_pct:.1f}",
            f"{pkt_rate:.0f}",
            "YES" if label != "BENIGN" else "NO"
        )
        _log_rows.append(row)
        tag = "attack_row" if label != "BENIGN" else "benign_row"

        flt = filter_var.get()
        show = (flt == "ALL"
                or (flt == "ATTACK"  and label != "BENIGN")
                or (flt == "BENIGN"  and label == "BENIGN"))
        if show:
            tree.insert("", 0, values=row, tags=(tag,))

        events_count_var.set(f"Events: {len(_log_rows)}")
        kpi_logs.config(text=str(len(_log_rows)))

    root.after(0, _insert)

def _apply_filter(*_):
    for item in tree.get_children():
        tree.delete(item)
    flt = filter_var.get()
    for row in reversed(_log_rows):
        lbl = row[1]
        show = (flt == "ALL"
                or (flt == "ATTACK" and lbl != "BENIGN")
                or (flt == "BENIGN" and lbl == "BENIGN"))
        if show:
            tag = "attack_row" if lbl != "BENIGN" else "benign_row"
            tree.insert("", "end", values=row, tags=(tag,))

filter_combo.bind("<<ComboboxSelected>>", _apply_filter)

# Execution log tab 
tk.Label(tab_log, text="System Execution Log",
         bg="white", fg="#595959",
         font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
text = tk.Text(tab_log, font=("Consolas", 10), bg="white", relief="flat")
text.pack(fill="both", expand=True, padx=10, pady=6)
log_sc = tk.Scrollbar(tab_log, command=text.yview)
log_sc.pack(side="right", fill="y")
text.config(yscrollcommand=log_sc.set)

def log(msg):
    text.insert(END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    text.see(END)
    root.update_idletasks()

def embed_plot(fig):
    for w in viz_frame.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=viz_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill="both")
    notebook.select(tab_viz)

# DataSet Upload 
def upload_dataset():
    global filename, dataset, labels
    text.delete("1.0", END)
    folder = filedialog.askdirectory(initialdir=".")
    if not folder:
        return

    filename = folder
    log(f"Loading dataset from: {folder}")
    dfs = []
    for csv_name in CSV_FILES:
        path = os.path.join(folder, csv_name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            dfs.append(df)
            log(f"  Loaded {csv_name}  ({len(df):,} rows)")
        else:
            log(f"  WARNING: {csv_name} not found — skipping")

    if not dfs:
        messagebox.showerror("Error", "No CSV files found in selected folder.")
        return

    dataset = pd.concat(dfs, ignore_index=True)
    dataset.columns = dataset.columns.str.strip()
    labels = sorted(dataset["Label"].unique().tolist())

    log(f"\nTotal records : {len(dataset):,}")
    log(f"Classes ({len(labels)}) : {labels}")

    dist = dataset["Label"].value_counts()
    _update_kpis()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#f0f2f5")
    dist.plot(kind="bar", ax=axes[0], color="#1890ff", edgecolor="white")
    axes[0].set_title("Class Distribution")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].set_facecolor("white")
    axes[1].pie(dist.values, labels=dist.index, autopct="%1.1f%%",
                startangle=140,
                colors=sns.color_palette("tab10", len(dist)))
    axes[1].set_title("Class Share")
    plt.tight_layout()
    embed_plot(fig)

# Data Preprocessing
def preprocess_dataset():
    global dataset, labels, X, Y, X_train, X_test, y_train, y_test
    global scaler, selector, label_encoders, train_columns

    if dataset is None:
        messagebox.showwarning("Warning", "Upload the dataset first.")
        return

    log("\n── Preprocessing ──────────────────────────")
    df = dataset.copy()

    drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    log(f"Dropped leakage columns: {drop_cols}")

    label_encoders = []
    for col in df.select_dtypes(include="object").columns:
        if col == "Label":
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders.append((col, le))

    le_label = LabelEncoder()
    le_label.fit(labels)
    Y = le_label.transform(df["Label"])
    df.drop(columns=["Label"], inplace=True)

    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.clip(-1e9, 1e9)

    X = df.values
    train_columns = list(df.columns)
    log(f"Features before selection : {X.shape[1]}")

    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(X)
    log(f"Features after threshold  : {X.shape[1]}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    log(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    if HAS_SMOTE:
        before = dict(zip(*np.unique(y_train, return_counts=True)))
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        after = dict(zip(*np.unique(y_train, return_counts=True)))
        log("SMOTE applied:")
        for cls_id in sorted(before.keys()):
            log(f"  {labels[cls_id]:<25} {before[cls_id]:>5} → {after[cls_id]:>5}")
    else:
        log("SMOTE not available (pip install imbalanced-learn).")

    _update_kpis()

    top20 = pd.DataFrame(X_train).iloc[:, :20]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(top20.corr(), ax=ax, cmap="coolwarm",
                center=0, linewidths=0.3, annot=False)
    ax.set_title("Feature Correlation Heatmap (top 20 features)")
    plt.tight_layout()
    embed_plot(fig)
    log("Preprocessing complete ✓")

#  Metric Calculation
def _calc_metrics(algo_name, y_pred, y_true=None):
    global metrics_dict, model_results, accuracy, precision, recall, fscore

    if y_true is None:
        y_true = y_test

    a = accuracy_score(y_true, y_pred) * 100
    p = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    r = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    accuracy.append(a); precision.append(p)
    recall.append(r);   fscore.append(f)
    model_results[algo_name] = a
    metrics_dict[algo_name]  = {"acc": a, "prec": p, "rec": r, "f1": f}

    log(f"\n── {algo_name} ──")
    log(f"  Accuracy  : {a:.2f}%  |  Precision: {p:.2f}%")
    log(f"  Recall    : {r:.2f}%  |  F1 Score : {f:.2f}%")
    _update_kpis()

    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f0f2f5")

    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=axes[0], linewidths=0.3)
    axes[0].set_title(f"{algo_name} — Confusion Matrix")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].set_facecolor("white")

    report = classification_report(y_true, y_pred, target_names=labels,
                                   output_dict=True, zero_division=0)
    cls_f1 = [report[l]["f1-score"] * 100 for l in labels]
    colors = ["#52c41a" if v >= 90 else "#faad14" if v >= 70 else "#ff4d4f"
              for v in cls_f1]
    axes[1].barh(labels, cls_f1, color=colors, edgecolor="white")
    axes[1].set_xlim(0, 105)
    axes[1].set_title(f"{algo_name} — Per-Class F1 (%)")
    axes[1].set_facecolor("white")
    for i, v in enumerate(cls_f1):
        axes[1].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    embed_plot(fig)

#  ML Models
def run_naive_bayes():
    if X_train is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    cached = _load_model("nb")
    if cached:
        log("Naive Bayes: loaded from cache"); nb = cached
    else:
        log("Training Naive Bayes…")
        nb = GaussianNB(); nb.fit(X_train, y_train); _save_model(nb, "nb")
    _calc_metrics("Naive Bayes", nb.predict(X_test))

def run_random_forest():
    global classifier
    if X_train is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    cached = _load_model("rf")
    if cached:
        log("Random Forest: loaded from cache"); rf = cached
    else:
        log("Training Random Forest…")
        rf = RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                    n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train); _save_model(rf, "rf")
    classifier = rf
    _calc_metrics("Random Forest", rf.predict(X_test))

    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([f"F{i}" for i in top_idx], importances[top_idx],
           color="#1890ff", edgecolor="white")
    ax.set_title("Random Forest — Top 20 Feature Importances")
    ax.tick_params(axis="x", rotation=45)
    ax.set_facecolor("white")
    plt.tight_layout(); embed_plot(fig)

def run_svm():
    if X_train is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    cached = _load_model("svm")
    if cached:
        log("LinearSVC: loaded from cache"); svc = cached
    else:
        log("Training LinearSVC…")
        svc = LinearSVC(max_iter=3000, class_weight="balanced", random_state=42)
        svc.fit(X_train, y_train); _save_model(svc, "svm")
    _calc_metrics("LinearSVC", svc.predict(X_test))

def run_xgboost():
    if X_train is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    cached = _load_model("xgb")
    if cached:
        log("XGBoost: loaded from cache"); xgb = cached
    else:
        log("Training XGBoost…")
        xgb = XGBClassifier(n_estimators=150, learning_rate=0.1,
                             tree_method="hist", verbosity=0,
                             eval_metric="mlogloss", n_jobs=-1,
                             random_state=42)
        xgb.fit(X_train, y_train); _save_model(xgb, "xgb")
    _calc_metrics("XGBoost", xgb.predict(X_test))

def run_ensemble():
    global classifier
    if X_train is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    log("Training Voting Ensemble (RF + XGB + LinearSVC)…")
    rf  = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                  n_jobs=-1, random_state=42)
    xgb = XGBClassifier(n_estimators=100, tree_method="hist", verbosity=0,
                         eval_metric="mlogloss", n_jobs=-1, random_state=42)
    svc = LinearSVC(max_iter=2000, class_weight="balanced", random_state=42)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("svc", svc)],
        voting="hard", n_jobs=-1)
    ensemble.fit(X_train, y_train)
    classifier = ensemble
    _calc_metrics("Voting Ensemble", ensemble.predict(X_test))

# DNN 
def run_dnn():
    global classifier
    if not HAS_TF:
        messagebox.showerror("Error", "pip install tensorflow"); return
    if X_train is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    log("Training DNN…")
    n = len(labels)
    model = Sequential([
        Dense(256, input_dim=X_train.shape[1], activation="relu"),
        BatchNormalization(), Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(), Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(n, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    cb = [EarlyStopping(patience=5, restore_best_weights=True),
          ReduceLROnPlateau(patience=3, factor=0.5)]
    history = model.fit(X_train, to_categorical(y_train, n),
                        validation_split=0.1, epochs=30,
                        batch_size=128, callbacks=cb, verbose=0)
    preds = np.argmax(model.predict(X_test), axis=1)
    _calc_metrics("DNN", preds)
    classifier = model

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#f0f2f5")
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history.history[key], label="train", color="#1890ff")
        ax.plot(history.history[f"val_{key}"], label="val", color="#ff4d4f")
        ax.set_title(f"DNN — {title}"); ax.legend(); ax.set_facecolor("white")
    plt.tight_layout(); embed_plot(fig)

# LSTM 
def run_lstm():
    global classifier
    if not HAS_TF:
        messagebox.showerror("Error", "pip install tensorflow"); return
    if X_train is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    log("Training LSTM…")
    n = len(labels)
    Xr = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    Xrt= X_test.reshape( X_test.shape[0],  1, X_test.shape[1])
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(1, X_train.shape[1])),
        Dropout(0.3), LSTM(32), BatchNormalization(), Dropout(0.2),
        Dense(64, activation="relu"), Dense(n, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    cb = [EarlyStopping(patience=5, restore_best_weights=True),
          ReduceLROnPlateau(patience=3, factor=0.5)]
    model.fit(Xr, to_categorical(y_train, n), validation_split=0.1,
              epochs=30, batch_size=128, callbacks=cb, verbose=0)
    preds = np.argmax(model.predict(Xrt), axis=1)
    _calc_metrics("LSTM", preds)
    classifier = model
    log("LSTM complete ✓")

# K-Fold 
def run_kfold():
    if X is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    log("5-Fold Stratified CV on Random Forest…")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    scores = cross_val_score(rf, X, Y,
                             cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                random_state=42),
                             scoring="f1_macro", n_jobs=-1)
    log(f"  Folds : {[f'{s*100:.2f}%' for s in scores]}")
    log(f"  Mean  : {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([f"Fold {i+1}" for i in range(5)], scores*100,
           color="#1890ff", edgecolor="white")
    ax.axhline(scores.mean()*100, color="#ff4d4f", linestyle="--",
               label=f"Mean={scores.mean()*100:.2f}%")
    ax.set_title("5-Fold CV — F1 Macro"); ax.set_ylim(0,105)
    ax.legend(); ax.set_facecolor("white")
    plt.tight_layout(); embed_plot(fig)

# SHAP 
def run_shap():
    if not HAS_SHAP:
        messagebox.showerror("Error", "pip install shap"); return
    if X_train is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    log("Computing SHAP values…")
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    sample = shap.sample(X_test, 500, random_state=42)
    shap_vals = shap.TreeExplainer(rf).shap_values(sample)
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    top_idx   = np.argsort(mean_shap)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([f"F{i}" for i in top_idx[::-1]], mean_shap[top_idx[::-1]],
            color="#722ed1", edgecolor="white")
    ax.set_title("SHAP — Top 20 Features"); ax.set_facecolor("white")
    plt.tight_layout(); embed_plot(fig)
    log("SHAP complete ✓")

# ROC 
def run_roc():
    if X_train is None:
        messagebox.showwarning("Warning", "Preprocess first."); return
    log("Computing ROC curves…")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)
    y_bin  = (y_test[:,None] == np.arange(len(labels))).astype(int)
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", len(labels))
    for i, lbl in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
        auc = roc_auc_score(y_bin[:,i], y_prob[:,i])
        ax.plot(fpr, tpr, color=palette[i], lw=1.5,
                label=f"{lbl} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=0.8)
    ax.set_title("ROC Curves (Random Forest)")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_facecolor("white")
    plt.tight_layout(); embed_plot(fig)

# Comparison 
def show_comparison():
    if len(metrics_dict) < 2:
        messagebox.showwarning("Warning", "Run at least 2 models first."); return
    names = list(metrics_dict.keys())
    accs  = [metrics_dict[n]["acc"]  for n in names]
    precs = [metrics_dict[n]["prec"] for n in names]
    recs  = [metrics_dict[n]["rec"]  for n in names]
    fs    = [metrics_dict[n]["f1"]   for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#f0f2f5")
    fig.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
    for ax, vals, title, col in zip(
        axes.flat,
        [accs, precs, recs, fs],
        ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"],
        ["#1890ff", "#52c41a", "#fa8c16", "#722ed1"]
    ):
        bars = ax.bar(names, vals, color=col, edgecolor="white")
        ax.set_title(title, fontsize=10); ax.set_ylim(0,105)
        ax.tick_params(axis="x", rotation=30); ax.set_facecolor("white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+1, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); embed_plot(fig)

    html = """<html><head><style>
    body{font-family:Segoe UI;background:#f0f2f5;padding:20px}
    h2{color:#001529;text-align:center}
    table{border-collapse:collapse;width:75%;margin:auto}
    th{background:#1890ff;color:white;padding:10px}
    td{padding:9px;border:1px solid #ddd;text-align:center}
    tr:nth-child(even){background:#fafafa}
    </style></head><body>
    <h2>DDoS Detection — Algorithm Benchmark </h2><table>
    <tr><th>Algorithm</th><th>Accuracy</th>
    <th>Precision</th><th>Recall</th><th>F1</th></tr>"""
    for n in names:
        m = metrics_dict[n]
        html += (f"<tr><td>{n}</td>"
                 f"<td>{m['acc']:.2f}%</td><td>{m['prec']:.2f}%</td>"
                 f"<td>{m['rec']:.2f}%</td><td>{m['f1']:.2f}%</td></tr>")
    html += "</table></body></html>"
    with open("comparison_report.html", "w") as fh:
        fh.write(html)
    webbrowser.open("comparison_report.html")

#  Predict form csv
def _preprocess_test_file(df_test):
    df_test = df_test.copy()
    df_test.columns = df_test.columns.str.strip()
    drop = [c for c in LEAKAGE_COLS if c in df_test.columns]
    df_test.drop(columns=drop+
                 (["Label"] if "Label" in df_test.columns else []),
                 inplace=True, errors="ignore")
    for col, le in label_encoders:
        if col in df_test.columns:
            df_test[col] = df_test[col].astype(str).map(
                lambda s, _le=le: _le.transform([s])[0]
                if s in _le.classes_ else 0)
    df_test = df_test.reindex(columns=train_columns, fill_value=0)
    df_test = df_test.apply(pd.to_numeric, errors="coerce").fillna(0).clip(-1e9, 1e9)
    arr = selector.transform(df_test.values)
    return scaler.transform(arr)

def predict_attack():
    if classifier is None:
        messagebox.showerror("Error", "Train a model first."); return
    f = filedialog.askopenfilename(
        title="Select test CSV",
        filetypes=[("CSV files","*.csv"),("All","*.*")])
    if not f: return
    try:
        df_test = pd.read_csv(f)
        X_p = _preprocess_test_file(df_test)
        if HAS_TF and isinstance(classifier, tf.keras.Model):
            inp = (X_p.reshape(X_p.shape[0],1,X_p.shape[1])
                   if len(classifier.input_shape)==3 else X_p)
            preds = np.argmax(classifier.predict(inp), axis=1)
            confs = None
        else:
            preds = classifier.predict(X_p)
            confs = (classifier.predict_proba(X_p)
                     if hasattr(classifier, "predict_proba") else None)

        pred_labels = [labels[p] for p in preds]
        unique, counts = np.unique(pred_labels, return_counts=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#f0f2f5")
        axes[0].pie(counts, labels=unique, autopct="%1.1f%%",
                    startangle=140,
                    colors=sns.color_palette("Set2", len(unique)),
                    wedgeprops={"edgecolor":"white"})
        axes[0].set_title(f"Threat Profile: {os.path.basename(f)}")
        axes[1].barh(unique, counts, color="#1890ff", edgecolor="white")
        axes[1].set_title("Prediction Counts")
        axes[1].set_facecolor("white")
        plt.tight_layout(); embed_plot(fig)

        log(f"\n── Prediction ({len(preds)} packets) ──")
        for lbl, cnt in zip(unique, counts):
            log(f"  {lbl:<25} {cnt:>5}  ({cnt/len(preds)*100:.1f}%)")

        # Write to session log
        if _log_writer:
            for i, lbl in enumerate(pred_labels):
                conf = float(np.max(confs[i])) if confs is not None else 0.9
                sev  = compute_severity(lbl, conf, 100, 0)
                _write_log_row(lbl, sev, conf*100, 100)
                _add_tree_row(lbl, sev, conf*100, 100)

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e)); log(f"ERROR: {e}")

#  Real-Time Detection
_pkt_count_this_sec = 0
_last_sec_ts        = time.time()

def _extract_packet_features(pkt):
    features = {}
    if IP in pkt:
        features["Protocol"]                    = pkt[IP].proto
        features["Total Fwd Packets"]           = 1
        features["Total Backward Packets"]      = 0
        features["Total Length of Fwd Packets"] = len(pkt)
        features["Flow Bytes/s"]                = len(pkt)
        features["Flow Packets/s"]              = 1
        if TCP in pkt:
            features["SYN Flag Count"] = 1 if pkt[TCP].flags & 0x02 else 0
            features["ACK Flag Count"] = 1 if pkt[TCP].flags & 0x10 else 0
            features["FIN Flag Count"] = 1 if pkt[TCP].flags & 0x01 else 0
            features["RST Flag Count"] = 1 if pkt[TCP].flags & 0x04 else 0
            features["PSH Flag Count"] = 1 if pkt[TCP].flags & 0x08 else 0
            features["Destination Port"] = pkt[TCP].dport
        elif SUDP in pkt:
            features["Destination Port"] = pkt[SUDP].dport
    return features

def _rt_inference_worker():
    """Background thread: batches 20 packets → predict → update all UI."""
    global recent_attack_count, last_attack_time
    global _pkt_count_this_sec, _last_sec_ts

    batch = []
    batch_start = time.time()

    while rt_running:
        try:
            feat_dict = rt_queue.get(timeout=0.5)
            batch.append(feat_dict)

            # Update timeline bucket counter
            with _timeline_lock:
                now = time.time()
                _pkt_count_this_sec += 1
                # every second, push a new bar to timeline
                if now - _last_sec_ts >= 1.0:
                    dominant = max(
                        _timeline_bucket, key=_timeline_bucket.get
                    ) if _timeline_bucket else "BENIGN"
                    timeline_counts.append(_pkt_count_this_sec)
                    timeline_labels.append(dominant)
                    _timeline_bucket.clear()
                    _pkt_count_this_sec = 0
                    _last_sec_ts = now

            if len(batch) < 20:
                continue

            # Build feature dataframe 
            df_b = (pd.DataFrame(batch)
                      .reindex(columns=train_columns, fill_value=0)
                      .apply(pd.to_numeric, errors="coerce")
                      .fillna(0))
            arr = selector.transform(df_b.values)
            arr = scaler.transform(arr)

            elapsed    = max(time.time() - batch_start, 0.001)
            pkt_rate   = len(batch) / elapsed
            batch_start = time.time()

            # Predict 
            if HAS_TF and isinstance(classifier, tf.keras.Model):
                inp  = (arr.reshape(arr.shape[0],1,arr.shape[1])
                        if len(classifier.input_shape)==3 else arr)
                prob = classifier.predict(inp, verbose=0)
                preds = np.argmax(prob, axis=1)
                confs = np.max(prob, axis=1)
            elif hasattr(classifier, "predict_proba"):
                prob  = classifier.predict_proba(arr)
                preds = np.argmax(prob, axis=1)
                confs = np.max(prob, axis=1)
            else:
                preds = classifier.predict(arr)
                confs = np.ones(len(preds)) * 0.85  

            # Process each prediction 
            for pred, conf in zip(preds, confs):
                lbl      = labels[pred]
                is_attack = lbl != "BENIGN"

                if is_attack:
                    recent_attack_count += 1
                    last_attack_time = time.time()
                else:
                    if time.time() - last_attack_time > 10:
                        recent_attack_count = max(0, recent_attack_count - 1)

                severity = compute_severity(
                    lbl, float(conf), pkt_rate, recent_attack_count
                )

                # Update timeline bucket
                with _timeline_lock:
                    _timeline_bucket[lbl] = \
                        _timeline_bucket.get(lbl, 0) + 1

                # Write to CSV log
                _write_log_row(lbl, severity, float(conf)*100, pkt_rate)

                # Queue UI updates to main thread
                _add_tree_row(lbl, severity, float(conf)*100, pkt_rate)

                root.after(0, lambda l=lbl, s=severity:
                    update_severity_ui(l, s))

                tag = "attack" if is_attack else "benign"
                sl  = severity_label(severity)
                root.after(0, lambda l=lbl, s=severity, sl=sl, t=tag:
                    rt_log_text.insert(
                        END,
                        f"[{time.strftime('%H:%M:%S')}]  {l:<22}"
                        f"  Severity: {s:>3}/100  [{sl}]\n",
                        t))
                root.after(0, rt_log_text.see, END)

                # Alert if high/critical
                if is_attack and severity >= 65:
                    _play_alert_sound(severity)
                    root.after(0, show_alert_toast, lbl, severity)

            batch.clear()

        except queue.Empty:
            continue
        except Exception as exc:
            root.after(0, rt_log_msg,
                       f"Inference error: {exc}", "warning")


def start_rt_detection():
    global rt_running
    if not HAS_SCAPY:
        messagebox.showerror("Error", "pip install scapy"); return
    if classifier is None:
        messagebox.showerror("Error", "Train a model first."); return

    _init_session_log()
    log_path_var.set(f"Log file: {_session_log_path}")

    rt_running = True
    iface = rt_iface_var.get()
    rt_status.config(text=f"Status: CAPTURING on {iface}", fg="#52c41a")
    rt_log_msg(f"Capture started — interface: {iface}", "info")
    rt_log_msg(f"Session log: {_session_log_path}",    "info")
    notebook.select(tab_rt)

    threading.Thread(target=_rt_inference_worker, daemon=True).start()

    def _capture():
        sniff(iface=iface,
              prn=lambda p: rt_queue.put(_extract_packet_features(p)),
              store=False,
              stop_filter=lambda _: not rt_running)

    threading.Thread(target=_capture, daemon=True).start()


def stop_rt_detection():
    global rt_running
    rt_running = False
    rt_status.config(text="Status: Stopped", fg="#ff4d4f")
    rt_log_msg("Capture stopped.", "warning")
    if _log_file:
        _log_file.flush()
    rt_log_msg(f"Log saved → {_session_log_path}", "info")

#  Sidebar
def _sec(txt):
    tk.Label(sidebar, text=txt, bg="#001529", fg="#ffd666",
             font=("Segoe UI", 8, "bold")).pack(
             fill="x", padx=18, pady=(12, 2))

def _btn(txt, cmd):
    b = tk.Button(sidebar, text=txt, command=cmd,
                  bg="#001529", fg="white",
                  font=("Segoe UI", 10), relief="flat",
                  pady=9, anchor="w", padx=18)
    b.pack(fill="x")
    b.bind("<Enter>", lambda e: b.config(bg="#1890ff"))
    b.bind("<Leave>", lambda e: b.config(bg="#001529"))

tk.Label(sidebar, text="DDoS Intelligence",
         bg="#001529", fg="white",
         font=("Segoe UI", 12, "bold")).pack(pady=(16, 4), padx=12)
tk.Frame(sidebar, bg="#0f2845", height=1).pack(fill="x", padx=12, pady=4)

_sec("DATA")
_btn("📂  Upload Dataset",      upload_dataset)
_btn("⚙️  Preprocess + SMOTE",  preprocess_dataset)

_sec("MACHINE LEARNING")
_btn("📊  Naive Bayes",         run_naive_bayes)
_btn("🌲  Random Forest",       run_random_forest)
_btn("⚡  XGBoost",             run_xgboost)
_btn("📐  LinearSVC",           run_svm)
_btn("🗳️  Voting Ensemble",     run_ensemble)

_sec("DEEP LEARNING")
_btn("🧠  Deep Neural Network",  run_dnn)
_btn("🔁  LSTM (temporal)",      run_lstm)

_sec("EVALUATION")
_btn("📉  K-Fold CV",            run_kfold)
_btn("🔍  SHAP Importance",      run_shap)
_btn("📈  ROC / AUC Curves",     run_roc)
_btn("📊  Comparison Graph",     show_comparison)

_sec("PREDICTION")
_btn("🔮  Predict from CSV",     predict_attack)

tk.Frame(sidebar, bg="#0f2845", height=1).pack(fill="x", padx=12, pady=8)

footer = (f"TF={'✓' if HAS_TF else '✗'}  "
          f"SMOTE={'✓' if HAS_SMOTE else '✗'}  "
          f"SHAP={'✓' if HAS_SHAP else '✗'}  "
          f"Scapy={'✓' if HAS_SCAPY else '✗'}")
tk.Label(sidebar, text=footer, bg="#001529", fg="#595959",
         font=("Segoe UI", 8)).pack(side="bottom", pady=10, padx=12)

#  Welocme Log
log("DDoS Detection Dashboard")
log("=" * 52)
log(f"TensorFlow   : {'✓ available' if HAS_TF    else '✗ pip install tensorflow'}")
log(f"SMOTE        : {'✓ available' if HAS_SMOTE else '✗ pip install imbalanced-learn'}")
log(f"SHAP         : {'✓ available' if HAS_SHAP  else '✗ pip install shap'}")
log(f"Scapy        : {'✓ available' if HAS_SCAPY else '✗ pip install scapy'}")
log(f"Logs folder  : {os.path.abspath(LOGS_DIR)}")
log("")
log("  • Threat severity gauge (0-100) in Real-Time tab")
log("  • Live scrolling attack timeline chart")
log("  • Session history table (Session History tab)")
log("  • CSV log auto-saved to logs/ folder every run")
log("  • Sound alert + toast popup for high/critical attacks")
log("")
log("Quick start:")
log("  1. Upload Dataset → Preprocess → Random Forest")
log("  2. Real-Time Detection tab → enter interface → Start")

root.mainloop()
