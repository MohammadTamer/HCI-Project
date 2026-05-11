"""gui.py
Tkinter GUI for the ECG Personal Photo Lock project.
No OOP. The logic is imported from logic.py.

Run:
    python gui.py
"""

from __future__ import annotations

import os
import queue
import threading
import traceback

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from PIL import Image, ImageTk, ImageDraw

import matplotlib

matplotlib.use("TkAgg")
from Logic import MODEL_STATE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Logic import (
    predict_subject_from_file,
    MODEL_STATE,
    FS,
    IDENTIFY_THRESHOLD,
    load_csv_signal,
    preprocess_signal,
    extract_heartbeats,
    extract_features,
    run_full_training, identify_subject,
)

# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DATA_ROOT = r"C:\Users\Mohamed\Desktop\HCI\PDB_csv"
PHOTOS_DIR = r"photos"
SUBJECT_IDS = ["s0001", "s0002", "s0003", "s0004", "s0005"]
HAS_HEADER = True
TARGET_COLUMN = None
TARGET_COLUMN_INDEX = None

# =============================================================================
# COLORS / FONTS
# =============================================================================

BG = "#0A0E1A"
PANEL = "#111827"
CARD = "#1A2235"
BORDER = "#1E2D45"
ACCENT = "#00D4FF"
ACCENT2 = "#6D28D9"
SUCCESS = "#22C55E"
WARN = "#F59E0B"
DANGER = "#EF4444"
TEXT = "#E2E8F0"
TEXT_DIM = "#64748B"

FNT_TITLE = ("Courier New", 20, "bold")
FNT_H2 = ("Courier New", 12, "bold")
FNT_MONO = ("Courier New", 9)
FNT_SMALL = ("Helvetica", 9)

# =============================================================================
# GLOBAL STATE (no OOP)
# =============================================================================

STATE = {
    "root": None,
    "clf": None,
    "scaler": None,
    "wavelet": "db1",
    "subject_ids": list(SUBJECT_IDS),
    "unlocked": set(),
    "meta": {},
    "all_results": {},
    "scanning": False,
    "training": False,
    "msg_queue": queue.Queue(),
    "photo_refs": [],
    "results_dir": "results",
    "photos_dir": PHOTOS_DIR,
    "data_root": DATA_ROOT,
    "has_header": HAS_HEADER,
    "target_column": TARGET_COLUMN,
    "target_column_index": TARGET_COLUMN_INDEX,
    "fs": FS,
}

# UI refs are stored here after build
UI = {}


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def make_rounded(img, radius=18):
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], radius=radius, fill=255)
    result = img.convert("RGBA")
    result.putalpha(mask)
    return result


def load_subject_photo(path, size=(200, 200)):
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    else:
        img = Image.new("RGB", size, (18, 30, 55))
        d = ImageDraw.Draw(img)
        d.ellipse([30, 30, size[0] - 30, size[1] - 30], fill=(30, 55, 95))
        d.text((size[0] // 2, size[1] // 2), "?", fill=(0, 200, 220), anchor="mm")
    return ImageTk.PhotoImage(make_rounded(img))


def locked_photo_img(size=(200, 200)):
    img = Image.new("RGB", size, (12, 18, 35))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=18, fill=(15, 22, 42))
    cx, cy = size[0] // 2, size[1] // 2 + 10
    d.arc([cx - 22, cy - 55, cx + 22, cy - 11], start=0, end=180, fill=(0, 180, 210), width=4)
    d.rounded_rectangle([cx - 28, cy - 15, cx + 28, cy + 30], radius=8,
                        fill=(22, 38, 65), outline=(0, 180, 210), width=2)
    d.ellipse([cx - 7, cy - 5, cx + 7, cy + 9], fill=(0, 180, 210))
    d.polygon([(cx, cy + 9), (cx - 5, cy + 22), (cx + 5, cy + 22)], fill=(0, 180, 210))
    return ImageTk.PhotoImage(make_rounded(img))


def style_ax(ax, fig):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor("#0D1520")
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)


# =============================================================================
# LOG / MODE / REFRESH HELPERS
# =============================================================================

def log(text):
    box = UI.get("log_box")
    if box is None:
        return
    box.configure(state="normal")
    box.insert("end", text + "\n")
    box.see("end")
    box.configure(state="disabled")


def set_mode(text, color):
    UI["mode_var"].set(text)
    UI["mode_lbl"].configure(fg=color)


def refresh_subject_list():
    frame = UI.get("subj_frame")
    if frame is None:
        return
    for w in frame.winfo_children():
        w.destroy()
    for sid in STATE["subject_ids"]:
        tk.Label(frame, text=f"• {sid}", bg=PANEL, fg=TEXT, font=FNT_SMALL).pack(anchor="w")


def find_photo(subject_id):
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = os.path.join(STATE["photos_dir"], subject_id + ext)
        if os.path.exists(p):
            return p
    return ""


def refresh_gallery():
    inner = UI.get("gallery_inner")
    if inner is None:
        return
    for w in inner.winfo_children():
        w.destroy()
    STATE["photo_refs"].clear()

    cols = 3
    for idx, sid in enumerate(STATE["subject_ids"]):
        r, c = divmod(idx, cols)
        card = tk.Frame(inner, bg=CARD, bd=1, relief="solid", padx=8, pady=8)
        card.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")
        inner.columnconfigure(c, weight=1)

        unlocked = sid in STATE["unlocked"]
        ref = load_subject_photo(find_photo(sid)) if unlocked else locked_photo_img()
        STATE["photo_refs"].append(ref)

        border_c = SUCCESS if unlocked else BORDER
        tk.Label(card, image=ref, bg=CARD, highlightbackground=border_c, highlightthickness=2).pack()
        status_txt = "🔓 UNLOCKED" if unlocked else "🔒 LOCKED"
        status_col = SUCCESS if unlocked else DANGER
        tk.Label(card, text=sid.upper(), bg=CARD, fg=TEXT, font=("Courier New", 8, "bold")).pack(pady=(6, 0))
        tk.Label(card, text=status_txt, bg=CARD, fg=status_col, font=FNT_SMALL).pack(pady=(0, 2))


def refresh_results_table(rows):
    tree = UI.get("results_tree")
    if tree is None:
        return
    for r in tree.get_children():
        tree.delete(r)
    for row in rows:
        tree.insert("", "end", values=(row["Wavelet"], row["Classifier"], row["Parameters"], row["Accuracy (%)"]))


# =============================================================================
# TRAINING
# =============================================================================

def start_training():
    if STATE["training"]:
        return
    refresh_subject_list()
    refresh_gallery()
    box = UI["log_box"]
    box.configure(state="normal")
    box.delete("1.0", "end")
    box.configure(state="disabled")
    refresh_results_table([])
    STATE["training"] = True
    UI["train_btn"].configure(
        state="disabled",
        text="[ TRAINING… ]"
    )
    set_mode("[ TRAINING… ]", WARN)
    t = threading.Thread(
        target=train_worker,
        daemon=True
    )
    t.start()
    STATE["root"].after(120, poll_messages)


def train_worker():
    try:
        df = run_full_training(
            STATE["data_root"],
            STATE["subject_ids"],
            fs=STATE["fs"],
            log_fn=lambda msg: STATE["msg_queue"].put(("log", msg)),
        )
        STATE["msg_queue"].put(("train_done", df))
    except Exception:
        STATE["msg_queue"].put(("train_error", traceback.format_exc()))


# =============================================================================
# SCANNING
# =============================================================================

def load_ecg_file():
    if MODEL_STATE["clf"] is None:
        messagebox.showwarning("No model", "Please train a model first.")
        return
    path = filedialog.askopenfilename(
        title="Select ECG CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if path:
        start_scan(csv_path=path)


def start_scan(csv_path=None):
    if STATE["scanning"]:
        return
    STATE["scanning"] = True
    UI["result_var"].set("Scanning…")
    UI["result_lbl"].configure(fg=WARN)
    UI["conf_var"].set("")
    UI["progress"]["value"] = 0
    UI["scan_status"].set("Starting scan…")
    t = threading.Thread(target=scan_worker, args=(csv_path,), daemon=True)
    t.start()
    STATE["root"].after(120, poll_messages)


def scan_worker(csv_path):
    q = STATE["msg_queue"]
    try:
        q.put(("scan_prog", 10, "Loading ECG signal…"))
        q.put(("scan_prog", 30, "Preprocessing…"))
        q.put(("scan_prog", 50, "Detecting R-peaks and segmenting…"))
        q.put(("scan_prog", 70, "Extracting features…"))
        q.put(("scan_prog", 88, "Classifying…"))
        result = predict_subject_from_file(
            csv_path,
            fs=STATE["fs"]
        )
        q.put(("ecg_plot", result["signal"][:3000]))
        q.put(("scan_prog", 100, "Done."))
        q.put((
            "scan_result",
            result["name"],
            result["confidence"],
            result["beats"].shape[0]
        ))
    except Exception:
        q.put(("scan_error", traceback.format_exc()))


# =============================================================================
# POLL MESSAGES
# =============================================================================

def poll_messages():
    still_busy = False
    try:
        while True:
            msg = STATE["msg_queue"].get_nowait()
            kind = msg[0]
            if kind == "log":
                log(msg[1])
                still_busy = True
            elif kind == "train_done":
                _, df = msg
                STATE["training"] = False
                UI["train_btn"].configure(state="normal", text="[ START TRAINING ]")
                best_row = df.loc[df["Accuracy (%)"].idxmax()]
                set_mode(f"[ MODEL OK  acc={best_row['Accuracy (%)']:.2f}% ]", SUCCESS)
                log(f"Model ready: "
                    f"{best_row['Classifier']} "
                    f"[{best_row['Parameters']}] "
                    f"acc={best_row['Accuracy (%)']:.2f}%"
                    )
                refresh_subject_list()
                refresh_gallery()
                refresh_results_table(df.to_dict("records"))
                UI["best_var"].set(
                    f"Best: "
                    f"{best_row['Classifier']} "
                    f"[{best_row['Parameters']}] "
                    f"| Acc={best_row['Accuracy (%)']:.2f}%"
                )

            elif kind == "train_error":
                STATE["training"] = False
                UI["train_btn"].configure(state="normal", text="[ START TRAINING ]")
                set_mode("[ TRAIN FAILED ]", DANGER)
                log("[ERROR]\n" + msg[1])

            elif kind == "scan_prog":
                _, pct, text = msg
                UI["progress"]["value"] = pct
                UI["scan_status"].set(text)
                still_busy = True

            elif kind == "ecg_plot":
                sig = msg[1]
                xs = np.arange(len(sig))
                UI["ecg_line"].set_data(xs, sig)
                UI["scan_ax"].set_xlim(0, len(sig))
                ymax = max(float(np.abs(sig).max()), 1.0) * 1.2
                UI["scan_ax"].set_ylim(-ymax, ymax)
                UI["scan_fig_canvas"].draw_idle()
                still_busy = True

            elif kind == "scan_result":
                _, name, conf, n_beats = msg
                STATE["scanning"] = False
                if name == "Unknown":
                    UI["result_var"].set("UNKNOWN")
                    UI["result_lbl"].configure(fg=DANGER)
                    UI["conf_var"].set(f"Confidence {conf:.1%} < {IDENTIFY_THRESHOLD:.0%} threshold")
                    UI["scan_status"].set("Subject not recognized.")
                else:
                    UI["result_var"].set(name.upper())
                    UI["result_lbl"].configure(fg=SUCCESS)
                    UI["conf_var"].set(f"Confidence {conf:.1%}")
                    UI["scan_status"].set(f"Identity confirmed – vault unlocked for {name}")
                    STATE["unlocked"].add(name)
                    refresh_gallery()
                    flash_unlock(name)

            elif kind == "scan_error":
                STATE["scanning"] = False
                UI["result_var"].set("ERROR")
                UI["result_lbl"].configure(fg=DANGER)
                UI["scan_status"].set("Scan failed – see log for details.")
                log("[SCAN ERROR]\n" + msg[1])
                UI["progress"]["value"] = 0

    except queue.Empty:
        pass

    if STATE["scanning"] or STATE["training"] or still_busy:
        STATE["root"].after(120, poll_messages)


# =============================================================================
# UTILITIES
# =============================================================================

def lock_all():
    STATE["unlocked"].clear()
    refresh_gallery()
    UI["result_var"].set("—")
    UI["result_lbl"].configure(fg=SUCCESS)
    UI["conf_var"].set("")
    UI["scan_status"].set("All photos locked.")


def flash_unlock(pid, n=6):
    def toggle(remaining):
        if remaining <= 0:
            UI["title_var"].set("▌ ECG PERSONAL PHOTO LOCK")
            return
        UI["title_var"].set(f"▌ {pid.upper()} UNLOCKED ✓")
        STATE["root"].after(280, lambda: UI["title_var"].set("▌ ECG PERSONAL PHOTO LOCK"))
        STATE["root"].after(560, lambda: toggle(remaining - 1))

    toggle(n)


def animate_cursor():
    cur = UI["title_var"].get()
    if cur.startswith("▌"):
        UI["title_var"].set(" " + cur[1:])
    else:
        UI["title_var"].set("▌" + cur[1:])
    STATE["root"].after(650, animate_cursor)


# =============================================================================
# BUILD UI
# =============================================================================

def build_ui(root):
    STATE["root"] = root
    root.title("ECG Personal Photo Lock – CSV Version")
    root.configure(bg=BG)
    root.geometry("1200x820")
    root.minsize(900, 700)
    os.makedirs(STATE["photos_dir"], exist_ok=True)
    hdr = tk.Frame(root, bg=BG, height=70)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    UI["title_var"] = tk.StringVar(value="▌ ECG PERSONAL PHOTO LOCK")
    tk.Label(hdr, textvariable=UI["title_var"], bg=BG, fg=ACCENT, font=FNT_TITLE).pack(side="left", padx=28, pady=16)
    UI["mode_var"] = tk.StringVar(value="[ NO MODEL ]")
    UI["mode_lbl"] = tk.Label(hdr, textvariable=UI["mode_var"], bg=BG, fg=DANGER, font=FNT_MONO)
    UI["mode_lbl"].pack(side="right", padx=24)
    tk.Frame(root, bg=BORDER, height=1).pack(fill="x")
    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True, padx=8, pady=8)
    left = tk.Frame(body, bg=PANEL, width=400)
    left.pack(side="left", fill="y", padx=(0, 6))
    left.pack_propagate(False)
    build_left(left)
    right = tk.Frame(body, bg=PANEL)
    right.pack(side="left", fill="both", expand=True)
    build_right(right)
    animate_cursor()


def build_left(parent):
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Dark.TNotebook", background=PANEL, borderwidth=0)
    style.configure("Dark.TNotebook.Tab", background=BORDER, foreground=TEXT_DIM, padding=[12, 5], font=FNT_MONO)
    style.map("Dark.TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", ACCENT)])
    style.configure("ECG.Horizontal.TProgressbar", troughcolor=BORDER, background=ACCENT,
                    lightcolor=ACCENT, darkcolor=ACCENT2, bordercolor=PANEL)
    nb = ttk.Notebook(parent, style="Dark.TNotebook")
    nb.pack(fill="both", expand=True, padx=6, pady=6)
    tab_scan = tk.Frame(nb, bg=PANEL)
    nb.add(tab_scan, text=" SCANNER ")
    build_scanner_tab(tab_scan)
    tab_train = tk.Frame(nb, bg=PANEL)
    nb.add(tab_train, text=" TRAINING ")
    build_training_tab(tab_train)
    tab_res = tk.Frame(nb, bg=PANEL)
    nb.add(tab_res, text=" RESULTS ")
    build_results_tab(tab_res)


def build_scanner_tab(parent):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    style_ax(ax, fig)
    ax.set_title("ECG Signal", color=TEXT_DIM, fontsize=8)
    ecg_line, = ax.plot([], [], color=ACCENT, lw=1.0)
    ax.set_xlim(0, 3000)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Samples", fontsize=7)
    ecg_canvas = FigureCanvasTkAgg(fig, master=parent)
    ecg_canvas.get_tk_widget().pack(fill="x", padx=8, pady=(10, 4))
    UI["scan_fig"] = fig
    UI["scan_ax"] = ax
    UI["ecg_line"] = ecg_line
    UI["scan_fig_canvas"] = ecg_canvas
    progress = ttk.Progressbar(parent, orient="horizontal", length=370,
                               mode="determinate", style="ECG.Horizontal.TProgressbar")
    progress.pack(padx=12, pady=4)
    UI["progress"] = progress
    scan_status = tk.StringVar(value="Ready. Load an ECG CSV file to identify a subject.")
    UI["scan_status"] = scan_status
    tk.Label(parent, textvariable=scan_status, bg=PANEL, fg=TEXT_DIM,
             font=FNT_SMALL, wraplength=370, justify="center").pack(padx=8, pady=2)
    rf = tk.Frame(parent, bg=CARD, bd=1, relief="solid")
    rf.pack(fill="x", padx=12, pady=8)
    tk.Label(rf, text="IDENTIFICATION RESULT", bg=CARD, fg=TEXT_DIM, font=FNT_MONO).pack(pady=(8, 0))
    result_var = tk.StringVar(value="—")
    result_lbl = tk.Label(rf, textvariable=result_var, bg=CARD, fg=SUCCESS, font=("Courier New", 18, "bold"))
    result_lbl.pack()
    conf_var = tk.StringVar(value="")
    tk.Label(rf, textvariable=conf_var, bg=CARD, fg=TEXT_DIM, font=FNT_SMALL).pack(pady=(0, 8))
    UI["result_var"] = result_var
    UI["result_lbl"] = result_lbl
    UI["conf_var"] = conf_var

    def btn(p, text, cmd, fg=ACCENT):
        b = tk.Button(p, text=text, command=cmd, bg=BORDER, fg=fg, font=FNT_MONO,
                      activebackground=ACCENT2, activeforeground="white",
                      relief="flat", cursor="hand2", padx=6, pady=7, bd=0)
        b.pack(fill="x", padx=12, pady=2)
        return b

    btn(parent, "[ LOAD ECG CSV ]", load_ecg_file)
    btn(parent, "[ LOCK ALL PHOTOS ]", lock_all, fg=DANGER)
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=8)
    tk.Label(parent, text="REGISTERED SUBJECTS:", bg=PANEL, fg=TEXT_DIM, font=FNT_MONO).pack(anchor="w", padx=14)
    subj_frame = tk.Frame(parent, bg=PANEL)
    subj_frame.pack(fill="x", padx=18)
    UI["subj_frame"] = subj_frame
    refresh_subject_list()


def build_training_tab(parent):
    tk.Label(
        parent,
        text="TRAINING",
        bg=PANEL,
        fg=ACCENT,
        font=FNT_H2
    ).pack(anchor="w", padx=12, pady=(12, 2))
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=6)
    train_btn = tk.Button(
        parent,
        text="[ START TRAINING ]",
        command=start_training,
        bg=ACCENT2,
        fg="white",
        font=FNT_H2,
        activebackground=ACCENT,
        activeforeground=BG,
        relief="flat",
        cursor="hand2",
        pady=8,
        bd=0
    )
    train_btn.pack(fill="x", padx=12, pady=4)
    UI["train_btn"] = train_btn
    tk.Label(parent, text="TRAINING LOG", bg=PANEL, fg=TEXT_DIM, font=FNT_MONO).pack(anchor="w", padx=12, pady=(8, 0))
    log_box = scrolledtext.ScrolledText(
        parent,
        bg="#0D1520",
        fg=SUCCESS,
        font=("Courier New", 8),
        insertbackground=ACCENT,
        relief="flat",
        bd=4,
        height=18,
        state="disabled"
    )
    log_box.pack(fill="both",expand=True,padx=10,pady=(2, 8))
    UI["log_box"] = log_box


def build_results_tab(parent):
    tk.Label(parent, text="ACCURACY TABLE", bg=PANEL, fg=ACCENT, font=FNT_H2).pack(anchor="w", padx=12, pady=(12, 4))
    style = ttk.Style()
    style.configure("Results.Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT, rowheight=22,
                    font=("Courier New", 8))
    style.configure("Results.Treeview.Heading", background=BORDER, foreground=ACCENT,
                    font=("Courier New", 8, "bold"))
    style.map("Results.Treeview", background=[("selected", ACCENT2)])
    cols = ("Wavelet", "Classifier", "Parameters", "Accuracy (%)")
    results_tree = ttk.Treeview(parent, columns=cols, show="headings", style="Results.Treeview", height=14)
    for c in cols:
        results_tree.heading(c, text=c)
        results_tree.column(c, width=80 if c != "Parameters" else 100, anchor="center")
    results_tree.pack(fill="both", expand=True, padx=10, pady=2)
    UI["results_tree"] = results_tree
    best_var = tk.StringVar(value="Train a model to see results.")
    UI["best_var"] = best_var
    tk.Label(parent, textvariable=best_var, bg=PANEL, fg=WARN, font=FNT_MONO, wraplength=370).pack(padx=10, pady=6)


def build_right(parent):
    tk.Label(parent, text="PHOTO VAULT", bg=PANEL, fg=ACCENT, font=FNT_H2).pack(anchor="w", padx=16, pady=(14, 4))
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=10)
    canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=0)
    vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=PANEL)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    UI["gallery_canvas"] = canvas
    UI["gallery_inner"] = inner
    refresh_gallery()


# =============================================================================
# MAIN / ENTRY
# =============================================================================

def main():
    root = tk.Tk()
    build_ui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
