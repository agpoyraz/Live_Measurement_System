import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
import cv2
import neoapi
from PIL import Image, ImageTk

# ---------------------------
# Optional: scipy varsa daha iyi hole-fill
# ---------------------------
try:
    from scipy import ndimage
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =========================================================
#  MATLAB-uyumlu yardımcılar
# =========================================================
def largest_component(binary_u8: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_u8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(binary_u8)
    out[labels == idx] = 255
    return out

def imfill_holes(binary_u8: np.ndarray) -> np.ndarray:
    if HAVE_SCIPY:
        filled = ndimage.binary_fill_holes(binary_u8.astype(bool))
        return (filled.astype(np.uint8) * 255)

    # --- scipy yoksa: floodfill ile arka planı doldur, tersle
    h, w = binary_u8.shape
    inv = cv2.bitwise_not(binary_u8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    ff = inv.copy()
    cv2.floodFill(ff, mask, (0, 0), 0)      # dış arka planı temizle
    holes = cv2.bitwise_not(ff)             # delikler 255 olur
    filled = cv2.bitwise_or(binary_u8, holes)
    return filled

def imclearborder_u8(binary_u8: np.ndarray) -> np.ndarray:
    h, w = binary_u8.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cleared = binary_u8.copy()

    for x in range(w):
        if cleared[0, x] == 255:
            cv2.floodFill(cleared, mask, (x, 0), 0)
        if cleared[h - 1, x] == 255:
            cv2.floodFill(cleared, mask, (x, h - 1), 0)

    for y in range(h):
        if cleared[y, 0] == 255:
            cv2.floodFill(cleared, mask, (0, y), 0)
        if cleared[y, w - 1] == 255:
            cv2.floodFill(cleared, mask, (w - 1, y), 0)

    return cleared

def imwarp_matlab_like(img_u8: np.ndarray, angle_deg: float, cx: float, cy: float) -> np.ndarray:
    h, w = img_u8.shape[:2]
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)

    R = np.array([[ c,  s, 0],
                  [-s,  c, 0],
                  [ 0,  0, 1]], dtype=np.float64)

    T1 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [-cx, -cy, 1]], dtype=np.float64)

    T2 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [ cx,  cy, 1]], dtype=np.float64)

    H_row = T1 @ R @ T2
    H_col = H_row.T
    Hinv = np.linalg.inv(H_col)
    M = Hinv[:2, :]

    return cv2.warpAffine(
        img_u8, M, (w, h),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

def measure_6_distances_from_frame(frame, mm_per_px=0.01, thresh_val=200, canny1=50, canny2=150,
                                   pos_offset_minus1=False):
    # Gray
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    BW_th = (gray > thresh_val).astype(np.uint8) * 255

    BW_largest = largest_component(BW_th)
    BW_largest1 = imfill_holes(BW_largest)

    pos = np.array([
        [1600, 900,  80, 350],
        [ 810, 1330, 80, 350],
        [ 810, 450,  80, 350]
    ], dtype=float)
    ang = np.array([0, 240, 120], dtype=float)

    if pos_offset_minus1:
        pos[:, 0:2] -= 1

    distances = {}

    # dış 3 -> No: 1,3,5
    for i in range(3):
        x, y, w, h = pos[i]
        cx = x + w/2.0
        cy = y + h/2.0

        Irot = imwarp_matlab_like(BW_largest1, float(ang[i]), cx, cy)
        Icrop = cv2.getRectSubPix(Irot, (int(round(w)), int(round(h))), (float(cx), float(cy)))

        edges = cv2.Canny(Icrop, canny1, canny2)
        rows, cols = np.where(edges > 0)
        if len(cols) < 10:
            distances[2*i + 1] = None
            continue

        x_pts = cols.astype(float)
        y_pts = rows.astype(float)
        y_thr = int(np.round(np.mean(y_pts)))

        upper = y_pts <= y_thr
        lower = y_pts >  y_thr

        x1, y1 = x_pts[upper], y_pts[upper]
        x2, y2 = x_pts[lower], y_pts[lower]
        if len(x1) < 2 or len(x2) < 2:
            distances[2*i + 1] = None
            continue

        m1, b1 = np.polyfit(x1, y1, 1)
        m2, b2 = np.polyfit(x2, y2, 1)

        A = float(m1)
        B = -1.0
        dist_px = abs(float(b2) - float(b1)) / np.sqrt(A*A + B*B)
        distances[2*i + 1] = dist_px * mm_per_px

    # iç 3 -> No: 2,4,6
    BW_largest_complement = 255 - BW_largest
    BW_th2 = imclearborder_u8(BW_largest_complement)
    BW_largest2 = largest_component(BW_th2)

    pos2 = np.array([
        [1265, 1247, 80, 350],
        [ 680,  900, 80, 350],
        [1270,  560, 80, 350]
    ], dtype=float)
    ang2 = np.array([300, 180, 60], dtype=float)

    if pos_offset_minus1:
        pos2[:, 0:2] -= 1

    for i in range(3):
        x, y, w, h = pos2[i]
        cx = x + w/2.0
        cy = y + h/2.0

        Irot = imwarp_matlab_like(BW_largest2, float(ang2[i]), cx, cy)
        Icrop = cv2.getRectSubPix(Irot, (int(round(w)), int(round(h))), (float(cx), float(cy)))

        edges = cv2.Canny(Icrop, canny1, canny2)
        rows, cols = np.where(edges > 0)
        if len(cols) < 10:
            distances[2*i + 2] = None
            continue

        x_pts = cols.astype(float)
        y_pts = rows.astype(float)
        y_thr = int(np.round(np.mean(y_pts)))

        upper = y_pts <= y_thr
        lower = y_pts >  y_thr

        x1, y1 = x_pts[upper], y_pts[upper]
        x2, y2 = x_pts[lower], y_pts[lower]
        if len(x1) < 2 or len(x2) < 2:
            distances[2*i + 2] = None
            continue

        m1, b1 = np.polyfit(x1, y1, 1)
        m2, b2 = np.polyfit(x2, y2, 1)

        A = float(m1)
        B = -1.0
        dist_px = abs(float(b2) - float(b1)) / np.sqrt(A*A + B*B)
        distances[2*i + 2] = dist_px * mm_per_px

    return distances


# =========================================================
#  Minimal Tkinter App
# =========================================================
class LiveMeasureApp(tk.Tk):
    def __init__(self, cam_serial="2825000092AD", display_width=1280):
        super().__init__()
        self.title("Live Measurement (NeoAPI)")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cam_serial = cam_serial
        self.display_width = display_width

        # ölçüm parametreleri
        self.mm_per_px = 0.01
        self.thresh_val = 200
        self.canny1 = 50
        self.canny2 = 150
        self.pos_offset_minus1 = False

        # tolerans default
        self.lsl = 1.230
        self.usl = 1.770

        # UI
        self._build_ui()

        # camera/thread
        self.camera = None
        self.running = True
        self.thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.thread.start()

    def _build_ui(self):
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        container.columnconfigure(0, weight=4)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # video
        self.video_label = ttk.Label(container, text="Kamera bekleniyor...")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # right panel
        right = ttk.Frame(container)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Ölçüm Sonuçları", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.result_text = tk.Text(right, height=12, width=35, font=("Consolas", 12))
        self.result_text.grid(row=1, column=0, sticky="ew")
        self.result_text.insert("end", "No 1..6 sonuçları burada görünecek.\n")
        self.result_text.configure(state="disabled")

        # basit ayarlar
        frm = ttk.LabelFrame(right, text="Parametreler")
        frm.grid(row=2, column=0, sticky="ew", pady=10)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="mm/px").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.mm_entry = ttk.Entry(frm)
        self.mm_entry.insert(0, str(self.mm_per_px))
        self.mm_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(frm, text="Threshold").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.th_entry = ttk.Entry(frm)
        self.th_entry.insert(0, str(self.thresh_val))
        self.th_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(frm, text="Canny1").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.c1_entry = ttk.Entry(frm)
        self.c1_entry.insert(0, str(self.canny1))
        self.c1_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(frm, text="Canny2").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.c2_entry = ttk.Entry(frm)
        self.c2_entry.insert(0, str(self.canny2))
        self.c2_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        # ---- tolerans satırları ----
        ttk.Label(frm, text="Alt Limit (mm)").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.lsl_entry = ttk.Entry(frm)
        self.lsl_entry.insert(0, str(self.lsl))
        self.lsl_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(frm, text="Üst Limit (mm)").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.usl_entry = ttk.Entry(frm)
        self.usl_entry.insert(0, str(self.usl))
        self.usl_entry.grid(row=5, column=1, sticky="ew", padx=5, pady=5)

        self.offset_var = tk.IntVar(value=1 if self.pos_offset_minus1 else 0)
        self.offset_check = ttk.Checkbutton(frm, text="pos/pos2 -1 px düzeltme", variable=self.offset_var)
        self.offset_check.grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        ttk.Button(frm, text="Uygula", command=self.apply_params).grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=8)

        # Büyük OK / NOT OK göstergesi
        self.big_indicator = tk.Label(
            right,
            text="---",
            font=("Segoe UI", 28, "bold"),
            fg="white",
            bg="gray",
            padx=10,
            pady=15
        )
        self.big_indicator.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        self.status_label = ttk.Label(right, text="Durum: Başlatılıyor...", foreground="blue")
        self.status_label.grid(row=4, column=0, sticky="ew", pady=(10, 0))

    def apply_params(self):
        try:
            self.mm_per_px = float(self.mm_entry.get())
            self.thresh_val = int(float(self.th_entry.get()))
            self.canny1 = int(float(self.c1_entry.get()))
            self.canny2 = int(float(self.c2_entry.get()))
            self.pos_offset_minus1 = bool(self.offset_var.get())

            self.lsl = float(self.lsl_entry.get())
            self.usl = float(self.usl_entry.get())
            if self.lsl >= self.usl:
                raise ValueError("Alt limit üst limitten küçük olmalı.")

            self.status_label.configure(text="Durum: Parametreler güncellendi.", foreground="green")
        except Exception as e:
            self.status_label.configure(text=f"Durum: Parametre hatası -> {e}", foreground="red")

    def connect_camera(self):
        cam = neoapi.Cam()
        cam.Connect(self.cam_serial)

        # mümkün olduğunca stabil set
        try: cam.f.ExposureAuto.Set(neoapi.ExposureAuto_Off)
        except: pass
        try: cam.f.GainAuto.Set(neoapi.GainAuto_Off)
        except: pass
        try: cam.f.BalanceWhiteAuto.Set(neoapi.BalanceWhiteAuto_Off)
        except: pass

        cam.f.ExposureTime.Set(5000)

        try: cam.f.PixelFormat.Set(neoapi.PixelFormat_BGR8)
        except:
            cam.f.PixelFormat.Set(neoapi.PixelFormat_Mono8)

        try: cam.f.TriggerMode.value = neoapi.TriggerMode_Off
        except: pass

        return cam

    def worker_loop(self):
        try:
            self.camera = self.connect_camera()
            self._ui_status("Durum: Kamera bağlı. Canlı akış başladı.", "green")
        except Exception as e:
            self._ui_status(f"Durum: Kamera bağlanamadı -> {e}", "red")
            return

        while self.running:
            try:
                img = self.camera.GetImage(1000)
                if img.IsEmpty():
                    continue

                frame = img.GetNPArray()
                # (H,W,1) -> (H,W)
                if frame.ndim == 3 and frame.shape[2] == 1:
                    frame = frame[:, :, 0]

                # ölçüm (orijinal frame üzerinde)
                distances = measure_6_distances_from_frame(
                    frame,
                    mm_per_px=self.mm_per_px,
                    thresh_val=self.thresh_val,
                    canny1=self.canny1,
                    canny2=self.canny2,
                    pos_offset_minus1=self.pos_offset_minus1
                )

                # overlay/gösterim için BGR’e çevir
                if frame.ndim == 2:
                    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    vis = frame.copy()

                y0 = 40
                cv2.putText(vis, "LIVE MEASUREMENT", (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                y0 += 40

                for k in range(1, 7):
                    v = distances.get(k, None)
                    is_ok = (v is not None) and (self.lsl <= v <= self.usl)
                    color = (0, 255, 0) if is_ok else (0, 0, 255)

                    txt = f"No {k}: {'---' if v is None else f'{v:.4f} mm'}"
                    cv2.putText(vis, txt, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    y0 += 35

                # GUI için resize (gösterim)
                h, w = vis.shape[:2]
                scale = self.display_width / w
                new_h = int(h * scale)
                vis_show = cv2.resize(vis, (self.display_width, new_h), interpolation=cv2.INTER_AREA)

                # Tkinter update (thread-safe: after ile)
                self.after(1, self.update_ui, vis_show, distances)

            except Exception as e:
                self._ui_status(f"Durum: Hata -> {e}", "red")
                time.sleep(0.2)

    def update_ui(self, bgr_image, distances):
        # görüntü
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.img_tk = img_tk
        self.video_label.configure(image=img_tk, text="")

        # sonuç metni + OK/NOT OK etiketi
        lines = []
        global_ok = True
        for k in range(1, 7):
            v = distances.get(k, None)
            is_ok = (v is not None) and (self.lsl <= v <= self.usl)
            if not is_ok:
                global_ok = False
            st = "OK" if is_ok else "NOT OK"
            lines.append(f"No {k}: {'---' if v is None else f'{v:.4f} mm'}   [{st}]")

        txt = "\n".join(lines)

        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", txt)
        self.result_text.configure(state="disabled")

        # büyük gösterge
        if global_ok:
            self.big_indicator.configure(text="OK", bg="green", fg="white")
        else:
            self.big_indicator.configure(text="NOT OK", bg="red", fg="white")

    def _ui_status(self, text, color):
        def _do():
            self.status_label.configure(text=text, foreground=color)
        self.after(1, _do)

    def on_close(self):
        self.running = False
        try:
            if self.camera:
                self.camera.Disconnect()
        except:
            pass
        self.destroy()


if __name__ == "__main__":
    app = LiveMeasureApp(cam_serial="2825000092AD", display_width=1280)
    app.mainloop()