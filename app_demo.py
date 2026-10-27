import base64
import csv
import html
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from functools import wraps
from urllib.parse import quote

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, request,
                   send_from_directory, session, url_for)
from paddleocr import PaddleOCR
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    zbar_decode = None

# ============================================================
# 1. CẤU HÌNH DEMO (không chứa thông tin camera thật)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "runtime_data")
DB_NAME = os.path.join(DATA_DIR, "warehouse_demo.db")
CSV_PATH = os.path.join(DATA_DIR, "ocr_demo_results.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "product_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Mặc định tắt camera IP để người clone repo không vô tình kết nối camera riêng.
# Muốn bật IP camera, hãy đặt USE_IP_CAMERA=true và cấu hình các biến CAMERA_*.
USE_IP_CAMERA = os.environ.get("USE_IP_CAMERA", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
CAMERA_IP = os.environ.get("CAMERA_IP", "").strip()
CAMERA_USER = os.environ.get("CAMERA_USER", "").strip()
CAMERA_PASS = os.environ.get("CAMERA_PASS", "")
RTSP_URL = None

if USE_IP_CAMERA:
    missing_camera_config = [
        name for name, value in (
            ("CAMERA_IP", CAMERA_IP),
            ("CAMERA_USER", CAMERA_USER),
            ("CAMERA_PASS", CAMERA_PASS),
        ) if not value
    ]
    if missing_camera_config:
        raise RuntimeError(
            "USE_IP_CAMERA=true nhưng chưa cấu hình: "
            + ", ".join(missing_camera_config)
        )
    RTSP_URL = (
        f"rtsp://{quote(CAMERA_USER, safe='')}:{quote(CAMERA_PASS, safe='')}"
        f"@{CAMERA_IP}:554/Streaming/Channels/101"
    )

# Mặc định chỉ cho truy cập từ máy đang chạy ứng dụng.
WEB_HOST = os.environ.get("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.environ.get("WEB_PORT", "5000"))
WEB_USER = os.environ.get("WEB_USER", "demo_admin")

# Không hard-code mật khẩu. Nếu chưa đặt WEB_PASS, tạo mật khẩu tạm thời
# ngẫu nhiên và in ra terminal khi khởi động để chạy demo local.
WEB_PASS = os.environ.get("WEB_PASS", "")
GENERATED_WEB_PASS = not bool(WEB_PASS)
if GENERATED_WEB_PASS:
    WEB_PASS = secrets.token_urlsafe(12)
WEB_PASS_HASH = generate_password_hash(WEB_PASS)
WEB_SECRET = os.environ.get("WEB_SECRET") or secrets.token_hex(32)

ROI_X1, ROI_X2, ROI_Y1, ROI_Y2 = 0.20, 0.80, 0.20, 0.80
OCR_INTERVAL_SEC = 0.25
COOLDOWN_SEC = 2.0            # (camera IP) thời gian nghỉ sau khi chốt
UNSEEN_TIMEOUT_SEC = 3.0      # (camera IP) mã lạ liên tục 3s thì chụp & mở form nhập liệu
IGNORE_SEC = 30.0             # sau khi bấm "Bỏ qua", không hỏi lại mã đó trong 30s
MIN_OCR_CONFIDENCE = 0.50
FORM_MIN_CONFIDENCE = 0.30
DUP_WARN_RATIO = 0.85

STREAM_WIDTH, STREAM_FPS, JPEG_QUALITY = 960, 12, 70


def _env_f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# ---- Cấu hình riêng cho luồng CAMERA ĐIỆN THOẠI (chỉnh bằng biến môi trường) ----
MOBILE_SESSION_TARGET_SEC = _env_f("MOBILE_SESSION_TARGET_SEC", 1.0)   # mục tiêu, KHÔNG phải giới hạn cứng
MOBILE_SESSION_MAX_SEC = _env_f("MOBILE_SESSION_MAX_SEC", 2.5)         # trần an toàn của một phiên
MOBILE_COOLDOWN_SEC = _env_f("MOBILE_COOLDOWN_SEC", 2.0)
MOBILE_IDLE_CHECK_INTERVAL = int(_env_f("MOBILE_IDLE_CHECK_INTERVAL", 500))  # ms, client kiểm tra "có vật" ở IDLE
MOBILE_SCAN_SEND_MS = int(_env_f("MOBILE_SCAN_SEND_MS", 250))          # ms giữa 2 khung gửi lên khi SCANNING
MOBILE_FRAME_WIDTH = int(_env_f("MOBILE_FRAME_WIDTH", 800))
MOBILE_JPEG_QUALITY = _env_f("MOBILE_JPEG_QUALITY", 0.70)
MOBILE_MIN_OCR_CONFIDENCE = _env_f("MOBILE_MIN_OCR_CONFIDENCE", 0.50)
MOBILE_PRESENCE_THR = _env_f("MOBILE_PRESENCE_THR", 0.05)    # tỉ lệ điểm ảnh có cạnh trong ROI (cần hiệu chỉnh thực tế)
MOBILE_EDGE_DIFF = int(_env_f("MOBILE_EDGE_DIFF", 40))       # ngưỡng gradient của 1 "điểm cạnh"
MOBILE_EMPTY_BACKOFF_SEC = _env_f("MOBILE_EMPTY_BACKOFF_SEC", 1.5)  # nghỉ sau phiên không ra mã (chống nhiễu nền)
MOBILE_STRONG_CONF = _env_f("MOBILE_STRONG_CONF", 0.80)      # chốt sớm nếu khớp ~tuyệt đối + conf cao
MOBILE_REPEAT_LOG_SEC = _env_f("MOBILE_REPEAT_LOG_SEC", 6.0) # cùng SKU trong khoảng này: hiện kết quả nhưng không ghi CSV lại
MOBILE_DEBUG = os.environ.get("MOBILE_DEBUG", "0") == "1"

# ============================================================
# 2. DATABASE
# ============================================================
def db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = db()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS warehouse (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT UNIQUE NOT NULL,
            product_name TEXT NOT NULL,
            location TEXT DEFAULT 'CHUA_CO',
            quantity INTEGER DEFAULT 0,
            updated_at TEXT
        )''')
    for col, ddl in [("brand", "TEXT DEFAULT ''"), ("series", "TEXT DEFAULT ''"),
                     ("image_path", "TEXT DEFAULT ''"), ("status", "TEXT DEFAULT 'OK'")]:
        try:
            cur.execute(f"ALTER TABLE warehouse ADD COLUMN {col} {ddl}")
        except sqlite3.OperationalError:
            pass

    seed = [
        ("ZLAN5143D", "ZLAN", "5143D", "Module chuyển đổi ZLAN5143D", "Kệ A1-01", 150),
        ("OMRONS8VS01505", "OMRON", "S8VS", "Nguồn Omron S8VS-01505", "Kệ B1-01", 90),
        ("AH165ZT", "FUJI", "AH165", "Nút nhấn công nghiệp AH165ZT", "Kệ C1-01", 25),
        ("CP30", "MITSUBISHI", "CP30", "MITSUBISHI CIRCUIT PROTECTOR", "Kệ D1-01", 100),
        ("125Y3N", "125Y3N", "125Y3N", "100PCS 125Y3N", "Kệ D1-03", 100),
        ("H3Y4", "OMRON", "H3Y4", "OMRON TIMER H3Y4", "Kệ D1-02", 100),
        ("SJOG", "SJOG", "SJ12AG", "SJ12AG / SJOG", "Kệ F1-01", 100),
        ("2744RP", "OMRON", "NRTC", "OMRON NRTC ", "Kệ E1-01", 100),
    ]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for sku, brand, series, name, loc, qty in seed:
        cur.execute('''INSERT OR IGNORE INTO warehouse
                    (sku, brand, series, product_name, location, quantity, updated_at)
                    VALUES (?,?,?,?,?,?,?)''', (sku, brand, series, name, loc, qty, now))
    conn.commit()
    conn.close()


init_database()

print("Đang khởi tạo PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
print("PaddleOCR sẵn sàng.")

# ============================================================
# 3. TRẠNG THÁI DÙNG CHUNG
# ============================================================
frame_lock = threading.Lock()     # frame camera IP
state_lock = threading.Lock()     # S (trạng thái hiển thị chung)
jpeg_lock = threading.Lock()
ocr_lock = threading.Lock()       # PaddleOCR không đảm bảo thread-safe: chỉ 1 OCR tại một thời điểm
rows_lock = threading.Lock()
stop_event = threading.Event()
ip_scan_enabled = threading.Event()
mobile_scan_enabled = threading.Event()
scan_enabled = ip_scan_enabled  # tương thích API cũ; chỉ điều khiển IP

latest_frame = None              # CHỈ dành cho camera IP
latest_jpeg = None
ignore_until = {}
_rows_cache = None

S = {
    "camera": ("Đang kết nối camera..." if USE_IP_CAMERA else "Camera IP đang tắt (demo)"),
    "source": "ip",      # "ip" hoặc "mobile"
    "code": None, "info": None, "ocr": "Chờ đọc...",
    "last_sku": None, "last_time": 0.0,
    "pending": None, "pending_by_source": {"ip": None, "mobile": None},
}
CAMERA_RESULTS = {
    "ip": {"code": None, "info": None, "ocr": "Chờ đọc...", "last_sku": None, "last_time": 0.0},
    "mobile": {"code": None, "info": None, "ocr": "Chờ đọc...", "last_sku": None, "last_time": 0.0},
}


def set_camera_ocr(source, value):
    with state_lock:
        CAMERA_RESULTS[source]["ocr"] = value
        if S["source"] == source:
            S["ocr"] = value


def set_camera_result(source, info, now):
    with state_lock:
        r = CAMERA_RESULTS[source]
        r.update(code=info["sku"], info=info, last_sku=info["sku"], last_time=now)
        if S["source"] == source:
            S.update(code=r["code"], info=r["info"], last_sku=r["last_sku"], last_time=r["last_time"])


# ---- State machine camera điện thoại ----
# IDLE / SCANNING / CONFIRMED / COOLDOWN / UNKNOWN_PENDING / STOPPED
mobile_cv = threading.Condition()       # bảo vệ M và MF, đồng thời dùng để đánh thức worker
M = {"state": "STOPPED", "sid": None, "t0": 0.0, "cooldown_until": 0.0, "rearm_at": 0.0,
     "object_present": False, "empty_frames": 0, "presence_frames": 0}
IP = {"state": "IDLE", "sid": None, "object_present": False, "present_frames": 0,
      "empty_frames": 0, "session_recorded": set(), "bg": None, "last_object_gray": None,
      "changed_object_frames": 0}
ip_state_lock = threading.RLock()
csv_lock = threading.Lock()
PRESENCE_CONFIRM_FRAMES = 3
EMPTY_CONFIRM_FRAMES = 5
IP_MIN_OBJECT_AREA = 0.012
MF = {"frame": None, "seq": 0, "t": 0.0}   # latest-frame-wins: chỉ giữ 1 frame mới nhất


def mlog(msg, debug_only=False):
    if debug_only and not MOBILE_DEBUG:
        return
    print(f"[{datetime.now():%H:%M:%S.%f}"[:-3] + f"] {msg}")


# ============================================================
# 4. XỬ LÝ ẢNH & TÌM KIẾM
# ============================================================
def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def roi_box(frame):
    h, w = frame.shape[:2]
    return int(w * ROI_X1), int(h * ROI_Y1), int(w * ROI_X2), int(h * ROI_Y2)


def get_roi(frame):
    x1, y1, x2, y2 = roi_box(frame)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def preprocess_roi(roi):
    # Tiền xử lý Mobile: giữ nguyên pipeline đang hoạt động ổn định.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    den = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    blur = cv2.GaussianBlur(den, (0, 0), 2.0)
    sharp = cv2.addWeighted(den, 1.5, blur, -0.5, 0)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(sharp)


def preprocess_ip_roi(roi, max_width=1440):
    """Pipeline nhẹ hơn riêng cho IP: tránh phóng ROI camera độ phân giải cao lên 2x."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("ROI camera IP rỗng")
    scale = min(1.5, max_width / float(w))
    if scale > 1.0:
        gray = cv2.resize(
            gray, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC
        )
    elif scale < 1.0:
        gray = cv2.resize(
            gray, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA
        )
    # Tránh bilateralFilter vốn khá nặng trên ROI lớn.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


NOISE = {"POWERSUPPLY", "POWER", "SUPPLY", "MODEL", "INPUT", "OUTPUT", "MADEIN",
         "WARNING", "CAUTION", "V", "A", "AC", "DC", "HZ", "CE", "ROHS"}


def is_noise_text(text):
    t = text.upper()
    return t in NOISE or len(t) < 2


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def presence_score(frame):
    """Tỉ lệ điểm ảnh có cạnh rõ trong ROI (ảnh thu nhỏ 96x72). Nền trơn -> thấp, nhãn có chữ -> cao.
    Công thức PHẢI trùng với presence() ở JavaScript."""
    x1, y1, x2, y2 = roi_box(frame)
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (96, 72), interpolation=cv2.INTER_AREA).astype(np.int16)
    gx = np.abs(g[:, 1:] - g[:, :-1])[:-1, :]
    gy = np.abs(g[1:, :] - g[:-1, :])[:, :-1]
    return float(((gx + gy) > MOBILE_EDGE_DIFF).mean())


def collect_dets(result, ox, oy, scale=2.0):
    dets = []
    for item in (result[0] if result and result[0] else []):
        try:
            pts, (text, conf) = item[0], item[1]
            conf = float(conf)
        except (IndexError, TypeError, ValueError):
            continue
        text = str(text).strip()
        if conf < FORM_MIN_CONFIDENCE or not text:
            continue
        xs = [p[0] / scale + ox for p in pts]
        ys = [p[1] / scale + oy for p in pts]
        dets.append({"text": text, "conf": round(conf, 3),
                     "box": [min(xs), min(ys), max(xs), max(ys)]})
    return dets


def load_rows():
    """Cache danh sách kho (kèm khoá đã chuẩn hoá). Chỉ truy vấn DB khi cache bị vô hiệu."""
    global _rows_cache
    with rows_lock:
        if _rows_cache is None:
            conn = db()
            try:
                rs = conn.execute("SELECT sku, brand, series, product_name, location, quantity, status "
                                  "FROM warehouse").fetchall()
            finally:
                conn.close()
            _rows_cache = [{"sku": r["sku"], "product_name": r["product_name"],
                            "location": r["location"], "quantity": r["quantity"],
                            "status": r["status"],
                            "n_sku": norm(r["sku"]), "n_brand": norm(r["brand"]),
                            "n_series": norm(r["series"])} for r in rs]
        return _rows_cache


def invalidate_rows():
    global _rows_cache
    with rows_lock:
        _rows_cache = None


def match_product(text, rows):
    c = norm(text)
    if len(c) < 2:
        return None
    best, top = None, 0.0
    for r in rows:
        sku, brand, series = r["n_sku"], r["n_brand"], r["n_series"]
        keys = [k for k in (sku, series, brand) if k]
        if c in keys:
            score = 1.0
        elif len(c) >= 3 and any(c in k or k in c for k in keys if len(k) >= 3):
            score = 0.90
        else:
            score = max([SequenceMatcher(None, c, k).ratio() for k in (sku, series) if k] or [0])
        if score > top:
            top = score
            best = {"sku": r["sku"], "product_name": r["product_name"],
                    "location": r["location"], "quantity": r["quantity"],
                    "status": r["status"], "found": True, "score": score,
                    "confidence_msg": f"Khớp (Score: {score*100:.0f}%)"}
    return best if top >= 0.65 else None


def analyze_dets(dets, rows, min_conf):
    """Dùng chung cho IP và mobile: trả (matched, conf, cands). Giữ nguyên logic nghiệp vụ cũ."""
    matched, best_conf, cands = None, 0.0, []
    for d in dets:
        cleaned = re.sub(r"\s+", "", d["text"])
        if d["conf"] < min_conf or is_noise_text(cleaned):
            continue
        if not (2 <= len(cleaned) <= 15 and any(ch.isalpha() for ch in cleaned)):
            continue
        cands.append((cleaned, d["conf"]))
        m = match_product(cleaned, rows)
        if m:
            matched, best_conf = m, d["conf"]
            break
    return matched, best_conf, cands


def run_ocr(img):
    with ocr_lock:
        return ocr.ocr(img, cls=False)


def save_result(code, conf, info, source=None, session_id=None):
    """Ghi CSV tuần tự; session_id ngăn cùng phiên ghi nhiều lần."""
    if session_id is not None:
        with ip_state_lock:
            if session_id in IP["session_recorded"]:
                return False
            IP["session_recorded"].add(session_id)
    try:
        with csv_lock:
            new_file = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
            with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["timestamp", "scanned_text", "product_name", "location",
                                "status", "confidence", "source"])
                w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), code,
                            info["product_name"], info["location"],
                            "FOUND" if info["found"] else "NOT_FOUND", f"{conf:.3f}", source or "unknown"])
        return True
    except Exception:
        if session_id is not None:
            with ip_state_lock:
                IP["session_recorded"].discard(session_id)
        raise


def get_pending(source=None):
    if source is None:
        with state_lock:
            source = S["source"]
    with state_lock:
        return S["pending_by_source"].get(source)


def set_pending(source, pending):
    with state_lock:
        S["pending_by_source"][source] = pending
        if S["source"] == source:
            S["pending"] = pending


def create_pending(cands, frame, dets, now, source="ip"):
    """Lưu ảnh và đặt S['pending'] cho workflow nhập liệu. Trả False nếu mã đang bị 'bỏ qua'."""
    if not cands or frame is None:
        return False
    if all(ignore_until.get((source, norm(c)), 0) > now for c in cands[:2]):
        return False
    barcodes = []
    if zbar_decode:
        try:
            barcodes = [b.data.decode("utf-8", "ignore") for b in zbar_decode(frame)]
        except Exception:
            pass
    pid = uuid.uuid4().hex[:10]
    filename = f"{norm(cands[0]) or 'NEW'}_{datetime.now():%Y%m%d_%H%M%S}_{pid}.jpg"
    cv2.imwrite(os.path.join(IMAGE_DIR, filename), frame)
    h, w = frame.shape[:2]
    set_pending(source, {"id": pid, "file": filename, "w": w, "h": h,
                         "dets": dets, "candidates": barcodes + [c for c in cands if c not in barcodes],
                         "barcodes": barcodes, "source": source})
    print(f"[{datetime.now():%H:%M:%S}] MÃ LẠ - chờ nhập liệu trên web: {cands[0]}")
    return True


# ============================================================
# 5. LUỒNG CAMERA IP / OCR IP / STREAM
# ============================================================
def camera_worker():
    global latest_frame
    if not USE_IP_CAMERA or not RTSP_URL:
        with state_lock:
            S["camera"] = "Camera IP đang tắt (demo)"
        return
    while not stop_event.is_set():
        cap = None
        try:
            with state_lock:
                S["camera"] = "Đang kết nối camera IP..."
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(3)
                continue
            with state_lock:
                S["camera"] = "Đã kết nối camera IP"
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                with frame_lock:
                    latest_frame = frame
        except Exception:
            pass
        finally:
            if cap:
                cap.release()
        time.sleep(2)


def stream_worker():
    global latest_jpeg
    delay = 1.0 / STREAM_FPS
    while not stop_event.is_set():
        with state_lock:
            is_mobile = (S["source"] == "mobile")
        if is_mobile:
            time.sleep(0.2)
            continue

        with frame_lock:
            f = None if latest_frame is None else latest_frame.copy()
        if f is None:
            time.sleep(0.2)
            continue
        x1, y1, x2, y2 = roi_box(f)
        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
        h, w = f.shape[:2]
        if w > STREAM_WIDTH:
            f = cv2.resize(f, (STREAM_WIDTH, int(h * STREAM_WIDTH / w)))
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with jpeg_lock:
                latest_jpeg = buf.tobytes()
        time.sleep(delay)


def confirm(info, conf, now, session_id):
    if save_result(info["sku"], conf, info, source="ip", session_id=session_id):
        set_camera_result("ip", info, now)
        print(f"[{datetime.now():%H:%M:%S}] CHỐT KHO: {info['product_name']} | {info['location']}")


def new_session(now):
    return {"start": now, "votes": {}, "best_frame": None, "best_dets": [], "best_score": -1.0}


def session_add(sess, frame, dets, cands):
    for text, conf in cands:
        v = sess["votes"].setdefault(norm(text), {"text": text, "n": 0, "conf": 0.0})
        v["n"] += 1
        v["conf"] += conf
    # Camera IP: đánh giá độ nét ở ROI sản phẩm, không phải toàn khung hình
    # (nền phía sau có thể nét trong khi nhãn sản phẩm đang nhòe).
    roi, _ = get_roi(frame)
    score = sharpness(roi)
    if score > sess["best_score"]:
        sess.update(best_score=score, best_frame=frame.copy(), best_dets=dets)


def finalize_session(sess, now):
    ranked = sorted(sess["votes"].values(),
                    key=lambda v: (v["n"], v["conf"] / v["n"]), reverse=True)
    # Chỉ giữ ứng viên OCR lặp lại ở ít nhất 2 lần đọc, tránh mở form do 1 frame nhòe.
    cands = [v["text"] for v in ranked if v["n"] >= 2][:8]
    if not cands or sess["best_frame"] is None:
        return False
    roi, _ = get_roi(sess["best_frame"])
    # Nếu nhãn còn nhòe, không tạo pending; tiếp tục chờ khung hình rõ hơn.
    if sharpness(roi) < 35.0:
        return False
    return create_pending(cands, sess["best_frame"], sess["best_dets"], now, source="ip")


def _ip_gray_roi(frame):
    """Ảnh xám ROI dùng riêng cho logic chuyển sản phẩm của camera IP."""
    roi, _ = get_roi(frame)
    return cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def _ip_object_changed(frame):
    """True khi vật trong ROI thay đổi rõ rệt so với vật IP vừa xử lý."""
    current = _ip_gray_roi(frame)
    with ip_state_lock:
        previous = IP.get("last_object_gray")
    if previous is None or previous.shape != current.shape:
        return False
    diff = cv2.absdiff(current, previous)
    _, mask = cv2.threshold(diff, 32, 255, cv2.THRESH_BINARY)
    changed_ratio = cv2.countNonZero(mask) / float(max(1, mask.shape[0] * mask.shape[1]))
    # Bỏ qua nhiễu nhỏ; yêu cầu khác biệt đủ lớn để coi là sản phẩm mới.
    return changed_ratio >= 0.10


def detect_ip_object(frame):
    """Phát hiện thay đổi ổn định trong ROI, không phân loại sản phẩm."""
    roi, _ = get_roi(frame)
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    with ip_state_lock:
        if IP["bg"] is None or IP["bg"].shape != gray.shape:
            # accumulateWeighted yêu cầu ảnh nền đích có kiểu float32/float64.
            IP["bg"] = gray.astype(np.float32)
            return False
        bg = IP["bg"].copy()
    # So sánh ảnh uint8 với ảnh nền chuyển về uint8; giữ bg float32 để cập nhật nền.
    diff = cv2.absdiff(gray, cv2.convertScaleAbs(bg))
    _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = max((cv2.contourArea(c) for c in contours), default=0.0)
    ratio = area / max(1, gray.shape[0] * gray.shape[1])
    present = ratio >= IP_MIN_OBJECT_AREA
    with ip_state_lock:
        # Không cập nhật nền khi có thay đổi đáng kể. Nếu cập nhật trong lúc vật
        # vừa xuất hiện (trước khi đủ 3 frame xác nhận), nền sẽ "nuốt" mất vật,
        # khiến present_frames bị reset và OCR không bao giờ khởi chạy.
        if not present and not IP["object_present"] and IP["bg"] is not None:
            cv2.accumulateWeighted(gray.astype(np.float32), IP["bg"], 0.01)
    return present


def ocr_worker():
    """Worker IP độc lập; OCR chỉ bắt đầu khi có vật thể mới ổn định."""
    sess, sid = None, None
    while not stop_event.is_set():
        if not ip_scan_enabled.is_set():
            sess = None
            with ip_state_lock:
                IP.update(state="IDLE", sid=None, present_frames=0, empty_frames=0,
                          object_present=False)
            time.sleep(0.05)
            continue
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        now = time.time()
        try:
            present = detect_ip_object(frame)
            with ip_state_lock:
                if present:
                    IP["present_frames"] += 1
                    IP["empty_frames"] = 0
                else:
                    IP["empty_frames"] += 1
                    IP["present_frames"] = 0
                if IP["state"] == "WAIT_OBJECT_REMOVAL":
                    if IP["empty_frames"] >= EMPTY_CONFIRM_FRAMES:
                        IP.update(state="IDLE", sid=None, object_present=False,
                                  present_frames=0, empty_frames=0,
                                  changed_object_frames=0, last_object_gray=None)
                        sess = None
                    elif present and _ip_object_changed(frame):
                        # Camera IP có thể không thấy ROI trống giữa hai sản phẩm.
                        # Nếu hình vật thể thay đổi rõ rệt liên tục, mở phiên mới.
                        IP["changed_object_frames"] += 1
                        if IP["changed_object_frames"] >= PRESENCE_CONFIRM_FRAMES:
                            IP.update(state="IDLE", sid=None, object_present=False,
                                      present_frames=PRESENCE_CONFIRM_FRAMES,
                                      empty_frames=0, changed_object_frames=0,
                                      last_object_gray=None)
                            sess = None
                        else:
                            time.sleep(OCR_INTERVAL_SEC)
                            continue
                    else:
                        IP["changed_object_frames"] = 0
                        time.sleep(OCR_INTERVAL_SEC)
                        continue
                elif IP["state"] == "SCANNING" and IP["empty_frames"] >= EMPTY_CONFIRM_FRAMES:
                    IP.update(state="IDLE", sid=None, object_present=False,
                              present_frames=0, empty_frames=0)
                    sess = None
                if IP["state"] == "IDLE" and get_pending("ip") is not None:
                    time.sleep(OCR_INTERVAL_SEC)
                    continue
                if IP["state"] == "IDLE" and IP["present_frames"] >= PRESENCE_CONFIRM_FRAMES:
                    sid = uuid.uuid4().hex
                    IP.update(state="SCANNING", sid=sid, object_present=True, empty_frames=0)
                    sess = new_session(now)
                elif IP["state"] != "SCANNING":
                    time.sleep(OCR_INTERVAL_SEC)
                    continue
                else:
                    sid = IP["sid"]

            roi, (ox, oy, _, _) = get_roi(frame)
            ip_pre = preprocess_ip_roi(roi)
            result = run_ocr(ip_pre)
            # Toạ độ box phải dùng đúng tỉ lệ resize của pipeline IP.
            ip_scale = ip_pre.shape[1] / float(max(1, roi.shape[1]))
            dets = collect_dets(result, ox, oy, scale=ip_scale)
            matched, best_conf, cands = analyze_dets(dets, load_rows(), MIN_OCR_CONFIDENCE)
            set_camera_ocr("ip", " | ".join(f"{t} ({c:.2f})" for t, c in cands[:3]) or "Đang chờ mã...")

            if matched:
                object_gray = _ip_gray_roi(frame)
                with ip_state_lock:
                    valid = IP["state"] == "SCANNING" and IP["sid"] == sid
                    if valid:
                        IP.update(state="WAIT_OBJECT_REMOVAL", empty_frames=0,
                                  last_object_gray=object_gray.copy(),
                                  changed_object_frames=0)
                if valid:
                    confirm(matched, best_conf, now, sid)
                sess = None
            elif cands and sess is not None:
                session_add(sess, frame, dets, cands)
                if now - sess["start"] >= UNSEEN_TIMEOUT_SEC:
                    # Chỉ kết thúc phiên nếu đã có ứng viên lặp lại và ảnh ROI đủ nét.
                    # Nếu đang cầm/di chuyển sản phẩm hoặc OCR đọc sai do nhòe, giữ phiên
                    # SCANNING để tiếp tục lấy frame rõ hơn, không bật form nhập liệu sớm.
                    roi_best = None if sess["best_frame"] is None else get_roi(sess["best_frame"])[0]
                    sharp_enough = roi_best is not None and sharpness(roi_best) >= 35.0
                    repeated_candidate = any(v["n"] >= 2 for v in sess["votes"].values())
                    if sharp_enough and repeated_candidate:
                        with ip_state_lock:
                            valid = IP["state"] == "SCANNING" and IP["sid"] == sid
                            if valid:
                                IP.update(state="WAIT_OBJECT_REMOVAL", empty_frames=0)
                        if valid:
                            finalize_session(sess, now)
                            with ip_state_lock:
                                IP["last_object_gray"] = _ip_gray_roi(frame).copy()
                                IP["changed_object_frames"] = 0
                            sess = None
                    else:
                        # Gia hạn cửa sổ lấy mẫu thay vì chốt ảnh mờ.
                        sess["start"] = now
        except Exception as exc:
            set_camera_ocr("ip", f"Lỗi OCR: {exc}")
        time.sleep(OCR_INTERVAL_SEC)


# ============================================================
# 5b. CAMERA ĐIỆN THOẠI: STATE MACHINE + WORKER
# ============================================================
def _m_tick_locked(now):
    """Gọi khi đang giữ mobile_cv. Hết cooldown -> IDLE (không dùng sleep)."""
    if M["state"] == "COOLDOWN" and now >= M["cooldown_until"]:
        M["state"], M["sid"] = "IDLE", None


def m_reset(state):
    """Đặt lại state machine (đổi nguồn / bắt đầu / dừng)."""
    with mobile_cv:
        M.update(state=state, sid=None, cooldown_until=0.0, rearm_at=0.0)
        MF.update(frame=None, seq=0, t=0.0)
        mobile_cv.notify_all()


def m_finish(sid, state, cooldown=0.0, backoff=0.0):
    """Chuyển trạng thái chỉ khi phiên `sid` còn hiệu lực (chống xử lý trùng/phiên cũ)."""
    now = time.time()
    with mobile_cv:
        if M["sid"] != sid or M["state"] != "SCANNING":
            return False
        if state == "CONFIRMED":                      # CONFIRMED là trạng thái chuyển tiếp -> COOLDOWN ngay
            state = "COOLDOWN"
            cooldown = cooldown or MOBILE_COOLDOWN_SEC
        M["state"] = state
        if state == "COOLDOWN":
            M["cooldown_until"] = now + cooldown
        elif state == "IDLE":
            M["sid"] = None
            M["rearm_at"] = now + backoff
        MF.update(frame=None)
        return True


def mobile_pending_done():
    """Gọi sau khi LƯU hoặc BỎ QUA pending Mobile."""
    with mobile_cv:
        if M["state"] == "UNKNOWN_PENDING":
            M["state"] = "COOLDOWN"
            M["cooldown_until"] = time.time() + MOBILE_COOLDOWN_SEC


def _decide(votes, unk, nframes, now, soft, hard):
    """Quyết định kết thúc phiên. Trả về None nếu cần thu thêm bằng chứng."""
    v = max(votes.values(), key=lambda x: (x["n"], x["conf"] / x["n"])) if votes else None
    if v:
        avg = v["conf"] / v["n"]
        if v["n"] >= 2 or (v["score"] >= 0.95 and avg >= MOBILE_STRONG_CONF):
            return ("confirm", v)
        if now >= soft and v["score"] >= 0.90:
            return ("confirm", v)
    ranked = sorted(unk.values(), key=lambda x: (x["n"], x["conf"] / x["n"]), reverse=True)
    if now >= soft and nframes > 0:
        if ranked and (ranked[0]["n"] >= 2 or now >= hard):
            return ("unknown", ranked)       # khớp yếu/không khớp -> để người dùng xác minh, không tự chốt
        if not ranked:
            return ("empty", None)
    if now >= hard:
        return ("unknown", ranked) if ranked else ("empty", None)
    return None


def run_mobile_session(sid, t0):
    votes, unk = {}, {}
    best = {"sharp": -1.0, "frame": None, "dets": []}
    st = {"frames": 0, "pre": 0.0, "ocr": 0.0, "match": 0.0, "err": 0}
    soft, hard = t0 + MOBILE_SESSION_TARGET_SEC, t0 + MOBILE_SESSION_MAX_SEC
    last_seq, decision = 0, None

    while decision is None and not stop_event.is_set():
        frame = None
        with mobile_cv:
            if M["sid"] != sid or M["state"] != "SCANNING":
                return                                 # bị dừng / đổi nguồn / phiên đã bị thay thế
            if MF["seq"] != last_seq and MF["frame"] is not None:
                frame, last_seq = MF["frame"], MF["seq"]
            else:
                mobile_cv.wait(0.05)
        now = time.time()
        if frame is None:
            if now >= hard:
                decision = _decide(votes, unk, st["frames"], now, soft, hard)
            continue

        try:
            a = time.time()
            roi, (ox, oy, _, _) = get_roi(frame)
            pre = preprocess_roi(roi)
            b = time.time()
            result = run_ocr(pre)
            c = time.time()
            dets = collect_dets(result, ox, oy)
            rows = load_rows()
            matched, bconf, cands = analyze_dets(dets, rows, MOBILE_MIN_OCR_CONFIDENCE)
            d = time.time()
        except Exception as exc:
            st["err"] += 1
            set_camera_ocr("mobile", f"Lỗi OCR/DB: {exc}")
            mlog(f"MOBILE {sid} lỗi xử lý frame: {exc}")
            if st["err"] >= 3:
                m_finish(sid, "IDLE", backoff=MOBILE_EMPTY_BACKOFF_SEC)    # không ghi nhận thành công giả
                return
            continue

        st["frames"] += 1
        st["pre"] += (b - a) * 1000
        st["ocr"] += (c - b) * 1000
        st["match"] += (d - c) * 1000
        mlog(f"MOBILE {sid} frame#{st['frames']} pre={(b-a)*1000:.0f}ms ocr={(c-b)*1000:.0f}ms "
             f"match={(d-c)*1000:.0f}ms cands={len(cands)} matched={matched['sku'] if matched else None}",
             debug_only=True)

        set_camera_ocr("mobile", " | ".join(f"{t} ({cf:.2f})" for t, cf in cands[:3]) or "Đang quét...")

        if matched:
            v = votes.setdefault(matched["sku"], {"info": matched, "n": 0, "conf": 0.0, "score": 0.0})
            v["n"] += 1
            v["conf"] += bconf
            v["score"] = max(v["score"], matched["score"])
        for text, cf in cands:
            u = unk.setdefault(norm(text), {"text": text, "n": 0, "conf": 0.0})
            u["n"] += 1
            u["conf"] += cf
        if cands:
            sh = sharpness(roi)
            if sh > best["sharp"]:
                best.update(sharp=sh, frame=frame.copy(), dets=dets)
        decision = _decide(votes, unk, st["frames"], time.time(), soft, hard)

    if decision is None:
        return
    kind, payload = decision
    now = time.time()
    t_agg = time.time()
    outcome = kind

    if kind == "confirm":
        info, conf = payload["info"], payload["conf"] / payload["n"]
        if m_finish(sid, "CONFIRMED", cooldown=MOBILE_COOLDOWN_SEC):
            try:
                saved = save_result(info["sku"], conf, info, source="mobile", session_id="mobile:" + sid)
                if saved:
                    set_camera_result("mobile", info, now)
                    print(f"[{datetime.now():%H:%M:%S}] CHỐT KHO (mobile): {info['product_name']} | {info['location']}")
            except Exception as exc:
                set_camera_ocr("mobile", f"Lỗi ghi CSV: {exc}")
    elif kind == "unknown":
        cands = [u["text"] for u in payload[:8]]
        ok = create_pending(cands, best["frame"], best["dets"], now, source="mobile")
        if ok:
            if not m_finish(sid, "UNKNOWN_PENDING"):          # phiên đã bị dừng giữa chừng -> huỷ pending vừa tạo
                set_pending("mobile", None)
        else:
            outcome = "ignored"
            m_finish(sid, "IDLE", backoff=MOBILE_EMPTY_BACKOFF_SEC)
    else:
        m_finish(sid, "IDLE", backoff=MOBILE_EMPTY_BACKOFF_SEC)

    mlog(f"MOBILE {sid} {outcome.upper()} total={(time.time()-t0)*1000:.0f}ms frames={st['frames']} "
         f"pre={st['pre']:.0f}ms ocr={st['ocr']:.0f}ms match={st['match']:.0f}ms "
         f"aggregate={(time.time()-t_agg)*1000:.0f}ms")


def mobile_worker():
    """Một worker duy nhất => không có nhiều phiên/OCR chồng lấn cho điện thoại."""
    while not stop_event.is_set():
        with mobile_cv:
            while not stop_event.is_set() and (
                    not mobile_scan_enabled.is_set() or M["state"] != "SCANNING"):
                mobile_cv.wait(0.5)
            if stop_event.is_set():
                break
            sid, t0 = M["sid"], M["t0"]
        try:
            run_mobile_session(sid, t0)
        except Exception as exc:
            mlog(f"MOBILE {sid} lỗi phiên: {exc}")
            m_finish(sid, "IDLE", backoff=MOBILE_EMPTY_BACKOFF_SEC)


# ============================================================
# 6. WEB (FLASK)
# ============================================================
app = Flask(__name__)
app.secret_key = WEB_SECRET



def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not session.get("auth"):
            if request.path.startswith("/api/"):
                return jsonify(error="unauthorized"), 401
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper


CSS = """<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:Arial,sans-serif;background:#202020;color:#fff;margin:0;padding:12px}
a{color:#ffaa00}.box{background:#111;border:2px groove #444;padding:12px;margin-top:10px;max-width:960px;border-radius:8px}
.code{color:#00ff88;font:bold 20px Consolas,monospace}.prod{color:#0ff;font-size:20px;font-weight:bold}
.loc{color:#f55;font-size:20px;font-weight:bold}.small{color:#aaa;font-size:13px}
img,canvas,video{max-width:100%;border-radius:8px;background:#000}
button,input,select{padding:9px;border-radius:6px;border:0;font-size:15px}
button{background:#00aa66;color:#fff;cursor:pointer;margin:3px}button.gray{background:#555}button.red{background:#c0392b}
input{width:100%;box-sizing:border-box;background:#fff;color:#000}
.row{display:flex;gap:8px;align-items:center;margin:5px 0}.row label{width:110px;flex:none}
.chip{display:inline-block;background:#333;border:1px solid #666;padding:4px 9px;margin:3px;border-radius:14px;cursor:pointer}
.banner{background:#7a4b00;border:2px solid #ffaa00;padding:12px;border-radius:8px;margin-top:10px;max-width:960px}
table{border-collapse:collapse;width:100%;max-width:1100px}td,th{border:1px solid #444;padding:6px;text-align:left}
.pend{color:#ffaa00}
</style>"""

LOGIN_HTML = CSS + """<title>Đăng nhập</title>
<form method="post" style="max-width:290px;margin:12vh auto;background:#111;padding:26px;border-radius:10px">
<h3>Truy xuất kho</h3><div style="color:#f66">%%ERR%%</div>
<input name="username" placeholder="Tài khoản" autofocus style="margin:6px 0">
<input name="password" type="password" placeholder="Mật khẩu" style="margin:6px 0">
<button style="width:100%">Đăng nhập</button></form>"""

INDEX_HTML = CSS + """<title>Nhận diện & truy xuất kho</title>
<h3>NHẬN DIỆN SẢN PHẨM & TRUY XUẤT KHO
<a href="/products" style="font-size:13px">[Danh sách kho]</a>
<a href="/logout" style="font-size:13px">[Đăng xuất]</a></h3>

<div style="margin-bottom:8px">
  <button id="btn_ip" onclick="switchSource('ip')">Dùng Camera IP</button>
  <button id="btn_mobile" class="gray" onclick="switchSource('mobile')">Dùng Camera Điện Thoại</button>
</div>

<div id="banner" class="banner" style="display:none">⚠ Phát hiện sản phẩm mới chưa có trong kho.
<a href="/new"><button>NHẬP LIỆU NGAY</button></a></div>

<!-- Luồng hiển thị Camera IP -->
<div id="view_ip">
  <img id="stream_img" src="/video_feed" style="width:100%;max-width:960px;margin-top:8px">
</div>

<!-- Luồng hiển thị Camera Điện thoại kèm khung ROI -->
<div id="view_mobile" style="display:none; position:relative; max-width:960px; margin-top:8px;">
  <video id="webcam" autoplay playsinline muted style="width:100%; border-radius:8px; display:block; background:#000;"></video>
  <div id="roi_box" style="position:absolute; top:20%; left:20%; width:60%; height:60%; border:2px solid #00ff88; pointer-events:none; border-radius:4px; box-sizing:border-box;"></div>
  <div id="mstate" style="position:absolute; top:8px; left:8px; background:rgba(0,0,0,.65); color:#fff; padding:4px 10px; border-radius:12px; font-weight:bold; font-size:14px;">ĐÃ DỪNG</div>
  <div style="margin-top:8px; display:flex; gap:8px;">
    <button onclick="scan('start')">BẮT ĐẦU QUÉT (ĐT)</button>
    <button class="gray" onclick="scan('stop')">DỪNG QUÉT (ĐT)</button>
  </div>
</div>

<div class="box">
<div class="small" id="status">...</div><div class="code" id="code">--</div>
<div class="prod" id="prod">--</div><div class="loc" id="loc">--</div>
<div class="small" id="ocr">--</div>
<button onclick="scan('start')">BẮT ĐẦU QUÉT</button><button class="gray" onclick="scan('stop')">DỪNG QUÉT</button>
</div>

<script>
const $=id=>document.getElementById(id);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const setText=(id,t)=>{const e=$(id);if(e.textContent!==t)e.textContent=t;};
let currentSource='ip', mediaStream=null, starting=false;
// MOB: trạng thái phía trình duyệt cho luồng điện thoại (cfg được server gửi qua /api/status)
const MOB={gen:0,state:'STOPPED',retryUntil:0,coolUntil:0,err:0,
  cfg:{pres_thr:0.05,edge_diff:40,idle_ms:500,scan_ms:250,frame_w:800,jpeg_q:0.7}};
const LABEL={IDLE:'ĐANG CHỜ',SCANNING:'ĐANG NHẬN DIỆN',CONFIRMED:'ĐÃ NHẬN DIỆN',COOLDOWN:'NGHỈ',
  UNKNOWN_PENDING:'PHÁT HIỆN SẢN PHẨM MỚI',STOPPED:'ĐÃ DỪNG'};
// canvas dùng lại, không tạo DOM mới mỗi khung
const pcv=document.createElement('canvas');pcv.width=96;pcv.height=72;
const pctx=pcv.getContext('2d',{willReadFrequently:true});
const fcv=document.createElement('canvas');const fctx=fcv.getContext('2d');

function applyMobileState(d){
  if(d.state)MOB.state=d.state;
  if(d.retry_ms>0)MOB.retryUntil=performance.now()+d.retry_ms;
  if(d.cooldown_left!=null)MOB.coolUntil=performance.now()+d.cooldown_left*1000;
  renderMobileState(d.has_code);
}
function renderMobileState(hasCode){
  let t=LABEL[MOB.state]||MOB.state;
  if(MOB.err>=2)t='LỖI KẾT NỐI';
  else if(MOB.state==='COOLDOWN'){
    const left=Math.max(0,(MOB.coolUntil-performance.now())/1000);
    t=(hasCode?'ĐÃ NHẬN DIỆN · ':'')+'NGHỈ '+left.toFixed(1)+'s';
  }
  setText('mstate',t);
}

async function switchSource(src){
  if(src===currentSource)return;
  currentSource=src;
  MOB.gen++;                       // kết thúc vòng lặp mobile hiện có (nếu có)
  await fetch('/api/source',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:src})});
  if(src==='mobile'){
    $('btn_mobile').className='';$('btn_ip').className='gray';
    $('view_ip').style.display='none';$('view_mobile').style.display='block';
    MOB.state='STOPPED';renderMobileState(false);
    startWebcam();
  } else {
    $('btn_ip').className='';$('btn_mobile').className='gray';
    $('view_mobile').style.display='none';$('view_ip').style.display='block';
    stopWebcam();
  }
}

async function startWebcam(){
  if(mediaStream||starting)return;   // tránh mở camera / tạo vòng lặp trùng
  starting=true;
  try{
    mediaStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1280},height:{ideal:720}}});
    if(currentSource!=='mobile'){stopWebcam();return;}
    $('webcam').srcObject=mediaStream;
    mobileLoop(++MOB.gen);
  }catch(e){alert('Không thể mở camera điện thoại: '+e);switchSource('ip');}
  finally{starting=false;}
}

function stopWebcam(){
  MOB.gen++;
  if(mediaStream){mediaStream.getTracks().forEach(t=>t.stop());mediaStream=null;}
  $('webcam').srcObject=null;
}

// Cùng công thức với presence_score() ở server: tỉ lệ điểm ảnh có cạnh trong ROI (96x72)
function presence(){
  const v=$('webcam');if(!v.videoWidth)return 0;
  pctx.drawImage(v,v.videoWidth*0.2,v.videoHeight*0.2,v.videoWidth*0.6,v.videoHeight*0.6,0,0,96,72);
  const d=pctx.getImageData(0,0,96,72).data,g=new Int16Array(96*72);
  for(let i=0,j=0;i<g.length;i++,j+=4)g[i]=Math.floor((299*d[j]+587*d[j+1]+114*d[j+2])/1000);
  const th=MOB.cfg.edge_diff;let n=0;
  for(let y=0;y<71;y++)for(let x=0;x<95;x++){const i=y*96+x;
    if(Math.abs(g[i+1]-g[i])+Math.abs(g[i+96]-g[i])>th)n++;}
  return n/(71*95);
}

async function sendFrame(){
  const v=$('webcam');if(!v.videoWidth)return;
  const w=Math.min(MOB.cfg.frame_w,v.videoWidth),h=Math.round(v.videoHeight*w/v.videoWidth);
  if(fcv.width!==w)fcv.width=w;if(fcv.height!==h)fcv.height=h;
  fctx.drawImage(v,0,0,w,h);
  const blob=await new Promise(r=>fcv.toBlob(r,'image/jpeg',MOB.cfg.jpeg_q));
  if(!blob)return;
  const ctl=new AbortController(),to=setTimeout(()=>ctl.abort(),4000);
  try{
    const r=await fetch('/api/mobile_frame',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob,signal:ctl.signal});
    if(r.status===401){location='/login';return;}
    applyMobileState(await r.json());MOB.err=0;
  }catch(e){MOB.err++;renderMobileState(false);}
  finally{clearTimeout(to);}
}

// Một vòng lặp duy nhất, tuần tự (await) => không bao giờ có 2 request chồng nhau
async function mobileLoop(gen){
  while(MOB.gen===gen){
    let wait=250;
    try{
      if(MOB.state==='COOLDOWN'&&performance.now()>=MOB.coolUntil)MOB.state='IDLE';
      if(MOB.state==='IDLE'){
        if(performance.now()>=MOB.retryUntil&&presence()>=MOB.cfg.pres_thr){await sendFrame();wait=MOB.cfg.scan_ms;}
        else wait=MOB.cfg.idle_ms;
      }else if(MOB.state==='SCANNING'){await sendFrame();wait=MOB.cfg.scan_ms;}
      if(MOB.err>0)wait=1000;
    }catch(e){wait=1000;}
    await sleep(wait);
  }
}

async function scan(a){
  try{
    const r=await fetch('/api/scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:a})});
    const d=await r.json();
    if(currentSource==='mobile'&&d.state){MOB.state=d.state;MOB.retryUntil=0;renderMobileState(false);}
  }catch(e){}
  tick();
}

async function tick(){try{const r=await fetch('/api/status');if(r.status===401){location='/login';return}
const d=await r.json();
if(d.cfg)MOB.cfg=d.cfg;
if(d.source==='mobile'){
  MOB.state=d.mstate;MOB.coolUntil=performance.now()+d.cooldown_left*1000;MOB.err=0;
  renderMobileState(!!d.code);
}
setText('status','Camera: '+d.camera+' | Nguồn: '+d.source.toUpperCase()+' | '+(d.scanning?'ĐANG QUÉT':'ĐÃ DỪNG'));
setText('code',d.code?'MÃ ĐÃ CHỐT: '+d.code:'Đang chờ đưa nhãn vào khung...');
setText('prod',d.code?'📦 '+d.product+(d.status==='PENDING'?' [chưa đủ thông tin]':''):'📦 --');
setText('loc',d.code?'📍 '+d.location:'📍 --');
setText('ocr','OCR đọc được: '+d.ocr);
const bn=$('banner').style,want=d.pending?'block':'none';if(bn.display!==want)bn.display=want;
}catch(e){if(currentSource==='mobile'){MOB.err++;renderMobileState(false);}}}
setInterval(tick,500);tick();
</script>"""

NEW_HTML = CSS + """<title>Thêm sản phẩm mới</title>
<h3>THÊM SẢN PHẨM MỚI <a href="/" style="font-size:13px">[Về trang chính]</a></h3>
<div class="small">Bấm vào ô cần điền, rồi bấm vào chữ (khung xanh) trên ảnh để điền tự động. Bấm nhiều chữ sẽ ghép lại.</div>
<canvas id="cv"></canvas>
<div class="box"><div class="small">Gợi ý (bấm để điền vào ô đang chọn):</div><div id="chips"></div>
<div id="form"></div>
<button onclick="save(false)">LƯU</button><button class="gray" onclick="skip()">BỎ QUA</button>
<div id="msg" class="small"></div></div>
<datalist id="dl_brand"></datalist><datalist id="dl_loc"></datalist>
<script>
const $=id=>document.getElementById(id);
const FIELDS=[['sku','SKU / Mã',''],['brand','Hãng','dl_brand'],['series','Series',''],['name','Tên sản phẩm',''],['location','Vị trí kệ','dl_loc'],['qty','Số lượng','']];
let P=null,focusId='sku',scale=1,img=new Image(),hit=new Set();
FIELDS.forEach(([k,t,dl])=>{const r=document.createElement('div');r.className='row';
r.innerHTML=`<label>${t}</label><input id="f_${k}" ${dl?`list="${dl}"`:''} autocomplete="off"><button class="gray" tabindex="-1">✕</button>`;
r.querySelector('button').onclick=()=>{$('f_'+k).value='';};
$('form').appendChild(r);$('f_'+k).addEventListener('focus',()=>focusId=k);});
$('f_qty').value='0';
function fill(text){const el=$('f_'+focusId);const v=el.value.trim();
el.value=(v&&!(focusId==='qty'))?v+' '+text:text;}
function draw(){const cv=$('cv'),ctx=cv.getContext('2d');
scale=Math.min(900/P.w,1);cv.width=P.w*scale;cv.height=P.h*scale;
ctx.drawImage(img,0,0,cv.width,cv.height);ctx.lineWidth=2;
P.dets.forEach((d,i)=>{const[b0,b1,b2,b3]=d.box.map(v=>v*scale);
ctx.strokeStyle=hit.has(i)?'#ffaa00':'#00ff88';ctx.strokeRect(b0,b1,b2-b0,b3-b1);});}
$('cv').addEventListener('click',e=>{const r=$('cv').getBoundingClientRect();
const x=(e.clientX-r.left)*($('cv').width/r.width)/scale,y=(e.clientY-r.top)*($('cv').height/r.height)/scale;
let best=-1,area=1e18;P.dets.forEach((d,i)=>{const[a,b,c,f]=d.box;
if(x>=a&&x<=c&&y>=b&&y<=f&&(c-a)*(f-b)<area){area=(c-a)*(f-b);best=i}});
if(best>=0){fill(P.dets[best].text);hit.add(best);draw();}});
async function init(){const r=await fetch('/api/pending');if(r.status===401){location='/login';return}
P=await r.json();if(!P.id){document.body.innerHTML='<p>Không có sản phẩm nào đang chờ. <a href="/">Về trang chính</a></p>';return}
P.brands.forEach(b=>$('dl_brand').appendChild(new Option(b)));P.locations.forEach(l=>$('dl_loc').appendChild(new Option(l)));
P.candidates.forEach(c=>{const s=document.createElement('span');s.className='chip';s.textContent=c;s.onclick=()=>fill(c);$('chips').appendChild(s)});
if(P.candidates.length)$('f_sku').value=P.candidates[0];
img.onload=draw;img.src=P.image;}
async function save(force){const body={id:P.id,force:force};
FIELDS.forEach(([k])=>body[k]=$('f_'+k).value.trim());
const r=await fetch('/api/products',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const d=await r.json();
if(r.ok){alert(d.message);location='/';return}
if(d.needs_confirm){if(confirm(d.message+'\\nVẫn muốn lưu thành sản phẩm mới?'))save(true);return}
$('msg').textContent=d.message||'Lỗi';}
async function skip(){await fetch('/api/pending/skip',{method:'POST'});location='/';}
init();
</script>"""

PRODUCTS_HTML = CSS + """<title>Danh sách kho</title>
<h3>DANH SÁCH KHO <a href="/" style="font-size:13px">[Về trang chính]</a></h3>
<table id="t"><thead><tr><th>SKU</th><th>Hãng</th><th>Series</th><th>Tên</th><th>Vị trí</th><th>SL</th><th>Trạng thái</th><th>Ảnh</th></tr></thead><tbody></tbody></table>
<script>
(async()=>{const r=await fetch('/api/products');if(r.status===401){location='/login';return}
const d=await r.json();const tb=document.querySelector('tbody');
d.forEach(p=>{const tr=document.createElement('tr');
[p.sku,p.brand,p.series,p.product_name,p.location,p.quantity,p.status==='PENDING'?'Chưa đủ thông tin':'Đầy đủ'].forEach((v,i)=>{
const td=document.createElement('td');td.textContent=v??'';if(i===6&&p.status==='PENDING')td.className='pend';tr.appendChild(td)});
const td=document.createElement('td');if(p.image_path){const a=document.createElement('a');a.href='/captures/'+p.image_path;a.target='_blank';a.textContent='xem';td.appendChild(a)}
tr.appendChild(td);tb.appendChild(tr)})})();
</script>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    err = ""
    if request.method == "POST":
        if (request.form.get("username") == WEB_USER and
                check_password_hash(WEB_PASS_HASH, request.form.get("password", ""))):
            session["auth"] = True
            return redirect(url_for("index"))
        err = "Sai tài khoản hoặc mật khẩu"
        time.sleep(1)
    return LOGIN_HTML.replace("%%ERR%%", html.escape(err))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return INDEX_HTML


@app.route("/new")
@login_required
def new_page():
    return NEW_HTML


@app.route("/products")
@login_required
def products_page():
    return PRODUCTS_HTML


@app.route("/captures/<path:filename>")
@login_required
def captures(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/video_feed")
@login_required
def video_feed():
    def gen():
        while not stop_event.is_set():
            with state_lock:
                is_mobile = S["source"] == "mobile"
            if is_mobile:                  # ngủ NGOÀI lock
                time.sleep(0.5)
                continue
            with jpeg_lock:
                buf = latest_jpeg
            if buf:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
            time.sleep(1.0 / STREAM_FPS)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/source", methods=["POST"])
@login_required
def api_source():
    global latest_frame
    data = request.get_json(silent=True) or {}
    src = "mobile" if data.get("source") == "mobile" else "ip"
    with state_lock:
        S["source"] = src
        S["pending"] = S["pending_by_source"].get(src)
        S["code"] = CAMERA_RESULTS[src]["code"]
        S["info"] = CAMERA_RESULTS[src]["info"]
        S["ocr"] = CAMERA_RESULTS[src]["ocr"]
        S["last_sku"] = CAMERA_RESULTS[src]["last_sku"]
        S["last_time"] = CAMERA_RESULTS[src]["last_time"]
        S["camera"] = (
            "Đang dùng Camera Điện thoại" if src == "mobile"
            else ("Đang kết nối camera IP..." if USE_IP_CAMERA
                  else "Camera IP đang tắt (demo)")
        )
    # Chỉ đổi nguồn hiển thị; không reset session hoặc dừng worker của camera còn lại.
    return jsonify(source=src)


@app.route("/api/mobile_frame", methods=["POST"])
@login_required
def api_mobile_frame():
    """Nhận JPEG nhị phân. Chỉ lưu frame mới nhất; OCR chạy ở mobile_worker (không chặn request)."""
    t_rx = time.time()
    raw = request.get_data(cache=False)
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    if frame is None:
        return jsonify(ok=False, error="ảnh không hợp lệ"), 400
    with state_lock:
        src = S["source"]
    pending = get_pending("mobile") is not None
    now = time.time()
    if src != "mobile" or not mobile_scan_enabled.is_set():
        return jsonify(ok=False, state="STOPPED")

    pres = None
    with mobile_cv:
        _m_tick_locked(now)
        state = M["state"]
    if state == "IDLE" and not pending:
        pres = presence_score(frame)               # tính ngoài lock

    retry_ms, started = 0, False
    with mobile_cv:
        _m_tick_locked(now)
        state = M["state"]
        if state == "SCANNING":
            MF.update(frame=frame, seq=MF["seq"] + 1, t=now)
            mobile_cv.notify_all()
        elif state == "IDLE" and not pending and pres is not None:
            if now < M["rearm_at"]:
                retry_ms = int((M["rearm_at"] - now) * 1000)
            elif pres >= MOBILE_PRESENCE_THR:
                M.update(state="SCANNING", sid=uuid.uuid4().hex[:6], t0=now)
                MF.update(frame=frame, seq=MF["seq"] + 1, t=now)
                started, state = True, "SCANNING"
                mobile_cv.notify_all()
        left = max(0.0, M["cooldown_until"] - now) if state == "COOLDOWN" else 0.0
        sid = M["sid"]
    if started:
        mlog(f"MOBILE {sid} BẮT ĐẦU presence={pres:.3f} rx+decode={(time.time()-t_rx)*1000:.0f}ms")
    with state_lock:
        has_code = S["code"] is not None
    return jsonify(ok=True, state="UNKNOWN_PENDING" if (pending and state == "IDLE") else state,
                   sid=sid, retry_ms=retry_ms, cooldown_left=round(left, 2), has_code=has_code)


@app.route("/api/status")
@login_required
def api_status():
    now = time.time()
    with mobile_cv:
        _m_tick_locked(now)
        mstate = M["state"]
        left = max(0.0, M["cooldown_until"] - now) if mstate == "COOLDOWN" else 0.0
    with state_lock:
        source = S["source"]
        mobile = source == "mobile"
        result_state = CAMERA_RESULTS[source]
        info = result_state["info"]
        return jsonify(camera=S["camera"], source=source,
                       scanning=mobile_scan_enabled.is_set() if mobile else ip_scan_enabled.is_set(),
                       mstate=mstate, cooldown_left=round(left, 2),
                       code=result_state["code"], ocr=result_state["ocr"],
                       product=info["product_name"] if info else "",
                       location=info["location"] if info else "",
                       status=info.get("status", "OK") if info else "",
                       found=bool(info and info["found"]),
                       pending=S["pending_by_source"].get(source) is not None,
                       cfg=dict(pres_thr=MOBILE_PRESENCE_THR, edge_diff=MOBILE_EDGE_DIFF,
                                idle_ms=MOBILE_IDLE_CHECK_INTERVAL, scan_ms=MOBILE_SCAN_SEND_MS,
                                frame_w=MOBILE_FRAME_WIDTH, jpeg_q=MOBILE_JPEG_QUALITY))


@app.route("/api/scan", methods=["POST"])
@login_required
def api_scan():
    action = (request.get_json(silent=True) or {}).get("action")
    with state_lock:
        mobile = S["source"] == "mobile"
    pending = get_pending("mobile" if mobile else "ip") is not None
    if mobile:
        if action == "start":
            mobile_scan_enabled.set()
            with mobile_cv:
                if M["state"] == "STOPPED":
                    M.update(state="UNKNOWN_PENDING" if pending else "IDLE", sid=None,
                             rearm_at=0.0, empty_frames=0, presence_frames=0)
                    MF.update(frame=None, seq=0)
                    mobile_cv.notify_all()
        elif action == "stop":
            mobile_scan_enabled.clear()
            m_reset("STOPPED")
        with mobile_cv:
            state = M["state"]
        return jsonify(scanning=mobile_scan_enabled.is_set(), state=state)

    if action == "start" and not USE_IP_CAMERA:
        with state_lock:
            S["camera"] = "Camera IP đang tắt; cấu hình USE_IP_CAMERA=true để bật"
        return jsonify(scanning=False, state="DISABLED",
                      message="Camera IP đang tắt. Hãy cấu hình biến môi trường trước khi bật.")

    if action == "start":
        ip_scan_enabled.set()
        with ip_state_lock:
            if IP["state"] != "WAIT_OBJECT_REMOVAL":
                IP.update(state="IDLE", sid=None, object_present=False,
                          present_frames=0, empty_frames=0, bg=None,
                          changed_object_frames=0, last_object_gray=None)
        with state_lock:
            CAMERA_RESULTS["ip"].update(code=None, info=None, ocr="Chờ đọc...")
            if S["source"] == "ip":
                S["code"], S["info"], S["ocr"] = None, None, "Chờ đọc..."
    elif action == "stop":
        ip_scan_enabled.clear()
        with ip_state_lock:
            if IP["object_present"] or IP["state"] in ("SCANNING", "WAIT_OBJECT_REMOVAL"):
                IP.update(state="WAIT_OBJECT_REMOVAL", sid=None, empty_frames=0)
            else:
                IP.update(state="IDLE", sid=None, present_frames=0, empty_frames=0, bg=None,
                          changed_object_frames=0, last_object_gray=None)
    with ip_state_lock:
        state = IP["state"]
    return jsonify(scanning=ip_scan_enabled.is_set(), state=state)


@app.route("/api/pending")
@login_required
def api_pending():
    p = get_pending()
    if not p:
        return jsonify(id=None)
    conn = db()
    brands = [r[0] for r in conn.execute("SELECT DISTINCT brand FROM warehouse WHERE brand<>'' ORDER BY brand")]
    locs = [r[0] for r in conn.execute("SELECT DISTINCT location FROM warehouse WHERE location<>'' ORDER BY location")]
    conn.close()
    return jsonify(id=p["id"], image=url_for("captures", filename=p["file"]), w=p["w"], h=p["h"],
                   dets=p["dets"], candidates=p["candidates"], barcodes=p["barcodes"],
                   brands=brands, locations=locs)


@app.route("/api/pending/skip", methods=["POST"])
@login_required
def api_pending_skip():
    with state_lock:
        source = S["source"]
    p = get_pending(source)
    set_pending(source, None)
    if p:
        until = time.time() + IGNORE_SEC
        for c in p["candidates"][:5]:
            ignore_until[(source, norm(c))] = until
    if source == "mobile":
        mobile_pending_done()
    return jsonify(ok=True)


@app.route("/api/products", methods=["GET"])
@login_required
def api_products_list():
    conn = db()
    rows = conn.execute("SELECT sku, brand, series, product_name, location, quantity, status, image_path "
                        "FROM warehouse ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/products", methods=["POST"])
@login_required
def api_products_add():
    d = request.get_json(silent=True) or {}
    sku = str(d.get("sku", "")).strip().upper()
    if not sku:
        return jsonify(message="SKU là bắt buộc."), 400
    try:
        qty = int(str(d.get("qty", "0")).strip() or 0)
    except ValueError:
        return jsonify(message="Số lượng phải là số nguyên."), 400

    brand = str(d.get("brand", "")).strip().upper()
    series = str(d.get("series", "")).strip().upper()
    name = str(d.get("name", "")).strip()
    loc_in = str(d.get("location", "")).strip()

    conn = db()
    rows = conn.execute("SELECT sku FROM warehouse").fetchall()
    if any(norm(r["sku"]) == norm(sku) for r in rows):
        conn.close()
        return jsonify(message=f"SKU {sku} đã tồn tại trong kho."), 409
    if not d.get("force"):
        similar = [r["sku"] for r in rows
                   if SequenceMatcher(None, norm(sku), norm(r["sku"])).ratio() >= DUP_WARN_RATIO]
        if similar:
            conn.close()
            return jsonify(needs_confirm=True,
                           message="Mã này rất giống: " + ", ".join(similar) + " (có thể OCR đọc sai)."), 409

    with state_lock:
        source = S["source"]
    p = get_pending(source)
    image_path = p["file"] if p and p["id"] == d.get("id") else ""
    status = "OK" if (name and loc_in) else "PENDING"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn.execute('''INSERT INTO warehouse
            (sku, brand, series, product_name, location, quantity, updated_at, image_path, status)
            VALUES (?,?,?,?,?,?,?,?,?)''',
                     (sku, brand, series, name or "(Chưa đặt tên)", loc_in or "CHUA_CO",
                      qty, now, image_path, status))
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify(message="SKU đã tồn tại."), 409
    finally:
        conn.close()

    invalidate_rows()                      # cập nhật cache kho ngay sau khi thêm
    cleared = False
    with state_lock:
        if p and p["id"] == d.get("id"):
            S["pending_by_source"][source] = None
            if S["source"] == source:
                S["pending"] = None
            cleared = True
        S["last_sku"], S["last_time"] = sku, time.time()
    if cleared and source == "mobile":
        mobile_pending_done()
    msg = "Đã lưu." if status == "OK" else "Đã lưu (chưa đủ thông tin, trạng thái PENDING)."
    return jsonify(message=msg)


# ============================================================
# 7. KHỞI CHẠY
# ============================================================
if __name__ == "__main__":
    if USE_IP_CAMERA:
        threading.Thread(target=camera_worker, daemon=True).start()
        threading.Thread(target=stream_worker, daemon=True).start()
        threading.Thread(target=ocr_worker, daemon=True).start()
    else:
        with state_lock:
            S["camera"] = "Camera IP đang tắt (demo)"
        print("Camera IP đang tắt. Có thể dùng Camera điện thoại trong trình duyệt.")

    threading.Thread(target=mobile_worker, daemon=True).start()

    print(f"Web server: http://127.0.0.1:{WEB_PORT}")
    print(f"Tài khoản demo: {WEB_USER}")
    if GENERATED_WEB_PASS:
        print(f"Mật khẩu demo tạm thời (chỉ hiển thị trong terminal): {WEB_PASS}")
        print("Đặt biến môi trường WEB_PASS để dùng mật khẩu riêng.")
    try:
        app.run(host=WEB_HOST, port=WEB_PORT, threaded=True, use_reloader=False)
    finally:
        stop_event.set()
        ip_scan_enabled.clear()
        mobile_scan_enabled.clear()
        with mobile_cv:
            mobile_cv.notify_all()