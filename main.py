"""
TASK 4: Object Detection and Tracking
======================================
- Real-time video input via webcam or video file (OpenCV)
- YOLOv8 pre-trained model with built-in ByteTrack tracking
- Bounding boxes drawn on each frame with motion trails
- Live object count panel per class
- CSV logging of every detection
- Video recording (toggle with R)
- Frame skipping for performance on CPU/laptop
- Display with labels and unique tracking IDs in real time

Dependencies (requirements.txt):
    numpy>=1.24
    opencv-python>=4.8
    ultralytics>=8.0

Usage:
    python main.py                            # webcam
    python main.py --source video.mp4         # video file
    python main.py --source 0 --conf 0.5      # custom confidence
    python main.py --skip 2                   # run YOLO every 2nd frame (faster)

Controls:
    Q or ESC  → Quit
    P         → Pause / Resume
    S         → Save screenshot
    R         → Start / Stop video recording
"""

import cv2
import numpy as np
import argparse
import time
import os
import csv
from collections import defaultdict


# ── Colour palette for track IDs ──────────────────────────────────────────────
_PALETTE = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31),
    (255, 178,  29), (207, 210,  49), ( 72, 249, 100),
    ( 14, 173, 156), ( 54,  67, 233), ( 72,  85, 121),
    (241,   3, 190), (  0, 255, 255), (255,   0, 255),
]

def track_color(track_id):
    return _PALETTE[int(track_id) % len(_PALETTE)]


# ── Overlay helpers ────────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, color, thickness=2):
    """Draw bounding box with corner accent lines."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    L = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    t = thickness + 1
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px + dx * L, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy * L), color, t)


def draw_label(frame, text, x, y, color):
    """Draw filled label badge with white text."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 4
    cv2.rectangle(frame, (x, y - th - pad * 2), (x + tw + pad * 2, y), color, -1)
    cv2.putText(frame, text, (x + pad, y - pad),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_hud(frame, fps, n_tracks, paused, recording, h, w):
    """Draw semi-transparent HUD panel in top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 105), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "OBJECT DETECTION + TRACKING", (8, 18),
                font, 0.40, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS    : {fps:5.1f}", (8, 38),
                font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Tracks : {n_tracks:3d}", (8, 57),
                font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Res    : {w}x{h}", (8, 76),
                font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Recording indicator — red when active, grey when off
    rec_color = (0, 0, 255) if recording else (100, 100, 100)
    rec_text  = "REC  " if recording else "REC  "
    cv2.putText(frame, rec_text, (8, 95), font, 0.45, rec_color, 1, cv2.LINE_AA)
    # Dot drawn separately so it can be a filled circle
    dot_color = (0, 0, 255) if recording else (80, 80, 80)
    cv2.circle(frame, (52, 91), 5, dot_color, -1)

    if paused:
        cv2.putText(frame, "[ PAUSED ]", (w // 2 - 55, 30),
                    font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    controls = "Q/ESC=quit  P=pause  S=shot  R=record"
    cv2.putText(frame, controls, (8, h - 8), font, 0.35, (160, 160, 160), 1, cv2.LINE_AA)


def draw_count_panel(frame, class_counts, h, w):
    """Draw per-class object count panel in the top-right corner."""
    if not class_counts:
        return
    font   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 6
    line_h = 20
    panel_w = 165
    panel_h = pad * 2 + line_h * (len(class_counts) + 1)
    x0 = w - panel_w - 8
    y0 = 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "OBJECT COUNTS", (x0 + pad, y0 + pad + line_h - 5),
                font, 0.38, (100, 255, 100), 1, cv2.LINE_AA)

    for i, (cls_name, count) in enumerate(sorted(class_counts.items())):
        y = y0 + pad + line_h * (i + 2) - 4
        cv2.putText(frame, f"{cls_name:<12}: {count:2d}",
                    (x0 + pad, y), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


# ── CSV logger ────────────────────────────────────────────────────────────────
def open_csv(save_dir):
    """Create a timestamped CSV file and write the header row."""
    os.makedirs(save_dir, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"detections_{ts}.csv")
    fh   = open(path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["frame", "timestamp", "track_id", "class",
                     "x1", "y1", "x2", "y2", "confidence"])
    print(f"[INFO] CSV logging  → {path}")
    return fh, writer


# ── Video writer helper ───────────────────────────────────────────────────────
def open_video_writer(save_dir, w, h, fps=20):
    """Open a timestamped .mp4 VideoWriter."""
    os.makedirs(save_dir, exist_ok=True)
    ts     = time.strftime("%Y%m%d_%H%M%S")
    path   = os.path.join(save_dir, f"recording_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw     = cv2.VideoWriter(path, fourcc, fps, (w, h))
    print(f"[INFO] Recording started → {path}")
    return vw, path


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run(source, conf, save_dir, skip_frames):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    print("[INFO] Loading YOLOv8n model ...")
    model = YOLO("yolov8n.pt")   # auto-downloads ~6 MB on first run

    # ── Open video source ─────────────────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source '{source}'.")
        print("  • Webcam busy? Check no other app is using it.")
        print("  • macOS: System Settings → Privacy & Security → Camera → allow Terminal/PyCharm")
        print("  • Video file? Check the path is correct.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce webcam latency

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Opened source : {source}  ({frame_w}x{frame_h})")
    print(f"[INFO] Frame skipping: every {skip_frames + 1} frame(s)")
    print("[INFO] Controls — Q/ESC: quit  |  P: pause  |  S: screenshot  |  R: record")

    # ── CSV always on ─────────────────────────────────────────────────────────
    csv_fh, csv_writer = open_csv(save_dir)

    # ── State variables ───────────────────────────────────────────────────────
    paused       = False
    recording    = False
    video_writer = None
    fps_smooth   = 0.0
    frame_idx    = 0
    trail        = defaultdict(list)   # track_id → [(cx, cy), ...]
    class_counts = {}

    # Cache last YOLO results so skipped frames still draw boxes
    last_boxes = last_ids = last_clses = last_confs = None

    while True:
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ──────────────────────────────────────────────────────
        if key in (ord('q'), 27):
            break

        if key == ord('p'):
            paused = not paused

        if key == ord('r'):
            if not recording:
                video_writer, _ = open_video_writer(save_dir, frame_w, frame_h)
                recording = True
            else:
                video_writer.release()
                video_writer = None
                recording = False
                print("[INFO] Recording stopped.")

        if paused:
            continue

        # ── Read frame ────────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break
        frame_idx += 1
        h, w = frame.shape[:2]

        # ── Frame skipping ────────────────────────────────────────────────────
        # Run YOLO on frame 1, 1+skip+1, 1+2*(skip+1), ...
        # On skipped frames, reuse the cached boxes from the last YOLO run.
        run_yolo = (frame_idx % (skip_frames + 1) == 1)

        if run_yolo:
            t0 = time.time()
            results = model.track(
                frame,
                conf=conf,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )[0]
            elapsed    = time.time() - t0
            fps_smooth = fps_smooth * 0.9 + (1.0 / max(elapsed, 1e-6)) * 0.1

            if results.boxes is not None and results.boxes.id is not None:
                last_boxes = results.boxes.xyxy.cpu().numpy()
                last_ids   = results.boxes.id.cpu().numpy().astype(int)
                last_clses = results.boxes.cls.cpu().numpy().astype(int)
                last_confs = results.boxes.conf.cpu().numpy()
            else:
                last_boxes = last_ids = last_clses = last_confs = None

        boxes  = last_boxes
        ids    = last_ids
        clses  = last_clses
        confs  = last_confs

        # ── Translucent detection fill ────────────────────────────────────────
        if boxes is not None and len(boxes):
            det_overlay = frame.copy()
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(det_overlay,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              (200, 200, 200), -1)
            cv2.addWeighted(det_overlay, 0.08, frame, 0.92, 0, frame)

        # ── Draw tracks, update counts, log CSV ───────────────────────────────
        n_tracks     = 0
        class_counts = {}
        ts_now       = time.strftime("%H:%M:%S")

        if boxes is not None and ids is not None:
            n_tracks = len(ids)

            for (x1, y1, x2, y2), tid, cls, conf_score in zip(boxes, ids, clses, confs):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = model.names[cls]
                color = track_color(tid)

                # --- Count panel ---
                class_counts[label] = class_counts.get(label, 0) + 1

                # --- CSV: write only on YOLO frames to avoid duplicate rows ---
                if run_yolo:
                    csv_writer.writerow([
                        frame_idx, ts_now, int(tid), label,
                        x1, y1, x2, y2, f"{float(conf_score):.3f}"
                    ])

                # --- Motion trail ---
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trail[tid].append((cx, cy))
                if len(trail[tid]) > 30:
                    trail[tid].pop(0)
                pts = trail[tid]
                for i in range(1, len(pts)):
                    alpha_t = i / len(pts)
                    tc = tuple(int(c * alpha_t) for c in color)
                    cv2.line(frame, pts[i - 1], pts[i], tc, 2)

                draw_box(frame, x1, y1, x2, y2, color)
                draw_label(frame, f"#{tid} {label}", x1, y1, color)

        # ── HUD + count panel ─────────────────────────────────────────────────
        draw_hud(frame, fps_smooth, n_tracks, paused, recording, h, w)
        draw_count_panel(frame, class_counts, h, w)

        # ── Screenshot ────────────────────────────────────────────────────────
        if key == ord('s'):
            path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot → {path}")

        # ── Video recording ───────────────────────────────────────────────────
        if recording and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("Object Detection & Tracking (YOLOv8 + ByteTrack)", frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print("[INFO] Recording saved.")
    csv_fh.flush()
    csv_fh.close()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Real-time Object Detection & Tracking — YOLOv8 + ByteTrack")
    ap.add_argument("--source",   default="0",
                    help="Webcam index (0, 1, …) or path to video file")
    ap.add_argument("--conf",     type=float, default=0.40,
                    help="Detection confidence threshold (default: 0.40)")
    ap.add_argument("--save-dir", default="output",
                    help="Folder for screenshots, recordings & CSV (default: output/)")
    ap.add_argument("--skip",     type=int, default=1,
                    help="Run YOLO every N+1 frames (0=every frame, 1=every 2nd, 2=every 3rd)")
    args = ap.parse_args()
    run(args.source, args.conf, args.save_dir, args.skip)