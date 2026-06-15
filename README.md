# Object Detection & Tracking

Real-time object detection and tracking using **YOLOv8 + ByteTrack**.

## Features
- Live webcam or video file input
- Unique tracking IDs with motion trails
- Per-class object count overlay
- CSV logging of all detections
- Screenshot & video recording support
- Frame skipping for CPU performance

## Installation

```bash
pip install numpy opencv-python ultralytics
```

## Usage

```bash
python main.py                          # webcam
python main.py --source video.mp4       # video file
python main.py --conf 0.5 --skip 2     # custom settings
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index or video file path |
| `--conf` | `0.40` | Confidence threshold |
| `--save-dir` | `output` | Output folder |
| `--skip` | `1` | Run YOLO every N+1 frames |

## Controls

| Key | Action |
|---|---|
| `Q / ESC` | Quit |
| `P` | Pause / Resume |
| `S` | Screenshot |
| `R` | Start / Stop recording |

## Output
All files saved to `output/` — CSV logs, MP4 recordings, and screenshots.

---
**CodeAlpha Internship — Task 4** | Arshbala khan, UET Peshawar
