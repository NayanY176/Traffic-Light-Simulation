#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess import CLASS_COLORS_BGR

W, H = 160, 240
SCALE = 4
ROAD_X0, ROAD_X1 = 56, 104
LANE_CX = (ROAD_X0 + ROAD_X1) // 2
STOP_Y = 118
INTERSECTION_Y0, INTERSECTION_Y1 = 100, 140
COMMIT_Y = STOP_Y + 2
CAR_W, CAR_H = 14, 22
MAX_SPEED = 1.35
ACCEL = 0.045
BRAKE = 0.08
CAM_W, CAM_H = 320, 180
VALID = ("red", "yellow", "green")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pixel demo driven by YOLO traffic-light detector")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--cams", default="assets/demo_cam")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--confirm", type=float, default=0.15, help="Min score to confirm user-selected light")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--record", default=None)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--auto",
        default="",
        help="Scripted user lights for recording, e.g. green:4,yellow:2,red:4,green:4",
    )
    return p.parse_args()


def put_text(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.45) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def load_cams(folder: Path) -> dict[str, np.ndarray]:
    cams: dict[str, np.ndarray] = {}
    for name in VALID:
        path = folder / f"{name}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Missing camera feed: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read {path}")
        cams[name] = img
    return cams


def class_scores(result) -> dict[str, float]:
    scores = {k: 0.0 for k in VALID}
    if result is None or result.boxes is None:
        return scores
    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        if name not in scores:
            continue
        score = float(box.conf[0])
        if score > scores[name]:
            scores[name] = score
    return scores


def resolve_detected(user_light: str, scores: dict[str, float], confirm: float) -> tuple[str | None, float]:
    if scores.get(user_light, 0.0) >= confirm:
        return user_light, scores[user_light]
    best_name = None
    best_conf = 0.0
    for name, score in scores.items():
        if score > best_conf:
            best_name, best_conf = name, score
    if best_conf <= 0:
        return None, 0.0
    return best_name, best_conf


def detect_light(model: YOLO, frame: np.ndarray, conf: float, imgsz: int, device: str):
    return model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]


def draw_cam(feed: np.ndarray, result, title: str) -> np.ndarray:
    view = cv2.resize(feed, (CAM_W, CAM_H))
    if result is not None and result.boxes is not None:
        sx = CAM_W / feed.shape[1]
        sy = CAM_H / feed.shape[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            name = result.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            p1 = (int(x1 * sx), int(y1 * sy))
            p2 = (int(x2 * sx), int(y2 * sy))
            color = CLASS_COLORS_BGR.get(cls_id, (255, 180, 0))
            cv2.rectangle(view, p1, p2, color, 2)
            put_text(view, f"{name} {score:.2f}", (p1[0], max(14, p1[1] - 4)), color, 0.4)
    put_text(view, title, (8, 18), (240, 240, 240), 0.45)
    return view


def draw_world(car_y: float, user_light: str) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (34, 92, 42)
    img[0:36, :] = (210, 170, 120)
    for bx in range(8, W, 28):
        cv2.rectangle(img, (bx, 6), (bx + 16, 28), (90, 70, 55), -1)
        cv2.rectangle(img, (bx + 2, 10), (bx + 6, 16), (180, 210, 230), -1)

    img[:, ROAD_X0:ROAD_X1] = (48, 48, 52)
    img[INTERSECTION_Y0:INTERSECTION_Y1, :] = (48, 48, 52)

    for y in range(0, H, 12):
        if INTERSECTION_Y0 - 4 <= y <= INTERSECTION_Y1:
            continue
        cv2.rectangle(img, (LANE_CX - 1, y), (LANE_CX + 1, y + 6), (200, 200, 200), -1)
    for x in range(0, W, 12):
        if ROAD_X0 - 2 <= x <= ROAD_X1 + 2:
            continue
        mid = (INTERSECTION_Y0 + INTERSECTION_Y1) // 2
        cv2.rectangle(img, (x, mid - 1), (x + 6, mid + 1), (200, 200, 200), -1)

    cv2.rectangle(img, (ROAD_X0 + 2, STOP_Y), (ROAD_X1 - 2, STOP_Y + 2), (240, 240, 240), -1)

    pole_x = ROAD_X1 + 10
    cv2.rectangle(img, (pole_x, 48), (pole_x + 3, 108), (30, 30, 30), -1)
    cv2.rectangle(img, (pole_x - 5, 42), (pole_x + 9, 78), (20, 20, 20), -1)
    for name, ly in (("red", 46), ("yellow", 56), ("green", 66)):
        on = name == user_light
        if name == "red":
            color = (0, 0, 255) if on else (40, 40, 80)
        elif name == "yellow":
            color = (0, 180, 255) if on else (40, 60, 70)
        else:
            color = (0, 220, 60) if on else (30, 70, 40)
        cv2.rectangle(img, (pole_x - 3, ly), (pole_x + 7, ly + 8), color, -1)

    cx = LANE_CX
    cy = int(round(car_y))
    x0, y0 = cx - CAR_W // 2, cy - CAR_H // 2
    cv2.rectangle(img, (x0, y0), (x0 + CAR_W, y0 + CAR_H), (30, 60, 210), -1)
    cv2.rectangle(img, (x0 + 2, y0 + 3), (x0 + CAR_W - 2, y0 + 9), (160, 210, 230), -1)
    cv2.rectangle(img, (x0 + 2, y0 + 14), (x0 + CAR_W - 2, y0 + 18), (20, 40, 140), -1)
    return img


def decide(state: str | None, car_y: float, car_v: float) -> tuple[float, str]:
    front = car_y - CAR_H / 2
    at_intersection = front <= COMMIT_Y
    past_stop = front <= STOP_Y + 1
    if state is None:
        return max(0.0, car_v - BRAKE), "NO DETECT"
    if state == "green":
        return min(MAX_SPEED, car_v + ACCEL), "DRIVE"
    if state == "yellow":
        if at_intersection or past_stop:
            return min(MAX_SPEED, car_v + ACCEL), "KEEP GOING"
        return max(0.0, car_v - BRAKE), "STOP BEFORE LINE"
    if past_stop:
        return min(MAX_SPEED, max(car_v, 0.4)), "CLEARING"
    if front > STOP_Y + 1:
        nxt = max(0.0, car_v - BRAKE)
        return nxt, "STOP" if nxt < 0.05 else "BRAKE"
    return 0.0, "STOP"


def parse_auto(spec: str) -> list[tuple[str, float]]:
    if not spec.strip():
        return []
    out: list[tuple[str, float]] = []
    for part in spec.split(","):
        name, secs = part.strip().split(":")
        name = name.strip().lower()
        if name not in VALID:
            raise ValueError(f"Bad auto light: {name}")
        out.append((name, float(secs)))
    return out


def compose(
    world: np.ndarray,
    cam: np.ndarray,
    user_light: str,
    detected: str | None,
    det_conf: float,
    action: str,
    car_v: float,
    scores: dict[str, float] | None = None,
) -> np.ndarray:
    world_big = cv2.resize(world, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
    panel_h = H * SCALE
    panel = np.zeros((panel_h, CAM_W + 24, 3), dtype=np.uint8)
    panel[:] = (18, 18, 22)
    cam_y = 40
    panel[cam_y : cam_y + CAM_H, 12 : 12 + CAM_W] = cam
    put_text(panel, "DETECTOR CAMERA (YOLO)", (12, 28), (220, 220, 220), 0.5)
    put_text(panel, "Keys: R red  Y yellow  G green  Q quit", (12, cam_y + CAM_H + 28), (180, 180, 180), 0.42)
    put_text(panel, f"USER SET: {user_light.upper()}", (12, cam_y + CAM_H + 52), (240, 240, 240), 0.5)
    det_label = detected.upper() if detected else "NONE"
    put_text(panel, f"DETECTED: {det_label} ({det_conf:.2f})", (12, cam_y + CAM_H + 76), (120, 255, 120), 0.5)
    put_text(panel, f"CAR: {action}   v={car_v:.2f}", (12, cam_y + CAM_H + 100), (180, 220, 255), 0.5)
    if scores:
        score_line = " ".join(f"{k[0].upper()}:{scores.get(k, 0):.2f}" for k in VALID)
        put_text(panel, score_line, (12, cam_y + CAM_H + 124), (200, 200, 200), 0.42)
    return np.hstack([world_big, panel])


def main() -> None:
    args = parse_args()
    cams = load_cams(Path(args.cams))
    model = YOLO(args.weights)
    user_light = "green"
    detected: str | None = None
    det_conf = 0.0
    last_result = None
    car_y = float(H - 20)
    car_v = 0.8
    t0 = time.perf_counter()
    auto = parse_auto(args.auto)
    auto_i = 0
    auto_t0 = 0.0

    writer = None
    out_path = Path(args.record) if args.record else None
    sample = compose(
        draw_world(car_y, user_light),
        draw_cam(cams[user_light], None, "warmup"),
        user_light,
        None,
        0.0,
        "START",
        car_v,
        {k: 0.0 for k in VALID},
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = sample.shape[:2]
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))

    win = "Traffic Light Detector Demo"
    if not args.no_show:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        now = time.perf_counter() - t0
        if auto:
            if auto_i < len(auto):
                name, hold = auto[auto_i]
                user_light = name
                if now - auto_t0 >= hold:
                    auto_i += 1
                    auto_t0 = now
            elif args.duration <= 0 and args.no_show:
                break

        feed = cams[user_light]
        last_result = detect_light(model, feed, args.conf, args.imgsz, args.device)
        scores = class_scores(last_result)
        detected, det_conf = resolve_detected(user_light, scores, args.confirm)
        car_v, action = decide(detected, car_y, car_v)
        car_y -= car_v
        if car_y < -CAR_H:
            car_y = float(H + CAR_H)
            car_v = 0.9

        world = draw_world(car_y, user_light)
        cam = draw_cam(feed, last_result, f"feed={user_light}")
        frame = compose(world, cam, user_light, detected, det_conf, action, car_v, scores)

        if writer is not None:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("1")):
                user_light = "red"
            elif key in (ord("y"), ord("2")):
                user_light = "yellow"
            elif key in (ord("g"), ord("3")):
                user_light = "green"
        else:
            time.sleep(0.02)

        if args.duration > 0 and now >= args.duration:
            break

    if writer is not None:
        writer.release()
        print(f"Wrote {out_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
