import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from model_for_getting_hidden_para.droplet_gnn import DropletGNN


def resolve_ckpt_path(ckpt_path):
    if ckpt_path.exists():
        return ckpt_path

    # Common fallback when running from droplet root.
    alt = Path("model_for_getting_hidden_para") / ckpt_path
    if alt.exists():
        return alt

    return ckpt_path


def detect_object_center(frame):
    """Detect the most prominent blob center in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    candidates = []
    for mask in (th1, th2):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 20:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = 4.0 * np.pi * area / (peri * peri)
            M = cv2.moments(c)
            if abs(M["m00"]) < 1e-6:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            score = area * (0.4 + circularity)
            candidates.append((score, (cx, cy), area))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def init_kalman(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
    )
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
    return kf


def choose_initial_point(first_frame, mode, init_x=None, init_y=None):
    if mode == "point":
        if init_x is None or init_y is None:
            raise RuntimeError("For --init point, both --init-x and --init-y are required.")
        return (float(init_x), float(init_y))

    if mode == "auto":
        p = detect_object_center(first_frame)
        if p is not None:
            return p
        raise RuntimeError(
            "Auto initialization failed to detect object. Re-run with --init point and coordinates."
        )

    try:
        bbox = cv2.selectROI("Select Object", first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Object")
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI is not available for --init click in this environment. "
            "Use --init point --init-x <x> --init-y <y>."
        ) from exc

    if bbox[2] <= 0 or bbox[3] <= 0:
        raise RuntimeError("No ROI selected. Cannot initialize tracker.")

    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    return (x, y)


def predict_path(kf_state, horizon):
    x, y, vx, vy = kf_state
    pts = []
    for step in range(1, horizon + 1):
        px = x + vx * step
        py = y + vy * step
        pts.append((int(round(px)), int(round(py))))
    return pts


def load_gnn_model(ckpt_path, device):
    model = DropletGNN().to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_path_gnn(model, current_state, horizon, device):
    x, y, vx, vy = current_state
    pts = []

    with torch.no_grad():
        for _ in range(horizon):
            # Single-node graph has no edges; model reduces to node update from node features.
            node_x = torch.tensor([[x, y, vx, vy]], dtype=torch.float32, device=device)
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32, device=device)

            pred_vel = model(node_x, edge_index, edge_attr)[0]
            vx = float(pred_vel[0].item())
            vy = float(pred_vel[1].item())

            x = x + vx
            y = y + vy
            pts.append((int(round(x)), int(round(y))))

    return pts


def render_panel(frame, history, current_state, pred_pts, title):
    panel = frame.copy()

    if len(history) > 1:
        pts_hist = np.array(history, dtype=np.int32)
        cv2.polylines(panel, [pts_hist], False, (0, 255, 0), 2)

    if len(pred_pts) > 1:
        cv2.polylines(
            panel,
            [np.array(pred_pts, dtype=np.int32)],
            False,
            (0, 0, 255),
            2,
        )

    cx, cy = int(round(current_state[0])), int(round(current_state[1]))
    cv2.circle(panel, (cx, cy), 4, (255, 255, 0), -1)

    speed = np.sqrt(current_state[2] ** 2 + current_state[3] ** 2)
    cv2.putText(
        panel,
        f"{title} | speed={speed:.2f} px/frame",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "green: past path  red: predicted path",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return panel


def process_video(
    input_path,
    output_path,
    horizon,
    trail,
    init_mode,
    predictor,
    ckpt_path,
    init_x,
    init_y,
):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model = None
    if predictor in ("gnn", "compare"):
        ckpt_path = resolve_ckpt_path(ckpt_path)
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
        gnn_model = load_gnn_model(ckpt_path, device)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_width = width * 2 if predictor == "compare" else width
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, height),
    )

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise RuntimeError("Video is empty.")

    x0, y0 = choose_initial_point(first_frame, init_mode, init_x=init_x, init_y=init_y)
    kf = init_kalman(x0, y0)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_pt = np.array([[[x0, y0]]], dtype=np.float32)

    history = deque(maxlen=trail)
    history.append((int(round(x0)), int(round(y0))))

    frame_idx = 0

    while True:
        frame = first_frame if frame_idx == 0 else cap.read()[1]
        if frame is None:
            break

        if frame_idx > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_pt,
                None,
                winSize=(31, 31),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )

            measurement = None
            if next_pt is not None and status is not None and status[0][0] == 1:
                mx, my = next_pt[0][0]
                if 0 <= mx < width and 0 <= my < height:
                    measurement = np.array([[mx], [my]], dtype=np.float32)
                    prev_pt = next_pt

            if measurement is None:
                redetect = detect_object_center(frame)
                if redetect is not None:
                    mx, my = redetect
                    measurement = np.array([[mx], [my]], dtype=np.float32)
                    prev_pt = np.array([[[mx, my]]], dtype=np.float32)

            kf.predict()
            if measurement is not None:
                state = kf.correct(measurement)
            else:
                state = kf.statePost

            cx, cy = int(round(float(state[0]))), int(round(float(state[1])))
            history.append((cx, cy))
            prev_gray = gray
        else:
            kf.predict()
            state = kf.correct(np.array([[x0], [y0]], dtype=np.float32))

        current_state = [
            float(kf.statePost[0]),
            float(kf.statePost[1]),
            float(kf.statePost[2]),
            float(kf.statePost[3]),
        ]

        pred_pts_kalman = predict_path(current_state, horizon)
        pred_pts_gnn = []
        if gnn_model is not None:
            pred_pts_gnn = predict_path_gnn(gnn_model, current_state, horizon, device)

        if predictor == "compare":
            left_panel = render_panel(
                frame,
                history,
                current_state,
                pred_pts_kalman,
                "Kalman",
            )
            right_panel = render_panel(
                frame,
                history,
                current_state,
                pred_pts_gnn,
                "GNN",
            )
            cv2.line(left_panel, (width - 1, 0), (width - 1, height - 1), (255, 255, 255), 2)
            out_frame = np.hstack([left_panel, right_panel])

            writer.write(out_frame)
            frame_idx += 1
            continue

        if predictor == "gnn" and gnn_model is not None:
            pred_pts = pred_pts_gnn
        else:
            pred_pts = pred_pts_kalman

        overlay = render_panel(
            frame,
            history,
            current_state,
            pred_pts,
            predictor.upper(),
        )
        cv2.putText(
            overlay,
            f"frame={frame_idx}",
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(overlay)
        frame_idx += 1

    cap.release()
    writer.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay tracked path and predicted future path for one object in a video."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .mp4 video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tracking_prediction_overlay.mp4"),
        help="Output .mp4 path",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Prediction horizon in frames",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=80,
        help="How many past points to keep in path overlay",
    )
    parser.add_argument(
        "--init",
        choices=["auto", "click", "point"],
        default="auto",
        help="Initialization mode: auto-detect center or click ROI",
    )
    parser.add_argument(
        "--init-x",
        type=float,
        default=None,
        help="Initial x coordinate when using --init point",
    )
    parser.add_argument(
        "--init-y",
        type=float,
        default=None,
        help="Initial y coordinate when using --init point",
    )
    parser.add_argument(
        "--predictor",
        choices=["kalman", "gnn", "compare"],
        default="gnn",
        help="Future-path predictor type",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("model_for_getting_hidden_para/droplet_gnn_clean.pt"),
        help="Path to GNN checkpoint (.pt). Used when --predictor gnn",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        args.input,
        args.output,
        args.horizon,
        args.trail,
        args.init,
        args.predictor,
        args.ckpt,
        args.init_x,
        args.init_y,
    )
    print(f"Saved overlay video to: {args.output}")
