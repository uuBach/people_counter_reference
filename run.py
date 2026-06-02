from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc: 
    raise SystemExit(
        "Ultralytics is not installed. Run: pip install -r requirements.txt"
    ) from exc


Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


@dataclass
class Track:
    track_id: int
    frames: List[int] = field(default_factory=list)
    centers: List[Point] = field(default_factory=list)
    boxes: List[Box] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def add(self, frame_idx: int, center: Point, box: Box, confidence: float) -> None:
        self.frames.append(frame_idx)
        self.centers.append(center)
        self.boxes.append(box)
        self.confidences.append(float(confidence))

    @property
    def first_frame(self) -> int:
        return self.frames[0]

    @property
    def last_frame(self) -> int:
        return self.frames[-1]

    @property
    def hits(self) -> int:
        return len(self.frames)

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return float(np.mean(self.confidences))


def mean_point(points: List[Point]) -> Point:
    arr = np.asarray(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def classify_track(
    track: Track,
    width: int,
    height: int,
    fps: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Return a classification dict for one completed track."""
    start_time_s = track.first_frame / fps
    end_time_s = track.last_frame / fps
    duration_s = end_time_s - start_time_s

    if track.hits < args.min_hits:
        return {
            "track_id": track.track_id,
            "direction": "unknown",
            "start_time_s": round(start_time_s, 3),
            "end_time_s": round(end_time_s, 3),
            "confidence": 0.1,
            "notes": f"too few detections: {track.hits} < {args.min_hits}",
        }

    if duration_s < args.min_track_seconds:
        return {
            "track_id": track.track_id,
            "direction": "unknown",
            "start_time_s": round(start_time_s, 3),
            "end_time_s": round(end_time_s, 3),
            "confidence": 0.1,
            "notes": f"track too short: {duration_s:.2f}s < {args.min_track_seconds:.2f}s",
        }

    k = min(args.endpoint_window, track.hits // 2)
    k = max(1, k)
    start_center = mean_point(track.centers[:k])
    end_center = mean_point(track.centers[-k:])

    dx = (end_center[0] - start_center[0]) / max(width, 1)
    dy = (end_center[1] - start_center[1]) / max(height, 1)
    movement_norm = math.sqrt(dx * dx + dy * dy)

    diagonal_projection = (dx + dy) / math.sqrt(2.0)
    alignment = abs(diagonal_projection) / max(movement_norm, 1e-9)

    start_score = start_center[0] / max(width, 1) + start_center[1] / max(height, 1)
    end_score = end_center[0] / max(width, 1) + end_center[1] / max(height, 1)
    crossed_midline = (start_score < 1.0 < end_score) or (end_score < 1.0 < start_score)

    if movement_norm < args.min_movement_norm:
        return {
            "track_id": track.track_id,
            "direction": "unknown",
            "start_time_s": round(start_time_s, 3),
            "end_time_s": round(end_time_s, 3),
            "confidence": round(clamp(track.mean_confidence * 0.35), 3),
            "notes": f"static/small movement: movement_norm={movement_norm:.3f}",
        }

    if abs(diagonal_projection) < args.min_diagonal_delta:
        return {
            "track_id": track.track_id,
            "direction": "unknown",
            "start_time_s": round(start_time_s, 3),
            "end_time_s": round(end_time_s, 3),
            "confidence": round(clamp(track.mean_confidence * 0.4), 3),
            "notes": f"weak diagonal displacement: projection={diagonal_projection:.3f}",
        }

    if alignment < args.min_alignment:
        return {
            "track_id": track.track_id,
            "direction": "unknown",
            "start_time_s": round(start_time_s, 3),
            "end_time_s": round(end_time_s, 3),
            "confidence": round(clamp(track.mean_confidence * 0.45), 3),
            "notes": f"movement not aligned with entry/exit diagonal: alignment={alignment:.3f}",
        }

    if args.require_midline_crossing and not crossed_midline:
        return {
            "track_id": track.track_id,
            "direction": "unknown",
            "start_time_s": round(start_time_s, 3),
            "end_time_s": round(end_time_s, 3),
            "confidence": round(clamp(track.mean_confidence * 0.5), 3),
            "notes": "did not cross the diagonal midpoint line",
        }

    direction = "entry" if diagonal_projection > 0 else "exit"

    movement_score = clamp(abs(diagonal_projection) / 0.25)
    hit_score = clamp(track.hits / max(args.min_hits * 3.0, 1.0))
    duration_score = clamp(duration_s / 5.0)
    det_score = clamp(track.mean_confidence)
    confidence = (
        0.35 * movement_score
        + 0.25 * clamp(alignment)
        + 0.20 * det_score
        + 0.10 * hit_score
        + 0.10 * duration_score
    )

    notes = (
        f"dx={dx:.3f}; dy={dy:.3f}; projection={diagonal_projection:.3f}; "
        f"alignment={alignment:.3f}; hits={track.hits}; mean_det_conf={track.mean_confidence:.3f}"
    )

    return {
        "track_id": track.track_id,
        "direction": direction,
        "start_time_s": round(start_time_s, 3),
        "end_time_s": round(end_time_s, 3),
        "confidence": round(clamp(confidence), 3),
        "notes": notes,
    }


def mark_duplicate_fragments(
    events: List[Dict[str, object]],
    tracks: Dict[int, Track],
    width: int,
    height: int,
    fps: float,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Conservative post-processing for obvious ID-switch fragments.

    If two accepted tracks have the same direction, the second starts very soon
    after the first ends, and its start point is close to the first end point,
    mark the second as unknown instead of counting it again.
    """
    diag_px = math.hypot(width, height)
    max_gap_frames = int(args.max_duplicate_gap_seconds * fps)
    max_distance_px = args.max_duplicate_distance_norm * diag_px

    by_id = {int(ev["track_id"]): ev for ev in events}
    accepted = [
        ev for ev in events if ev["direction"] in {"entry", "exit"} and int(ev["track_id"]) in tracks
    ]
    accepted.sort(key=lambda ev: tracks[int(ev["track_id"])].first_frame)

    kept_ids: List[int] = []
    for ev in accepted:
        tid = int(ev["track_id"])
        tr = tracks[tid]
        duplicate_of: Optional[int] = None

        for kept_id in kept_ids:
            kept_ev = by_id[kept_id]
            kept_tr = tracks[kept_id]
            if kept_ev["direction"] != ev["direction"]:
                continue

            gap = tr.first_frame - kept_tr.last_frame
            if gap < 0 or gap > max_gap_frames:
                continue

            end_a = kept_tr.centers[-1]
            start_b = tr.centers[0]
            dist_px = math.hypot(start_b[0] - end_a[0], start_b[1] - end_a[1])
            if dist_px <= max_distance_px:
                duplicate_of = kept_id
                break

        if duplicate_of is None:
            kept_ids.append(tid)
        else:
            ev["direction"] = "unknown"
            ev["confidence"] = 0.2
            ev["notes"] = f"possible duplicate/id-switch fragment of track {duplicate_of}; not counted"

    return events


def draw_debug_boxes(
    frame: np.ndarray,
    detections: List[Tuple[int, Box, float]],
) -> np.ndarray:
    canvas = frame.copy()
    for track_id, box, conf in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            canvas,
            f"ID {track_id} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def save_track_visualization(
    output_path: Path,
    tracks: Dict[int, Track],
    events: List[Dict[str, object]],
    width: int,
    height: int,
) -> None:
    scale = min(1.0, 1400.0 / max(width, height))
    canvas_w = max(1, int(width * scale))
    canvas_h = max(1, int(height * scale))
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    event_by_id = {int(ev["track_id"]): ev for ev in events}
    for tid, track in tracks.items():
        if track.hits < 2:
            continue
        ev = event_by_id.get(tid, {"direction": "unknown"})
        direction = ev["direction"]
        if direction == "entry":
            color = (40, 160, 40)
        elif direction == "exit":
            color = (40, 40, 220)
        else:
            color = (150, 150, 150)

        pts = [(int(x * scale), int(y * scale)) for x, y in track.centers]
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, p1, p2, color, 2)
        cv2.arrowedLine(canvas, pts[0], pts[-1], color, 3, tipLength=0.08)
        cv2.putText(
            canvas,
            str(tid),
            pts[-1],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def write_events_csv(events: List[Dict[str, object]], path: Path) -> None:
    fieldnames = ["track_id", "direction", "start_time_s", "end_time_s", "confidence", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in sorted(events, key=lambda x: (float(x["start_time_s"]), int(x["track_id"]))):
            writer.writerow({key: ev.get(key, "") for key in fieldnames})


def write_summary_json(
    events: List[Dict[str, object]],
    path: Path,
    video_path: Path,
    args: argparse.Namespace,
    processed_frames: int,
    total_frames: int,
) -> None:
    entries = sum(1 for ev in events if ev["direction"] == "entry")
    exits = sum(1 for ev in events if ev["direction"] == "exit")
    unknown = sum(1 for ev in events if ev["direction"] == "unknown")

    summary = {
        "video": video_path.name,
        "entries": entries,
        "exits": exits,
        "unknown_tracks": unknown,
        "frame_sample_rate": args.frame_sample_rate,
        "processed_frames": processed_frames,
        "total_video_frames": total_frames,
        "notes": (
            "YOLO person detector + ByteTrack. Direction is inferred from normalized "
            "track displacement along the image diagonal: upper-left to lower-right is entry, "
            "the reverse is exit. Short, static, weakly diagonal, and possible duplicate "
            "fragments are marked unknown. Counts are heuristic and should be validated on "
            "annotated debug frames."
        ),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count site entries/exits from an entry-camera video.")
    parser.add_argument("--video", default="data/entry_camera.mp4", help="Path to input video")
    parser.add_argument("--output-dir", default="results", help="Directory for summary/events/debug artifacts")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model path/name")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config")
    parser.add_argument("--device", default=None, help="Device for YOLO, e.g. cpu, 0, mps. Default: auto")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="YOLO NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    parser.add_argument(
        "--frame-sample-rate",
        type=int,
        default=5,
        help="Run detection/tracking every Nth frame. Use 1 for maximum tracking quality.",
    )
    parser.add_argument("--limit-frames", type=int, default=0, help="Process only first N frames; 0 means full video")
    parser.add_argument("--min-hits", type=int, default=5, help="Minimum detections required for a countable track")
    parser.add_argument("--min-track-seconds", type=float, default=1.0, help="Minimum track duration in seconds")
    parser.add_argument("--endpoint-window", type=int, default=5, help="Number of first/last centers averaged for direction")
    parser.add_argument(
        "--min-movement-norm",
        type=float,
        default=0.06,
        help="Minimum normalized Euclidean displacement relative to image dimensions",
    )
    parser.add_argument(
        "--min-diagonal-delta",
        type=float,
        default=0.05,
        help="Minimum normalized displacement along upper-left/lower-right diagonal",
    )
    parser.add_argument(
        "--min-alignment",
        type=float,
        default=0.45,
        help="Minimum alignment with the entry/exit diagonal; 1.0 is perfect diagonal movement",
    )
    parser.add_argument(
        "--require-midline-crossing",
        action="store_true",
        help="Only count tracks that cross the diagonal midpoint line. Stricter but can miss partial tracks.",
    )
    parser.add_argument(
        "--max-duplicate-gap-seconds",
        type=float,
        default=2.0,
        help="Max temporal gap for treating a later track as an ID-switch duplicate fragment",
    )
    parser.add_argument(
        "--max-duplicate-distance-norm",
        type=float,
        default=0.08,
        help="Max spatial gap as fraction of image diagonal for duplicate-fragment suppression",
    )
    parser.add_argument("--save-debug", action="store_true", help="Save selected annotated frames and track map")
    parser.add_argument("--debug-every-seconds", type=float, default=20.0, help="Save one debug frame every N seconds")
    parser.add_argument("--max-debug-frames", type=int, default=120, help="Maximum annotated debug frames to save")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_sample_rate < 1:
        raise SystemExit("--frame-sample-rate must be >= 1")

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}. Expected default: data/entry_camera.mp4")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}; FPS: {fps:.3f}; Frames: {total_frames}")
    print(f"Loading model: {args.model}")

    model = YOLO(args.model)

    tracks: Dict[int, Track] = {}
    processed_frames = 0
    frame_idx = 0
    saved_debug = 0
    next_debug_frame = 0
    debug_step = max(1, int(args.debug_every_seconds * fps))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.limit_frames and frame_idx >= args.limit_frames:
            break

        if frame_idx % args.frame_sample_rate != 0:
            frame_idx += 1
            continue

        result = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            classes=[0],  
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        detections_for_debug: List[Tuple[int, Box, float]] = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().numpy()

            for box_arr, track_id, conf in zip(xyxy, ids, confs):
                x1, y1, x2, y2 = [float(v) for v in box_arr]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                box = (x1, y1, x2, y2)
                track = tracks.setdefault(int(track_id), Track(track_id=int(track_id)))
                track.add(frame_idx, (cx, cy), box, float(conf))
                detections_for_debug.append((int(track_id), box, float(conf)))

        if (
            args.save_debug
            and saved_debug < args.max_debug_frames
            and frame_idx >= next_debug_frame
        ):
            annotated = draw_debug_boxes(frame, detections_for_debug)
            cv2.imwrite(str(debug_dir / f"frame_{frame_idx:06d}.jpg"), annotated)
            saved_debug += 1
            next_debug_frame = frame_idx + debug_step

        processed_frames += 1
        if processed_frames % 100 == 0:
            print(f"Processed sampled frames: {processed_frames}; video frame: {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()

    print(f"Classifying {len(tracks)} tracks...")
    events = [classify_track(track, width, height, fps, args) for track in tracks.values()]
    events = mark_duplicate_fragments(events, tracks, width, height, fps, args)

    write_events_csv(events, output_dir / "events.csv")
    write_summary_json(events, output_dir / "summary.json", video_path, args, processed_frames, total_frames)

    if args.save_debug:
        save_track_visualization(output_dir / "tracks_map.jpg", tracks, events, width, height)

    entries = sum(1 for ev in events if ev["direction"] == "entry")
    exits = sum(1 for ev in events if ev["direction"] == "exit")
    unknown = sum(1 for ev in events if ev["direction"] == "unknown")

    print("Done.")
    print(f"entries={entries}; exits={exits}; unknown_tracks={unknown}")
    print(f"Wrote: {output_dir / 'summary.json'}")
    print(f"Wrote: {output_dir / 'events.csv'}")
    if args.save_debug:
        print(f"Wrote debug frames to: {debug_dir}")
        print(f"Wrote: {output_dir / 'tracks_map.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
