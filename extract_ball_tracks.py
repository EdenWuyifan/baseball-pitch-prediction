from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
from baseballcv.functions import LoadTools
from ultralytics import YOLO


CLASS_ID_TO_NAME: dict[int, str] = {
    0: "glove",
    1: "homeplate",
    2: "baseball",
    3: "rubber",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the YOLOv9 advanced_ball_tracking detector on each trimmed video and "
            "store the baseball bounding boxes (XYXY) in a CSV."
        )
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/test"),
        help="Directory containing the trimmed videos to analyse.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ball_detections_test.csv"),
        help="Path to the CSV file that will store the detections.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.15,
        help="Minimum YOLO confidence to keep a detection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device for YOLO (e.g. 'cpu', 'cuda', '0', '0,1').",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit on the number of videos to process (useful for smoke tests).",
    )
    return parser.parse_args()


def collect_ball_detections(
    model,
    video_paths: List[Path],
    confidence_threshold: float,
    device: str,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    total = len(video_paths)

    for idx, video_path in enumerate(video_paths, start=1):
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"[{idx}/{total}] Processing {video_path.name}")

        try:
            results = model.predict(
                source=str(video_path),
                conf=confidence_threshold,
                device=device,
                save=True,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"Failed on {video_path.name}: {exc}", file=sys.stderr)
            continue

        for frame_idx, frame in enumerate(results):
            boxes = getattr(frame, "boxes", None)
            if boxes is None:
                continue

            xyxy = boxes.xyxy
            if xyxy is None or xyxy.numel() == 0:
                continue

            xyxy = xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()

            for det_idx, (coords, cls_id, score) in enumerate(zip(xyxy, cls, conf)):
                cls_int = int(cls_id)

                records.append(
                    {
                        "file_name": video_path.name,
                        "frame_index": frame_idx,
                        "detection_index": det_idx,
                        "x_min": float(coords[0]),
                        "y_min": float(coords[1]),
                        "x_max": float(coords[2]),
                        "y_max": float(coords[3]),
                        "class_id": cls_int,
                        "class_name": CLASS_ID_TO_NAME.get(cls_int, str(cls_int)),
                        "confidence": float(score),
                    }
                )

    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()

    if not args.video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")

    video_paths = sorted(args.video_dir.glob("*.mp4"))
    if args.max_videos is not None:
        video_paths = video_paths[: args.max_videos]

    if not video_paths:
        raise RuntimeError(f"No mp4 files found in {args.video_dir}")

    load_tools = LoadTools()
    model = YOLO(load_tools.load_model("ball_tracking"))

    detections_df = collect_ball_detections(
        model=model,
        video_paths=video_paths,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    detections_df.to_csv(args.output, index=False)
    print(f"Saved {len(detections_df)} baseball detections to {args.output}")


if __name__ == "__main__":
    main()
