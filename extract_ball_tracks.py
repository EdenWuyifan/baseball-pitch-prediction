from __future__ import annotations

import argparse
import sys
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

import pandas as pd  # type: ignore
from baseballcv.functions import LoadTools  # type: ignore
from baseballcv.model import YOLOv9  # type: ignore
from ultralytics import YOLO  # type: ignore


DEFAULT_IOU_THRESHOLD = 0.45


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    class_map: Optional[dict[int, str]] = None
    allowed_ids: Optional[set[int]] = None


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "ball_tracking": ModelConfig(
        backend="yolov8",
        class_map={
            15: "homeplate",
            16: "baseball",
            17: "rubber",
        },
    ),
    "ball_trackingv4": ModelConfig(
        backend="yolov8",
        class_map={2: "baseball"},
        allowed_ids={2},
    ),
    "homeplate_tracking": ModelConfig(
        backend="yolov9",
        class_map={0: "homeplate"},
    ),
}

SUPPORTED_MODELS = tuple(MODEL_REGISTRY.keys())

CollectorFn = Callable[
    [Any, List[Path], float, str, ModelConfig],
    pd.DataFrame,
]

YOLOV9_PICKLE_GLOBALS: tuple[str, ...] = (
    "models.common.ADown",
    "models.common.CBFuse",
    "models.common.CBLinear",
    "models.common.Concat",
    "models.common.Conv",
    "models.common.DFL",
    "models.common.RepConvN",
    "models.common.RepNBottleneck",
    "models.common.RepNCSP",
    "models.common.RepNCSPELAN4",
    "models.common.Silence",
    "models.common.SP",
    "models.common.SPPELAN",
    "models.yolo.DetectionModel",
    "models.yolo.DualDDetect",
    "torch.nn.modules.activation.SiLU",
    "torch.nn.modules.batchnorm.BatchNorm2d",
    "torch.nn.modules.container.ModuleList",
    "torch.nn.modules.container.Sequential",
    "torch.nn.modules.conv.Conv2d",
    "torch.nn.modules.linear.Identity",
    "torch.nn.modules.pooling.MaxPool2d",
    "torch.nn.modules.upsampling.Upsample",
)


_YOLOV9_ROOT_ADDED = False


def _import_attr(module_path: str, attribute: str) -> Any | None:
    try:
        module = import_module(module_path)
    except Exception:
        return None
    return getattr(module, attribute, None)


def _ensure_yolov9_on_sys_path() -> None:
    global _YOLOV9_ROOT_ADDED
    if _YOLOV9_ROOT_ADDED:
        return

    try:
        import yolov9  # type: ignore
    except Exception:
        return

    root = Path(getattr(yolov9, "__file__", "")).resolve().parent
    root_str = str(root)
    if root_str and root_str not in sys.path:
        sys.path.append(root_str)

    _YOLOV9_ROOT_ADDED = True


def _resolve_safe_global(global_name: str) -> tuple[Any, str] | None:
    module_name, _, attr = global_name.rpartition(".")
    if not module_name:
        return None

    obj: Any | None = None
    if module_name.startswith("models."):
        _ensure_yolov9_on_sys_path()
        obj = _import_attr(module_name, attr)
        if obj is None:
            obj = _import_attr(f"yolov9.{module_name}", attr)
    else:
        obj = _import_attr(module_name, attr)

    if obj is None:
        return None

    return obj, global_name


def append_yolov9_prediction_records(
    predictions: Sequence[Mapping[str, Any]],
    video_name: str,
    allowed_class_map: Mapping[int, str],
    allowed_ids: Optional[set[int]],
    model_names: Mapping[int, str] | Sequence[str],
    records: list[dict[str, float | int | str]],
) -> None:
    frame_counts: dict[int, int] = {}

    for idx, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            continue

        cls_value = prediction.get("class_id", prediction.get("class"))
        if cls_value is None:
            continue

        cls_int = int(cls_value)
        if allowed_ids is not None and cls_int not in allowed_ids:
            continue

        x1 = float(prediction.get("x", prediction.get("x_min", 0.0)))
        y1 = float(prediction.get("y", prediction.get("y_min", 0.0)))

        width = prediction.get("width")
        height = prediction.get("height")

        if width is None:
            x2 = float(prediction.get("x_max", x1))
        else:
            x2 = x1 + float(width)

        if height is None:
            y2 = float(prediction.get("y_max", y1))
        else:
            y2 = y1 + float(height)

        frame_idx = int(prediction.get("frame_index", idx))
        det_idx = (
            int(prediction["detection_index"])
            if "detection_index" in prediction
            else frame_counts.get(frame_idx, 0)
        )
        frame_counts[frame_idx] = det_idx + 1

        score = float(prediction.get("confidence", prediction.get("score", 0.0)))

        records.append(
            {
                "file_name": video_name,
                "frame_index": frame_idx,
                "detection_index": det_idx,
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
                "class_id": cls_int,
                "class_name": resolve_class_name(
                    cls_int, allowed_class_map, model_names
                ),
                "confidence": score,
            }
        )


def register_yolov9_safe_globals() -> None:
    try:
        from torch.serialization import add_safe_globals
    except (ImportError, AttributeError):  # pragma: no cover - torch availability
        return

    safe_globals: list[tuple[Any, str]] = []
    for global_name in YOLOV9_PICKLE_GLOBALS:
        resolved = _resolve_safe_global(global_name)
        if resolved is not None:
            safe_globals.append(resolved)

    if not safe_globals:
        return

    try:
        add_safe_globals(safe_globals)
    except Exception:
        pass


def extract_model_names(model: Any) -> Mapping[int, str] | Sequence[str]:
    names = getattr(model, "names", None)
    if names:
        return names

    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        inner_names = getattr(inner_model, "names", None)
        if inner_names:
            return inner_names

    return {}


def resolve_class_name(
    cls_id: int,
    class_map: Optional[dict[int, str]],
    model_names: Mapping[int, str] | Sequence[str],
) -> str:
    if class_map and cls_id in class_map:
        return class_map[cls_id]

    if isinstance(model_names, Mapping) and cls_id in model_names:
        return str(model_names[cls_id])

    if (
        isinstance(model_names, Sequence)
        and not isinstance(model_names, (str, bytes))
        and 0 <= cls_id < len(model_names)
    ):
        return str(model_names[cls_id])

    return str(cls_id)


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        data = value.tolist()
        if isinstance(data, list):
            return data
        return [data]
    return [value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the YOLO ball_tracking detector on each trimmed video and "
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
        default=0.25,
        help="Minimum YOLO confidence to keep a detection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device for YOLO (e.g. 'cpu', 'cuda', '0', '0,1').",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit on the number of videos to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default="ball_tracking",
        choices=SUPPORTED_MODELS,
        help="Registered model alias to load via LoadTools.",
    )
    return parser.parse_args()


def collect_yolov8_detections(
    model,
    video_paths: List[Path],
    confidence_threshold: float,
    device: str,
    config: ModelConfig,
) -> pd.DataFrame:
    allowed_class_map = config.class_map or {}
    allowed_ids = config.allowed_ids
    model_names = extract_model_names(model)
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
        except Exception as exc:  # pragma: no cover
            print(f"Failed on {video_path.name}: {exc}", file=sys.stderr)
            continue

        for frame_idx, frame in enumerate(results):
            boxes = getattr(frame, "boxes", None)
            if boxes is None:
                continue

            xyxy = boxes.xyxy
            if xyxy is None or xyxy.numel() == 0:
                continue

            xyxy_np = xyxy.cpu().numpy()
            cls_np = boxes.cls.cpu().numpy()
            conf_np = boxes.conf.cpu().numpy()

            for det_idx, (coords, cls_id, score) in enumerate(
                zip(xyxy_np, cls_np, conf_np)
            ):
                cls_int = int(cls_id)
                if allowed_ids is not None and cls_int not in allowed_ids:
                    continue

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
                        "class_name": resolve_class_name(
                            cls_int, allowed_class_map, model_names
                        ),
                        "confidence": float(score),
                    }
                )

    return pd.DataFrame.from_records(records)


def collect_yolov9_detections(
    model,
    video_paths: List[Path],
    confidence_threshold: float,
    device: str,
    config: ModelConfig,
) -> pd.DataFrame:
    _ = device  # Device is handled at model construction time for YOLOv9.
    allowed_class_map = config.class_map or {}
    allowed_ids = config.allowed_ids
    model_names = extract_model_names(model)
    records: list[dict[str, float | int | str]] = []
    total = len(video_paths)

    for idx, video_path in enumerate(video_paths, start=1):
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"[{idx}/{total}] Processing {video_path.name}")

        try:
            results = model.inference(
                source=str(video_path),
                conf_thres=confidence_threshold,
                iou_thres=DEFAULT_IOU_THRESHOLD,
                vid_stride=1,
                nosave=True,
                save_txt=False,
                save_conf=False,
                save_crop=False,
            )
        except Exception as exc:  # pragma: no cover
            print(f"Failed on {video_path.name}: {exc}", file=sys.stderr)
            continue

        if results is None:
            continue

        if isinstance(results, Mapping) and "predictions" in results:
            predictions = ensure_list(results.get("predictions"))
            append_yolov9_prediction_records(
                predictions,
                video_path.name,
                allowed_class_map,
                allowed_ids,
                model_names,
                records,
            )
            continue

        for frame_idx, detection in enumerate(results):
            boxes = detection.get("boxes")
            scores = detection.get("scores") or detection.get("conf") or detection.get("confidences")
            labels = detection.get("classes") or detection.get("labels") or detection.get("class_ids")

            if boxes is None or scores is None or labels is None:
                continue

            boxes_seq = ensure_list(boxes)
            scores_seq = ensure_list(scores)
            labels_seq = ensure_list(labels)

            if not boxes_seq or not scores_seq or not labels_seq:
                continue

            for det_idx, (box, score, cls_id) in enumerate(
                zip(boxes_seq, scores_seq, labels_seq)
            ):
                cls_int = int(cls_id)
                if allowed_ids is not None and cls_int not in allowed_ids:
                    continue

                try:
                    x1, y1, x2, y2 = (float(coord) for coord in box)
                except TypeError:
                    continue

                records.append(
                    {
                        "file_name": video_path.name,
                        "frame_index": frame_idx,
                        "detection_index": det_idx,
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "class_id": cls_int,
                        "class_name": resolve_class_name(
                            cls_int, allowed_class_map, model_names
                        ),
                        "confidence": float(score),
                    }
                )

    return pd.DataFrame.from_records(records)


COLLECTORS: dict[str, CollectorFn] = {
    "yolov8": collect_yolov8_detections,
    "yolov9": collect_yolov9_detections,
}


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
    config = MODEL_REGISTRY[args.model_alias]
    model_path = Path(load_tools.load_model(args.model_alias))

    if config.backend == "yolov9":
        register_yolov9_safe_globals()
        # Load YOLOv9 model using alias
        if model_path.is_file():
            model_dir = model_path.parent
            weights_dir = model_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            destination = weights_dir / model_path.name
            if not destination.exists():
                shutil.copy2(model_path, destination)
        else:
            model_dir = model_path

        # YOLOv9 uses different device format than YOLOv8
        yolov9_device = args.device
        if args.device == "cuda":
            yolov9_device = "0"
        elif args.device == "cpu":
            yolov9_device = "cpu"
        # Keep numeric device strings as-is (e.g., "0", "0,1")

        model = YOLOv9(
            device=yolov9_device,
            model_path=str(model_dir),
            name=args.model_alias,
        )
    else:
        model = YOLO(str(model_path))

    collector = COLLECTORS[config.backend]
    detections_df = collector(
        model=model,
        video_paths=video_paths,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        config=config,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    detections_df.to_csv(args.output, index=False)
    print(f"Saved {len(detections_df)} baseball detections to {args.output}")


if __name__ == "__main__":
    main()
