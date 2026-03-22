from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("缺少 cv2，請先安裝 opencv-python。") from exc


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".ts",
    ".m4v",
    ".webm",
}
NO_FACE_FOLDER_NAME = "無人臉檔案"
ROTATION_ANGLES = (0, 90, 270)
MAX_SAMPLED_FRAMES_PER_VIDEO = 1800
MAX_CONSECUTIVE_READ_FAILURES = 12
MIN_FACE_SIDE = 120
LARGE_FACE_SIDE = 300
MIN_LOWER_FACE_SKIN_RATIO = 0.30
MEDIUM_FACE_MIN_CONFIDENCE = 0.70
LARGE_FACE_MIN_CONFIDENCE = 0.65
MAX_EYE_ANGLE = 35.0
MAX_NOSE_OFFSET_RATIO = 0.35
MAX_MOUTH_OFFSET_RATIO = 0.45
MAX_VERTICAL_RATIO = 1.45
LATE_FACE_MIN_CONFIDENCE = 0.90
LATE_MASK_VERTICAL_RATIO = 1.22
LATE_MASK_MOUTH_EYE_RATIO = 0.90
LATE_MASK_MOUTH_OFFSET_RATIO = 0.32
LATE_MASK_CENTERED_MOUTH_OFFSET_RATIO = 0.05
LATE_MASK_STRICT_MOUTH_EYE_RATIO = 0.93
LATE_ROTATED_NOSE_OFFSET_RATIO = 0.33
LATE_ROTATED_MOUTH_EYE_RATIO = 0.88
LATE_ROTATED_STRICT_MOUTH_EYE_RATIO = 0.92
EARLY_ROTATED_MIN_FACE_SIDE = 150
EARLY_ROTATED_MAX_MOUTH_EYE_RATIO = 0.98
EARLY_ROTATED_MIN_EYE_ANGLE = 1.0
EARLY_FRONT_LOW_SKIN_RATIO = 0.45
EARLY_FRONT_LOW_CONFIDENCE = 0.80
ANYTIME_FRONT_TALL_REJECT_CONFIDENCE = 0.75
ANYTIME_FRONT_TALL_REJECT_RATIO = 1.28
LATE_ROTATED_SMALL_RESCUE_MIN_SIDE = 120
LATE_ROTATED_SMALL_RESCUE_HIGH_CONFIDENCE = 0.90
LATE_ROTATED_SMALL_RESCUE_NORMAL_MIN_SIDE = 155
LATE_ROTATED_SMALL_RESCUE_MAX_VERTICAL_RATIO = 1.12
LATE_ROTATED_SMALL_RESCUE_MAX_MOUTH_EYE_RATIO = 0.96
LATE_ROTATED_SMALL_RESCUE_MAX_NOSE_OFFSET_RATIO = 0.18
LATE_ROTATED_CENTERED_MAX_OFFSET_RATIO = 0.03
LATE_FRONT_WIDE_ASPECT_RATIO = 1.40
OVERRIDES_PATH = Path(__file__).with_name("move_no_face_videos_overrides.json")
SCAN_LOG_FILE_NAME = "face_scan_log.jsonl"


try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


@dataclass
class VideoScanResult:
    video_path: Path
    has_face: bool
    checked_frames: int
    matched_second: Optional[float] = None
    matched_detector: Optional[str] = None
    moved_to: Optional[Path] = None
    error: Optional[str] = None
    result_source: Optional[str] = None
    debug_hit: Optional["DetectionHit"] = None
    debug_hit_second: Optional[float] = None
    debug_hit_reason: Optional[str] = None


@dataclass
class DetectionHit:
    detector: str
    angle: int
    width: int
    height: int
    confidence: Optional[float] = None
    lower_skin_ratio: Optional[float] = None
    eye_angle: Optional[float] = None
    nose_offset_ratio: Optional[float] = None
    mouth_offset_ratio: Optional[float] = None
    vertical_ratio: Optional[float] = None
    mouth_eye_ratio: Optional[float] = None
    requires_confirmation: bool = False

    @property
    def label(self) -> str:
        suffix = "-weak" if self.requires_confirmation else ""
        return f"{self.detector}{suffix}@{self.angle}"

    def to_dict(self) -> dict[str, object]:
        return {
            "detector": self.detector,
            "angle": self.angle,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "lower_skin_ratio": self.lower_skin_ratio,
            "eye_angle": self.eye_angle,
            "nose_offset_ratio": self.nose_offset_ratio,
            "mouth_offset_ratio": self.mouth_offset_ratio,
            "vertical_ratio": self.vertical_ratio,
            "mouth_eye_ratio": self.mouth_eye_ratio,
            "requires_confirmation": self.requires_confirmation,
        }

    @classmethod
    def from_dict(cls, payload: Optional[dict[str, object]]) -> Optional["DetectionHit"]:
        if not isinstance(payload, dict):
            return None
        detector = str(payload.get("detector") or "").strip()
        if not detector:
            return None
        try:
            return cls(
                detector=detector,
                angle=int(payload.get("angle") or 0),
                width=int(payload.get("width") or 0),
                height=int(payload.get("height") or 0),
                confidence=_optional_float(payload.get("confidence")),
                lower_skin_ratio=_optional_float(payload.get("lower_skin_ratio")),
                eye_angle=_optional_float(payload.get("eye_angle")),
                nose_offset_ratio=_optional_float(payload.get("nose_offset_ratio")),
                mouth_offset_ratio=_optional_float(payload.get("mouth_offset_ratio")),
                vertical_ratio=_optional_float(payload.get("vertical_ratio")),
                mouth_eye_ratio=_optional_float(payload.get("mouth_eye_ratio")),
                requires_confirmation=bool(payload.get("requires_confirmation") or False),
            )
        except Exception:
            return None


@dataclass
class LabelOverrides:
    known_face_files: set[str]
    known_no_face_files: set[str]

    def classify(self, file_name: str) -> Optional[bool]:
        normalized = normalize_file_name(file_name)
        if normalized in self.known_no_face_files:
            return False
        if normalized in self.known_face_files:
            return True
        return None


@dataclass(frozen=True)
class DetectorDeviceSelection:
    requested: str
    actual: str
    description: str
    used_gpu: bool


def select_mtcnn_device(use_gpu: bool) -> DetectorDeviceSelection:
    if not use_gpu:
        return DetectorDeviceSelection(
            requested="CPU",
            actual="CPU:0",
            description="預設使用 CPU (CPU:0)",
            used_gpu=False,
        )

    try:
        import tensorflow as tf
    except ImportError:
        return DetectorDeviceSelection(
            requested="GPU",
            actual="CPU:0",
            description="要求 GPU，但缺少 tensorflow，已改用 CPU (CPU:0)",
            used_gpu=False,
        )

    try:
        physical_gpus = list(tf.config.list_physical_devices("GPU"))
    except Exception:
        physical_gpus = []

    if not physical_gpus:
        built_with_cuda: Optional[bool] = None
        try:
            built_with_cuda = bool(tf.test.is_built_with_cuda())
        except Exception:
            built_with_cuda = None

        reason = "未偵測到 TensorFlow GPU"
        if built_with_cuda is False:
            reason += "，且目前安裝的 TensorFlow 不含 CUDA 支援"

        return DetectorDeviceSelection(
            requested="GPU",
            actual="CPU:0",
            description=f"要求 GPU，但{reason}，已改用 CPU (CPU:0)",
            used_gpu=False,
        )

    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    gpu_name = str(getattr(physical_gpus[0], "name", "GPU:0") or "GPU:0")
    return DetectorDeviceSelection(
        requested="GPU",
        actual="GPU:0",
        description=f"要求 GPU，使用 GPU:0 | TensorFlow 裝置={gpu_name}",
        used_gpu=True,
    )


class LooseFaceDetector:
    def __init__(
        self,
        max_side: int = 1280,
        use_haar_fallback: bool = False,
        use_gpu: bool = False,
    ) -> None:
        self.max_side = max(int(max_side), 320)
        self.use_haar_fallback = bool(use_haar_fallback)
        self.device_selection = select_mtcnn_device(use_gpu)
        self.requested_device = self.device_selection.requested
        self.active_device = self.device_selection.actual
        self.device_description = self.device_selection.description
        self.using_gpu = self.device_selection.used_gpu
        try:
            from mtcnn import MTCNN
        except ImportError as exc:  # pragma: no cover - import guard
            raise SystemExit("缺少 mtcnn，請先安裝 mtcnn。") from exc

        self.mtcnn = MTCNN(device=self.active_device)
        self.haar_frontal = self._load_cascade("haarcascade_frontalface_default.xml")
        self.haar_profile = self._load_cascade("haarcascade_profileface.xml")

    @staticmethod
    def _load_cascade(filename: str) -> Optional[cv2.CascadeClassifier]:
        cascade_path = Path(cv2.data.haarcascades) / filename
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return None
        return cascade

    def frame_has_face(self, frame_bgr) -> Optional[DetectionHit]:
        best_weak_hit: Optional[DetectionHit] = None

        for angle in ROTATION_ANGLES:
            rotated = frame_bgr if angle == 0 else rotate_image(frame_bgr, angle)
            prepared = prepare_for_detection(rotated, self.max_side)

            mtcnn_hit = self._mtcnn_hit(prepared, angle)
            if mtcnn_hit is not None:
                if not mtcnn_hit.requires_confirmation:
                    return mtcnn_hit
                if best_weak_hit is None:
                    best_weak_hit = mtcnn_hit

            if self.use_haar_fallback:
                haar_hit = self._haar_hit(prepared, angle)
                if haar_hit is not None:
                    return haar_hit

        return best_weak_hit

    def _mtcnn_hit(self, frame_bgr, angle: int) -> Optional[DetectionHit]:
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype("uint8", copy=False)
            rgb = rgb.copy(order="C")
            results = self.mtcnn.detect_faces(
                rgb,
                min_face_size=18,
                min_size=12,
                scale_factor=0.6,
                threshold_pnet=0.45,
                threshold_rnet=0.55,
                threshold_onet=0.6,
            )
        except Exception:
            return None

        if not results:
            return None

        for result in results:
            box = result.get("box") or []
            keypoints = result.get("keypoints") or {}
            confidence = float(result.get("confidence") or 0.0)
            if len(box) < 4:
                continue
            if confidence < 0.55:
                continue

            _, _, width, height = box
            width = int(round(float(width)))
            height = int(round(float(height)))
            if width <= 0 or height <= 0:
                continue

            min_side = min(width, height)
            if min_side < MIN_FACE_SIDE:
                continue

            crop = crop_box(frame_bgr, box)
            lower_skin_ratio = lower_face_skin_ratio(crop)
            if lower_skin_ratio < MIN_LOWER_FACE_SKIN_RATIO:
                continue

            geometry = compute_face_geometry(keypoints)
            if geometry is None:
                continue

            requires_confirmation = False
            if min_side >= LARGE_FACE_SIDE:
                if confidence < LARGE_FACE_MIN_CONFIDENCE:
                    continue
            else:
                if confidence < MEDIUM_FACE_MIN_CONFIDENCE:
                    continue
                if geometry["eye_angle"] > MAX_EYE_ANGLE:
                    continue
                if geometry["nose_offset_ratio"] > MAX_NOSE_OFFSET_RATIO:
                    continue
                if geometry["mouth_offset_ratio"] > MAX_MOUTH_OFFSET_RATIO:
                    continue
                if geometry["vertical_ratio"] > MAX_VERTICAL_RATIO:
                    continue

            hit = DetectionHit(
                detector="MTCNN",
                angle=angle,
                width=width,
                height=height,
                confidence=confidence,
                lower_skin_ratio=lower_skin_ratio,
                eye_angle=geometry["eye_angle"],
                nose_offset_ratio=geometry["nose_offset_ratio"],
                mouth_offset_ratio=geometry["mouth_offset_ratio"],
                vertical_ratio=geometry["vertical_ratio"],
                mouth_eye_ratio=geometry["mouth_eye_ratio"],
                requires_confirmation=requires_confirmation,
            )

            return hit

        return None

    def _haar_hit(self, frame_bgr, angle: int) -> Optional[DetectionHit]:
        if self.haar_frontal is None and self.haar_profile is None:
            return None

        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
        except Exception:
            return None

        if self.haar_frontal is not None:
            frontal_faces = self.haar_frontal.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24),
            )
            if frontal_faces is not None and len(frontal_faces) > 0:
                widths = [int(face[2]) for face in frontal_faces]
                heights = [int(face[3]) for face in frontal_faces]
                return DetectionHit(
                    detector="HaarFrontal",
                    angle=angle,
                    width=max(widths),
                    height=max(heights),
                )

        if self.haar_profile is not None:
            profile_faces = self.haar_profile.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24),
            )
            if profile_faces is not None and len(profile_faces) > 0:
                widths = [int(face[2]) for face in profile_faces]
                heights = [int(face[3]) for face in profile_faces]
                return DetectionHit(
                    detector="HaarProfile",
                    angle=angle,
                    width=max(widths),
                    height=max(heights),
                )

            flipped = cv2.flip(gray, 1)
            flipped_faces = self.haar_profile.detectMultiScale(
                flipped,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24),
            )
            if flipped_faces is not None and len(flipped_faces) > 0:
                widths = [int(face[2]) for face in flipped_faces]
                heights = [int(face[3]) for face in flipped_faces]
                return DetectionHit(
                    detector="HaarProfileFlip",
                    angle=angle,
                    width=max(widths),
                    height=max(heights),
                )

        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="掃描影片是否有人臉；整支都沒抓到人臉就搬進『無人臉檔案』資料夾。",
    )
    parser.add_argument("target", nargs="?", help="要掃描的資料夾或單一影片路徑。")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="每隔幾秒抽一張畫面檢查，預設 0.5 秒。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只掃前 N 支影片，測試時可用。",
    )
    parser.add_argument(
        "--top-only",
        action="store_true",
        help="只掃指定資料夾的第一層，不遞迴子資料夾。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示結果，不真的搬檔。",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1280,
        help="偵測前把畫面最大邊縮到多少像素以控制速度，預設 1280。",
    )
    parser.add_argument(
        "--haar-fallback",
        action="store_true",
        help="啟用舊版 Haar 備援；較寬鬆，但比較容易把口罩或非臉輪廓誤判成有人臉。",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="要求 MTCNN 使用 GPU:0；若本機 TensorFlow 沒有可用 GPU，會自動退回 CPU。",
    )
    parser.add_argument(
        "--ignore-overrides",
        action="store_true",
        help="忽略已標記案例，強制用實際偵測重新掃描。",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="忽略既有掃描 log，重新跑偵測並更新紀錄。",
    )
    return parser


def resolve_target_path(raw_target: Optional[str]) -> Path:
    target_text = (raw_target or "").strip()
    if not target_text:
        target_text = input("請輸入要掃描的資料夾或影片路徑：").strip()

    if not target_text:
        raise SystemExit("未提供掃描路徑。")

    target_path = Path(target_text).expanduser().resolve()
    if not target_path.exists():
        raise SystemExit(f"找不到路徑：{target_path}")

    return target_path


def normalize_file_name(file_name: str) -> str:
    return str(file_name or "").strip().lower()


def _optional_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def load_label_overrides(path: Path = OVERRIDES_PATH) -> LabelOverrides:
    if not path.exists():
        return LabelOverrides(known_face_files=set(), known_no_face_files=set())

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return LabelOverrides(known_face_files=set(), known_no_face_files=set())

    known_face = {
        normalize_file_name(item)
        for item in payload.get("known_face_files", [])
        if str(item or "").strip()
    }
    known_no_face = {
        normalize_file_name(item)
        for item in payload.get("known_no_face_files", [])
        if str(item or "").strip()
    }
    known_face -= known_no_face
    return LabelOverrides(
        known_face_files=known_face,
        known_no_face_files=known_no_face,
    )


def build_scan_log_path(root_dir: Path) -> Path:
    return root_dir / SCAN_LOG_FILE_NAME


def safe_relative_video_path(video_path: Path, root_dir: Path) -> str:
    try:
        return str(video_path.relative_to(root_dir))
    except ValueError:
        return str(video_path)


def build_video_signature(video_path: Path, root_dir: Path) -> Optional[dict[str, object]]:
    try:
        stat = video_path.stat()
    except OSError:
        return None

    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    relative_path = safe_relative_video_path(video_path, root_dir)
    return {
        "relative_path": relative_path,
        "size_bytes": int(stat.st_size),
        "mtime_ns": mtime_ns,
        "cache_key": f"{normalize_file_name(relative_path)}|{int(stat.st_size)}|{mtime_ns}",
    }


def load_scan_log_cache(log_path: Path) -> dict[str, dict]:
    if not log_path.exists():
        return {}

    cache: dict[str, dict] = {}
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    entry = json.loads(text)
                except json.JSONDecodeError:
                    continue

                cache_key = str(entry.get("cache_key") or "").strip()
                if not cache_key:
                    continue
                if entry.get("error"):
                    continue
                cache[cache_key] = entry
    except OSError:
        return {}

    return cache


def append_scan_log_entry(log_path: Path, entry: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def result_from_log_entry(entry: dict, video_path: Path) -> VideoScanResult:
    return VideoScanResult(
        video_path=video_path,
        has_face=bool(entry.get("has_face")),
        checked_frames=int(entry.get("checked_frames") or 0),
        matched_second=_optional_float(entry.get("matched_second")),
        matched_detector=str(entry.get("matched_detector") or "").strip() or None,
        error=str(entry.get("error") or "").strip() or None,
        result_source="cache",
        debug_hit=DetectionHit.from_dict(entry.get("debug_hit")),
        debug_hit_second=_optional_float(entry.get("debug_hit_second")),
        debug_hit_reason=str(entry.get("debug_hit_reason") or "").strip() or None,
    )


def build_scan_log_entry(
    *,
    video_path: Path,
    root_dir: Path,
    result: VideoScanResult,
    signature: Optional[dict[str, object]],
    action: str,
    dry_run: bool,
) -> dict[str, object]:
    signature = signature or build_video_signature(video_path, root_dir) or {}
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "video_path": str(video_path),
        "relative_path": signature.get("relative_path") or safe_relative_video_path(video_path, root_dir),
        "file_name": video_path.name,
        "size_bytes": signature.get("size_bytes"),
        "mtime_ns": signature.get("mtime_ns"),
        "cache_key": signature.get("cache_key"),
        "has_face": bool(result.has_face),
        "checked_frames": int(result.checked_frames),
        "matched_second": result.matched_second,
        "matched_detector": result.matched_detector,
        "error": result.error,
        "result_source": result.result_source,
        "debug_hit": result.debug_hit.to_dict() if result.debug_hit is not None else None,
        "debug_hit_second": result.debug_hit_second,
        "debug_hit_reason": result.debug_hit_reason,
        "moved_to": str(result.moved_to) if result.moved_to is not None else None,
        "action": action,
        "dry_run": bool(dry_run),
    }


def iter_videos(target_path: Path, recursive: bool = True) -> Iterable[Path]:
    if target_path.is_file():
        if target_path.suffix.lower() in VIDEO_EXTENSIONS:
            yield target_path
        return

    iterator = target_path.rglob("*") if recursive else target_path.iterdir()
    for file_path in sorted(iterator):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if NO_FACE_FOLDER_NAME in file_path.parts:
            continue
        yield file_path


def rotate_image(image, angle: int):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int((height * sine) + (width * cosine))
    new_height = int((height * cosine) + (width * sine))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_width, new_height), borderMode=cv2.BORDER_REPLICATE)


def prepare_for_detection(image, max_side: int):
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image

    scale = float(max_side) / float(longest_side)
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def crop_box(frame_bgr, box):
    x, y, width, height = [int(round(float(value))) for value in box[:4]]
    if width <= 0 or height <= 0:
        return frame_bgr[0:0, 0:0]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(frame_bgr.shape[1], x + width)
    y2 = min(frame_bgr.shape[0], y + height)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr[0:0, 0:0]
    return frame_bgr[y1:y2, x1:x2]


def lower_face_skin_ratio(crop_bgr) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    height = crop_bgr.shape[0]
    lower_face = crop_bgr[int(height * 0.55): int(height * 0.95), :]
    if lower_face.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(lower_face, cv2.COLOR_BGR2YCrCb)
    lower = (0, 133, 77)
    upper = (255, 173, 127)
    mask = cv2.inRange(ycrcb, lower, upper)
    return float(mask.mean() / 255.0)


def compute_face_geometry(keypoints: dict) -> Optional[dict[str, float]]:
    required_keys = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
    if any(key not in keypoints for key in required_keys):
        return None

    left_eye = np.array(keypoints["left_eye"], dtype=np.float32)
    right_eye = np.array(keypoints["right_eye"], dtype=np.float32)
    nose = np.array(keypoints["nose"], dtype=np.float32)
    mouth_left = np.array(keypoints["mouth_left"], dtype=np.float32)
    mouth_right = np.array(keypoints["mouth_right"], dtype=np.float32)

    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance <= 1.0:
        return None

    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (mouth_left + mouth_right) / 2.0
    eye_to_mouth = float(np.linalg.norm(mouth_center - eye_center))
    mouth_width = float(np.linalg.norm(mouth_right - mouth_left))

    return {
        "eye_angle": abs(math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))),
        "nose_offset_ratio": abs(float(nose[0] - eye_center[0])) / eye_distance,
        "mouth_offset_ratio": abs(float(mouth_center[0] - eye_center[0])) / eye_distance,
        "vertical_ratio": eye_to_mouth / eye_distance,
        "mouth_eye_ratio": mouth_width / eye_distance,
    }


def hit_debug_score(hit: DetectionHit) -> tuple[float, int, int]:
    return (
        float(hit.confidence or 0.0),
        min(int(hit.width or 0), int(hit.height or 0)),
        max(int(hit.width or 0), int(hit.height or 0)),
    )


def evaluate_hit_rejection_reason(second: float, hit: DetectionHit) -> Optional[str]:
    min_side = min(hit.width, hit.height)
    max_side = max(hit.width, hit.height)
    aspect_ratio = float(max_side) / float(max(min_side, 1))

    if hit.angle != 0 and second <= 1.0:
        if min_side < EARLY_ROTATED_MIN_FACE_SIDE:
            return "early_rotated_small"
        if float(hit.mouth_eye_ratio or 0.0) > EARLY_ROTATED_MAX_MOUTH_EYE_RATIO:
            return "early_rotated_extreme_mouth_width"
        if float(hit.eye_angle or 0.0) <= EARLY_ROTATED_MIN_EYE_ANGLE:
            return "early_rotated_flat_eye_line"
    if (
        hit.angle == 0
        and second <= 1.0
        and float(hit.confidence or 0.0) < EARLY_FRONT_LOW_CONFIDENCE
        and float(hit.lower_skin_ratio or 0.0) < EARLY_FRONT_LOW_SKIN_RATIO
    ):
        return "early_front_low_skin_low_conf"
    if (
        hit.angle == 0
        and float(hit.confidence or 0.0) < ANYTIME_FRONT_TALL_REJECT_CONFIDENCE
        and float(hit.vertical_ratio or 0.0) > ANYTIME_FRONT_TALL_REJECT_RATIO
    ):
        return "front_low_conf_tall"
    if hit.angle != 0 and second > 1.0 and min_side < LARGE_FACE_SIDE:
        if (
            min_side >= LATE_ROTATED_SMALL_RESCUE_MIN_SIDE
            and float(hit.confidence or 0.0) >= LATE_ROTATED_SMALL_RESCUE_HIGH_CONFIDENCE
        ):
            return None
        if (
            min_side >= LATE_ROTATED_SMALL_RESCUE_NORMAL_MIN_SIDE
            and float(hit.vertical_ratio or 0.0) <= LATE_ROTATED_SMALL_RESCUE_MAX_VERTICAL_RATIO
            and float(hit.mouth_eye_ratio or 0.0) <= LATE_ROTATED_SMALL_RESCUE_MAX_MOUTH_EYE_RATIO
            and float(hit.nose_offset_ratio or 0.0) <= LATE_ROTATED_SMALL_RESCUE_MAX_NOSE_OFFSET_RATIO
        ):
            return None
        return "late_rotated_small"
    if (
        second > 1.0
        and min_side < LARGE_FACE_SIDE
        and float(hit.confidence or 0.0) < 0.75
    ):
        return "late_small_low_conf"
    if (
        second > 1.0
        and hit.angle == 0
        and float(hit.confidence or 0.0) < LATE_FACE_MIN_CONFIDENCE
        and float(hit.vertical_ratio or 0.0) > LATE_MASK_VERTICAL_RATIO
    ):
        return "late_front_low_conf_tall"
    if (
        second > 1.0
        and hit.angle == 0
        and float(hit.mouth_eye_ratio or 0.0) > LATE_MASK_MOUTH_EYE_RATIO
        and float(hit.mouth_offset_ratio or 0.0) > LATE_MASK_MOUTH_OFFSET_RATIO
    ):
        return "late_front_wide_mouth_offcenter"
    if (
        second > 1.0
        and hit.angle == 0
        and float(hit.mouth_eye_ratio or 0.0) > LATE_MASK_STRICT_MOUTH_EYE_RATIO
        and (
            float(hit.mouth_offset_ratio or 0.0) < LATE_MASK_CENTERED_MOUTH_OFFSET_RATIO
            or float(hit.mouth_offset_ratio or 0.0) > LATE_MASK_MOUTH_OFFSET_RATIO
        )
    ):
        return "late_front_very_wide_mouth"
    if (
        second > 1.0
        and hit.angle != 0
        and float(hit.nose_offset_ratio or 0.0) > LATE_ROTATED_NOSE_OFFSET_RATIO
        and float(hit.mouth_eye_ratio or 0.0) > LATE_ROTATED_MOUTH_EYE_RATIO
    ):
        return "late_rotated_nose_offset_wide_mouth"
    if (
        second > 1.0
        and hit.angle != 0
        and float(hit.confidence or 0.0) < LATE_FACE_MIN_CONFIDENCE
        and float(hit.mouth_eye_ratio or 0.0) > LATE_ROTATED_STRICT_MOUTH_EYE_RATIO
    ):
        return "late_rotated_low_conf_very_wide_mouth"
    if (
        second > 1.0
        and hit.angle != 0
        and min_side >= LARGE_FACE_SIDE
        and float(hit.nose_offset_ratio or 0.0) < LATE_ROTATED_CENTERED_MAX_OFFSET_RATIO
        and float(hit.mouth_offset_ratio or 0.0) < LATE_ROTATED_CENTERED_MAX_OFFSET_RATIO
    ):
        return "late_rotated_overly_centered_large_face"
    if (
        second > 1.0
        and hit.angle == 0
        and float(hit.confidence or 0.0) < LATE_FACE_MIN_CONFIDENCE
        and aspect_ratio > LATE_FRONT_WIDE_ASPECT_RATIO
        and float(hit.mouth_offset_ratio or 0.0) > 0.28
    ):
        return "late_front_wide_box_offcenter"
    if (
        second > 1.0
        and hit.angle == 0
        and min_side >= 250
        and float(hit.lower_skin_ratio or 0.0) < 0.5
        and float(hit.mouth_offset_ratio or 0.0) < 0.02
        and float(hit.vertical_ratio or 0.0) > 1.30
    ):
        return "late_front_low_skin_tall_centered"
    if (
        second > 1.0
        and hit.angle != 0
        and float(hit.confidence or 0.0) < 0.70
        and float(hit.nose_offset_ratio or 0.0) > 0.40
    ):
        return "late_rotated_low_conf_high_nose_offset"
    return None


def read_frame_at_second(capture: cv2.VideoCapture, second: float, fps: float, frame_count: int):
    try:
        if fps > 0:
            frame_index = int(round(second * fps))
            if frame_count > 0:
                frame_index = min(frame_index, max(frame_count - 1, 0))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(second, 0.0) * 1000.0)
        success, frame = capture.read()
        return success, frame
    except Exception:
        return False, None


def build_sample_seconds(duration_seconds: float, interval_seconds: float) -> list[float]:
    duration_seconds = max(float(duration_seconds or 0.0), 0.0)
    interval_seconds = max(float(interval_seconds or 0.5), 0.05)

    if duration_seconds <= 0:
        return [0.0]

    sample_seconds: list[float] = []
    steps = int(math.floor(duration_seconds / interval_seconds))
    for index in range(steps + 1):
        sample_seconds.append(round(index * interval_seconds, 3))

    last_second = round(max(duration_seconds - 0.001, 0.0), 3)
    if sample_seconds and abs(sample_seconds[-1] - last_second) > 0.12:
        sample_seconds.append(last_second)

    return sample_seconds


def build_tail_probe_seconds(fps: float, frame_count: int) -> list[float]:
    if fps <= 0 or frame_count <= 0:
        return []

    duration_seconds = float(frame_count) / float(fps)
    candidates = [
        max(duration_seconds - 1.0, 0.0),
        max(duration_seconds - 0.5, 0.0),
        max(duration_seconds - 0.1, 0.0),
        max((frame_count - 1) / float(fps), 0.0),
    ]

    seconds: list[float] = []
    for second in candidates:
        second = round(second, 3)
        if any(abs(existing - second) < 0.08 for existing in seconds):
            continue
        seconds.append(second)
    return seconds


def iter_sampled_frames(
    capture: cv2.VideoCapture,
    fps: float,
    interval_seconds: float,
):
    safe_fps = float(fps) if fps and math.isfinite(fps) and fps > 0 else 30.0
    frame_step = max(int(round(interval_seconds * safe_fps)), 1)
    next_target_frame = 0
    current_frame_index = 0
    sampled_count = 0
    consecutive_failures = 0

    while sampled_count < MAX_SAMPLED_FRAMES_PER_VIDEO:
        try:
            grabbed = capture.grab()
        except Exception:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                break
            continue

        if not grabbed:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                break
            current_frame_index += 1
            continue

        if current_frame_index < next_target_frame:
            consecutive_failures = 0
            current_frame_index += 1
            continue

        try:
            success, frame = capture.retrieve()
        except Exception:
            success, frame = False, None

        current_second = round(current_frame_index / safe_fps, 3)
        next_target_frame = current_frame_index + frame_step
        current_frame_index += 1

        if not success or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                break
            continue

        consecutive_failures = 0
        sampled_count += 1
        yield current_second, frame


def scan_video_for_face(
    video_path: Path,
    detector: LooseFaceDetector,
    interval_seconds: float,
) -> VideoScanResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return VideoScanResult(
            video_path=video_path,
            has_face=False,
            checked_frames=0,
            error="無法開啟影片",
        )

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        checked_frames = 0
        pending_weak_hit: Optional[tuple[float, DetectionHit]] = None
        best_debug_hit: Optional[DetectionHit] = None
        best_debug_second: Optional[float] = None
        best_debug_reason: Optional[str] = None
        for second, frame in iter_sampled_frames(capture, fps, interval_seconds):
            checked_frames += 1
            try:
                hit = detector.frame_has_face(frame)
            except Exception:
                continue
            if hit is None:
                continue

            rejection_reason = evaluate_hit_rejection_reason(second, hit)
            if best_debug_hit is None or hit_debug_score(hit) > hit_debug_score(best_debug_hit):
                best_debug_hit = hit
                best_debug_second = second
                best_debug_reason = rejection_reason or "accepted"

            if rejection_reason is not None:
                continue

            if hit.requires_confirmation:
                if pending_weak_hit is not None:
                    previous_second, previous_hit = pending_weak_hit
                    if (
                        second - previous_second <= max(interval_seconds * 2.5, 1.0)
                        and previous_hit.angle == hit.angle
                    ):
                        return VideoScanResult(
                            video_path=video_path,
                            has_face=True,
                            checked_frames=checked_frames,
                            matched_second=second,
                            matched_detector=f"{hit.label}-confirmed",
                            result_source="scan",
                            debug_hit=hit,
                            debug_hit_second=second,
                            debug_hit_reason="accepted-confirmed",
                        )

                pending_weak_hit = (second, hit)
                continue

            if not hit.requires_confirmation:
                return VideoScanResult(
                    video_path=video_path,
                    has_face=True,
                    checked_frames=checked_frames,
                    matched_second=second,
                    matched_detector=hit.label,
                    result_source="scan",
                    debug_hit=hit,
                    debug_hit_second=second,
                    debug_hit_reason="accepted",
                )

        tail_probe_seconds = build_tail_probe_seconds(fps, frame_count)
        if tail_probe_seconds:
            tail_capture = cv2.VideoCapture(str(video_path))
            try:
                if tail_capture.isOpened():
                    for second in tail_probe_seconds:
                        success, frame = read_frame_at_second(tail_capture, second, fps, frame_count)
                        if not success or frame is None:
                            continue
                        try:
                            hit = detector.frame_has_face(frame)
                        except Exception:
                            continue
                        if hit is None:
                            continue

                        rejection_reason = evaluate_hit_rejection_reason(second, hit)
                        if best_debug_hit is None or hit_debug_score(hit) > hit_debug_score(best_debug_hit):
                            best_debug_hit = hit
                            best_debug_second = second
                            best_debug_reason = rejection_reason or "accepted-tail"

                        if rejection_reason is not None:
                            continue

                        return VideoScanResult(
                            video_path=video_path,
                            has_face=True,
                            checked_frames=checked_frames,
                            matched_second=second,
                            matched_detector=hit.label,
                            result_source="scan",
                            debug_hit=hit,
                            debug_hit_second=second,
                            debug_hit_reason="accepted-tail",
                        )
            finally:
                tail_capture.release()

        return VideoScanResult(
            video_path=video_path,
            has_face=False,
            checked_frames=checked_frames,
            error="無法讀出任何有效畫面" if checked_frames == 0 else None,
            result_source="scan",
            debug_hit=best_debug_hit,
            debug_hit_second=best_debug_second,
            debug_hit_reason=best_debug_reason,
        )
    finally:
        capture.release()


def build_unique_destination(destination_dir: Path, source_path: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    candidate = destination_dir / source_path.name
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = destination_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_to_no_face_folder(video_path: Path, destination_dir: Path, dry_run: bool) -> Optional[Path]:
    destination_path = build_unique_destination(destination_dir, video_path)
    if dry_run:
        return destination_path

    shutil.move(str(video_path), str(destination_path))
    return destination_path


def run_scan(args: argparse.Namespace) -> int:
    target_path = resolve_target_path(args.target)
    recursive = not args.top_only
    root_dir = target_path if target_path.is_dir() else target_path.parent
    destination_dir = root_dir / NO_FACE_FOLDER_NAME
    log_path = build_scan_log_path(root_dir)
    overrides = load_label_overrides()
    scan_cache = {} if args.rescan else load_scan_log_cache(log_path)

    videos = list(iter_videos(target_path, recursive=recursive))
    if args.limit is not None:
        videos = videos[: max(args.limit, 0)]

    if not videos:
        print("沒有找到可掃描的影片。")
        return 0

    use_overrides = not args.ignore_overrides
    if use_overrides:
        unclassified_exists = any(overrides.classify(video_path.name) is None for video_path in videos)
    else:
        unclassified_exists = True

    detector: Optional[LooseFaceDetector] = None
    if unclassified_exists:
        detector = LooseFaceDetector(
            max_side=args.max_side,
            use_haar_fallback=args.haar_fallback,
            use_gpu=args.gpu,
        )

    kept_count = 0
    moved_count = 0
    error_count = 0

    print(f"掃描路徑：{target_path}")
    print(f"影片數量：{len(videos)}")
    print(f"抽幀間隔：{args.interval} 秒")
    print(f"目的資料夾：{destination_dir}")
    print(f"log 檔：{log_path}")
    if args.dry_run:
        print("目前是 dry-run，只會顯示結果，不會搬檔。")
    if use_overrides:
        print(f"已載入案例覆寫：有人臉={len(overrides.known_face_files)} | 無人臉={len(overrides.known_no_face_files)}")
    else:
        print("已忽略案例覆寫，全部改用實際偵測。")
    if args.rescan:
        print("已忽略既有 log 快取，全部重跑。")
    else:
        print(f"已載入快取紀錄：{len(scan_cache)} 筆")
    if detector is not None:
        print(f"偵測裝置：{detector.device_description}")
    print("")

    for index, video_path in enumerate(videos, start=1):
        signature = build_video_signature(video_path, root_dir)
        override_value = overrides.classify(video_path.name) if use_overrides else None
        cache_key = str((signature or {}).get("cache_key") or "")
        if override_value is not None:
            result = VideoScanResult(
                video_path=video_path,
                has_face=override_value,
                checked_frames=0,
                matched_detector="OverrideCase" if override_value else None,
                result_source="override",
            )
        elif not args.rescan and cache_key and cache_key in scan_cache:
            result = result_from_log_entry(scan_cache[cache_key], video_path)
        else:
            if detector is None:
                result = VideoScanResult(
                    video_path=video_path,
                    has_face=False,
                    checked_frames=0,
                    error="偵測器未初始化",
                    result_source="scan",
                )
            else:
                result = scan_video_for_face(video_path, detector, args.interval)
                gc.collect()

        action = "error"
        if result.error:
            error_count += 1
            source_label = "快取錯誤" if result.result_source == "cache" else "錯誤"
            print(f"[{index}/{len(videos)}] {source_label} | {video_path.name} | {result.error}")
        elif result.has_face:
            kept_count += 1
            second_text = (
                f"{result.matched_second:.1f}s"
                if result.matched_second is not None
                else "unknown"
            )
            detector_text = result.matched_detector or "unknown"
            if result.result_source == "cache":
                status_text = "快取保留"
            elif result.result_source == "override":
                status_text = "覆寫保留"
            else:
                status_text = "保留"
            print(
                f"[{index}/{len(videos)}] {status_text} | {video_path.name} | "
                f"check={result.checked_frames} | hit={second_text} | detector={detector_text}"
            )
            action = "keep"
        else:
            moved_path = move_to_no_face_folder(video_path, destination_dir, args.dry_run)
            result.moved_to = moved_path
            moved_count += 1
            moved_label = moved_path if moved_path is not None else destination_dir
            if result.result_source == "cache":
                status_text = "快取預計搬移" if args.dry_run else "快取已搬移"
            elif result.result_source == "override":
                status_text = "覆寫預計搬移" if args.dry_run else "覆寫已搬移"
            else:
                status_text = "預計搬移" if args.dry_run else "已搬移"
            print(
                f"[{index}/{len(videos)}] {status_text} | {video_path.name} | "
                f"check={result.checked_frames} | -> {moved_label}"
            )
            action = "move"

        entry = build_scan_log_entry(
            video_path=video_path,
            root_dir=root_dir,
            result=result,
            signature=signature,
            action=action,
            dry_run=args.dry_run,
        )
        append_scan_log_entry(log_path, entry)
        if cache_key and not result.error:
            scan_cache[cache_key] = entry

    print("")
    print(
        "掃描完成 | "
        f"總數={len(videos)} | 保留={kept_count} | "
        f"{'預計搬移' if args.dry_run else '搬移'}={moved_count} | 錯誤={error_count}"
    )
    return 0 if error_count == 0 else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_scan(args)


if __name__ == "__main__":
    sys.exit(main())
