from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
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
OVERRIDES_PATH = Path(__file__).with_name("move_no_face_videos_overrides.json")


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


class LooseFaceDetector:
    def __init__(self, max_side: int = 1280, use_haar_fallback: bool = False) -> None:
        self.max_side = max(int(max_side), 320)
        self.use_haar_fallback = bool(use_haar_fallback)
        try:
            from mtcnn import MTCNN
        except ImportError as exc:  # pragma: no cover - import guard
            raise SystemExit("缺少 mtcnn，請先安裝 mtcnn。") from exc

        self.mtcnn = MTCNN(device="CPU:0")
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
        "--ignore-overrides",
        action="store_true",
        help="忽略已標記案例，強制用實際偵測重新掃描。",
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
        checked_frames = 0
        pending_weak_hit: Optional[tuple[float, DetectionHit]] = None
        for second, frame in iter_sampled_frames(capture, fps, interval_seconds):
            checked_frames += 1
            try:
                hit = detector.frame_has_face(frame)
            except Exception:
                continue
            if hit is None:
                continue

            if hit.angle != 0 and second > 1.0 and min(hit.width, hit.height) < LARGE_FACE_SIDE:
                continue
            if (
                second > 1.0
                and min(hit.width, hit.height) < LARGE_FACE_SIDE
                and float(hit.confidence or 0.0) < 0.75
            ):
                continue
            if (
                second > 1.0
                and hit.angle == 0
                and float(hit.confidence or 0.0) < LATE_FACE_MIN_CONFIDENCE
                and float(hit.vertical_ratio or 0.0) > LATE_MASK_VERTICAL_RATIO
            ):
                continue
            if (
                second > 1.0
                and hit.angle == 0
                and float(hit.mouth_eye_ratio or 0.0) > LATE_MASK_MOUTH_EYE_RATIO
                and float(hit.mouth_offset_ratio or 0.0) > LATE_MASK_MOUTH_OFFSET_RATIO
            ):
                continue
            if (
                second > 1.0
                and hit.angle == 0
                and float(hit.mouth_eye_ratio or 0.0) > LATE_MASK_STRICT_MOUTH_EYE_RATIO
                and (
                    float(hit.mouth_offset_ratio or 0.0) < LATE_MASK_CENTERED_MOUTH_OFFSET_RATIO
                    or float(hit.mouth_offset_ratio or 0.0) > LATE_MASK_MOUTH_OFFSET_RATIO
                )
            ):
                continue
            if (
                second > 1.0
                and hit.angle != 0
                and float(hit.nose_offset_ratio or 0.0) > LATE_ROTATED_NOSE_OFFSET_RATIO
                and float(hit.mouth_eye_ratio or 0.0) > LATE_ROTATED_MOUTH_EYE_RATIO
            ):
                continue
            if (
                second > 1.0
                and hit.angle != 0
                and float(hit.confidence or 0.0) < LATE_FACE_MIN_CONFIDENCE
                and float(hit.mouth_eye_ratio or 0.0) > LATE_ROTATED_STRICT_MOUTH_EYE_RATIO
            ):
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
                )

        return VideoScanResult(
            video_path=video_path,
            has_face=False,
            checked_frames=checked_frames,
            error="無法讀出任何有效畫面" if checked_frames == 0 else None,
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
    overrides = load_label_overrides()

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
        )

    kept_count = 0
    moved_count = 0
    error_count = 0

    print(f"掃描路徑：{target_path}")
    print(f"影片數量：{len(videos)}")
    print(f"抽幀間隔：{args.interval} 秒")
    print(f"目的資料夾：{destination_dir}")
    if args.dry_run:
        print("目前是 dry-run，只會顯示結果，不會搬檔。")
    if use_overrides:
        print(f"已載入案例覆寫：有人臉={len(overrides.known_face_files)} | 無人臉={len(overrides.known_no_face_files)}")
    else:
        print("已忽略案例覆寫，全部改用實際偵測。")
    print("")

    for index, video_path in enumerate(videos, start=1):
        override_value = overrides.classify(video_path.name) if use_overrides else None
        if override_value is None:
            if detector is None:
                result = VideoScanResult(
                    video_path=video_path,
                    has_face=False,
                    checked_frames=0,
                    error="偵測器未初始化",
                )
            else:
                result = scan_video_for_face(video_path, detector, args.interval)
                gc.collect()
        else:
            result = VideoScanResult(
                video_path=video_path,
                has_face=override_value,
                checked_frames=0,
                matched_detector="OverrideCase" if override_value else None,
            )

        if result.error:
            error_count += 1
            print(f"[{index}/{len(videos)}] 錯誤 | {video_path.name} | {result.error}")
            continue

        if result.has_face:
            kept_count += 1
            second_text = (
                f"{result.matched_second:.1f}s"
                if result.matched_second is not None
                else "unknown"
            )
            detector_text = result.matched_detector or "unknown"
            print(
                f"[{index}/{len(videos)}] 保留 | {video_path.name} | "
                f"check={result.checked_frames} | hit={second_text} | detector={detector_text}"
            )
            continue

        moved_path = move_to_no_face_folder(video_path, destination_dir, args.dry_run)
        result.moved_to = moved_path
        moved_count += 1
        moved_label = moved_path if moved_path is not None else destination_dir
        status_text = "預計搬移" if args.dry_run else "已搬移"
        print(
            f"[{index}/{len(videos)}] {status_text} | {video_path.name} | "
            f"check={result.checked_frames} | -> {moved_label}"
        )

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
