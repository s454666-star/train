from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("缺少 cv2，請先安裝 opencv-python。") from exc

try:
    from mtcnn import MTCNN
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("缺少 mtcnn，請先安裝 mtcnn。") from exc


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


class LooseFaceDetector:
    def __init__(self, max_side: int = 1280) -> None:
        self.max_side = max(int(max_side), 320)
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

    def frame_has_face(self, frame_bgr) -> tuple[bool, Optional[str]]:
        for angle in ROTATION_ANGLES:
            rotated = frame_bgr if angle == 0 else rotate_image(frame_bgr, angle)
            prepared = prepare_for_detection(rotated, self.max_side)

            if self._mtcnn_has_face(prepared):
                return True, f"MTCNN@{angle}"

            if self._haar_has_face(prepared):
                return True, f"Haar@{angle}"

        return False, None

    def _mtcnn_has_face(self, frame_bgr) -> bool:
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
            return False

        if not results:
            return False

        for result in results:
            box = result.get("box") or []
            confidence = float(result.get("confidence") or 0.0)
            if len(box) < 4:
                continue
            if confidence < 0.55:
                continue

            _, _, width, height = box
            if width > 0 and height > 0:
                return True

        return False

    def _haar_has_face(self, frame_bgr) -> bool:
        if self.haar_frontal is None and self.haar_profile is None:
            return False

        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
        except Exception:
            return False

        if self.haar_frontal is not None:
            frontal_faces = self.haar_frontal.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24),
            )
            if frontal_faces is not None and len(frontal_faces) > 0:
                return True

        if self.haar_profile is not None:
            profile_faces = self.haar_profile.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24),
            )
            if profile_faces is not None and len(profile_faces) > 0:
                return True

            flipped = cv2.flip(gray, 1)
            flipped_faces = self.haar_profile.detectMultiScale(
                flipped,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24),
            )
            if flipped_faces is not None and len(flipped_faces) > 0:
                return True

        return False


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


def read_frame_at_second(capture: cv2.VideoCapture, second: float, fps: float, frame_count: int):
    if fps > 0:
        frame_index = int(round(second * fps))
        if frame_count > 0:
            frame_index = min(frame_index, max(frame_count - 1, 0))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    else:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(second, 0.0) * 1000.0)
    success, frame = capture.read()
    return success, frame


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
        duration_seconds = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        sample_seconds = build_sample_seconds(duration_seconds, interval_seconds)

        checked_frames = 0
        for second in sample_seconds:
            success, frame = read_frame_at_second(capture, second, fps, frame_count)
            if not success or frame is None:
                continue

            checked_frames += 1
            has_face, matched_detector = detector.frame_has_face(frame)
            if has_face:
                return VideoScanResult(
                    video_path=video_path,
                    has_face=True,
                    checked_frames=checked_frames,
                    matched_second=second,
                    matched_detector=matched_detector,
                )

        return VideoScanResult(
            video_path=video_path,
            has_face=False,
            checked_frames=checked_frames,
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

    videos = list(iter_videos(target_path, recursive=recursive))
    if args.limit is not None:
        videos = videos[: max(args.limit, 0)]

    if not videos:
        print("沒有找到可掃描的影片。")
        return 0

    detector = LooseFaceDetector(max_side=args.max_side)
    kept_count = 0
    moved_count = 0
    error_count = 0

    print(f"掃描路徑：{target_path}")
    print(f"影片數量：{len(videos)}")
    print(f"抽幀間隔：{args.interval} 秒")
    print(f"目的資料夾：{destination_dir}")
    if args.dry_run:
        print("目前是 dry-run，只會顯示結果，不會搬檔。")
    print("")

    for index, video_path in enumerate(videos, start=1):
        result = scan_video_for_face(video_path, detector, args.interval)

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
