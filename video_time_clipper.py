"""A small PyQt5 video player for collecting copy-ready time ranges."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import ctypes
import faulthandler
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, QRect, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

try:
    import cv2
except ImportError:  # pragma: no cover - only used on machines without OpenCV.
    cv2 = None

try:
    import av
except ImportError:  # pragma: no cover - only used on machines without PyAV.
    av = None

for _vlc_dir in (r"C:\Program Files\VideoLAN\VLC", r"C:\Program Files (x86)\VideoLAN\VLC"):
    if os.path.isdir(_vlc_dir):
        os.environ["PATH"] = _vlc_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_vlc_dir)
        break

try:
    import vlc
except ImportError:  # pragma: no cover - only used on machines without python-vlc.
    vlc = None

if cv2 is not None:
    cv2.setNumThreads(1)


VIDEO_FILTER = "影片檔案 (*.mp4 *.mkv *.mov *.avi *.wmv *.m4v);;所有檔案 (*.*)"
TEXT_FILTER = "文字檔案 (*.txt);;所有檔案 (*.*)"
DEFAULT_OUTPUT_LIST_PATH = r"C:\Users\User\Documents\清單.txt"
WINDOWS_APP_ID = "Star.VideoTimeClipper"
SLIDER_STEP_MS = 1000
PLAYBACK_MAX_FPS = 24.0
SEEK_DEBOUNCE_MS = 45
VLC_SCRUB_THROTTLE_MS = 25
THUMBNAIL_COUNT = 9
THUMBNAIL_MAX_WIDTH = 180
THUMBNAIL_MAX_HEIGHT = 76
THUMBNAIL_REFRESH_DELAY_MS = 80
THUMBNAIL_HANDLE_THRESHOLD = 28
SEEK_SLIDER_HIT_HEIGHT = 42
VIDEO_READ_LOCK = threading.RLock()
PLAYBACK_CATCHUP_FRAME_LIMIT = 8
PLAYBACK_SEEK_CATCHUP_MS = 1200
_CRASH_LOG_HANDLE = None


def ffmpeg_executable() -> Optional[str]:
    return shutil.which("ffmpeg") or (r"C:\ffmpeg\bin\ffmpeg.exe" if os.path.isfile(r"C:\ffmpeg\bin\ffmpeg.exe") else None)


def read_video_image_with_ffmpeg(video_path: str, position_ms: int, max_width: int, timeout: int = 8) -> QImage:
    ffmpeg_path = ffmpeg_executable()
    if not ffmpeg_path:
        return QImage()
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0, position_ms) / 1000:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-vf",
        f"scale={max(1, int(max_width))}:-2:force_original_aspect_ratio=decrease",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception:
        return QImage()
    if result.returncode != 0 or not result.stdout:
        return QImage()
    image = QImage()
    if image.loadFromData(result.stdout):
        return image.copy()
    return QImage()


def enable_crash_logging() -> None:
    global _CRASH_LOG_HANDLE
    if _CRASH_LOG_HANDLE is not None:
        return
    try:
        base_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "CodexTools" / "VideoTimeClipper"
        base_dir.mkdir(parents=True, exist_ok=True)
        _CRASH_LOG_HANDLE = open(base_dir / "crash.log", "a", encoding="utf-8")
        _CRASH_LOG_HANDLE.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        _CRASH_LOG_HANDLE.flush()
        faulthandler.enable(_CRASH_LOG_HANDLE, all_threads=True)
    except Exception:
        _CRASH_LOG_HANDLE = None


class AvVideoReader:
    def __init__(self, path: str) -> None:
        if av is None:
            raise RuntimeError("PyAV is not available")
        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.time_base = float(self.stream.time_base or 0) or (1 / 30)
        self.fps = self._read_fps()
        self.frame_count = int(self.stream.frames or 0)
        self.duration_ms = self._read_duration_ms()
        self.last_position_ms = 0
        self._packets = self.container.demux(self.stream)

    def _read_fps(self) -> float:
        rate = self.stream.average_rate or self.stream.base_rate or self.stream.guessed_rate
        if rate:
            fps = float(rate)
            if 1 <= fps <= 240:
                return fps
        return 30.0

    def _read_duration_ms(self) -> int:
        if self.stream.duration:
            return max(0, int(float(self.stream.duration) * self.time_base * 1000))
        if self.container.duration:
            return max(0, int(self.container.duration / 1000))
        if self.frame_count > 0:
            return max(0, int((self.frame_count / self.fps) * 1000))
        return 0

    def read(self):
        for packet in self._packets:
            for frame in packet.decode():
                self._remember_position(frame)
                return True, frame.to_ndarray(format="bgr24")
        return False, None

    def seek_read(self, position_ms: int):
        position_ms = max(0, int(position_ms))
        target_pts = int((position_ms / 1000) / self.time_base)
        self.container.seek(target_pts, any_frame=False, backward=True, stream=self.stream)
        self._packets = self.container.demux(self.stream)
        self.last_position_ms = position_ms

        fallback_frame = None
        max_decode = max(8, min(120, int(self.fps * 4)))
        for _ in range(max_decode):
            ok, frame = self.read()
            if not ok:
                break
            fallback_frame = frame
            if self.last_position_ms + (1000 / self.fps) >= position_ms:
                return True, frame
        if fallback_frame is not None:
            return True, fallback_frame
        return False, None

    def _remember_position(self, frame) -> None:
        if frame.pts is not None:
            self.last_position_ms = max(0, int(frame.pts * self.time_base * 1000))
        else:
            self.last_position_ms += int(1000 / self.fps)

    def release(self) -> None:
        self.container.close()


class CvVideoReader:
    def __init__(self, path: str) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is not available")
        self.path = path
        self.capture = None
        self.backend_name = "OpenCV"
        self._open_capture(path)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0)
        if self.fps <= 1 or self.fps > 240:
            self.fps = 30.0
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_ms = max(0, int((self.frame_count / self.fps) * 1000)) if self.frame_count > 0 else 0
        self.last_position_ms = 0

    def _open_capture(self, path: str) -> None:
        backends = []
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("OpenCV MSMF", cv2.CAP_MSMF))

        for name, backend in backends:
            capture = cv2.VideoCapture(path, backend)
            if capture.isOpened():
                try:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self.capture = capture
                self.backend_name = name
                return
            capture.release()
        raise RuntimeError("OpenCV MSMF cannot open video")

    def read(self):
        ok, frame = self.capture.read()
        if ok:
            self.last_position_ms = int(self.capture.get(cv2.CAP_PROP_POS_MSEC) or 0)
            if self.last_position_ms <= 0:
                frame_index = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                self.last_position_ms = int((frame_index / self.fps) * 1000)
        return ok, frame

    def seek_read(self, position_ms: int):
        self.capture.set(cv2.CAP_PROP_POS_MSEC, max(0, int(position_ms)))
        ok, frame = self.read()
        if not ok:
            self.last_position_ms = max(0, int(position_ms))
        return ok, frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def open_video_reader(path: str):
    errors = []
    if av is not None:
        try:
            return AvVideoReader(path), "PyAV"
        except Exception as exc:
            errors.append(f"PyAV: {exc}")

    if cv2 is not None:
        try:
            reader = CvVideoReader(path)
            return reader, reader.backend_name
        except Exception as exc:
            errors.append(f"OpenCV: {exc}")

    raise RuntimeError("; ".join(errors) or "No video reader is available")


def resize_frame_to_fit(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0 or max_width <= 0 or max_height <= 0:
        return frame
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 0.999:
        return frame
    target_width = max(1, int(width * scale))
    target_height = max(1, int(height * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def resize_frame_to_cover(frame, target_width: int, target_height: int):
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0:
        return frame

    scale = max(target_width / width, target_height / height)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    if scale < 0.999:
        interpolation = cv2.INTER_AREA
    elif scale > 1.001:
        interpolation = getattr(cv2, "INTER_LINEAR_EXACT", cv2.INTER_LINEAR)
    else:
        interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)

    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return resized[top : top + target_height, left : left + target_width].copy()


@dataclass(frozen=True)
class ClipSegment:
    start_ms: int
    end_ms: int

    def normalized(self) -> "ClipSegment":
        if self.start_ms <= self.end_ms:
            return self
        return ClipSegment(self.end_ms, self.start_ms)

    def label(self) -> str:
        segment = self.normalized()
        return f"{format_time(segment.start_ms)}~{format_time(segment.end_ms)}"


class ClickableSlider(QSlider):
    """Slider that seeks to the clicked position instead of only dragging."""

    scanStarted = pyqtSignal()
    scanMoved = pyqtSignal(int)
    scanFinished = pyqtSignal(int)

    def __init__(self, orientation, parent=None) -> None:
        super().__init__(orientation, parent)
        if orientation == Qt.Horizontal:
            self.setFixedHeight(SEEK_SLIDER_HIT_HEIGHT)
        self.setCursor(Qt.PointingHandCursor)

    def _value_from_event(self, event) -> int:
        if self.orientation() == Qt.Horizontal:
            ratio = event.x() / max(1, self.width() - 1)
        else:
            ratio = 1 - (event.y() / max(1, self.height() - 1))
        ratio = max(0.0, min(1.0, ratio))
        value = round(self.minimum() + ratio * (self.maximum() - self.minimum()))
        return max(self.minimum(), min(self.maximum(), value))

    def _move_to_event_position(self, event) -> int:
        value = self._value_from_event(event)
        self.setSliderPosition(value)
        self.setValue(value)
        return value

    def mousePressEvent(self, event):  # noqa: N802 - Qt override name
        if event.button() == Qt.LeftButton and self.maximum() > self.minimum():
            self.setSliderDown(True)
            self.scanStarted.emit()
            value = self._move_to_event_position(event)
            self.scanMoved.emit(value)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override name
        if self.isSliderDown() and event.buttons() & Qt.LeftButton and self.maximum() > self.minimum():
            value = self._move_to_event_position(event)
            self.scanMoved.emit(value)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override name
        if event.button() == Qt.LeftButton and self.isSliderDown():
            value = self._move_to_event_position(event)
            self.setSliderDown(False)
            self.scanFinished.emit(value)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ThumbnailRangeStrip(QWidget):
    rangeChanged = pyqtSignal(int, int)
    handleMoved = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("thumbnailStrip")
        self.setMinimumHeight(76)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.thumbnails: list[tuple[int, QPixmap]] = []
        self.duration_ms = 0
        self.view_start_ms = 0
        self.view_end_ms = 0
        self.range_start_ms: Optional[int] = None
        self.range_end_ms: Optional[int] = None
        self._dragging_handle: Optional[str] = None
        self._drag_anchor_x = 0
        self._drag_anchor_start_ms = 0
        self._drag_anchor_end_ms = 0

    def mousePressEvent(self, event):  # noqa: N802 - Qt override name
        if event.button() == Qt.LeftButton:
            handle = self._nearest_handle(event.x())
            if handle is not None:
                self._dragging_handle = handle
                self._move_handle_to_x(event.x())
                event.accept()
                return
            if self._is_inside_range(event.x()):
                self._dragging_handle = "range"
                self._drag_anchor_x = event.x()
                self._drag_anchor_start_ms, self._drag_anchor_end_ms = self._normalized_range()
                self.setCursor(Qt.SizeAllCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override name
        if self._dragging_handle and event.buttons() & Qt.LeftButton:
            self._move_handle_to_x(event.x())
            event.accept()
            return
        hover_handle = self._nearest_handle(event.x())
        if hover_handle:
            self.setCursor(Qt.SizeHorCursor)
        elif self._is_inside_range(event.x()):
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override name
        if event.button() == Qt.LeftButton and self._dragging_handle:
            self._move_handle_to_x(event.x())
            self._dragging_handle = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):  # noqa: N802 - Qt override name
        if not self._dragging_handle:
            self.unsetCursor()
        super().leaveEvent(event)

    def paintEvent(self, event):  # noqa: N802 - Qt override name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, QColor("#05070d"))

        if not self.thumbnails:
            painter.setPen(QColor("#64748b"))
            painter.drawText(rect, Qt.AlignCenter, "設定 [ 和 ] 後可拖左右藍線微調")
            painter.end()
            return

        thumb_count = len(self.thumbnails)
        thumb_width = max(1, rect.width() / thumb_count)
        for index, (_, pixmap) in enumerate(self.thumbnails):
            target = QRect(
                int(rect.left() + index * thumb_width),
                rect.top(),
                int(thumb_width) + 1,
                rect.height(),
            )
            painter.drawPixmap(target, pixmap)

        start_x, end_x = self._range_pixels()
        painter.fillRect(rect.left(), rect.top(), max(0, start_x - rect.left()), rect.height(), QColor(0, 0, 0, 145))
        painter.fillRect(end_x, rect.top(), max(0, rect.right() - end_x + 1), rect.height(), QColor(0, 0, 0, 145))

        if self.range_start_ms is not None and self.range_end_ms is not None:
            selected_rect = QRect(start_x, rect.top(), max(2, end_x - start_x), rect.height())
            painter.setPen(QPen(QColor("#38bdf8"), 2))
            painter.drawRect(selected_rect.adjusted(0, 1, 0, -2))
            self._draw_handle(painter, start_x, rect, "[")
            self._draw_handle(painter, end_x, rect, "]")

        painter.end()

    def set_duration(self, duration_ms: int) -> None:
        self.duration_ms = max(0, int(duration_ms))
        self.view_start_ms = 0
        self.view_end_ms = self.duration_ms
        self.update()

    def set_view(self, start_ms: int, end_ms: int) -> None:
        self.view_start_ms = max(0, int(start_ms))
        self.view_end_ms = max(self.view_start_ms + SLIDER_STEP_MS, int(end_ms))
        if self.duration_ms > 0:
            self.view_end_ms = min(self.duration_ms, self.view_end_ms)
            self.view_start_ms = min(self.view_start_ms, max(0, self.view_end_ms - SLIDER_STEP_MS))
        self.update()

    def set_thumbnails(self, thumbnails: list[tuple[int, QPixmap]]) -> None:
        self.thumbnails = thumbnails
        self.update()

    def set_range(self, start_ms: Optional[int], end_ms: Optional[int]) -> None:
        self.range_start_ms = start_ms
        self.range_end_ms = end_ms
        self.update()

    def _time_to_x(self, position_ms: int) -> int:
        if self.view_end_ms <= self.view_start_ms:
            return self.rect().left()
        rect = self.rect().adjusted(0, 0, -1, -1)
        ratio = (position_ms - self.view_start_ms) / max(1, self.view_end_ms - self.view_start_ms)
        ratio = max(0.0, min(1.0, ratio))
        return int(rect.left() + ratio * rect.width())

    def _x_to_time(self, x: int) -> int:
        if self.view_end_ms <= self.view_start_ms:
            return 0
        rect = self.rect().adjusted(0, 0, -1, -1)
        ratio = (x - rect.left()) / max(1, rect.width())
        ratio = max(0.0, min(1.0, ratio))
        return int(self.view_start_ms + ratio * (self.view_end_ms - self.view_start_ms))

    def _range_pixels(self) -> tuple[int, int]:
        if self.range_start_ms is None or self.range_end_ms is None:
            return self.rect().left(), self.rect().right()
        start_ms, end_ms = self._normalized_range()
        return self._time_to_x(start_ms), self._time_to_x(end_ms)

    def _normalized_range(self) -> tuple[int, int]:
        if self.range_start_ms is None or self.range_end_ms is None:
            return 0, 0
        return tuple(sorted((int(self.range_start_ms), int(self.range_end_ms))))

    def _is_inside_range(self, x: int) -> bool:
        if self.range_start_ms is None or self.range_end_ms is None:
            return False
        start_x, end_x = self._range_pixels()
        return start_x + THUMBNAIL_HANDLE_THRESHOLD < x < end_x - THUMBNAIL_HANDLE_THRESHOLD

    def _nearest_handle(self, x: int) -> Optional[str]:
        if self.range_start_ms is None or self.range_end_ms is None:
            return None
        start_x, end_x = self._range_pixels()
        threshold = THUMBNAIL_HANDLE_THRESHOLD
        start_distance = abs(x - start_x)
        end_distance = abs(x - end_x)
        if start_distance <= threshold or end_distance <= threshold:
            return "start" if start_distance <= end_distance else "end"
        return None

    def _move_handle_to_x(self, x: int) -> None:
        if self.range_start_ms is None or self.range_end_ms is None or self._dragging_handle is None:
            return
        position_ms = self._x_to_time(x)
        start_ms, end_ms = self._normalized_range()
        if self._dragging_handle == "range":
            start_ms, end_ms, preview_ms = self._moved_range_values(x)
        elif self._dragging_handle == "start":
            start_ms = min(position_ms, end_ms - SLIDER_STEP_MS)
            start_ms = max(0, start_ms)
            preview_ms = start_ms
        else:
            end_ms = max(position_ms, start_ms + SLIDER_STEP_MS)
            end_ms = min(self.duration_ms, end_ms)
            preview_ms = end_ms
        self.range_start_ms = start_ms
        self.range_end_ms = end_ms
        self.rangeChanged.emit(start_ms, end_ms)
        self.handleMoved.emit(preview_ms)
        self.update()

    def _moved_range_values(self, x: int) -> tuple[int, int, int]:
        range_duration = max(SLIDER_STEP_MS, self._drag_anchor_end_ms - self._drag_anchor_start_ms)
        rect = self.rect().adjusted(0, 0, -1, -1)
        view_duration = max(1, self.view_end_ms - self.view_start_ms)
        delta_ms = int(((x - self._drag_anchor_x) / max(1, rect.width())) * view_duration)
        upper_bound = self.duration_ms if self.duration_ms > 0 else self.view_end_ms
        max_start = max(0, upper_bound - range_duration)
        start_ms = max(0, min(max_start, self._drag_anchor_start_ms + delta_ms))
        end_ms = start_ms + range_duration
        return int(start_ms), int(end_ms), int(start_ms)

    def _draw_handle(self, painter: QPainter, x: int, rect: QRect, label: str) -> None:
        painter.setPen(QPen(QColor("#38bdf8"), 3))
        painter.drawLine(x, rect.top(), x, rect.bottom())
        handle_rect = QRect(x - 11, rect.top(), 22, 20)
        painter.fillRect(handle_rect, QColor("#38bdf8"))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(handle_rect, Qt.AlignCenter, label)


class ThumbnailWorker(QObject):
    finished = pyqtSignal(int, int, int, list)

    def __init__(self, request_id: int, video_path: str, view_start_ms: int, view_end_ms: int) -> None:
        super().__init__()
        self.request_id = request_id
        self.video_path = video_path
        self.view_start_ms = view_start_ms
        self.view_end_ms = view_end_ms

    def run(self) -> None:
        thumbnails: list[tuple[int, QImage]] = []
        ffmpeg_path = ffmpeg_executable()
        if not ffmpeg_path:
            self.finished.emit(self.request_id, self.view_start_ms, self.view_end_ms, thumbnails)
            return

        positions = self._thumbnail_positions()
        thumbnails = self._read_thumbnail_sheet(ffmpeg_path, positions)
        if thumbnails:
            self.finished.emit(self.request_id, self.view_start_ms, self.view_end_ms, thumbnails)
            return

        for position_ms in positions:
            if QThread.currentThread().isInterruptionRequested():
                break
            image = read_video_image_with_ffmpeg(
                self.video_path, position_ms, THUMBNAIL_MAX_WIDTH, timeout=2
            )
            if not image.isNull():
                thumbnails.append((position_ms, image))

        self.finished.emit(self.request_id, self.view_start_ms, self.view_end_ms, thumbnails)

    def _thumbnail_positions(self) -> list[int]:
        if THUMBNAIL_COUNT <= 1 or self.view_end_ms <= self.view_start_ms:
            return [self.view_start_ms]
        return [
            int(self.view_start_ms + ((self.view_end_ms - self.view_start_ms) * index) / (THUMBNAIL_COUNT - 1))
            for index in range(THUMBNAIL_COUNT)
        ]

    def _read_thumbnail_sheet(self, ffmpeg_path: str, positions: list[int]) -> list[tuple[int, QImage]]:
        if not positions:
            return []
        duration_ms = max(SLIDER_STEP_MS, self.view_end_ms - self.view_start_ms)
        duration_sec = max(0.5, duration_ms / 1000)
        fps_value = THUMBNAIL_COUNT / duration_sec
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-noaccurate_seek",
            "-ss",
            f"{max(0, self.view_start_ms) / 1000:.3f}",
            "-i",
            self.video_path,
            "-t",
            f"{duration_sec:.3f}",
            "-vf",
            (
                f"fps={fps_value:.6f},"
                f"scale={THUMBNAIL_MAX_WIDTH}:-2:force_original_aspect_ratio=decrease,"
                f"tile={THUMBNAIL_COUNT}x1:padding=0:margin=0"
            ),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=4,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception:
            return []
        if result.returncode != 0 or not result.stdout:
            return []
        sheet = QImage()
        if not sheet.loadFromData(result.stdout) or sheet.isNull():
            return []

        cell_width = max(1, sheet.width() // THUMBNAIL_COUNT)
        thumbnails: list[tuple[int, QImage]] = []
        for index, position_ms in enumerate(positions[:THUMBNAIL_COUNT]):
            cell = sheet.copy(index * cell_width, 0, cell_width, sheet.height())
            if not cell.isNull():
                thumbnails.append((position_ms, cell.copy()))
        return thumbnails


class VideoDropFilter(QObject):
    dropped = pyqtSignal(str)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt override name
        if event.type() in (QEvent.DragEnter, QEvent.DragMove):
            drag_event: QDragEnterEvent = event
            if drag_event.mimeData().hasUrls():
                drag_event.acceptProposedAction()
                return True
        if event.type() == QEvent.Drop:
            drop_event: QDropEvent = event
            for url in drop_event.mimeData().urls():
                if url.isLocalFile():
                    self.dropped.emit(url.toLocalFile())
                    drop_event.acceptProposedAction()
                    return True
        return super().eventFilter(obj, event)


def format_time(milliseconds: int) -> str:
    total_seconds = max(0, int(milliseconds // 1000))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def file_uri(path: str) -> str:
    absolute = os.path.abspath(path).replace("\\", "/")
    if not absolute.startswith("/"):
        absolute = "/" + absolute
    return "file://" + absolute


def set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception:
        return


class SegmentList(QListWidget):
    def keyPressEvent(self, event):  # noqa: N802 - Qt override name
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in self.selectedItems():
                self.takeItem(self.row(item))
            return
        super().keyPressEvent(event)


class VideoTimeClipper(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("影片時間擷取複製工具")
        self.resize(1320, 820)
        self.setMinimumSize(860, 560)
        self.setAcceptDrops(True)

        self.video_path: Optional[str] = None
        self.duration_ms = 0
        self.pending_start_ms: Optional[int] = None
        self.pending_end_ms: Optional[int] = None
        self._slider_is_pressed = False
        self._resume_after_slider = False
        self._fallback_active = False
        self._fallback_playing = False
        self._fallback_capture = None
        self._fallback_backend = ""
        self._fallback_fps = 30.0
        self._fallback_position_ms = 0
        self._playback_clock_started_at = 0.0
        self._playback_clock_position_ms = 0
        self._vlc_instance = None
        self._vlc_player = None
        self._vlc_active = False
        self._vlc_playing = False

        self.fallback_timer = QTimer(self)
        self.fallback_timer.timeout.connect(self._render_fallback_frame)
        self.vlc_timer = QTimer(self)
        self.vlc_timer.timeout.connect(self._sync_vlc_position)
        self.vlc_scrub_timer = QTimer(self)
        self.vlc_scrub_timer.setSingleShot(True)
        self.vlc_scrub_timer.timeout.connect(self._apply_pending_vlc_scrub)
        self.seek_timer = QTimer(self)
        self.seek_timer.setSingleShot(True)
        self.seek_timer.timeout.connect(self._apply_pending_seek)
        self._pending_vlc_scrub_ms: Optional[int] = None
        self._pending_vlc_scrub_refresh_thumbnails = False
        self._vlc_terminal_seek_retry_active = False
        self._vlc_terminal_seek_retry_ms: Optional[int] = None
        self._last_vlc_scrub_at = 0.0
        self._pending_seek_ms: Optional[int] = None
        self._pending_seek_refresh_thumbnails = False
        self.thumbnail_timer = QTimer(self)
        self.thumbnail_timer.setSingleShot(True)
        self.thumbnail_timer.timeout.connect(self._refresh_thumbnail_strip)
        self._thumbnail_pending_center_ms = 0
        self._thumbnail_refresh_running = False
        self._thumbnail_refresh_queued = False
        self._thumbnail_request_id = 0
        self._thumbnail_thread: Optional[QThread] = None
        self._thumbnail_worker: Optional[ThumbnailWorker] = None

        self.drop_filter = VideoDropFilter(self)
        self.drop_filter.dropped.connect(self.open_video)
        self.installEventFilter(self.drop_filter)

        self._build_ui()
        self._enable_drop_everywhere()
        self._wire_events()
        self._apply_style()
        self._refresh_capture_labels()

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setAcceptDrops(True)
        central.installEventFilter(self.drop_filter)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        player_panel = QFrame(self)
        player_panel.setObjectName("playerPanel")
        player_layout = QVBoxLayout(player_panel)
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.setSpacing(0)

        self.frame_label = QLabel("拖曳影片到這裡")
        self.frame_label.setObjectName("videoSurface")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_label.setAcceptDrops(True)
        self.frame_label.setAttribute(Qt.WA_NativeWindow, True)
        self.frame_label.setAttribute(Qt.WA_DontCreateNativeAncestors, True)
        self.frame_label.installEventFilter(self.drop_filter)
        player_layout.addWidget(self.frame_label, stretch=1)

        bottom_panel = QFrame(self)
        bottom_panel.setObjectName("bottomPanel")
        bottom_panel.setAcceptDrops(True)
        bottom_panel.installEventFilter(self.drop_filter)
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(10, 8, 10, 9)
        bottom_layout.setSpacing(7)

        self.file_label = QLabel("拖曳影片到這裡，或按「開啟影片」")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        timeline = QHBoxLayout()
        timeline.setSpacing(10)
        self.current_time_label = QLabel("0:00")
        self.current_time_label.setObjectName("timeLabel")
        self.seek_slider = ClickableSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSingleStep(1)
        self.seek_slider.setPageStep(5)
        self.seek_slider.setTracking(True)
        self.seek_slider.setToolTip("拖曳可逐秒掃描畫面")
        self.total_time_label = QLabel("0:00")
        self.total_time_label.setObjectName("timeLabel")
        timeline.addWidget(self.current_time_label)
        timeline.addWidget(self.seek_slider, stretch=1)
        timeline.addWidget(self.total_time_label)

        self.thumbnail_strip = ThumbnailRangeStrip()
        self.thumbnail_strip.rangeChanged.connect(self._on_thumbnail_range_changed)
        self.thumbnail_strip.handleMoved.connect(self._on_thumbnail_handle_moved)

        main_button_row = QHBoxLayout()
        main_button_row.setSpacing(7)
        self.open_button = QPushButton("開啟影片")
        self.play_button = QPushButton("播放")
        self.rewind_button = QPushButton("-5秒")
        self.forward_button = QPushButton("+5秒")
        self.set_start_button = QPushButton("[")
        self.set_end_button = QPushButton("]")
        self.set_start_button.setToolTip("擷取左邊開頭 ([)")
        self.set_end_button.setToolTip("擷取右邊結尾 (])")
        self.play_button.setEnabled(False)
        self.rewind_button.setEnabled(False)
        self.forward_button.setEnabled(False)
        self.set_start_button.setEnabled(False)
        self.set_end_button.setEnabled(False)

        self.add_segment_button = QPushButton("加入")
        self.add_segment_button.setEnabled(False)
        self.delete_segment_button = QPushButton("刪除")
        self.clear_segments_button = QPushButton("清空")
        self.delete_segment_button.setEnabled(False)
        self.clear_segments_button.setEnabled(False)
        self.copy_button = QPushButton("複製")
        self.copy_button.setObjectName("primaryButton")
        self.copy_button.setEnabled(False)
        self.write_button = QPushButton("寫入")
        self.write_button.setObjectName("primaryButton")
        self.write_button.setEnabled(False)

        self.start_value_label = QLabel("--:--")
        self.start_value_label.setObjectName("captureValue")
        self.end_value_label = QLabel("--:--")
        self.end_value_label.setObjectName("captureValue")
        self.capture_total_title = QLabel("總長")
        self.capture_total_title.setObjectName("caption")
        self.capture_total_label = QLabel("0.0 分")
        self.capture_total_label.setObjectName("captureValue")
        self.capture_total_label.setToolTip("已加入片段加上目前 [ ] 擷取範圍的總長")

        self.segment_list = SegmentList(self)
        self.segment_list.setObjectName("segmentList")
        self.segment_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.segment_list.setMaximumHeight(64)
        self.segment_list.setMinimumHeight(46)

        self.output_preview = QLabel("")
        self.output_preview.setObjectName("outputPreview")
        self.output_preview.setWordWrap(True)
        self.output_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.output_preview.setMaximumHeight(64)

        self.status_label = QLabel("尚未載入影片")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)

        output_path_row = QHBoxLayout()
        output_path_row.setSpacing(7)
        self.output_path_label = QLabel("寫入檔案")
        self.output_path_label.setObjectName("caption")
        self.output_path_edit = QLineEdit(DEFAULT_OUTPUT_LIST_PATH)
        self.output_path_edit.setObjectName("outputPathEdit")
        self.output_path_edit.setMinimumWidth(360)
        self.output_path_edit.setToolTip("按「寫入」會把目前複製內容追加到這個檔案最下面")
        self.pick_output_path_button = QPushButton("選擇")
        self.pick_output_path_button.setToolTip("選擇要追加寫入的清單檔案")
        output_path_row.addWidget(self.output_path_label)
        output_path_row.addWidget(self.output_path_edit, stretch=1)
        output_path_row.addWidget(self.pick_output_path_button)

        main_button_row.addWidget(self.open_button)
        main_button_row.addWidget(self.play_button)
        main_button_row.addWidget(self.rewind_button)
        main_button_row.addWidget(self.forward_button)
        main_button_row.addSpacing(8)
        main_button_row.addWidget(self.set_start_button)
        main_button_row.addWidget(self.start_value_label)
        main_button_row.addWidget(self.set_end_button)
        main_button_row.addWidget(self.end_value_label)
        main_button_row.addWidget(self.capture_total_title)
        main_button_row.addWidget(self.capture_total_label)
        main_button_row.addWidget(self.add_segment_button)
        main_button_row.addWidget(self.delete_segment_button)
        main_button_row.addWidget(self.clear_segments_button)
        main_button_row.addWidget(self.copy_button)
        main_button_row.addWidget(self.write_button)
        main_button_row.addStretch(1)
        main_button_row.addWidget(self.status_label, stretch=1)

        bottom_grid = QGridLayout()
        bottom_grid.setHorizontalSpacing(8)
        bottom_grid.setVerticalSpacing(5)
        bottom_grid.addWidget(self.segment_list, 0, 0)
        bottom_grid.addWidget(self.output_preview, 0, 1)
        bottom_grid.setColumnStretch(0, 3)
        bottom_grid.setColumnStretch(1, 2)

        bottom_layout.addWidget(self.file_label)
        bottom_layout.addLayout(timeline)
        bottom_layout.addWidget(self.thumbnail_strip)
        bottom_layout.addLayout(main_button_row)
        bottom_layout.addLayout(output_path_row)
        bottom_layout.addLayout(bottom_grid)

        root.addWidget(player_panel, stretch=1)
        root.addWidget(bottom_panel, stretch=0)
        self.setCentralWidget(central)

        self.open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.rewind_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.forward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.copy_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.write_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))

    def _enable_drop_everywhere(self) -> None:
        for widget in [self, self.centralWidget(), *self.findChildren(QWidget)]:
            if widget is None:
                continue
            widget.setAcceptDrops(True)
            widget.installEventFilter(self.drop_filter)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt override name
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt override name
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt override name
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_video(url.toLocalFile())
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def _wire_events(self) -> None:
        self.open_button.clicked.connect(self.pick_video)
        self.play_button.clicked.connect(self.toggle_playback)
        self.rewind_button.clicked.connect(lambda: self.seek_relative(-5000))
        self.forward_button.clicked.connect(lambda: self.seek_relative(5000))
        self.set_start_button.clicked.connect(self.capture_start)
        self.set_end_button.clicked.connect(self.capture_end)
        self.add_segment_button.clicked.connect(self.add_segment)
        self.copy_button.clicked.connect(self.copy_output)
        self.write_button.clicked.connect(self.write_output_to_file)
        self.pick_output_path_button.clicked.connect(self.pick_output_file)
        self.delete_segment_button.clicked.connect(self.delete_selected_segments)
        self.clear_segments_button.clicked.connect(self.clear_segments)

        self.seek_slider.scanStarted.connect(self._on_slider_pressed)
        self.seek_slider.scanMoved.connect(self._on_slider_moved)
        self.seek_slider.scanFinished.connect(self._on_slider_released)
        self.segment_list.itemSelectionChanged.connect(self.update_actions)
        self.segment_list.itemDoubleClicked.connect(self.seek_to_segment_start)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_playback)
        QShortcut(QKeySequence("S"), self, activated=self.capture_start)
        QShortcut(QKeySequence("E"), self, activated=self.capture_end)
        QShortcut(QKeySequence("["), self, activated=self.capture_start)
        QShortcut(QKeySequence("]"), self, activated=self.capture_end)
        QShortcut(QKeySequence("A"), self, activated=self.add_segment)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self.copy_output)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.write_output_to_file)

    def _apply_style(self) -> None:
        QApplication.instance().setFont(QFont("Microsoft JhengHei UI", 10))
        self.setStyleSheet(
            """
            QMainWindow {
                background: #111827;
                color: #e5e7eb;
            }
            QFrame#playerPanel, QFrame#bottomPanel {
                background: #172033;
                border: 1px solid #273247;
                border-radius: 8px;
            }
            QWidget#videoSurface, QLabel#videoSurface {
                background: #05070d;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                color: #64748b;
                font-size: 18px;
                font-weight: 700;
            }
            QFrame#bottomPanel {
                background: #101827;
            }
            QLabel#fileLabel {
                color: #c7d2fe;
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#timeLabel {
                color: #d1d5db;
                min-width: 58px;
                font-variant-numeric: tabular-nums;
            }
            QLabel#panelTitle {
                color: #f9fafb;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#panelSubtitle, QLabel#statusLabel {
                color: #9ca3af;
                font-size: 11px;
            }
            QLabel#caption {
                color: #93c5fd;
                font-weight: 700;
                font-size: 11px;
            }
            QLabel#captureValue {
                color: #ffffff;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 5px 6px;
                font-size: 14px;
                font-weight: 700;
                font-variant-numeric: tabular-nums;
                min-width: 72px;
            }
            QWidget#thumbnailStrip {
                background: #05070d;
                border: 1px solid #273247;
                border-radius: 4px;
                color: #64748b;
                font-size: 10px;
                font-variant-numeric: tabular-nums;
            }
            QLabel#outputPreview {
                color: #dbeafe;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 7px;
                min-height: 42px;
                font-family: Consolas, "Microsoft JhengHei UI";
                font-size: 11px;
            }
            QListWidget#segmentList {
                color: #e5e7eb;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 7px;
                padding: 4px;
                outline: 0;
                font-size: 13px;
                font-variant-numeric: tabular-nums;
            }
            QListWidget#segmentList::item {
                padding: 5px 7px;
                border-radius: 5px;
                margin: 1px;
            }
            QListWidget#segmentList::item:selected {
                background: #2563eb;
                color: #ffffff;
            }
            QPushButton {
                color: #e5e7eb;
                background: #25324a;
                border: 1px solid #3b4b68;
                border-radius: 6px;
                padding: 7px 8px;
                font-weight: 650;
                font-size: 12px;
            }
            QPushButton:hover:!disabled {
                background: #30415f;
                border-color: #5a72a0;
            }
            QPushButton:pressed:!disabled {
                background: #1d4ed8;
            }
            QPushButton:disabled {
                color: #6b7280;
                background: #1f2937;
                border-color: #2b3546;
            }
            QPushButton#primaryButton {
                background: #16a34a;
                border-color: #22c55e;
                color: #ffffff;
                font-size: 14px;
                padding: 9px;
            }
            QPushButton#primaryButton:hover:!disabled {
                background: #15803d;
            }
            QLineEdit#outputPathEdit {
                color: #dbeafe;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 8px;
                selection-background-color: #2563eb;
            }
            QSlider::groove:horizontal {
                height: 10px;
                background: #334155;
                border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: #60a5fa;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #f9fafb;
                border: 2px solid #60a5fa;
                width: 24px;
                margin: -8px 0;
                border-radius: 12px;
            }
            """
        )

    def pick_video(self) -> None:
        start_dir = str(Path(self.video_path).parent) if self.video_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "選擇影片", start_dir, VIDEO_FILTER)
        if path:
            self.open_video(path)

    def pick_output_file(self) -> None:
        current_path = self.output_path_edit.text().strip() or DEFAULT_OUTPUT_LIST_PATH
        path, _ = QFileDialog.getSaveFileName(self, "選擇寫入檔案", current_path, TEXT_FILTER)
        if path:
            self.output_path_edit.setText(path)

    def open_video(self, path: str) -> None:
        if not os.path.isfile(path):
            QMessageBox.warning(self, "無法開啟", "找不到影片檔案。")
            return

        self._cancel_thumbnail_worker()
        self._stop_vlc_playback()
        self._stop_fallback_playback()
        self.video_path = os.path.abspath(path)
        self.pending_start_ms = None
        self.pending_end_ms = None
        self.segment_list.clear()
        self.duration_ms = 0
        self._fallback_position_ms = 0
        self.frame_label.setText("正在載入影片...")
        self._clear_thumbnail_strip()

        self.file_label.setText(self.video_path)
        self.status_label.setText("正在載入影片")
        self.play_button.setEnabled(True)
        self.rewind_button.setEnabled(True)
        self.forward_button.setEnabled(True)
        self.set_start_button.setEnabled(True)
        self.set_end_button.setEnabled(True)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSingleStep(1)
        self.seek_slider.setPageStep(5)
        self.current_time_label.setText("0:00")
        self.total_time_label.setText("0:00")
        self._refresh_capture_labels()
        self.update_actions()

        QTimer.singleShot(0, lambda: self._start_fallback_playback("逐秒掃描播放模式已啟動"))

    def current_position_ms(self) -> int:
        if self._vlc_active and self._vlc_player is not None:
            position_ms = int(self._vlc_player.get_time() or 0)
            return max(0, position_ms)
        if self._fallback_active:
            return int(self._fallback_position_ms)
        return 0

    def _slider_value_to_ms(self, slider_value: int) -> int:
        return max(0, int(slider_value) * SLIDER_STEP_MS)

    def _position_to_slider_value(self, position_ms: int) -> int:
        return max(0, int(position_ms) // SLIDER_STEP_MS)

    def _duration_to_slider_max(self, duration_ms: int) -> int:
        return max(0, int((max(0, duration_ms) + SLIDER_STEP_MS - 1) // SLIDER_STEP_MS))

    def _set_slider_duration(self, duration_ms: int) -> None:
        previous_duration = self.duration_ms
        self.duration_ms = max(0, int(duration_ms))
        slider_max = self._duration_to_slider_max(self.duration_ms)
        self.seek_slider.setRange(0, slider_max)
        self.seek_slider.setSingleStep(1)
        self.seek_slider.setPageStep(5)
        self.total_time_label.setText(format_time(self.duration_ms))
        self.thumbnail_strip.set_duration(self.duration_ms)
        if previous_duration <= 0 < self.duration_ms and not self.thumbnail_strip.thumbnails:
            self._schedule_thumbnail_preview(0)

    def _clear_thumbnail_strip(self) -> None:
        self.thumbnail_strip.set_thumbnails([])
        self.thumbnail_strip.set_range(None, None)

    def _schedule_thumbnail_refresh(self, center_ms: Optional[int] = None) -> None:
        if not self.video_path or self.pending_start_ms is None or self.pending_end_ms is None:
            return
        self._thumbnail_pending_center_ms = self.current_position_ms() if center_ms is None else max(0, int(center_ms))
        view_start_ms, view_end_ms = self._thumbnail_view_window()
        self.thumbnail_strip.set_view(view_start_ms, view_end_ms)
        self.thumbnail_strip.set_range(self.pending_start_ms, self.pending_end_ms)
        self.thumbnail_timer.start(THUMBNAIL_REFRESH_DELAY_MS)

    def _schedule_thumbnail_preview(self, center_ms: Optional[int] = None) -> None:
        if not self.video_path or self.duration_ms <= 0:
            return
        self._thumbnail_pending_center_ms = self.current_position_ms() if center_ms is None else max(0, int(center_ms))
        view_start_ms, view_end_ms = self._thumbnail_view_window()
        self.thumbnail_strip.set_view(view_start_ms, view_end_ms)
        self.thumbnail_strip.set_range(self.pending_start_ms, self.pending_end_ms)
        self.thumbnail_timer.start(THUMBNAIL_REFRESH_DELAY_MS)

    def _refresh_thumbnail_strip(self) -> None:
        if (
            not self.video_path
            or self.duration_ms <= 0
        ):
            return

        if self._thumbnail_refresh_running:
            self._thumbnail_request_id += 1
            self._thumbnail_refresh_queued = True
            if self._thumbnail_thread is not None:
                self._thumbnail_thread.requestInterruption()
            return

        view_start_ms, view_end_ms = self._thumbnail_view_window()
        self.thumbnail_strip.set_view(view_start_ms, view_end_ms)
        self._thumbnail_request_id += 1
        request_id = self._thumbnail_request_id
        self._thumbnail_refresh_running = True

        thread = QThread(self)
        worker = ThumbnailWorker(request_id, self.video_path, view_start_ms, view_end_ms)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_thumbnail_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thumbnail_thread_finished)
        self._thumbnail_thread = thread
        self._thumbnail_worker = worker
        thread.start()

    def _on_thumbnail_worker_finished(
        self, request_id: int, view_start_ms: int, view_end_ms: int, thumbnails: list[tuple[int, QImage]]
    ) -> None:
        if request_id != self._thumbnail_request_id:
            return
        self.thumbnail_strip.set_view(view_start_ms, view_end_ms)
        self.thumbnail_strip.set_thumbnails([(position_ms, QPixmap.fromImage(image)) for position_ms, image in thumbnails])
        self.thumbnail_strip.set_range(self.pending_start_ms, self.pending_end_ms)

    def _on_thumbnail_thread_finished(self) -> None:
        self._thumbnail_refresh_running = False
        self._thumbnail_thread = None
        self._thumbnail_worker = None
        if self._thumbnail_refresh_queued:
            self._thumbnail_refresh_queued = False
            QTimer.singleShot(0, self._refresh_thumbnail_strip)

    def _cancel_thumbnail_worker(self, wait: bool = False) -> None:
        self.thumbnail_timer.stop()
        self._thumbnail_request_id += 1
        self._thumbnail_refresh_queued = False
        if self._thumbnail_thread is not None and self._thumbnail_thread.isRunning():
            self._thumbnail_thread.requestInterruption()
            self._thumbnail_thread.quit()
            if wait:
                self._thumbnail_thread.wait(6500)
                if not self._thumbnail_thread.isRunning():
                    self._thumbnail_refresh_running = False
                    self._thumbnail_thread = None
                    self._thumbnail_worker = None

    def _thumbnail_view_window(self) -> tuple[int, int]:
        if self.duration_ms <= 0:
            return 0, 0

        if self.pending_start_ms is not None and self.pending_end_ms is not None:
            start_ms, end_ms = sorted((self.pending_start_ms, self.pending_end_ms))
            selected_duration = max(SLIDER_STEP_MS, end_ms - start_ms)
            window_ms = max(60_000, selected_duration * 3)
            center_ms = (start_ms + end_ms) // 2
        else:
            window_ms = min(max(60_000, self.duration_ms // 12), self.duration_ms)
            center_ms = self.current_position_ms()

        window_ms = min(window_ms, self.duration_ms)
        view_start = max(0, center_ms - window_ms // 2)
        view_end = view_start + window_ms
        if view_end > self.duration_ms:
            view_end = self.duration_ms
            view_start = max(0, view_end - window_ms)
        return int(view_start), int(view_end)

    def _is_currently_playing(self) -> bool:
        if self._vlc_active:
            return self._vlc_playing
        if self._fallback_active:
            return self._fallback_playing
        return False

    def _pause_for_slider_scan(self) -> None:
        if self._vlc_active:
            return
        elif self._fallback_active:
            self._pause_fallback_playback()

    def _resume_after_slider_scan(self) -> None:
        if not self._resume_after_slider:
            return
        self._resume_after_slider = False
        if self._vlc_active:
            self._resume_vlc_playback()
        elif self._fallback_active:
            self._resume_fallback_playback()

    def _schedule_player_position(self, position_ms: int, refresh_thumbnails: bool = False) -> None:
        upper_bound = self.duration_ms if self.duration_ms > 0 else max(position_ms, 0)
        if self._vlc_active:
            self._schedule_vlc_scrub(
                max(0, min(int(position_ms), upper_bound)),
                refresh_thumbnails=refresh_thumbnails,
            )
            return
        self._pending_seek_ms = max(0, min(int(position_ms), upper_bound))
        self._pending_seek_refresh_thumbnails = refresh_thumbnails
        self.seek_timer.start(SEEK_DEBOUNCE_MS)

    def _schedule_vlc_scrub(self, position_ms: int, refresh_thumbnails: bool = False) -> None:
        self._pending_vlc_scrub_ms = int(position_ms)
        self._pending_vlc_scrub_refresh_thumbnails = self._pending_vlc_scrub_refresh_thumbnails or refresh_thumbnails
        elapsed_ms = (time.perf_counter() - self._last_vlc_scrub_at) * 1000
        if elapsed_ms >= VLC_SCRUB_THROTTLE_MS:
            self._apply_pending_vlc_scrub()
            return
        if not self.vlc_scrub_timer.isActive():
            self.vlc_scrub_timer.start(max(1, int(VLC_SCRUB_THROTTLE_MS - elapsed_ms)))

    def _apply_pending_vlc_scrub(self) -> None:
        if self._pending_vlc_scrub_ms is None:
            return
        position_ms = self._pending_vlc_scrub_ms
        refresh_thumbnails = self._pending_vlc_scrub_refresh_thumbnails
        self._pending_vlc_scrub_ms = None
        self._pending_vlc_scrub_refresh_thumbnails = False
        self._last_vlc_scrub_at = time.perf_counter()
        self._seek_vlc(position_ms, refresh_thumbnails=refresh_thumbnails)

    def _apply_pending_seek(self) -> None:
        if self._pending_seek_ms is None:
            return
        position_ms = self._pending_seek_ms
        refresh_thumbnails = self._pending_seek_refresh_thumbnails
        self._pending_seek_ms = None
        self._pending_seek_refresh_thumbnails = False
        self.set_player_position(position_ms, refresh_thumbnails=refresh_thumbnails)

    def _seek_player_position_now(self, position_ms: int, refresh_thumbnails: bool = False) -> None:
        self.seek_timer.stop()
        self.vlc_scrub_timer.stop()
        self._pending_seek_ms = None
        self._pending_seek_refresh_thumbnails = False
        self._pending_vlc_scrub_ms = None
        self._pending_vlc_scrub_refresh_thumbnails = False
        self.set_player_position(position_ms, refresh_thumbnails=refresh_thumbnails)

    def set_player_position(self, position_ms: int, refresh_thumbnails: bool = False) -> None:
        upper_bound = self.duration_ms if self.duration_ms > 0 else max(position_ms, 0)
        position_ms = max(0, min(int(position_ms), upper_bound))
        if self._vlc_active:
            self._seek_vlc(position_ms, refresh_thumbnails=refresh_thumbnails)
            return
        if self._fallback_active:
            self._seek_fallback(position_ms, refresh_thumbnails=refresh_thumbnails)
            return
        self._update_position_labels(position_ms)

    def _start_vlc_playback(self, reason: str) -> bool:
        if vlc is None or not self.video_path:
            return False
        try:
            if self._vlc_instance is None:
                self._vlc_instance = vlc.Instance(
                    "--quiet",
                    "--no-video-title-show",
                    "--avcodec-hw=any",
                    "--file-caching=220",
                    "--input-fast-seek",
                    "--drop-late-frames",
                    "--skip-frames",
                )
            if self._vlc_instance is None:
                return False

            self._show_safe_preview_frame(0)
            media = self._vlc_instance.media_new(self.video_path)
            media.add_option(":file-caching=220")
            media.add_option(":input-fast-seek")
            player = self._vlc_instance.media_player_new()
            player.set_media(media)
            player.set_hwnd(int(self.frame_label.winId()))
            player.video_set_key_input(False)
            player.video_set_mouse_input(False)
            self._vlc_player = player
            self._vlc_active = True
            self._vlc_playing = True
            self._fallback_active = False
            self._fallback_playing = False
            self._fallback_backend = "VLC"
            self._set_play_button_state(is_playing=True)
            self.status_label.setText(f"{reason}（VLC 硬體播放器）")
            self._apply_vlc_crop()
            if player.play() == -1:
                self._stop_vlc_playback()
                return False
            self.vlc_timer.start(100)
            QTimer.singleShot(250, self._sync_vlc_position)
            return True
        except Exception as exc:
            self._stop_vlc_playback()
            self.status_label.setText(f"VLC 播放器啟動失敗，改用備援：{exc}")
            return False

    def _stop_vlc_playback(self) -> None:
        self.vlc_timer.stop()
        if self._vlc_player is not None:
            try:
                self._vlc_player.stop()
                self._vlc_player.release()
            except Exception:
                pass
        self._vlc_player = None
        self._vlc_active = False
        self._vlc_playing = False

    def _pause_vlc_playback(self) -> None:
        if not self._vlc_player:
            return
        self._vlc_player.set_pause(1)
        self._vlc_playing = False
        self._set_play_button_state(is_playing=False)

    def _resume_vlc_playback(self) -> None:
        if not self._vlc_player:
            return
        try:
            state = self._vlc_player.get_state()
        except Exception:
            state = None
        if vlc is not None and state in (vlc.State.Ended, vlc.State.Stopped):
            try:
                target_ms = self._fallback_position_ms
                if self.duration_ms > 0 and target_ms >= self.duration_ms - 500:
                    target_ms = 0
                self._vlc_player.play()
                QApplication.processEvents()
                self._vlc_player.set_time(max(0, min(target_ms, self.duration_ms)))
            except Exception:
                return
        else:
            self._vlc_player.set_pause(0)
        self._vlc_playing = True
        self._set_play_button_state(is_playing=True)

    def _seek_vlc(self, position_ms: int, refresh_thumbnails: bool = True) -> None:
        if not self._vlc_player:
            return
        position_ms = max(0, min(int(position_ms), self.duration_ms if self.duration_ms > 0 else int(position_ms)))
        should_pause_after_seek = not self._vlc_playing
        terminal_state = False
        try:
            state = self._vlc_player.get_state()
        except Exception:
            state = None

        if vlc is not None and state in (vlc.State.Ended, vlc.State.Stopped):
            terminal_state = True
            should_pause_after_seek = True
            try:
                self._vlc_player.play()
                QApplication.processEvents()
            except Exception:
                pass

        self._vlc_player.set_time(position_ms)
        self._fallback_position_ms = position_ms
        self._update_position_labels(self._fallback_position_ms)
        if should_pause_after_seek:
            try:
                self._vlc_player.set_pause(1)
                self._vlc_playing = False
                self._set_play_button_state(is_playing=False)
                self._vlc_player.next_frame()
            except Exception:
                pass
        if terminal_state:
            self._queue_vlc_terminal_seek_retry(position_ms)
        if refresh_thumbnails:
            self._schedule_thumbnail_refresh(position_ms)

    def _queue_vlc_terminal_seek_retry(self, position_ms: int) -> None:
        self._vlc_terminal_seek_retry_ms = max(0, int(position_ms))
        if self._vlc_terminal_seek_retry_active:
            return
        self._vlc_terminal_seek_retry_active = True
        QTimer.singleShot(80, self._retry_vlc_terminal_seek)

    def _retry_vlc_terminal_seek(self) -> None:
        self._vlc_terminal_seek_retry_active = False
        if not self._vlc_active or self._vlc_player is None or self._vlc_terminal_seek_retry_ms is None:
            return
        position_ms = self._vlc_terminal_seek_retry_ms
        self._vlc_terminal_seek_retry_ms = None
        try:
            self._vlc_player.set_time(position_ms)
            self._fallback_position_ms = position_ms
            self._update_position_labels(position_ms)
            if not self._vlc_playing:
                self._vlc_player.set_pause(1)
                self._vlc_player.next_frame()
        except Exception:
            pass

    def _sync_vlc_position(self) -> None:
        if not self._vlc_active or self._vlc_player is None:
            return
        try:
            state = self._vlc_player.get_state()
            length_ms = int(self._vlc_player.get_length() or 0)
            position_ms = int(self._vlc_player.get_time() or 0)
        except Exception:
            return

        if length_ms > 0 and abs(length_ms - self.duration_ms) > 500:
            self._set_slider_duration(length_ms)
        if position_ms >= 0:
            self._fallback_position_ms = position_ms
            self._update_position_labels(position_ms)

        if vlc is not None and state == vlc.State.Error:
            self._vlc_playing = False
            self._set_play_button_state(is_playing=False)
            self.status_label.setText("VLC 播放這支影片時發生錯誤")
        elif vlc is not None and state in (vlc.State.Ended, vlc.State.Stopped):
            self._vlc_playing = False
            self._set_play_button_state(is_playing=False)

    def _apply_vlc_crop(self) -> None:
        if not self._vlc_active or self._vlc_player is None:
            return
        size = self.frame_label.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        try:
            self._vlc_player.video_set_crop_geometry(f"{size.width()}:{size.height()}")
        except Exception:
            pass

    def _show_safe_preview_frame(self, position_ms: int) -> None:
        if not self.video_path:
            return
        max_width = max(640, min(1920, self.frame_label.width() or 1280))
        image = read_video_image_with_ffmpeg(self.video_path, position_ms, max_width, timeout=5)
        if image.isNull():
            self.frame_label.clear()
            self.frame_label.setText("正在載入影片...")
            return
        self.frame_label.setPixmap(QPixmap.fromImage(image))

    def _start_fallback_playback(self, reason: str) -> None:
        if self._fallback_active or not self.video_path:
            return
        tried_vlc = vlc is not None
        if self._start_vlc_playback(reason):
            return
        if tried_vlc:
            self.frame_label.setText("VLC 無法播放這支影片")
            self.status_label.setText("VLC 播放器無法開啟這支影片，已停止以避免 PyAV/OpenCV 解碼崩潰。")
            self._set_play_button_state(is_playing=False)
            return
        if cv2 is None:
            self.status_label.setText(f"{reason}，但目前 Python 沒有 OpenCV 可用。")
            return

        try:
            with VIDEO_READ_LOCK:
                capture, backend_name = open_video_reader(self.video_path)
        except Exception as exc:
            message = f"{reason}，但相容模式也無法開啟這支影片：{exc}"
            self.frame_label.clear()
            self.frame_label.setText("無法播放這支影片")
            self.status_label.setText(message)
            self._set_play_button_state(is_playing=False)
            return

        self._fallback_capture = capture
        self._fallback_backend = backend_name
        self._fallback_active = True
        self._fallback_playing = True
        self._fallback_position_ms = 0

        fps = float(getattr(capture, "fps", 0) or 0)
        if fps <= 1 or fps > 240:
            fps = 30.0
        self._fallback_fps = fps

        duration_ms = int(getattr(capture, "duration_ms", 0) or 0)
        if duration_ms > 0 and self.duration_ms <= 0:
            self._set_slider_duration(duration_ms)

        self.status_label.setText(f"{reason}（{backend_name}）")
        self._set_play_button_state(is_playing=True)
        self._sync_playback_clock()
        self._render_fallback_frame()
        self._sync_playback_clock()
        self.fallback_timer.start(self._playback_interval_ms())

    def _stop_fallback_playback(self) -> None:
        self.fallback_timer.stop()
        if self._fallback_capture is not None:
            with VIDEO_READ_LOCK:
                self._fallback_capture.release()
        self._fallback_capture = None
        self._fallback_backend = ""
        self._fallback_active = False
        self._fallback_playing = False
        self.frame_label.clear()
        self.frame_label.setText("拖曳影片到這裡")

    def _pause_fallback_playback(self) -> None:
        self.fallback_timer.stop()
        self._fallback_playing = False
        self._sync_playback_clock()
        self._set_play_button_state(is_playing=False)

    def _resume_fallback_playback(self) -> None:
        if not self._fallback_capture:
            return
        self._fallback_playing = True
        self._sync_playback_clock()
        self._set_play_button_state(is_playing=True)
        self.fallback_timer.start(self._playback_interval_ms())

    def _seek_fallback(self, position_ms: int, refresh_thumbnails: bool = True) -> None:
        if not self._fallback_capture:
            return
        was_playing = self._fallback_playing
        self.fallback_timer.stop()
        try:
            with VIDEO_READ_LOCK:
                ok, frame = self._fallback_capture.seek_read(position_ms)
                current_ms = int(getattr(self._fallback_capture, "last_position_ms", position_ms) or position_ms)
        except Exception as exc:
            self._fallback_position_ms = position_ms
            self._update_position_labels(position_ms)
            self.status_label.setText(f"跳轉時解碼失敗：{exc}")
            ok = False
            frame = None
            current_ms = position_ms

        if ok and frame is not None:
            self._fallback_position_ms = max(0, current_ms)
            self._show_fallback_frame(frame)
            self._update_position_labels(self._fallback_position_ms)
        else:
            self._fallback_position_ms = position_ms
            self._update_position_labels(position_ms)
        if refresh_thumbnails:
            self._schedule_thumbnail_refresh(position_ms)
        if was_playing:
            self._sync_playback_clock()
            self.fallback_timer.start(self._playback_interval_ms())

    def _playback_interval_ms(self) -> int:
        fps = min(max(self._fallback_fps, 1.0), PLAYBACK_MAX_FPS)
        return max(30, int(1000 / fps))

    def _sync_playback_clock(self) -> None:
        self._playback_clock_position_ms = int(self._fallback_position_ms)
        self._playback_clock_started_at = time.perf_counter()

    def _playback_target_position_ms(self) -> int:
        if not self._fallback_playing or self._playback_clock_started_at <= 0:
            return int(self._fallback_position_ms)
        elapsed_ms = int((time.perf_counter() - self._playback_clock_started_at) * 1000)
        target_ms = self._playback_clock_position_ms + max(0, elapsed_ms)
        if self.duration_ms > 0:
            target_ms = min(target_ms, self.duration_ms)
        return max(0, target_ms)

    def _render_fallback_frame(self) -> None:
        if not self._fallback_capture:
            return
        if not VIDEO_READ_LOCK.acquire(blocking=False):
            return
        error_message = ""
        try:
            target_ms = self._playback_target_position_ms()
            ok, frame, position_ms = self._read_playback_frame_for_target(target_ms)
        except Exception as exc:
            ok = False
            frame = None
            position_ms = self._fallback_position_ms
            error_message = str(exc)
        finally:
            VIDEO_READ_LOCK.release()

        if error_message:
            self._pause_fallback_playback()
            self.status_label.setText(f"影片解碼失敗：{error_message}")
            return
        if not ok:
            self._pause_fallback_playback()
            self.status_label.setText("影片播放到結尾")
            return

        self._fallback_position_ms = max(0, position_ms)
        self._show_fallback_frame(frame)
        self._update_position_labels(self._fallback_position_ms)

    def _read_playback_frame_for_target(self, target_ms: int):
        if (
            self.duration_ms > 0
            and target_ms >= self.duration_ms
            and self._fallback_position_ms >= self.duration_ms - 500
        ):
            return False, None, self.duration_ms

        if target_ms - self._fallback_position_ms > PLAYBACK_SEEK_CATCHUP_MS:
            ok, frame = self._fallback_capture.seek_read(target_ms)
            position_ms = int(getattr(self._fallback_capture, "last_position_ms", 0) or 0)
            return ok, frame, position_ms

        frame = None
        ok = False
        position_ms = self._fallback_position_ms
        frame_margin_ms = max(12, int(500 / max(self._fallback_fps, 1.0)))
        for _ in range(PLAYBACK_CATCHUP_FRAME_LIMIT):
            ok, next_frame = self._fallback_capture.read()
            if not ok:
                break
            frame = next_frame
            position_ms = int(getattr(self._fallback_capture, "last_position_ms", position_ms) or position_ms)
            if position_ms + frame_margin_ms >= target_ms:
                break

        return ok and frame is not None, frame, position_ms

    def _show_fallback_frame(self, frame) -> None:
        label_size = self.frame_label.size()
        if not label_size.isEmpty():
            frame = resize_frame_to_cover(frame, label_size.width(), label_size.height())
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.frame_label.setPixmap(QPixmap.fromImage(image))

    def _update_position_labels(self, position_ms: int) -> None:
        if not self._slider_is_pressed:
            self._set_slider_value_without_feedback(self._position_to_slider_value(position_ms))
        self.current_time_label.setText(format_time(position_ms))

    def _set_play_button_state(self, is_playing: bool) -> None:
        if is_playing:
            self.play_button.setText("暫停")
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setText("播放")
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def toggle_playback(self) -> None:
        if not self.video_path:
            return
        if self._vlc_active:
            if self._vlc_playing:
                self._pause_vlc_playback()
            else:
                self._resume_vlc_playback()
            return
        if self._fallback_active:
            if self._fallback_playing:
                self._pause_fallback_playback()
            else:
                self._resume_fallback_playback()
            return

    def seek_relative(self, delta_ms: int) -> None:
        if not self.video_path:
            return
        new_position = max(0, min(self.duration_ms, self.current_position_ms() + delta_ms))
        self.set_player_position(new_position)

    def capture_start(self) -> None:
        if not self.video_path:
            return
        self.pending_start_ms = self.current_position_ms()
        self._refresh_capture_labels()
        self.thumbnail_strip.set_range(self.pending_start_ms, self.pending_end_ms)
        self._schedule_thumbnail_refresh(self.pending_start_ms)
        self.status_label.setText(f"[ 左邊開頭：{format_time(self.pending_start_ms)}")
        self.update_actions()

    def capture_end(self) -> None:
        if not self.video_path:
            return
        self.pending_end_ms = self.current_position_ms()
        self._refresh_capture_labels()
        self.thumbnail_strip.set_range(self.pending_start_ms, self.pending_end_ms)
        self._schedule_thumbnail_refresh(self.pending_end_ms)
        self.status_label.setText(f"] 右邊結尾：{format_time(self.pending_end_ms)}")
        self.update_actions()

    def add_segment(self) -> None:
        if self.pending_start_ms is None or self.pending_end_ms is None:
            return
        segment = ClipSegment(self.pending_start_ms, self.pending_end_ms).normalized()
        if segment.start_ms == segment.end_ms:
            QMessageBox.information(self, "時間相同", "開始和結束時間相同，請先移動播放位置。")
            return

        item = QListWidgetItem(segment.label())
        item.setData(Qt.UserRole, segment)
        self.segment_list.addItem(item)
        self.pending_start_ms = None
        self.pending_end_ms = None
        self._refresh_capture_labels()
        self.thumbnail_strip.set_range(None, None)
        self._schedule_thumbnail_refresh()
        self.status_label.setText(f"已加入：{segment.label()}")
        self.update_actions()

    def delete_selected_segments(self) -> None:
        for item in self.segment_list.selectedItems():
            self.segment_list.takeItem(self.segment_list.row(item))
        self.update_actions()

    def clear_segments(self) -> None:
        self.segment_list.clear()
        self.update_actions()

    def seek_to_segment_start(self, item: QListWidgetItem) -> None:
        segment = item.data(Qt.UserRole)
        if isinstance(segment, ClipSegment):
            self.set_player_position(segment.normalized().start_ms)

    def copy_output(self) -> None:
        output = self._prepare_output_text()
        if not output:
            return
        QApplication.clipboard().setText(output)
        self.status_label.setText("已複製到剪貼簿")
        self.output_preview.setText(output)

    def write_output_to_file(self) -> None:
        output = self._prepare_output_text()
        if not output:
            return

        target_text = self.output_path_edit.text().strip().strip('"')
        if not target_text:
            QMessageBox.warning(self, "缺少檔案", "請先設定要寫入的清單檔案。")
            return

        target_path = Path(os.path.expandvars(os.path.expanduser(target_text)))
        try:
            self._append_text_block(target_path, output)
        except OSError as exc:
            QMessageBox.warning(self, "寫入失敗", f"無法寫入檔案：\n{target_path}\n\n{exc}")
            return

        self.status_label.setText(f"已寫入：{target_path}")
        self.output_preview.setText(output)

    def _prepare_output_text(self) -> str:
        if not self.video_path:
            return ""
        if self.segment_list.count() == 0 and self.pending_start_ms is not None and self.pending_end_ms is not None:
            self.add_segment()
        return self.build_output_text()

    def _append_text_block(self, target_path: Path, text: str) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        newline = self._detect_text_newline(target_path)
        prefix = ""
        if target_path.exists() and target_path.stat().st_size > 0:
            tail = target_path.read_bytes()[-8:]
            newline_bytes = newline.encode("utf-8")
            double_newline = newline_bytes + newline_bytes
            if tail.endswith(double_newline):
                prefix = ""
            elif tail.endswith(newline_bytes):
                prefix = newline
            else:
                prefix = newline + newline

        with target_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(prefix + text.rstrip() + newline + newline)

    def _detect_text_newline(self, target_path: Path) -> str:
        if not target_path.exists() or target_path.stat().st_size == 0:
            return "\n"
        with target_path.open("rb") as handle:
            sample = handle.read(4096)
        return "\r\n" if b"\r\n" in sample else "\n"

    def build_output_text(self) -> str:
        if not self.video_path or self.segment_list.count() == 0:
            return ""
        lines = [file_uri(self.video_path)]
        for index in range(self.segment_list.count()):
            item = self.segment_list.item(index)
            segment = item.data(Qt.UserRole)
            if isinstance(segment, ClipSegment):
                lines.append(segment.label())
        return "\n".join(lines)

    def _captured_duration_ms(self) -> int:
        total_ms = 0
        for index in range(self.segment_list.count()):
            item = self.segment_list.item(index)
            segment = item.data(Qt.UserRole)
            if isinstance(segment, ClipSegment):
                normalized = segment.normalized()
                total_ms += max(0, normalized.end_ms - normalized.start_ms)
        if self.pending_start_ms is not None and self.pending_end_ms is not None:
            total_ms += abs(self.pending_end_ms - self.pending_start_ms)
        return total_ms

    def _refresh_capture_total_label(self) -> None:
        total_ms = self._captured_duration_ms()
        minutes = total_ms / 60_000
        self.capture_total_label.setText(f"{minutes:.1f} 分")
        self.capture_total_label.setToolTip(f"目前擷取總長：{format_time(total_ms)}")

    def _refresh_capture_labels(self) -> None:
        self.start_value_label.setText(format_time(self.pending_start_ms) if self.pending_start_ms is not None else "--:--")
        self.end_value_label.setText(format_time(self.pending_end_ms) if self.pending_end_ms is not None else "--:--")
        self._refresh_capture_total_label()

    def _on_slider_pressed(self) -> None:
        self._slider_is_pressed = True
        self._resume_after_slider = self._is_currently_playing()
        self._pause_for_slider_scan()

    def _set_slider_value_without_feedback(self, slider_value: int) -> None:
        if self.seek_slider.value() == slider_value:
            return
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(slider_value)
        self.seek_slider.blockSignals(False)

    def _on_slider_moved(self, slider_value: int) -> None:
        self._set_slider_value_without_feedback(slider_value)
        target_ms = self._slider_value_to_ms(slider_value)
        self._schedule_player_position(target_ms)
        self.current_time_label.setText(format_time(target_ms))

    def _on_thumbnail_range_changed(self, start_ms: int, end_ms: int) -> None:
        self.pending_start_ms = start_ms
        self.pending_end_ms = end_ms
        self._refresh_capture_labels()
        self.update_actions()

    def _on_thumbnail_handle_moved(self, position_ms: int) -> None:
        self._seek_player_position_now(position_ms, refresh_thumbnails=False)

    def _on_slider_released(self, slider_value: Optional[int] = None) -> None:
        self._slider_is_pressed = False
        if slider_value is not None:
            self._set_slider_value_without_feedback(slider_value)
        self.seek_timer.stop()
        self.vlc_scrub_timer.stop()
        self._pending_seek_ms = None
        self._pending_vlc_scrub_ms = None
        self._pending_vlc_scrub_refresh_thumbnails = False
        self.set_player_position(self._slider_value_to_ms(self.seek_slider.value()))
        self._resume_after_slider_scan()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        self._cancel_thumbnail_worker(wait=True)
        self._stop_vlc_playback()
        self._stop_fallback_playback()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override name
        super().resizeEvent(event)
        self._apply_vlc_crop()

    def update_actions(self) -> None:
        has_video = bool(self.video_path)
        has_pending_segment = self.pending_start_ms is not None and self.pending_end_ms is not None
        has_segments = self.segment_list.count() > 0
        self.add_segment_button.setEnabled(has_video and has_pending_segment)
        self.copy_button.setEnabled(has_video and (has_segments or has_pending_segment))
        self.write_button.setEnabled(has_video and (has_segments or has_pending_segment))
        self.delete_segment_button.setEnabled(bool(self.segment_list.selectedItems()))
        self.clear_segments_button.setEnabled(has_segments)
        self._refresh_capture_total_label()
        self.output_preview.setText(self.build_output_text())


def main() -> int:
    set_windows_app_id()
    enable_crash_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("影片時間擷取工具")
    app.setApplicationDisplayName("影片時間擷取工具")
    app.setWindowIcon(QIcon())
    window = VideoTimeClipper()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.open_video(sys.argv[1])
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
