"""A small PyQt5 video player for collecting copy-ready time ranges."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon, QKeySequence
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
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


VIDEO_FILTER = "影片檔案 (*.mp4 *.mkv *.mov *.avi *.wmv *.m4v);;所有檔案 (*.*)"


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

    clickedValue = pyqtSignal(int)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override name
        if event.button() == Qt.LeftButton and self.maximum() > self.minimum():
            if self.orientation() == Qt.Horizontal:
                ratio = event.x() / max(1, self.width())
            else:
                ratio = 1 - (event.y() / max(1, self.height()))
            value = round(self.minimum() + ratio * (self.maximum() - self.minimum()))
            value = max(self.minimum(), min(self.maximum(), value))
            self.setValue(value)
            self.clickedValue.emit(value)
            event.accept()
            return
        super().mousePressEvent(event)


class VideoDropFilter(QObject):
    dropped = pyqtSignal(str)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt override name
        if event.type() == QEvent.DragEnter:
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
        self.resize(1180, 760)
        self.setMinimumSize(920, 620)
        self.setAcceptDrops(True)

        self.video_path: Optional[str] = None
        self.duration_ms = 0
        self.pending_start_ms: Optional[int] = None
        self.pending_end_ms: Optional[int] = None
        self._slider_is_pressed = False

        self.player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

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
        root = QHBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(16)

        player_panel = QFrame(self)
        player_panel.setObjectName("playerPanel")
        player_layout = QVBoxLayout(player_panel)
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.setSpacing(0)

        self.video_widget.setObjectName("videoSurface")
        self.video_widget.setAcceptDrops(True)
        self.video_widget.installEventFilter(self.drop_filter)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        player_layout.addWidget(self.video_widget, stretch=1)

        controls = QFrame(self)
        controls.setObjectName("controls")
        controls.setAcceptDrops(True)
        controls.installEventFilter(self.drop_filter)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 14, 18, 16)
        controls_layout.setSpacing(12)

        self.file_label = QLabel("拖曳影片到這裡，或按「開啟影片」")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        timeline = QHBoxLayout()
        timeline.setSpacing(10)
        self.current_time_label = QLabel("0:00")
        self.current_time_label.setObjectName("timeLabel")
        self.seek_slider = ClickableSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.total_time_label = QLabel("0:00")
        self.total_time_label.setObjectName("timeLabel")
        timeline.addWidget(self.current_time_label)
        timeline.addWidget(self.seek_slider, stretch=1)
        timeline.addWidget(self.total_time_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.open_button = QPushButton("開啟影片")
        self.play_button = QPushButton("播放")
        self.rewind_button = QPushButton("-5秒")
        self.forward_button = QPushButton("+5秒")
        self.play_button.setEnabled(False)
        self.rewind_button.setEnabled(False)
        self.forward_button.setEnabled(False)
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.play_button)
        button_row.addWidget(self.rewind_button)
        button_row.addWidget(self.forward_button)
        button_row.addStretch(1)

        controls_layout.addWidget(self.file_label)
        controls_layout.addLayout(timeline)
        controls_layout.addLayout(button_row)
        player_layout.addWidget(controls)

        side_panel = QFrame(self)
        side_panel.setObjectName("sidePanel")
        side_panel.setAcceptDrops(True)
        side_panel.installEventFilter(self.drop_filter)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(14)

        title = QLabel("時間段")
        title.setObjectName("panelTitle")
        subtitle = QLabel("設定開始和結束後加入清單，可一次複製多段。")
        subtitle.setObjectName("panelSubtitle")
        subtitle.setWordWrap(True)

        capture_grid = QGridLayout()
        capture_grid.setHorizontalSpacing(10)
        capture_grid.setVerticalSpacing(10)
        start_caption = QLabel("開始")
        start_caption.setObjectName("caption")
        end_caption = QLabel("結束")
        end_caption.setObjectName("caption")
        self.start_value_label = QLabel("--:--")
        self.start_value_label.setObjectName("captureValue")
        self.end_value_label = QLabel("--:--")
        self.end_value_label.setObjectName("captureValue")
        self.set_start_button = QPushButton("設為開始")
        self.set_end_button = QPushButton("設為結束")
        self.add_segment_button = QPushButton("加入時間段")
        self.set_start_button.setEnabled(False)
        self.set_end_button.setEnabled(False)
        self.add_segment_button.setEnabled(False)
        capture_grid.addWidget(start_caption, 0, 0)
        capture_grid.addWidget(self.start_value_label, 0, 1)
        capture_grid.addWidget(self.set_start_button, 0, 2)
        capture_grid.addWidget(end_caption, 1, 0)
        capture_grid.addWidget(self.end_value_label, 1, 1)
        capture_grid.addWidget(self.set_end_button, 1, 2)
        capture_grid.addWidget(self.add_segment_button, 2, 0, 1, 3)

        self.segment_list = SegmentList(self)
        self.segment_list.setObjectName("segmentList")
        self.segment_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        segment_actions = QHBoxLayout()
        segment_actions.setSpacing(10)
        self.delete_segment_button = QPushButton("刪除選取")
        self.clear_segments_button = QPushButton("清空")
        self.delete_segment_button.setEnabled(False)
        self.clear_segments_button.setEnabled(False)
        segment_actions.addWidget(self.delete_segment_button)
        segment_actions.addWidget(self.clear_segments_button)

        self.preview_label = QLabel("複製預覽")
        self.preview_label.setObjectName("caption")
        self.output_preview = QLabel("")
        self.output_preview.setObjectName("outputPreview")
        self.output_preview.setWordWrap(True)
        self.output_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.copy_button = QPushButton("複製")
        self.copy_button.setObjectName("primaryButton")
        self.copy_button.setEnabled(False)
        self.status_label = QLabel("尚未載入影片")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)

        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)
        side_layout.addLayout(capture_grid)
        side_layout.addWidget(self.segment_list, stretch=1)
        side_layout.addLayout(segment_actions)
        side_layout.addWidget(self.preview_label)
        side_layout.addWidget(self.output_preview)
        side_layout.addWidget(self.copy_button)
        side_layout.addWidget(self.status_label)

        root.addWidget(player_panel, stretch=3)
        root.addWidget(side_panel, stretch=1)
        self.setCentralWidget(central)

        self.open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.rewind_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.forward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.copy_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))

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
        self.delete_segment_button.clicked.connect(self.delete_selected_segments)
        self.clear_segments_button.clicked.connect(self.clear_segments)

        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.stateChanged.connect(self.on_state_changed)
        self.player.error.connect(self.on_player_error)

        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.sliderMoved.connect(self.player.setPosition)
        self.seek_slider.clickedValue.connect(self.player.setPosition)
        self.segment_list.itemSelectionChanged.connect(self.update_actions)
        self.segment_list.itemDoubleClicked.connect(self.seek_to_segment_start)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_playback)
        QShortcut(QKeySequence("S"), self, activated=self.capture_start)
        QShortcut(QKeySequence("E"), self, activated=self.capture_end)
        QShortcut(QKeySequence("A"), self, activated=self.add_segment)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self.copy_output)

    def _apply_style(self) -> None:
        QApplication.instance().setFont(QFont("Microsoft JhengHei UI", 10))
        self.setStyleSheet(
            """
            QMainWindow {
                background: #111827;
                color: #e5e7eb;
            }
            QFrame#playerPanel, QFrame#sidePanel {
                background: #172033;
                border: 1px solid #273247;
                border-radius: 8px;
            }
            QVideoWidget#videoSurface {
                background: #05070d;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QFrame#controls {
                background: #101827;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
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
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#panelSubtitle, QLabel#statusLabel {
                color: #9ca3af;
            }
            QLabel#caption {
                color: #93c5fd;
                font-weight: 700;
            }
            QLabel#captureValue {
                color: #ffffff;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 17px;
                font-weight: 700;
                font-variant-numeric: tabular-nums;
                min-width: 88px;
            }
            QLabel#outputPreview {
                color: #dbeafe;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 10px;
                min-height: 72px;
                font-family: Consolas, "Microsoft JhengHei UI";
            }
            QListWidget#segmentList {
                color: #e5e7eb;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 7px;
                padding: 6px;
                outline: 0;
                font-size: 16px;
                font-variant-numeric: tabular-nums;
            }
            QListWidget#segmentList::item {
                padding: 9px 10px;
                border-radius: 5px;
                margin: 2px;
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
                padding: 9px 12px;
                font-weight: 650;
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
                font-size: 16px;
                padding: 12px;
            }
            QPushButton#primaryButton:hover:!disabled {
                background: #15803d;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #334155;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #60a5fa;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #f9fafb;
                border: 2px solid #60a5fa;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            """
        )

    def pick_video(self) -> None:
        start_dir = str(Path(self.video_path).parent) if self.video_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "選擇影片", start_dir, VIDEO_FILTER)
        if path:
            self.open_video(path)

    def open_video(self, path: str) -> None:
        if not os.path.isfile(path):
            QMessageBox.warning(self, "無法開啟", "找不到影片檔案。")
            return

        self.video_path = os.path.abspath(path)
        self.pending_start_ms = None
        self.pending_end_ms = None
        self.segment_list.clear()
        self.duration_ms = 0

        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))
        self.file_label.setText(self.video_path)
        self.status_label.setText("影片已載入")
        self.play_button.setEnabled(True)
        self.rewind_button.setEnabled(True)
        self.forward_button.setEnabled(True)
        self.set_start_button.setEnabled(True)
        self.set_end_button.setEnabled(True)
        self.seek_slider.setRange(0, 0)
        self.current_time_label.setText("0:00")
        self.total_time_label.setText("0:00")
        self._refresh_capture_labels()
        self.update_actions()

        QTimer.singleShot(80, self.player.play)

    def toggle_playback(self) -> None:
        if not self.video_path:
            return
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def seek_relative(self, delta_ms: int) -> None:
        if not self.video_path:
            return
        new_position = max(0, min(self.duration_ms, self.player.position() + delta_ms))
        self.player.setPosition(new_position)

    def capture_start(self) -> None:
        if not self.video_path:
            return
        self.pending_start_ms = self.player.position()
        self._refresh_capture_labels()
        self.status_label.setText(f"開始時間：{format_time(self.pending_start_ms)}")
        self.update_actions()

    def capture_end(self) -> None:
        if not self.video_path:
            return
        self.pending_end_ms = self.player.position()
        self._refresh_capture_labels()
        self.status_label.setText(f"結束時間：{format_time(self.pending_end_ms)}")
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
            self.player.setPosition(segment.normalized().start_ms)

    def copy_output(self) -> None:
        if not self.video_path:
            return
        if self.segment_list.count() == 0 and self.pending_start_ms is not None and self.pending_end_ms is not None:
            self.add_segment()
        output = self.build_output_text()
        if not output:
            return
        QApplication.clipboard().setText(output)
        self.status_label.setText("已複製到剪貼簿")
        self.output_preview.setText(output)

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

    def _refresh_capture_labels(self) -> None:
        self.start_value_label.setText(format_time(self.pending_start_ms) if self.pending_start_ms is not None else "--:--")
        self.end_value_label.setText(format_time(self.pending_end_ms) if self.pending_end_ms is not None else "--:--")

    def _on_slider_pressed(self) -> None:
        self._slider_is_pressed = True

    def _on_slider_released(self) -> None:
        self._slider_is_pressed = False
        self.player.setPosition(self.seek_slider.value())

    def on_duration_changed(self, duration: int) -> None:
        self.duration_ms = max(0, duration)
        self.seek_slider.setRange(0, self.duration_ms)
        self.total_time_label.setText(format_time(self.duration_ms))

    def on_position_changed(self, position: int) -> None:
        if not self._slider_is_pressed:
            self.seek_slider.setValue(position)
        self.current_time_label.setText(format_time(position))

    def on_state_changed(self, state: QMediaPlayer.State) -> None:
        if state == QMediaPlayer.PlayingState:
            self.play_button.setText("暫停")
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setText("播放")
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def on_player_error(self) -> None:
        error_text = self.player.errorString() or "播放時發生錯誤。"
        self.status_label.setText(error_text)

    def update_actions(self) -> None:
        has_video = bool(self.video_path)
        has_pending_segment = self.pending_start_ms is not None and self.pending_end_ms is not None
        has_segments = self.segment_list.count() > 0
        self.add_segment_button.setEnabled(has_video and has_pending_segment)
        self.copy_button.setEnabled(has_video and (has_segments or has_pending_segment))
        self.delete_segment_button.setEnabled(bool(self.segment_list.selectedItems()))
        self.clear_segments_button.setEnabled(has_segments)
        self.output_preview.setText(self.build_output_text())


def main() -> int:
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon())
    window = VideoTimeClipper()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.open_video(sys.argv[1])
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
