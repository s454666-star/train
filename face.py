import threading
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from customtkinter import CTk, CTkLabel, CTkButton, CTkProgressBar, CTkEntry, CTkOptionMenu
import numpy as np
import math
import cv2  # 用於影片讀取和處理

# ===== TensorFlow / MTCNN 穩定性設定 =====
# 1) 關閉 oneDNN（在某些版本的 TF + oneDNN 組合下，MTCNN 可能會在無候選框時觸發 shape error）
# 2) 降低 TF 原生層 log 噪音
# 注意：必須在 import tensorflow / mtcnn 前設定才有效
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from mtcnn import MTCNN
import re
import traceback
import shutil
import json
import hashlib
import mysql.connector
import sys
import subprocess
import logging
import faulthandler
import atexit
import platform
from urllib import request as urllib_request
from urllib import error as urllib_error

from typing import Optional, List, Dict, Any, Tuple, Set
from datetime import datetime, date
from db_env import load_db_config


# ===== OpenCV 穩定性設定（只為降低偶發 native crash，不影響功能）=====
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass


LOG_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
APP_LOG_PATH = os.path.join(LOG_DIR, "face_extractor_runtime.log")
FAULT_LOG_PATH = os.path.join(LOG_DIR, "face_extractor_faulthandler.log")

_FAULT_FP = None


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("FaceExtractor")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.debug("Logging file: %s", APP_LOG_PATH)
    except Exception as e:
        logger.debug("無法建立 log 檔案 %s，將只輸出到 console。原因: %s", APP_LOG_PATH, e)

    logger.propagate = False
    return logger


LOGGER = _configure_logger()


def _flush_logs() -> None:
    try:
        for h in list(getattr(LOGGER, "handlers", [])):
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass


def _enable_faulthandler() -> None:
    global _FAULT_FP
    try:
        _FAULT_FP = open(FAULT_LOG_PATH, "a", buffering=1, encoding="utf-8", errors="ignore")
        faulthandler.enable(file=_FAULT_FP, all_threads=True)
        LOGGER.debug("Faulthandler 已啟用，log: %s", FAULT_LOG_PATH)
        _flush_logs()
    except Exception as e:
        try:
            LOGGER.debug("Faulthandler 啟用失敗: %s", e)
            _flush_logs()
        except Exception:
            pass


def _install_global_exception_hooks() -> None:
    def _sys_hook(exc_type, exc, tb):
        try:
            LOGGER.critical("主執行緒未捕捉例外", exc_info=(exc_type, exc, tb))
            _flush_logs()
        except Exception:
            pass

    sys.excepthook = _sys_hook

    if hasattr(threading, "excepthook"):

        def _thread_hook(args):
            try:
                LOGGER.critical(
                    "背景執行緒未捕捉例外: %s",
                    getattr(args.thread, "name", "unknown"),
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                )
                _flush_logs()
            except Exception:
                pass

        threading.excepthook = _thread_hook


def _register_atexit() -> None:
    def _on_exit():
        try:
            LOGGER.info("程式結束（atexit）")
            _flush_logs()
        except Exception:
            pass

    atexit.register(_on_exit)


_enable_faulthandler()
_install_global_exception_hooks()
_register_atexit()


class FaceExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("影片人臉擷取工具")
        self.root.geometry("800x600")

        try:
            LOGGER.info("FaceExtractorApp 初始化開始")
            LOGGER.info("Python: %s", sys.version.replace("\n", " "))
            LOGGER.info("Platform: %s", platform.platform())
            LOGGER.info("PID: %s", os.getpid())
            _flush_logs()
        except Exception:
            pass

        try:
            def _tk_exception_hook(exc, val, tb):
                try:
                    LOGGER.critical("Tkinter callback 例外", exc_info=(exc, val, tb))
                    _flush_logs()
                except Exception:
                    pass
                try:
                    self._update_current_file(f"UI錯誤：{val}")
                except Exception:
                    pass

            self.root.report_callback_exception = _tk_exception_hook
            LOGGER.info("已安裝 Tkinter report_callback_exception")
            _flush_logs()
        except Exception as e:
            try:
                LOGGER.exception("安裝 Tkinter exception hook 失敗: %s", e)
                _flush_logs()
            except Exception:
                pass

        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        except Exception:
            pass

        self.root.configure(bg="#f0f0f0")

        self.video_list = []
        self.is_running = False
        self.current_index = 0
        self.total_videos = 0
        self.frame_count = 5

        CTkLabel(root, text="影片人臉擷取工具", font=("Arial", 20)).pack(pady=10)

        top_btn_frame = tk.Frame(root, bg="#f0f0f0")
        top_btn_frame.pack(pady=5)

        self.select_button = CTkButton(top_btn_frame, text="選擇影片", command=self.select_videos)
        self.select_button.pack(side=tk.LEFT, padx=6)

        self.scan_button = CTkButton(top_btn_frame, text="掃描", command=self.scan_f_drive_videos)
        self.scan_button.pack(side=tk.LEFT, padx=6)

        self.video_listbox = tk.Listbox(root, height=10, width=80, selectmode=tk.MULTIPLE)
        self.video_listbox.pack(pady=10)
        self.video_listbox.bind("<Delete>", self.delete_selected_videos)

        self.frame_count_label = CTkLabel(root, text="選擷取張數（1-1000，可留空=5）：")
        self.frame_count_label.pack(pady=5)
        self.frame_count_entry = CTkEntry(root, width=100, placeholder_text="例如：10，或留空=5")
        self.frame_count_entry.insert(0, "5")
        self.frame_count_entry.pack(pady=5)

        self.video_type_label = CTkLabel(root, text="影片類別：")
        self.video_type_label.pack(pady=5)
        self.video_type_var = tk.StringVar(value="1")
        self.video_type_optionmenu = CTkOptionMenu(
            root,
            variable=self.video_type_var,
            values=["1", "2", "3", "4"]
        )
        self.video_type_optionmenu.pack(pady=5)

        self.progress_label = CTkLabel(root, text="進度：0%")
        self.progress_label.pack(pady=5)
        self.current_file_label = CTkLabel(root, text="目前處理檔案：無")
        self.current_file_label.pack(pady=5)
        self.progress_bar = CTkProgressBar(root, width=400)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)

        button_frame = tk.Frame(root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        self.start_button = CTkButton(button_frame, text="執行", command=self.start_extraction)
        self.start_button.pack(side=tk.LEFT, padx=20)

        self.pause_button = CTkButton(button_frame, text="暫停", command=self.pause_extraction, state='disabled')
        self.pause_button.pack(side=tk.LEFT, padx=20)

        self.stop_button = CTkButton(button_frame, text="停止", command=self.stop_extraction, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=20)

        # ===== 修復自動關閉核心：MTCNN 不可跨執行緒共用 =====
        # 改成 thread-local 延遲初始化（每個 thread 自己建立一次）
        self.detector = None
        self._mtcnn_local = threading.local()

        # 偵測前若畫面太大，先縮小再偵測（降低 TF/MTCNN native crash 機率）
        # 0 表示不縮放；建議 1280
        self.mtcnn_detect_max_side = 1280

        # ===== 備援：OpenCV Haar Cascade（當 MTCNN / TensorFlow 異常時仍可繼續） =====
        self._haar_face = None
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self._haar_face = cv2.CascadeClassifier(cascade_path)
            if self._haar_face.empty():
                self._haar_face = None
        except Exception:
            self._haar_face = None

        try:
            db_config = load_db_config(prefix="STAR")
            LOGGER.info("準備連接資料庫 host=%s port=%s database=%s", db_config["host"], db_config["port"], db_config["database"])
            _flush_logs()
        except Exception:
            pass
        try:
            db_config = load_db_config(prefix="STAR")
            self.db_connection = mysql.connector.connect(**db_config)
            self.db_cursor = self.db_connection.cursor()
            print("成功連接到資料庫")
        except mysql.connector.Error as err:
            print(f"資料庫連線錯誤: {err}")
            self._update_current_file(f"資料庫連線失敗：{err}")
            self.db_connection = None
            self.db_cursor = None
        except ValueError as err:
            print(f"資料庫 ENV 設定錯誤: {err}")
            self._update_current_file(f"資料庫 ENV 設定錯誤：{err}")
            self.db_connection = None
            self.db_cursor = None

    # ===== 修復：每個 thread 都有自己的 detector =====
    def _get_mtcnn_detector(self) -> MTCNN:
        detector = getattr(self._mtcnn_local, "detector", None)
        if detector is not None:
            return detector

    def _reset_mtcnn_detector(self) -> None:
        try:
            if hasattr(self, "_mtcnn_local") and getattr(self._mtcnn_local, "detector", None) is not None:
                try:
                    delattr(self._mtcnn_local, "detector")
                except Exception:
                    self._mtcnn_local.detector = None
        except Exception:
            pass


        try:
            LOGGER.info("建立 MTCNN detector（thread=%s）", threading.current_thread().name)
            _flush_logs()
        except Exception:
            pass

        detector = MTCNN()
        self._mtcnn_local.detector = detector
        return detector

    def _prepare_for_detection(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        try:
            max_side = int(self.mtcnn_detect_max_side or 0)
            if max_side <= 0:
                return image, 1.0

            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                return image, 1.0

            if max(h, w) <= max_side:
                return image, 1.0

            scale = float(max_side) / float(max(h, w))
            new_w = max(int(round(w * scale)), 1)
            new_h = max(int(round(h * scale)), 1)

            # MTCNN 在某些 TF 組合下，遇到特定尺寸更容易出現不穩定情況
            # 這裡把尺寸調成偶數，並設最小邊界，降低 pyramid / resize 的極端 corner case
            if new_w % 2 != 0:
                new_w += 1
            if new_h % 2 != 0:
                new_h += 1
            new_w = max(new_w, 64)
            new_h = max(new_h, 64)

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, scale
        except Exception:
            return image, 1.0

    # ===== 公用工具 =====

    def sanitize_filename(self, filename: str) -> str:
        filename = filename.replace("..", "_")
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = filename.strip().strip(".")
        if not filename:
            filename = "unnamed"
        return filename

    def get_clean_folder_name(self, video_name: str) -> str:
        base = os.path.splitext(os.path.basename(video_name))[0]
        base = re.sub(r'\s*\(.*?\)\s*', '', base)
        base = base.replace(" ", "")
        base = self.sanitize_filename(base)
        return base

    def get_unique_output_dir_starting_from_1(self, parent_dir: str, base_name: str) -> str:
        parent_dir = os.path.abspath(parent_dir)
        counter = 1
        while True:
            candidate = os.path.join(parent_dir, f"{base_name}_{counter}")
            if not os.path.exists(candidate):
                return candidate
            counter += 1

    def ensure_dir(self, path: str) -> None:
        abs_path = os.path.abspath(path)
        os.makedirs(abs_path, exist_ok=True)

    def build_output_video_name(self, output_dir: str, original_video_path: str) -> str:
        output_base_name = os.path.basename(os.path.abspath(output_dir))
        original_ext = os.path.splitext(original_video_path)[1].lower() or ".mp4"
        return self.sanitize_filename(f"{output_base_name}{original_ext}")

    def build_db_relative_path(self, output_dir: str, filename: str) -> str:
        folder_name = self.sanitize_filename(os.path.basename(os.path.abspath(output_dir)))
        safe_filename = self.sanitize_filename(os.path.basename(filename))
        return f"\\{folder_name}\\{safe_filename}"

    def build_feature_capture_plan(self, duration: float) -> List[Dict[str, Any]]:
        duration = max(float(duration or 0.0), 0.0)

        if duration < 10.0:
            capture_second = min(3.0, max(duration - 0.25, 0.0))
            return [{
                "capture_order": 1,
                "label_second": 3.0,
                "capture_second": round(capture_second, 3),
            }]

        plan: List[Dict[str, float]] = []
        for index, target in enumerate([10.0, 20.0, 30.0, 40.0], start=1):
            if duration + 0.001 < target:
                continue
            plan.append({
                "capture_order": index,
                "label_second": target,
                "capture_second": round(min(target, max(duration - 0.25, 0.0)), 3),
            })
        return plan

    def build_feature_filename(self, video_path: str, capture_order: int, label_second: float) -> str:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        base_name = re.sub(r'[<>:"/\\|?*]+', '_', base_name) or "video"
        label = f"{label_second:.3f}".rstrip("0").rstrip(".").replace(".", "_")
        return f"{base_name}_feature_{capture_order:02d}_{label}s.jpg"

    def compute_dhash_hex(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize((9, 8), Image.BILINEAR)
            pixels = list(resized.getdata())

        bytes_out = [0] * 8
        bit_index = 0

        for y in range(8):
            for x in range(8):
                left = pixels[(y * 9) + x]
                right = pixels[(y * 9) + x + 1]

                left_gray = (int(left[0]) + int(left[1]) + int(left[2])) // 3
                right_gray = (int(right[0]) + int(right[1]) + int(right[2])) // 3
                bit = 1 if left_gray > right_gray else 0

                byte_pos = bit_index // 8
                bit_pos = 7 - (bit_index % 8)
                if bit == 1:
                    bytes_out[byte_pos] |= 1 << bit_pos

                bit_index += 1

        return "".join(f"{byte & 255:02x}" for byte in bytes_out)

    def probe_video_duration(self, video_path: str) -> float:
        ffprobe_bin = os.environ.get("FFPROBE_BIN", "ffprobe")
        command = [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            video_path,
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except FileNotFoundError as err:
            raise RuntimeError(f"ffprobe 不存在：{ffprobe_bin}") from err
        except subprocess.TimeoutExpired as err:
            raise RuntimeError(f"ffprobe 執行逾時：{video_path}") from err

        if result.returncode != 0:
            output = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"ffprobe 失敗：{output}")

        output = (result.stdout or "").strip()
        if not output:
            raise RuntimeError("ffprobe 未回傳有效的 duration")

        try:
            duration = float(output)
        except ValueError as err:
            raise RuntimeError("ffprobe 未回傳有效的 duration") from err

        return max(duration, 0.0)

    def capture_feature_frame(
        self,
        video_path: str,
        capture_second: float,
        output_path: str,
        excluded_capture_second_keys: Optional[Set[str]] = None,
    ) -> float:
        ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
        failure_messages: List[str] = []
        excluded_capture_second_keys = excluded_capture_second_keys or set()

        for candidate_second in self.build_feature_capture_second_candidates(capture_second, excluded_capture_second_keys):
            last_output = ""

            for force_compatible_colorspace in (False, True):
                if force_compatible_colorspace and not self.should_retry_feature_frame_with_compatible_colorspace(last_output):
                    break

                last_output = self.run_feature_frame_attempt(
                    ffmpeg_bin,
                    video_path,
                    candidate_second,
                    output_path,
                    force_compatible_colorspace,
                )

                if last_output == "":
                    return candidate_second

                mode_label = "compatible_colorspace" if force_compatible_colorspace else "default"
                failure_messages.append(f"[{candidate_second:.3f}s/{mode_label}] {last_output}")

        last_failure = failure_messages[-1] if failure_messages else "ffmpeg 未輸出任何畫面"
        raise RuntimeError(f"ffmpeg 擷取截圖失敗：{last_failure}")

    def run_feature_frame_attempt(
        self,
        ffmpeg_bin: str,
        video_path: str,
        capture_second: float,
        output_path: str,
        force_compatible_colorspace: bool,
    ) -> str:
        if os.path.isfile(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

        command = self.build_feature_frame_command(
            ffmpeg_bin,
            video_path,
            capture_second,
            output_path,
            force_compatible_colorspace,
        )

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except FileNotFoundError as err:
            raise RuntimeError(f"ffmpeg 不存在：{ffmpeg_bin}") from err
        except subprocess.TimeoutExpired as err:
            raise RuntimeError(f"ffmpeg 擷取截圖逾時：{video_path}") from err

        if result.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            return ""

        output = (result.stderr or result.stdout or "").strip()
        if output:
            return output

        if result.returncode == 0:
            return "ffmpeg 未輸出任何畫面（可能命中損毀影格或過近 EOF）"

        return f"ffmpeg 失敗，未回傳錯誤訊息 (exit_code={result.returncode})"

    def build_feature_capture_second_candidates(
        self,
        capture_second: float,
        excluded_capture_second_keys: Optional[Set[str]] = None,
    ) -> List[float]:
        capture_second = max(round(float(capture_second), 3), 0.0)
        excluded_capture_second_keys = excluded_capture_second_keys or set()
        raw_candidates = [
            capture_second,
            math.floor(capture_second),
        ]

        if abs(capture_second - math.floor(capture_second)) >= 0.001:
            raw_candidates.append(capture_second - 0.5)

        for offset in range(1, 16):
            raw_candidates.append(math.floor(capture_second) - offset)

        candidates: List[float] = []
        seen = set()

        for candidate in raw_candidates:
            normalized = round(max(float(candidate), 0.0), 3)
            key = f"{normalized:.3f}"
            if key in seen or key in excluded_capture_second_keys:
                continue
            seen.add(key)
            candidates.append(normalized)

        return candidates

    def build_feature_frame_command(
        self,
        ffmpeg_bin: str,
        video_path: str,
        capture_second: float,
        output_path: str,
        force_compatible_colorspace: bool,
    ) -> List[str]:
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-ss",
            f"{capture_second:.3f}",
            "-i",
            video_path,
        ]

        if force_compatible_colorspace:
            command.extend([
                "-vf",
                "colorspace=all=bt709:iall=bt709,format=yuv420p",
            ])

        command.extend([
            "-frames:v",
            "1",
            "-q:v",
            "3",
            output_path,
        ])

        return command

    def should_retry_feature_frame_with_compatible_colorspace(self, output: str) -> bool:
        output = (output or "").lower()
        return (
            "impossible to convert between the formats supported by the filter" in output
            or ("auto_scale" in output and "function not implemented" in output)
        )

    def find_master_face_id(self, video_master_id: int) -> Optional[int]:
        if not self.db_cursor or not video_master_id:
            return None

        sql = """
            SELECT vfs.id
            FROM video_face_screenshots vfs
            INNER JOIN video_screenshots vs ON vs.id = vfs.video_screenshot_id
            WHERE vs.video_master_id = %s
              AND vfs.is_master = 1
            LIMIT 1
        """
        self.db_cursor.execute(sql, (video_master_id,))
        row = self.db_cursor.fetchone()
        if not row:
            return None
        return int(row[0])

    def clear_existing_video_features(self, video_master_id: int) -> None:
        if not self.db_cursor or not video_master_id:
            return

        select_sql = """
            SELECT vff.video_screenshot_id, vff.screenshot_path
            FROM video_feature_frames vff
            INNER JOIN video_features vf ON vf.id = vff.video_feature_id
            WHERE vf.video_master_id = %s
        """
        self.db_cursor.execute(select_sql, (video_master_id,))
        rows = self.db_cursor.fetchall() or []

        for screenshot_id, screenshot_path in rows:
            if screenshot_path:
                screenshot_abs_path = os.path.abspath(os.path.join(r"D:\video", str(screenshot_path).lstrip("\\/").replace("\\", os.sep)))
                if os.path.exists(screenshot_abs_path):
                    try:
                        os.remove(screenshot_abs_path)
                    except Exception:
                        pass

            if screenshot_id:
                self.db_cursor.execute("DELETE FROM video_screenshots WHERE id = %s", (int(screenshot_id),))

        self.db_cursor.execute("DELETE FROM video_features WHERE video_master_id = %s", (video_master_id,))

    def update_video_feature_error(self, video_master_id: int, video_name: str, relative_video_path: str, duration: float, message: str) -> None:
        if not self.db_cursor or not video_master_id:
            return

        directory_path = os.path.dirname(relative_video_path.lstrip("\\/")).replace("/", "\\").strip("\\")
        file_name = os.path.basename(relative_video_path.replace("\\", os.sep))
        path_sha1 = hashlib.sha1(relative_video_path.lower().encode("utf-8")).hexdigest()

        sql = """
            INSERT INTO video_features
            (
                video_master_id,
                master_face_screenshot_id,
                video_name,
                video_path,
                directory_path,
                file_name,
                path_sha1,
                file_size_bytes,
                duration_seconds,
                file_created_at,
                file_modified_at,
                screenshot_count,
                feature_version,
                capture_rule,
                extracted_at,
                last_error,
                created_at,
                updated_at
            )
            VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
                master_face_screenshot_id = VALUES(master_face_screenshot_id),
                video_name = VALUES(video_name),
                video_path = VALUES(video_path),
                directory_path = VALUES(directory_path),
                file_name = VALUES(file_name),
                path_sha1 = VALUES(path_sha1),
                duration_seconds = VALUES(duration_seconds),
                capture_rule = VALUES(capture_rule),
                feature_version = VALUES(feature_version),
                extracted_at = NOW(),
                last_error = VALUES(last_error),
                updated_at = NOW()
        """

        self.db_cursor.execute(
            sql,
            (
                video_master_id,
                self.find_master_face_id(video_master_id),
                video_name,
                relative_video_path,
                directory_path or None,
                file_name,
                path_sha1,
                None,
                round(float(duration or 0.0), 3),
                None,
                None,
                0,
                "v1",
                "10s_x4",
                message[:65535],
            ),
        )
        self.db_connection.commit()

    def create_video_feature_records(
        self,
        video_master_id: Optional[int],
        output_dir: str,
        destination_path: str,
        clean_folder_name: str,
        duration: float,
    ) -> None:
        if not self.db_cursor or not self.db_connection or not video_master_id:
            return

        relative_video_path = self.build_db_relative_path(output_dir, os.path.basename(destination_path))
        video_name = os.path.basename(destination_path)
        feature_duration = round(float(duration or 0.0), 3)
        created_files: List[str] = []

        try:
            feature_duration = round(self.probe_video_duration(destination_path), 3)
            if feature_duration <= 0:
                raise RuntimeError(f"ffprobe 無法取得影片時長：{destination_path}")

            capture_plan = self.build_feature_capture_plan(feature_duration)
            if not capture_plan:
                raise RuntimeError("無法建立影片特徵截圖計畫")

            self.clear_existing_video_features(video_master_id)

            directory_path = os.path.dirname(relative_video_path.lstrip("\\/")).replace("/", "\\").strip("\\")
            file_name = os.path.basename(destination_path)
            path_sha1 = hashlib.sha1(relative_video_path.lower().encode("utf-8")).hexdigest()
            file_size_bytes = os.path.getsize(destination_path) if os.path.exists(destination_path) else None
            file_created_at = datetime.fromtimestamp(os.path.getctime(destination_path)) if os.path.exists(destination_path) else None
            file_modified_at = datetime.fromtimestamp(os.path.getmtime(destination_path)) if os.path.exists(destination_path) else None
            capture_rule = "lt_10s_at_3s" if feature_duration < 10.0 else "10s_x4"

            feature_sql = """
                INSERT INTO video_features
                (
                    video_master_id,
                    master_face_screenshot_id,
                    video_name,
                    video_path,
                    directory_path,
                    file_name,
                    path_sha1,
                    file_size_bytes,
                    duration_seconds,
                    file_created_at,
                    file_modified_at,
                    screenshot_count,
                    feature_version,
                    capture_rule,
                    extracted_at,
                    last_error,
                    created_at,
                    updated_at
                )
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NULL, NOW(), NOW())
                ON DUPLICATE KEY UPDATE
                    master_face_screenshot_id = VALUES(master_face_screenshot_id),
                    video_name = VALUES(video_name),
                    video_path = VALUES(video_path),
                    directory_path = VALUES(directory_path),
                    file_name = VALUES(file_name),
                    path_sha1 = VALUES(path_sha1),
                    file_size_bytes = VALUES(file_size_bytes),
                    duration_seconds = VALUES(duration_seconds),
                    file_created_at = VALUES(file_created_at),
                    file_modified_at = VALUES(file_modified_at),
                    screenshot_count = VALUES(screenshot_count),
                    feature_version = VALUES(feature_version),
                    capture_rule = VALUES(capture_rule),
                    extracted_at = NOW(),
                    last_error = NULL,
                    updated_at = NOW()
            """

            self.db_cursor.execute(
                feature_sql,
                (
                    video_master_id,
                    self.find_master_face_id(video_master_id),
                    video_name,
                    relative_video_path,
                    directory_path or None,
                    file_name,
                    path_sha1,
                    file_size_bytes,
                    feature_duration,
                    file_created_at,
                    file_modified_at,
                    len(capture_plan),
                    "v1",
                    capture_rule,
                ),
            )

            self.db_cursor.execute("SELECT id FROM video_features WHERE video_master_id = %s", (video_master_id,))
            feature_row = self.db_cursor.fetchone()
            if not feature_row:
                raise RuntimeError(f"找不到 video_features：video_master_id={video_master_id}")
            video_feature_id = int(feature_row[0])

            inserted_count = 0
            used_capture_second_keys: Set[str] = set()

            for plan in capture_plan:
                capture_order = int(plan["capture_order"])
                label_second = float(plan["label_second"])
                capture_second = float(plan["capture_second"])

                feature_filename = self.build_feature_filename(destination_path, capture_order, label_second)
                feature_path = os.path.abspath(os.path.join(output_dir, feature_filename))
                self.ensure_dir(os.path.dirname(feature_path))
                capture_second = self.capture_feature_frame(
                    destination_path,
                    capture_second,
                    feature_path,
                    used_capture_second_keys,
                )
                used_capture_second_keys.add(f"{capture_second:.3f}")
                created_files.append(feature_path)

                screenshot_db_path = self.build_db_relative_path(output_dir, os.path.basename(feature_path))

                self.db_cursor.execute(
                    """
                        INSERT INTO video_screenshots (video_master_id, screenshot_path)
                        VALUES (%s, %s)
                    """,
                    (video_master_id, screenshot_db_path),
                )
                screenshot_id = int(self.db_cursor.lastrowid)

                dhash_hex = self.compute_dhash_hex(feature_path)
                with Image.open(feature_path) as feature_image:
                    image_width, image_height = feature_image.size
                frame_sha1 = hashlib.sha1()
                with open(feature_path, "rb") as image_fp:
                    frame_sha1.update(image_fp.read())

                self.db_cursor.execute(
                    """
                        INSERT INTO video_feature_frames
                        (
                            video_feature_id,
                            video_screenshot_id,
                            capture_order,
                            capture_second,
                            screenshot_path,
                            dhash_hex,
                            dhash_prefix,
                            frame_sha1,
                            image_width,
                            image_height,
                            created_at,
                            updated_at
                        )
                        VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    """,
                    (
                        video_feature_id,
                        screenshot_id,
                        capture_order,
                        round(capture_second, 3),
                        screenshot_db_path,
                        dhash_hex,
                        dhash_hex[:2],
                        frame_sha1.hexdigest(),
                        image_width,
                        image_height,
                    ),
                )

                inserted_count += 1

            if inserted_count <= 0:
                raise RuntimeError("無法建立任何影片特徵截圖")

            self.db_cursor.execute(
                """
                    UPDATE video_features
                    SET screenshot_count = %s,
                        master_face_screenshot_id = %s,
                        last_error = NULL,
                        extracted_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                """,
                (
                    inserted_count,
                    self.find_master_face_id(video_master_id),
                    video_feature_id,
                ),
            )

            self.db_connection.commit()
            print(f"已建立影片特徵資料，video_master_id={video_master_id}, feature_frames={inserted_count}")
        except Exception as err:
            self.db_connection.rollback()
            for created_file in created_files:
                try:
                    if created_file and os.path.exists(created_file):
                        os.remove(created_file)
                except Exception:
                    pass
            try:
                self.update_video_feature_error(video_master_id, video_name, relative_video_path, feature_duration, str(err))
            except Exception:
                pass
            raise

    def eagle_api_request(self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"http://localhost:41595{endpoint}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib_request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib_request.urlopen(request, timeout=10) as response:
                body = response.read().decode("utf-8")
        except urllib_error.URLError as err:
            raise RuntimeError(f"Eagle API connection failed: {err}") from err

        try:
            result = json.loads(body)
        except json.JSONDecodeError as err:
            raise RuntimeError(f"Eagle API returned non-JSON: {body}") from err

        if result.get("status") != "success":
            raise RuntimeError(f"Eagle API returned failure: {result}")
        return result

    def import_video_to_eagle_retry_library(self, video_path: str, item_name: str) -> Dict[str, Any]:
        target_library_name = "重跑資源"
        library_info = self.eagle_api_request("GET", "/api/library/info")
        current_library = (library_info.get("data") or {}).get("library") or {}
        current_library_name = current_library.get("name")
        current_library_path = current_library.get("path")
        target_library_path = current_library_path if current_library_name == target_library_name else None

        if not target_library_path:
            history = self.eagle_api_request("GET", "/api/library/history").get("data") or []
            for library_path in history:
                library_basename = os.path.splitext(os.path.basename(library_path))[0]
                if library_basename == target_library_name:
                    target_library_path = library_path
                    break

        if not target_library_path:
            raise FileNotFoundError(f"Eagle library not found: {target_library_name}")

        switched_library = current_library_path != target_library_path
        if switched_library:
            self.eagle_api_request("POST", "/api/library/switch", {"libraryPath": target_library_path})

        try:
            return self.eagle_api_request(
                "POST",
                "/api/item/addFromPath",
                {
                    "path": video_path,
                    "name": item_name,
                },
            )
        finally:
            if switched_library and current_library_path:
                try:
                    self.eagle_api_request("POST", "/api/library/switch", {"libraryPath": current_library_path})
                except Exception as err:
                    try:
                        LOGGER.exception("Failed to restore Eagle library: %s", err)
                        _flush_logs()
                    except Exception:
                        pass

    # ===== 事件/UI =====

    def _on_window_close(self):
        try:
            LOGGER.info("收到視窗關閉事件（WM_DELETE_WINDOW）")
            _flush_logs()
        except Exception:
            pass

        try:
            self.is_running = False
        except Exception:
            pass

        try:
            if hasattr(self, "db_connection") and self.db_connection and self.db_connection.is_connected():
                try:
                    if hasattr(self, "db_cursor") and self.db_cursor:
                        self.db_cursor.close()
                except Exception:
                    pass
                try:
                    self.db_connection.close()
                except Exception:
                    pass
                try:
                    LOGGER.info("視窗關閉時已關閉資料庫連線")
                    _flush_logs()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception as e:
            try:
                LOGGER.exception("關閉視窗時發生錯誤: %s", e)
                _flush_logs()
            except Exception:
                pass

    def select_videos(self):
        try:
            LOGGER.info("觸發 UI：選擇影片")
            _flush_logs()
        except Exception:
            pass
        files = filedialog.askopenfilenames(title="選擇影片", filetypes=[("影片檔案", "*.mp4;*.avi;*.mov;*.mkv")])
        for file in files:
            abs_file = os.path.abspath(file)
            if abs_file not in self.video_list:
                self.video_list.append(abs_file)
                self.video_listbox.insert(tk.END, os.path.basename(abs_file))
        print(f"選擇的影片列表: {self.video_list}")

    def scan_f_drive_videos(self):
        try:
            LOGGER.info("觸發 UI：掃描影片資料夾（scan_f_drive_videos）")
            _flush_logs()
        except Exception:
            pass
        """
        只掃描 D:\\video 底下（不遞迴）的 .mp4 檔案：
        - 直接使用 os.listdir("D:\\video")
        - 僅加入檔案且副檔名為 .mp4（大小寫不拘）
        - 不掃描任何子資料夾的內容（因為處理過的檔案會被搬入子資料夾）
        - 不跳出任何 alert，結果顯示在下方標籤
        """
        root_dir = r"D:\video"
        if not os.path.exists(root_dir):
            msg = f"掃描失敗：路徑不存在 {root_dir}"
            print(msg)
            self._update_current_file(msg)
            return

        added = 0
        try:
            for name in os.listdir(root_dir):
                full_path = os.path.abspath(os.path.join(root_dir, name))
                if os.path.isfile(full_path) and os.path.splitext(name)[1].lower() == ".mp4":
                    if full_path not in self.video_list:
                        self.video_list.append(full_path)
                        self.video_listbox.insert(tk.END, os.path.basename(full_path))
                        added += 1
        except Exception as e:
            traceback.print_exc()
            self._update_current_file(f"掃描時發生錯誤：{e}")
            return

        self._update_current_file(f"掃描完成：新增 {added} 個 .mp4（來源：{root_dir}）")

    def delete_selected_videos(self, event=None):
        try:
            LOGGER.info("觸發 UI：刪除選取影片（Delete）")
            _flush_logs()
        except Exception:
            pass
        selected_indices = self.video_listbox.curselection()
        for index in selected_indices[::-1]:
            print(f"刪除影片: {self.video_list[index]}")
            self.video_listbox.delete(index)
            del self.video_list[index]
        print(f"更新後的影片列表: {self.video_list}")

    def start_extraction(self):
        try:
            LOGGER.info("觸發 UI：開始執行（start_extraction），目前清單數量=%s", len(self.video_list))
            _flush_logs()
        except Exception:
            pass
        if not self.video_list:
            self._update_current_file("請先選擇或掃描影片")
            return

        frame_count_input = (self.frame_count_entry.get() or "").strip()
        if frame_count_input:
            try:
                frame_count = int(frame_count_input)
                if frame_count < 1 or frame_count > 1000:
                    raise ValueError
                self.frame_count = frame_count
            except ValueError:
                self._update_current_file("擷取張數無效（需 1-1000），或留空=5")
                return
        else:
            self.frame_count = 5

        self.is_running = True
        self.current_index = 0
        self.total_videos = len(self.video_list)

        print(f"開始擷取，擷取張數: {self.frame_count}, 總影片數: {self.total_videos}")

        self.start_button.configure(state='disabled')
        self.pause_button.configure(state='normal')
        self.stop_button.configure(state='normal')

        try:
            LOGGER.info("啟動背景執行緒：process_videos")
            _flush_logs()
        except Exception:
            pass
        threading.Thread(target=self._process_videos_entry, daemon=True, name="process_videos_thread").start()

    # ===== 核心流程 =====

    def _process_videos_entry(self):
        try:
            LOGGER.info("背景執行緒開始：process_videos")
            _flush_logs()
        except Exception:
            pass

        # 修復：在背景執行緒先初始化 MTCNN，避免第一次偵測才初始化導致更容易 crash
        try:
            _ = self._get_mtcnn_detector()
            LOGGER.info("背景執行緒 MTCNN detector 初始化完成")
            _flush_logs()
        except Exception as e:
            try:
                LOGGER.exception("背景執行緒初始化 MTCNN 失敗: %s", e)
                _flush_logs()
            except Exception:
                pass

        try:
            self.process_videos()
        except Exception as e:
            try:
                LOGGER.exception("process_videos 發生未捕捉例外: %s", e)
                _flush_logs()
            except Exception:
                pass
            try:
                self.is_running = False
                self._update_current_file(f"背景執行緒錯誤：{e}")
            except Exception:
                pass
        finally:
            try:
                LOGGER.info("背景執行緒結束：process_videos")
                _flush_logs()
            except Exception:
                pass

    def process_videos(self):
        try:
            LOGGER.info("進入 process_videos 迴圈，is_running=%s, video_list=%s", self.is_running, len(self.video_list))
            _flush_logs()
        except Exception:
            pass
        while self.is_running and self.video_list:
            if self.current_index >= len(self.video_list):
                self.current_index = 0

            video_path = os.path.abspath(self.video_list[self.current_index])
            video_name = os.path.basename(video_path)

            clean_folder_name = self.get_clean_folder_name(video_name)
            parent_dir = os.path.dirname(video_path)
            output_dir = self.get_unique_output_dir_starting_from_1(parent_dir, clean_folder_name)
            output_dir = os.path.abspath(output_dir)
            output_video_name = self.build_output_video_name(output_dir, video_path)

            try:
                LOGGER.info("開始處理影片 index=%s/%s path=%s", self.current_index + 1, max(len(self.video_list), 1), video_path)
                _flush_logs()
            except Exception:
                pass
            print(f"處理影片: {video_path}")
            print(f"輸出目錄: {output_dir}")

            try:
                self.ensure_dir(output_dir)
                print(f"已確保輸出目錄: {output_dir}")
            except Exception as e:
                print(f"無法創建輸出目錄: {output_dir}, 錯誤: {e}")
                self.current_index += 1
                continue

            cap = cv2.VideoCapture(video_path)
            try:
                LOGGER.info("VideoCapture 建立完成，isOpened=%s", cap.isOpened())
                _flush_logs()
            except Exception:
                pass
            if not cap.isOpened():
                print(f"無法打開影片檔案: {video_name}")
                self._update_current_file(f"無法打開影片：{video_name}")
                self.current_index += 1
                continue

            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                fallback_duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0
                try:
                    duration = self.probe_video_duration(video_path)
                except Exception as duration_err:
                    duration = fallback_duration
                    print(f"ffprobe 取得時長失敗，改用 OpenCV 推估: {duration_err}")

                frame_count = self.frame_count
                frame_indices = self._compute_frame_indices(total_frames, frame_count)

                print(f"影片 FPS: {fps}, 總幀數: {total_frames}, 持續時間(秒): {duration}, 擷取張數: {frame_count}")
                print(f"將擷取幀位索引: {frame_indices}")

                self._update_progress(0, "進度：0%")

                video_type = self.video_type_var.get() if self.video_type_var.get() else "1"

                video_master_id = None
                if self.db_cursor:
                    try:
                        insert_video = """
                            INSERT INTO video_master (video_name, video_path, duration, video_type)
                            VALUES (%s, %s, %s, %s)
                        """
                        relative_video_path = f"\\{os.path.basename(output_dir)}\\{output_video_name}"
                        self.db_cursor.execute(insert_video, (output_video_name, relative_video_path, round(duration, 3), video_type))
                        self.db_connection.commit()
                        video_master_id = self.db_cursor.lastrowid
                        print(f"已插入 video_master, ID: {video_master_id}")
                    except mysql.connector.Error as err:
                        try:
                            LOGGER.exception("插入 video_master 失敗: %s", err)
                            _flush_logs()
                        except Exception:
                            pass
                        print(f"插入 video_master 時發生錯誤: {err}")
                        traceback.print_exc()

                saved = 0

                for idx, frame_idx in enumerate(frame_indices, start=1):
                    if not self.is_running:
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"無法讀取幀位 {frame_idx}，略過")
                        continue

                    frame_filename = self.sanitize_filename(f"{clean_folder_name}_{idx}.jpg")
                    frame_path = os.path.abspath(os.path.join(output_dir, frame_filename))

                    try:
                        parent_check = os.path.dirname(frame_path)
                        if not os.path.abspath(parent_check).startswith(os.path.abspath(output_dir)):
                            frame_path = os.path.abspath(os.path.join(output_dir, self.sanitize_filename(os.path.basename(frame_filename))))
                        self.ensure_dir(os.path.dirname(frame_path))
                    except Exception as e:
                        print(f"校驗輸出路徑失敗: {e}")
                        continue

                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        pil_image.save(frame_path)
                        print(f"成功儲存截圖至: {frame_path}")
                    except Exception as e:
                        print(f"儲存影像時發生錯誤: {e}")
                        traceback.print_exc()
                        continue

                    screenshot_id = None
                    if self.db_cursor and video_master_id:
                        try:
                            insert_screenshot = """
                                INSERT INTO video_screenshots (video_master_id, screenshot_path)
                                VALUES (%s, %s)
                            """
                            screenshot_db_path = f"\\{os.path.basename(output_dir)}\\{os.path.basename(frame_path)}"
                            self.db_cursor.execute(insert_screenshot, (video_master_id, screenshot_db_path))
                            self.db_connection.commit()
                            screenshot_id = self.db_cursor.lastrowid
                            print(f"已插入 video_screenshots, ID: {screenshot_id}")
                        except mysql.connector.Error as err:
                            try:
                                LOGGER.exception("插入 video_screenshots 失敗: %s", err)
                                _flush_logs()
                            except Exception:
                                pass
                            print(f"插入 video_screenshots 時發生錯誤: {err}")
                            traceback.print_exc()

                    self._update_current_file(os.path.basename(frame_path))

                    try:
                        LOGGER.debug("開始人臉偵測：frame_idx=%s", frame_idx)
                        _flush_logs()
                    except Exception:
                        pass

                    faces = self.detect_faces(frame)

                    if faces:
                        for fidx, face in enumerate(faces, start=1):
                            face_filename = self.sanitize_filename(f"{clean_folder_name}_face_{idx}_{fidx}.jpg")
                            face_path = os.path.abspath(os.path.join(output_dir, face_filename))

                            try:
                                if not os.path.abspath(os.path.dirname(face_path)).startswith(output_dir):
                                    face_path = os.path.abspath(os.path.join(output_dir, os.path.basename(face_filename)))
                                self.ensure_dir(os.path.dirname(face_path))

                                if face is not None and face.size > 0:
                                    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                                    pil_face = Image.fromarray(rgb_face)
                                    pil_face.save(face_path)
                                    print(f"成功儲存人臉至: {face_path}")

                                    if self.db_cursor and screenshot_id:
                                        try:
                                            insert_face = """
                                                INSERT INTO video_face_screenshots (video_screenshot_id, face_image_path)
                                                VALUES (%s, %s)
                                            """
                                            face_db_path = f"\\{os.path.basename(output_dir)}\\{os.path.basename(face_path)}"
                                            self.db_cursor.execute(insert_face, (screenshot_id, face_db_path))
                                            self.db_connection.commit()
                                            print(f"已插入 video_face_screenshots")
                                        except mysql.connector.Error as err:
                                            try:
                                                LOGGER.exception("插入 video_face_screenshots 失敗: %s", err)
                                                _flush_logs()
                                            except Exception:
                                                pass
                                            print(f"插入 video_face_screenshots 時發生錯誤: {err}")
                                            traceback.print_exc()
                            except Exception as e:
                                print(f"儲存人臉時發生錯誤: {e}")
                                traceback.print_exc()

                    saved += 1
                    progress = saved / float(frame_count)
                    percent = int(progress * 100)
                    self._update_progress(progress, f"進度：{percent}%")

                cap.release()
                print(f"完成處理影片: {video_path}")

                try:
                    destination_path = os.path.abspath(os.path.join(output_dir, output_video_name))
                    if os.path.exists(destination_path):
                        raise FileExistsError(f"輸出影片已存在: {destination_path}")

                    shutil.move(video_path, destination_path)
                    print(f"已移動影片檔案至: {destination_path}")

                    try:
                        self.create_video_feature_records(
                            video_master_id=video_master_id,
                            output_dir=output_dir,
                            destination_path=destination_path,
                            clean_folder_name=clean_folder_name,
                            duration=duration,
                        )
                    except Exception as feature_err:
                        try:
                            LOGGER.exception("建立影片特徵資料失敗: %s", feature_err)
                            _flush_logs()
                        except Exception:
                            pass
                        print(f"建立影片特徵資料時發生錯誤: {feature_err}")
                        traceback.print_exc()

                    retry_dir = os.path.abspath(r"Z:\video(重跑)")
                    self.ensure_dir(retry_dir)
                    retry_copy_path = os.path.abspath(os.path.join(retry_dir, output_video_name))
                    shutil.copy2(destination_path, retry_copy_path)
                    print(f"已複製影片檔案至 Z:\\video(重跑): {retry_copy_path}")

                    eagle_result = self.import_video_to_eagle_retry_library(destination_path, output_video_name)
                    print(f"已匯入 Eagle 重跑資源: {eagle_result}")
                except Exception as e:
                    try:
                        LOGGER.exception("移動影片檔案發生錯誤: %s", e)
                        _flush_logs()
                    except Exception:
                        pass
                    print(f"移動影片檔案時發生錯誤: {e}")
                    traceback.print_exc()

                self.remove_processed_video(self.current_index)

            except Exception as e:
                print(f"處理影片時發生未預期錯誤: {e}")
                traceback.print_exc()
                try:
                    cap.release()
                except Exception:
                    pass
                self.remove_processed_video(self.current_index)

        try:
            LOGGER.info("離開 process_videos：準備呼叫 processing_complete()")
            _flush_logs()
        except Exception:
            pass
        self.processing_complete()

    def _compute_frame_indices(self, total_frames: int, frame_count: int):
        if total_frames <= 0 or frame_count <= 0:
            return []
        if frame_count == 1:
            return [max(total_frames // 2, 0)]
        step = total_frames / float(frame_count + 1)
        indices = [int(round(step * i)) for i in range(1, frame_count + 1)]
        indices = [min(max(idx, 0), total_frames - 1) for idx in indices]
        indices = sorted(list(dict.fromkeys(indices)))
        return indices

    def remove_processed_video(self, index):
        if index < len(self.video_list):
            print(f"從清單中移除已處理的影片: {self.video_list[index]}")
            del self.video_list[index]
            self.video_listbox.delete(index)

    def processing_complete(self):
        try:
            LOGGER.info("processing_complete()：所有影片已處理完成或已停止，remaining=%s", len(self.video_list))
            _flush_logs()
        except Exception:
            pass
        print("所有影片已處理完成")
        self.progress_bar.set(1.0)
        self.progress_label.configure(text="進度：100%")
        self.current_file_label.configure(text="目前處理檔案：無")
        self.is_running = False
        self.start_button.configure(state='normal')
        self.pause_button.configure(state='disabled')
        self.stop_button.configure(state='disabled')
        self.root.after(1000, lambda: (self.progress_bar.set(0.0), self.progress_label.configure(text="進度：0%")))

    # ===== 人臉偵測與影像處理 =====

    def _detect_faces_with_haar(self, frame: np.ndarray) -> List[np.ndarray]:
        faces: List[np.ndarray] = []

        if self._haar_face is None:
            return faces

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return faces

        try:
            # scaleFactor / minNeighbors 偏保守，降低誤判
            rects = self._haar_face.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
        except Exception:
            return faces

        if rects is None:
            return faces

        try:
            for (x, y, w, h) in rects:
                if w <= 0 or h <= 0:
                    continue
                face = self.extract_face(frame, int(x), int(y), int(w), int(h), 0)
                if face is not None:
                    faces.append(face)
        except Exception:
            return faces

        return faces

    def _safe_mtcnn_detect_faces(self, detector: MTCNN, rgb: np.ndarray) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        try:
            # 確保資料連續、dtype 正確，減少底層實作踩雷機率
            if rgb is None:
                return None, "rgb is None"
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8, copy=False)
            rgb = np.ascontiguousarray(rgb)

            results = detector.detect_faces(rgb)
            return results, None
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            try:
                LOGGER.exception("MTCNN detect_faces 發生例外：%s", msg)
                _flush_logs()
            except Exception:
                pass

            # 這種錯誤常見於 TF runtime 在某些 edge case（例如候選框為空）時的 shape 推導
            # 重新初始化 detector 往往可以讓後續影格繼續跑
            self._reset_mtcnn_detector()
            return None, msg

    def detect_faces(self, frame):
        try:
            LOGGER.debug("detect_faces() 進入，frame shape=%s", getattr(frame, "shape", None))
            _flush_logs()
        except Exception:
            pass

        faces: List[np.ndarray] = []
        if frame is None or not hasattr(frame, "shape"):
            return faces

        angles = [0, 45, 90, 135, 180, 225, 270, 315]

        detector: Optional[MTCNN] = None
        try:
            detector = self._get_mtcnn_detector()
        except Exception as e:
            try:
                LOGGER.exception("取得 MTCNN detector 失敗: %s", e)
                _flush_logs()
            except Exception:
                pass
            return self._detect_faces_with_haar(frame)

        mtcnn_failed = False

        try:
            for angle in angles:
                if angle == 0:
                    src = frame
                else:
                    src = self.rotate_image(frame, angle)

                detect_src, scale = self._prepare_for_detection(src)

                try:
                    rgb = cv2.cvtColor(detect_src, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"cvtColor 失敗: {e}")
                    traceback.print_exc()
                    continue

                try:
                    LOGGER.debug("detect_faces() 呼叫 detector.detect_faces() 前 angle=%s scale=%s", angle, scale)
                    _flush_logs()
                except Exception:
                    pass

                results, err = self._safe_mtcnn_detect_faces(detector, rgb)
                if err is not None:
                    mtcnn_failed = True
                    # detector 可能已被 reset，取新的再繼續
                    try:
                        detector = self._get_mtcnn_detector()
                    except Exception:
                        detector = None
                    if detector is None:
                        break
                    continue

                if results:
                    for result in results:
                        x, y, w, h = result.get("box", [0, 0, 0, 0])

                        if scale and scale != 1.0:
                            try:
                                x = int(round(x / scale))
                                y = int(round(y / scale))
                                w = int(round(w / scale))
                                h = int(round(h / scale))
                            except Exception:
                                pass

                        if w <= 0 or h <= 0:
                            continue

                        face = self.extract_face(src, x, y, w, h, angle)
                        if face is not None:
                            faces.append(face)

                    if faces:
                        break

            if not faces:
                # 如果 MTCNN 在本次影格中發生過 TF/shape 例外，啟用 Haar 備援，避免整支程式被拖下去
                if mtcnn_failed:
                    try:
                        LOGGER.warning("MTCNN 本次影格偵測失敗，改用 Haar Cascade 備援偵測")
                        _flush_logs()
                    except Exception:
                        pass
                    faces = self._detect_faces_with_haar(frame)

            print(f"偵測到 {len(faces)} 張人臉")
        except Exception as e:
            print(f"人臉偵測時發生錯誤: {e}")
            traceback.print_exc()

        return faces


    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        rotated = cv2.warpAffine(image, M, (nW, nH))
        return rotated

    def extract_face(self, rotated_image, x, y, width, height, angle):
        try:
            expansion_factor = 2.0
            new_width = int(width * expansion_factor)
            new_height = int(height * expansion_factor)

            center_x = x + width // 2
            center_y = y + height // 2
            new_x = int(center_x - new_width // 2)
            new_y = int(center_y - new_height // 2)

            new_x = max(new_x, 0)
            new_y = max(new_y, 0)
            new_width = min(new_width, rotated_image.shape[1] - new_x)
            new_height = min(new_height, rotated_image.shape[0] - new_y)

            if new_width <= 0 or new_height <= 0:
                return None

            face = rotated_image[new_y:new_y + new_height, new_x:new_x + new_width]
            if angle != 0:
                face = self.rotate_face_back(face, angle)
            return face
        except Exception as e:
            print(f"提取人臉時發生錯誤: {e}")
            traceback.print_exc()
            return None

    def rotate_face_back(self, face_image, angle):
        try:
            (h, w) = face_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated_back = cv2.warpAffine(face_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return rotated_back
        except Exception as e:
            print(f"旋轉人臉回正面時發生錯誤: {e}")
            traceback.print_exc()
            return face_image

    # ===== 控制 =====

    def pause_extraction(self):
        try:
            LOGGER.info("觸發 UI：暫停（pause_extraction），is_running=%s", self.is_running)
            _flush_logs()
        except Exception:
            pass
        if self.is_running:
            self.is_running = False
            self.current_file_label.configure(text="目前處理檔案：暫停中")
            print("擷取已暫停")
            self.pause_button.configure(state='disabled')
            self.start_button.configure(state='normal')

    def stop_extraction(self):
        try:
            LOGGER.info("觸發 UI：停止（stop_extraction），is_running=%s, current_index=%s, list=%s", self.is_running, self.current_index, len(self.video_list))
            _flush_logs()
        except Exception:
            pass
        self.is_running = False
        self.progress_bar.set(0)
        self.progress_label.configure(text="進度：0%")
        self.current_file_label.configure(text="目前處理檔案：無")
        self.current_index = len(self.video_list)
        print("擷取已停止")
        self.start_button.configure(state='normal')
        self.pause_button.configure(state='disabled')
        self.stop_button.configure(state='disabled')

    # ===== UI 輔助 =====

    def _update_progress(self, progress: float, text: str):
        try:
            self.root.after(0, lambda: (self.progress_bar.set(progress), self.progress_label.configure(text=text)))
        except Exception:
            pass

    def _update_current_file(self, filename: str):
        try:
            self.root.after(0, lambda: self.current_file_label.configure(text=f"目前處理檔案：{filename}"))
        except Exception:
            pass

    # ===== 資源釋放 =====

    def __del__(self):
        try:
            if hasattr(self, 'db_connection') and self.db_connection and self.db_connection.is_connected():
                if hasattr(self, 'db_cursor') and self.db_cursor:
                    self.db_cursor.close()
                self.db_connection.close()
                print("資料庫連線已關閉")
        except Exception:
            pass


if __name__ == "__main__":
    app = CTk()
    extractor = FaceExtractorApp(app)
    app.mainloop()
