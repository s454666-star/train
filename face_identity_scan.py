from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)

import cv2
import mysql.connector
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

import face_identity_config as config


try:
    cv2.setLogLevel(0)
except Exception:  # noqa: BLE001
    pass


@dataclass
class DiscoveredVideo:
    root_label: str
    root_path: Path
    video_path: Path


@dataclass
class FaceCandidate:
    capture_second: float
    detector_score: float
    blur_score: float
    frontal_score: float
    quality_score: float
    bounding_box: list[float]
    landmarks: list[list[float]]
    embedding: np.ndarray
    crop_bgr: np.ndarray


@dataclass
class CandidateCluster:
    samples: list[FaceCandidate] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None

    def add(self, candidate: FaceCandidate) -> None:
        self.samples.append(candidate)
        vectors = np.stack([item.embedding for item in self.samples], axis=0)
        self.centroid = normalize_vector(vectors.mean(axis=0))


class FaceIdentityScanner:
    def __init__(self, force: bool = False, limit: Optional[int] = None, paths: Optional[list[str]] = None) -> None:
        self.force = force
        self.limit = limit
        self.override_paths = paths or []
        self.logger = self._configure_logger()
        self.device = torch.device("cpu")
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            image_size=config.MTCNN_IMAGE_SIZE,
            margin=config.MTCNN_MARGIN,
            min_face_size=config.MTCNN_MIN_FACE_SIZE,
            post_process=True,
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.connection = self._connect_database()

    def run(self) -> int:
        self._assert_schema()
        discovered = self._discover_videos()

        if not discovered:
            self.logger.info("沒有找到任何待掃描影片。")
            return 0

        total = len(discovered)
        processed = 0

        for index, item in enumerate(discovered, start=1):
            self.logger.info("[%s/%s] %s", index, total, item.video_path)
            try:
                self._process_video(item)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("處理影片失敗：%s", item.video_path)
                try:
                    self._persist_error(item, str(exc))
                except Exception:  # noqa: BLE001
                    self.logger.exception("寫入錯誤狀態失敗：%s", item.video_path)

        self.logger.info("掃描完成，成功處理 %s / %s 部影片。", processed, total)
        return 0

    def close(self) -> None:
        try:
            self.connection.close()
        except Exception:  # noqa: BLE001
            pass

    def _configure_logger(self) -> logging.Logger:
        config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("face_identity_scan")
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False
        return logger

    def _connect_database(self) -> mysql.connector.MySQLConnection:
        env_values = parse_env_file(config.BLOG_ENV_PATH)
        db_host = env_values.get("DB_HOST")
        db_port = int(env_values.get("DB_PORT", "3306"))
        db_name = env_values.get("DB_DATABASE")
        db_user = env_values.get("DB_USERNAME")
        db_password = env_values.get("DB_PASSWORD", "")

        if not all([db_host, db_name, db_user]):
            raise RuntimeError(f"無法從 {config.BLOG_ENV_PATH} 解析 blog DB 連線資訊")

        return mysql.connector.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            charset="utf8mb4",
            autocommit=True,
        )

    def _assert_schema(self) -> None:
        tables = (
            "face_identity_people",
            "face_identity_videos",
            "face_identity_samples",
            "face_identity_group_changes",
        )
        cursor = self.connection.cursor()
        try:
            for table_name in tables:
                cursor.execute("SHOW TABLES LIKE %s", (table_name,))
                if cursor.fetchone() is None:
                    raise RuntimeError(
                        f"blog DB 缺少資料表 {table_name}，請先在 C:\\www\\blog 執行 php artisan migrate"
                    )
        finally:
            cursor.close()

    def _discover_videos(self) -> list[DiscoveredVideo]:
        items: list[DiscoveredVideo] = []

        for entry in self._iter_scan_targets():
            root_label = entry["label"]
            target_path = Path(entry["path"])
            if not target_path.exists():
                self.logger.warning("找不到掃描路徑：%s", target_path)
                continue

            if target_path.is_file():
                if target_path.suffix.lower() in config.VIDEO_EXTENSIONS:
                    items.append(
                        DiscoveredVideo(
                            root_label=root_label,
                            root_path=target_path.parent,
                            video_path=target_path,
                        )
                    )
                continue

            for file_path in sorted(target_path.rglob("*")):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in config.VIDEO_EXTENSIONS:
                    continue

                items.append(
                    DiscoveredVideo(
                        root_label=root_label,
                        root_path=target_path,
                        video_path=file_path,
                    )
                )

                if self.limit is not None and len(items) >= self.limit:
                    return items

        return items

    def _iter_scan_targets(self) -> Iterable[dict[str, Any]]:
        if self.override_paths:
            for raw_path in self.override_paths:
                path = Path(raw_path)
                yield {
                    "label": path.stem or path.name or "override",
                    "path": path,
                }
            return

        for item in config.VIDEO_ROOTS:
            yield item

    def _process_video(self, item: DiscoveredVideo) -> None:
        absolute_path = item.video_path.resolve()
        file_stat = absolute_path.stat()
        path_sha1 = sha1_path(absolute_path)
        modified_at = datetime.fromtimestamp(file_stat.st_mtime).replace(microsecond=0)
        existing = self._find_existing_video(path_sha1)

        if existing and bool(existing.get("group_locked")) and config.SKIP_LOCKED_VIDEOS:
            self.logger.info("略過手動鎖定作品：%s", absolute_path)
            return

        if existing and config.SKIP_COMPLETED_VIDEOS and not self.force:
            same_size = int(existing.get("file_size_bytes") or 0) == int(file_stat.st_size)
            same_mtime = existing.get("file_modified_at") == modified_at
            same_status = str(existing.get("scan_status") or "") in {"complete", "no_face"}

            if same_size and same_mtime and same_status:
                self.logger.info("略過已完成且未變更影片：%s", absolute_path)
                return

        duration_seconds = probe_duration_seconds(absolute_path)
        self.logger.info(
            "開始分析 %s | duration=%.1fs | interval=%ss | success_skip=%ss | max_samples=%s",
            absolute_path.name,
            duration_seconds,
            config.FRAME_INTERVAL_SECONDS,
            config.SECONDS_TO_SKIP_AFTER_SUCCESS,
            config.MAX_SAMPLES_PER_VIDEO,
        )
        selected_samples, metadata = self._extract_video_faces(absolute_path, duration_seconds)

        if not selected_samples:
            self.logger.info("沒有抓到符合條件的清晰正面人臉：%s", absolute_path)
        else:
            self.logger.info("擷取到 %s 張有效樣本：%s", len(selected_samples), absolute_path.name)

        self.logger.info(
            "影片分析摘要 %s | checked=%s | detected=%s | accepted=%s | dominant_cluster=%s | rejected=%s",
            absolute_path.name,
            metadata.get("checked_frame_count", 0),
            metadata.get("detected_face_frame_count", 0),
            len(selected_samples),
            metadata.get("dominant_cluster_size", 0),
            metadata.get("rejected_counts", {}),
        )
        if metadata.get("accepted_seconds"):
            self.logger.info("有效樣本秒數 %s | %s", absolute_path.name, metadata["accepted_seconds"])

        self._persist_result(
            item=item,
            absolute_path=absolute_path,
            file_stat=file_stat,
            modified_at=modified_at,
            duration_seconds=duration_seconds,
            path_sha1=path_sha1,
            existing=existing,
            selected_samples=selected_samples,
            metadata=metadata,
        )

    def _extract_video_faces(
        self,
        video_path: Path,
        duration_seconds: float,
    ) -> tuple[list[FaceCandidate], dict[str, Any]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")

        clusters: list[CandidateCluster] = []
        rejected_counts: dict[str, int] = {}
        checked_seconds: list[float] = []
        debug_events: list[dict[str, Any]] = []
        accepted_seconds: list[float] = []
        detected_face_frame_count = 0
        current_second = float(config.FIRST_SAMPLE_OFFSET_SECONDS)
        checked_frame_count = 0
        rejection_log_count = 0
        rejection_log_suppressed = False

        try:
            while current_second <= duration_seconds and checked_frame_count < config.MAX_FRAME_CHECKS_PER_VIDEO:
                best_candidate: Optional[FaceCandidate] = None
                best_reason: Optional[str] = None
                best_debug_info: Optional[dict[str, Any]] = None

                for offset in config.NEARBY_FRAME_OFFSETS_SECONDS:
                    if checked_frame_count >= config.MAX_FRAME_CHECKS_PER_VIDEO:
                        break

                    target_second = round(min(max(current_second + float(offset), 0.0), duration_seconds), 3)
                    checked_seconds.append(target_second)
                    checked_frame_count += 1

                    success, frame = capture_frame(capture, target_second)
                    if not success or frame is None:
                        best_reason = best_reason or "frame_read_failed"
                        event = {
                            "second": target_second,
                            "status": "frame_read_failed",
                            "reason": "frame_read_failed",
                            "faces_detected": 0,
                        }
                        append_debug_event(debug_events, event)
                        rejection_log_count, rejection_log_suppressed = self._log_frame_event(
                            video_path.name,
                            event,
                            rejection_log_count,
                            rejection_log_suppressed,
                        )
                        continue

                    candidate, reason, debug_info = self._pick_best_face(frame, target_second)
                    if int(debug_info.get("faces_detected") or 0) > 0:
                        detected_face_frame_count += 1
                    if candidate is None:
                        best_reason = best_reason or reason or "no_face"
                        event = {
                            "second": target_second,
                            "status": "rejected" if (reason or "no_face") != "no_face" else "no_face",
                            "reason": reason or "no_face",
                            **debug_info,
                        }
                        append_debug_event(debug_events, event)
                        rejection_log_count, rejection_log_suppressed = self._log_frame_event(
                            video_path.name,
                            event,
                            rejection_log_count,
                            rejection_log_suppressed,
                        )
                        continue

                    if best_candidate is None or candidate.quality_score > best_candidate.quality_score:
                        best_candidate = candidate
                        best_debug_info = debug_info

                if best_candidate is None:
                    key = best_reason or "no_face"
                    rejected_counts[key] = rejected_counts.get(key, 0) + 1
                    current_second += float(config.FRAME_INTERVAL_SECONDS)
                    continue

                self._assign_candidate_to_cluster(clusters, best_candidate)
                best_cluster = pick_best_cluster(clusters)
                accepted_seconds.append(round(best_candidate.capture_second, 3))
                event = {
                    "second": best_candidate.capture_second,
                    "status": "accepted",
                    "reason": None,
                    "cluster_size": len(best_cluster.samples) if best_cluster else 1,
                    **(best_debug_info or {}),
                }
                append_debug_event(debug_events, event)
                rejection_log_count, rejection_log_suppressed = self._log_frame_event(
                    video_path.name,
                    event,
                    rejection_log_count,
                    rejection_log_suppressed,
                    always=True,
                )
                current_second += float(config.SECONDS_TO_SKIP_AFTER_SUCCESS)
                if best_cluster and len(best_cluster.samples) >= config.MAX_SAMPLES_PER_VIDEO:
                    break
        finally:
            capture.release()

        best_cluster = pick_best_cluster(clusters)
        selected_samples: list[FaceCandidate] = []
        if best_cluster is not None:
            selected_samples = sorted(
                best_cluster.samples,
                key=lambda sample: (-sample.quality_score, sample.capture_second),
            )[: config.MAX_SAMPLES_PER_VIDEO]
            selected_samples = sorted(selected_samples, key=lambda sample: sample.capture_second)

        metadata = {
            "checked_frame_seconds": checked_seconds,
            "checked_frame_count": len(checked_seconds),
            "detected_face_frame_count": detected_face_frame_count,
            "candidate_cluster_count": len(clusters),
            "dominant_cluster_size": len(best_cluster.samples) if best_cluster else 0,
            "accepted_seconds": accepted_seconds,
            "rejected_counts": rejected_counts,
            "debug_events": debug_events,
        }
        return selected_samples, metadata

    def _pick_best_face(
        self,
        frame_bgr: np.ndarray,
        capture_second: float,
    ) -> tuple[Optional[FaceCandidate], Optional[str], dict[str, Any]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        boxes, probabilities, landmarks = self.mtcnn.detect(frame_pil, landmarks=True)
        if boxes is None or probabilities is None or landmarks is None:
            return None, "no_face", {
                "faces_detected": 0,
                "reason": "no_face",
            }

        aligned = self.mtcnn.extract(frame_pil, boxes, None)
        if aligned is None:
            return None, "extract_failed", {
                "faces_detected": len(boxes),
                "reason": "extract_failed",
            }

        if isinstance(aligned, torch.Tensor):
            face_tensors = aligned
            if face_tensors.ndim == 3:
                face_tensors = face_tensors.unsqueeze(0)
        elif isinstance(aligned, list):
            tensors = [tensor for tensor in aligned if isinstance(tensor, torch.Tensor)]
            if not tensors:
                return None, "extract_failed", {
                    "faces_detected": len(boxes),
                    "reason": "extract_failed",
                }
            face_tensors = torch.stack(tensors)
        else:
            return None, "extract_failed", {
                "faces_detected": len(boxes),
                "reason": "extract_failed",
            }

        with torch.no_grad():
            embeddings = self.resnet(face_tensors.to(self.device)).cpu().numpy()

        best_candidate: Optional[FaceCandidate] = None
        best_reason: Optional[str] = None
        best_debug_info: Optional[dict[str, Any]] = None
        best_rejected_info: Optional[dict[str, Any]] = None

        for index in range(len(boxes)):
            probability = float(probabilities[index] or 0.0)
            box = boxes[index].tolist()
            point_set = landmarks[index].tolist()

            crop_bgr = crop_face_region(frame_bgr, box)
            metrics = score_face_crop(crop_bgr, point_set, probability)
            debug_info = {
                "faces_detected": len(boxes),
                "detector_score": float(metrics["detector_score"]),
                "blur_score": float(metrics["blur_score"]),
                "frontal_score": float(metrics["frontal_score"]),
                "quality_score": float(metrics["quality_score"]),
                "brightness": float(metrics["brightness"]),
                "lower_std": float(metrics["lower_std"]),
                "lower_texture": float(metrics["lower_texture"]),
                "width": int(metrics["width"]),
                "height": int(metrics["height"]),
                "eye_angle": float(metrics["eye_angle"]),
                "nose_offset_ratio": float(metrics["nose_offset_ratio"]),
                "mouth_offset_ratio": float(metrics["mouth_offset_ratio"]),
                "vertical_ratio": float(metrics["vertical_ratio"]),
            }

            if not metrics["accepted"]:
                rejected_info = {
                    **debug_info,
                    "reason": metrics["reason"],
                    "reasons": metrics["reasons"],
                }
                if best_rejected_info is None or rejected_info["quality_score"] > best_rejected_info["quality_score"]:
                    best_rejected_info = rejected_info
                    best_reason = metrics["reason"]
                continue

            embedding = normalize_vector(embeddings[index])
            candidate = FaceCandidate(
                capture_second=capture_second,
                detector_score=probability,
                blur_score=float(metrics["blur_score"]),
                frontal_score=float(metrics["frontal_score"]),
                quality_score=float(metrics["quality_score"]),
                bounding_box=[float(value) for value in box],
                landmarks=[[float(value) for value in pair] for pair in point_set],
                embedding=embedding,
                crop_bgr=crop_bgr,
            )

            if best_candidate is None or candidate.quality_score > best_candidate.quality_score:
                best_candidate = candidate
                best_debug_info = debug_info

        if best_candidate is None:
            return None, best_reason or "no_valid_face", best_rejected_info or {
                "faces_detected": len(boxes),
                "reason": best_reason or "no_valid_face",
            }

        return best_candidate, None, best_debug_info or {
            "faces_detected": len(boxes),
        }

    def _log_frame_event(
        self,
        video_name: str,
        event: dict[str, Any],
        log_count: int,
        suppressed: bool,
        always: bool = False,
    ) -> tuple[int, bool]:
        if not always and log_count >= config.DEBUG_FRAME_LOG_LIMIT:
            if not suppressed:
                self.logger.info(
                    "逐幀偵錯 log 已達上限 %s 筆，後續只保留 accepted 與統計摘要：%s",
                    config.DEBUG_FRAME_LOG_LIMIT,
                    video_name,
                )
            return log_count, True

        self.logger.info("FrameCheck %s | %s", video_name, format_frame_event(event))
        if always:
            return log_count, suppressed

        return log_count + 1, suppressed

    def _assign_candidate_to_cluster(self, clusters: list[CandidateCluster], candidate: FaceCandidate) -> None:
        best_cluster = None
        best_similarity = -1.0

        for cluster in clusters:
            if cluster.centroid is None:
                continue

            similarity = cosine_similarity(candidate.embedding, cluster.centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is not None and best_similarity >= config.VIDEO_CLUSTER_MATCH_THRESHOLD:
            best_cluster.add(candidate)
            return

        new_cluster = CandidateCluster()
        new_cluster.add(candidate)
        clusters.append(new_cluster)

    def _persist_result(
        self,
        item: DiscoveredVideo,
        absolute_path: Path,
        file_stat: Any,
        modified_at: datetime,
        duration_seconds: float,
        path_sha1: str,
        existing: Optional[dict[str, Any]],
        selected_samples: list[FaceCandidate],
        metadata: dict[str, Any],
    ) -> None:
        cursor = self.connection.cursor(dictionary=True)
        old_person_id = int(existing["person_id"]) if existing and existing.get("person_id") else None
        old_sample_paths: list[str] = []
        sample_paths: list[str] = []

        try:
            if existing:
                cursor.execute(
                    "SELECT image_path FROM face_identity_samples WHERE video_id = %s ORDER BY capture_order",
                    (existing["id"],),
                )
                old_sample_paths = [str(row["image_path"]) for row in cursor.fetchall()]

            if selected_samples:
                sample_paths = save_sample_images(path_sha1, selected_samples)

            self.connection.start_transaction()

            person_id = None
            match_confidence = None
            assignment_source = "auto"
            group_locked = bool(existing.get("group_locked")) if existing else False

            if group_locked and old_person_id:
                person_id = old_person_id
                assignment_source = "manual"
            elif selected_samples:
                centroid = normalize_vector(np.mean([sample.embedding for sample in selected_samples], axis=0))
                person_id, match_confidence, is_new_person = self._match_or_create_person(cursor, centroid)
                assignment_source = "auto_new" if is_new_person else "auto"

            relative_path = normalize_relative_path(item.root_path, absolute_path)
            relative_directory = str(Path(relative_path).parent).replace("\\", "/")
            if relative_directory == ".":
                relative_directory = None

            preview_sample_path = sample_paths[0] if sample_paths else None
            scan_status = "complete" if selected_samples else "no_face"
            last_scanned_at = datetime.now().replace(microsecond=0)
            metadata_json = json.dumps(
                {
                    **metadata,
                    "source_root_label": item.root_label,
                    "source_root_path": str(item.root_path),
                    "feature_model": config.FEATURE_MODEL,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

            if existing:
                cursor.execute(
                    """
                    UPDATE face_identity_videos
                    SET person_id = %s,
                        feature_model = %s,
                        source_root_label = %s,
                        source_root_path = %s,
                        relative_directory = %s,
                        relative_path = %s,
                        absolute_path = %s,
                        file_name = %s,
                        file_size_bytes = %s,
                        file_modified_at = %s,
                        duration_seconds = %s,
                        frame_interval_seconds = %s,
                        accepted_sample_count = %s,
                        preview_sample_path = %s,
                        match_confidence = %s,
                        assignment_source = %s,
                        group_locked = %s,
                        scan_status = %s,
                        last_scanned_at = %s,
                        last_error = NULL,
                        metadata_json = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        person_id,
                        config.FEATURE_MODEL,
                        item.root_label,
                        str(item.root_path),
                        relative_directory,
                        relative_path,
                        str(absolute_path),
                        absolute_path.name,
                        int(file_stat.st_size),
                        modified_at,
                        round(duration_seconds, 3),
                        config.FRAME_INTERVAL_SECONDS,
                        len(selected_samples),
                        preview_sample_path,
                        match_confidence,
                        assignment_source,
                        int(group_locked),
                        scan_status,
                        last_scanned_at,
                        metadata_json,
                        existing["id"],
                    ),
                )
                video_id = int(existing["id"])
                cursor.execute("DELETE FROM face_identity_samples WHERE video_id = %s", (video_id,))
            else:
                cursor.execute(
                    """
                    INSERT INTO face_identity_videos (
                        person_id,
                        feature_model,
                        source_root_label,
                        source_root_path,
                        relative_directory,
                        relative_path,
                        absolute_path,
                        file_name,
                        path_sha1,
                        file_size_bytes,
                        file_modified_at,
                        duration_seconds,
                        frame_interval_seconds,
                        accepted_sample_count,
                        preview_sample_path,
                        match_confidence,
                        assignment_source,
                        group_locked,
                        scan_status,
                        last_scanned_at,
                        metadata_json,
                        created_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    """,
                    (
                        person_id,
                        config.FEATURE_MODEL,
                        item.root_label,
                        str(item.root_path),
                        relative_directory,
                        relative_path,
                        str(absolute_path),
                        absolute_path.name,
                        path_sha1,
                        int(file_stat.st_size),
                        modified_at,
                        round(duration_seconds, 3),
                        config.FRAME_INTERVAL_SECONDS,
                        len(selected_samples),
                        preview_sample_path,
                        match_confidence,
                        assignment_source,
                        0,
                        scan_status,
                        last_scanned_at,
                        metadata_json,
                    ),
                )
                video_id = int(cursor.lastrowid)

            for capture_order, (sample, relative_image_path) in enumerate(zip(selected_samples, sample_paths), start=1):
                cursor.execute(
                    """
                    INSERT INTO face_identity_samples (
                        video_id,
                        person_id,
                        feature_model,
                        capture_order,
                        capture_second,
                        image_path,
                        embedding_json,
                        embedding_sha1,
                        detector_score,
                        quality_score,
                        blur_score,
                        frontal_score,
                        bbox_json,
                        landmarks_json,
                        created_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    """,
                    (
                        video_id,
                        person_id,
                        config.FEATURE_MODEL,
                        capture_order,
                        round(sample.capture_second, 3),
                        relative_image_path,
                        vector_to_json(sample.embedding),
                        vector_sha1(sample.embedding),
                        round(sample.detector_score, 4),
                        round(sample.quality_score, 4),
                        round(sample.blur_score, 4),
                        round(sample.frontal_score, 4),
                        json.dumps(sample.bounding_box, separators=(",", ":")),
                        json.dumps(sample.landmarks, separators=(",", ":")),
                    ),
                )

            if person_id is not None:
                self._refresh_person_summary(cursor, person_id)
            if old_person_id is not None and old_person_id != person_id:
                self._refresh_person_summary(cursor, old_person_id)

            self.connection.commit()

        except Exception:  # noqa: BLE001
            self.connection.rollback()
            delete_relative_paths(sample_paths)
            raise
        finally:
            cursor.close()

        delete_relative_paths(old_sample_paths)

    def _match_or_create_person(
        self,
        cursor: mysql.connector.cursor.MySQLCursorDict,
        centroid_embedding: np.ndarray,
    ) -> tuple[int, Optional[float], bool]:
        cursor.execute(
            """
            SELECT id, centroid_embedding_json
            FROM face_identity_people
            WHERE feature_model = %s
              AND centroid_embedding_json IS NOT NULL
              AND video_count > 0
            ORDER BY id
            """,
            (config.FEATURE_MODEL,),
        )

        best_person_id: Optional[int] = None
        best_similarity = -1.0

        for row in cursor.fetchall():
            existing_vector = json_to_vector(row["centroid_embedding_json"])
            if existing_vector is None:
                continue
            similarity = cosine_similarity(centroid_embedding, existing_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_person_id = int(row["id"])

        if best_person_id is not None and best_similarity >= config.PERSON_MATCH_THRESHOLD:
            return best_person_id, round(best_similarity, 4), False

        cursor.execute(
            """
            INSERT INTO face_identity_people (feature_model, created_at, updated_at)
            VALUES (%s, NOW(), NOW())
            """,
            (config.FEATURE_MODEL,),
        )
        return int(cursor.lastrowid), None, True

    def _refresh_person_summary(
        self,
        cursor: mysql.connector.cursor.MySQLCursorDict,
        person_id: int,
    ) -> None:
        cursor.execute(
            """
            SELECT preview_sample_path, last_scanned_at
            FROM face_identity_videos
            WHERE person_id = %s
            ORDER BY last_scanned_at
            """,
            (person_id,),
        )
        videos = cursor.fetchall()

        if not videos:
            cursor.execute("DELETE FROM face_identity_people WHERE id = %s", (person_id,))
            return

        cursor.execute(
            """
            SELECT embedding_json, image_path
            FROM face_identity_samples
            WHERE person_id = %s
            ORDER BY capture_order
            """,
            (person_id,),
        )
        samples = cursor.fetchall()

        centroid_vectors = [
            vector
            for vector in (json_to_vector(row["embedding_json"]) for row in samples)
            if vector is not None
        ]
        centroid_json = None
        if centroid_vectors:
            centroid = normalize_vector(np.mean(np.stack(centroid_vectors, axis=0), axis=0))
            centroid_json = vector_to_json(centroid)

        cover_sample_path = next(
            (str(row["preview_sample_path"]) for row in videos if row.get("preview_sample_path")),
            None,
        ) or next((str(row["image_path"]) for row in samples if row.get("image_path")), None)

        first_seen_at = next((row["last_scanned_at"] for row in videos if row.get("last_scanned_at") is not None), None)
        last_seen_at = next(
            (row["last_scanned_at"] for row in reversed(videos) if row.get("last_scanned_at") is not None),
            None,
        )

        cursor.execute(
            """
            UPDATE face_identity_people
            SET cover_sample_path = %s,
                video_count = %s,
                sample_count = %s,
                first_seen_at = %s,
                last_seen_at = %s,
                centroid_embedding_json = %s,
                updated_at = NOW()
            WHERE id = %s
            """,
            (
                cover_sample_path,
                len(videos),
                len(samples),
                first_seen_at,
                last_seen_at,
                centroid_json,
                person_id,
            ),
        )

    def _persist_error(self, item: DiscoveredVideo, message: str) -> None:
        absolute_path = item.video_path.resolve()
        file_stat = absolute_path.stat()
        modified_at = datetime.fromtimestamp(file_stat.st_mtime).replace(microsecond=0)
        path_sha1 = sha1_path(absolute_path)
        existing = self._find_existing_video(path_sha1)
        cursor = self.connection.cursor(dictionary=True)

        try:
            payload = json.dumps(
                {
                    "source_root_label": item.root_label,
                    "source_root_path": str(item.root_path),
                    "error": message,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            relative_path = normalize_relative_path(item.root_path, absolute_path)
            relative_directory = str(Path(relative_path).parent).replace("\\", "/")
            if relative_directory == ".":
                relative_directory = None

            self.connection.start_transaction()
            if existing:
                cursor.execute(
                    """
                    UPDATE face_identity_videos
                    SET source_root_label = %s,
                        source_root_path = %s,
                        relative_directory = %s,
                        relative_path = %s,
                        absolute_path = %s,
                        file_name = %s,
                        file_size_bytes = %s,
                        file_modified_at = %s,
                        scan_status = 'error',
                        last_scanned_at = %s,
                        last_error = %s,
                        metadata_json = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        item.root_label,
                        str(item.root_path),
                        relative_directory,
                        relative_path,
                        str(absolute_path),
                        absolute_path.name,
                        int(file_stat.st_size),
                        modified_at,
                        datetime.now().replace(microsecond=0),
                        message,
                        payload,
                        existing["id"],
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO face_identity_videos (
                        feature_model,
                        source_root_label,
                        source_root_path,
                        relative_directory,
                        relative_path,
                        absolute_path,
                        file_name,
                        path_sha1,
                        file_size_bytes,
                        file_modified_at,
                        frame_interval_seconds,
                        scan_status,
                        last_scanned_at,
                        last_error,
                        metadata_json,
                        created_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'error', %s, %s, %s, NOW(), NOW())
                    """,
                    (
                        config.FEATURE_MODEL,
                        item.root_label,
                        str(item.root_path),
                        relative_directory,
                        relative_path,
                        str(absolute_path),
                        absolute_path.name,
                        path_sha1,
                        int(file_stat.st_size),
                        modified_at,
                        config.FRAME_INTERVAL_SECONDS,
                        datetime.now().replace(microsecond=0),
                        message,
                        payload,
                    ),
                )

            self.connection.commit()
        finally:
            cursor.close()

    def _find_existing_video(self, path_sha1: str) -> Optional[dict[str, Any]]:
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT id, person_id, file_size_bytes, file_modified_at, scan_status, group_locked
                FROM face_identity_videos
                WHERE path_sha1 = %s
                LIMIT 1
                """,
                (path_sha1,),
            )
            row = cursor.fetchone()
            return row
        finally:
            cursor.close()


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise RuntimeError(f"找不到 blog env：{path}")

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()

        values[key.strip()] = value

    return values


def build_sample_schedule(duration_seconds: float) -> list[float]:
    if duration_seconds <= 0:
        return [0.0]

    schedule: list[float] = []
    current_second = float(config.FIRST_SAMPLE_OFFSET_SECONDS)

    while current_second <= duration_seconds and len(schedule) < config.MAX_FRAME_CHECKS_PER_VIDEO:
        schedule.append(round(current_second, 3))
        current_second += float(config.FRAME_INTERVAL_SECONDS)

    if not schedule:
        schedule = [0.0]

    return schedule


def capture_frame(capture: cv2.VideoCapture, capture_second: float) -> tuple[bool, Optional[np.ndarray]]:
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(capture_second, 0.0) * 1000.0)
        ok, frame = capture.read()
        if ok and frame is not None:
            return True, frame

        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        if fps > 0:
            frame_index = max(int(round(capture_second * fps)), 0)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                return True, frame
    except cv2.error:
        return False, None

    return False, None


def probe_duration_seconds(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return max(float(completed.stdout.strip()), 0.0)
    except Exception:  # noqa: BLE001
        capture = cv2.VideoCapture(str(video_path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if fps > 0 and frame_count > 0:
                return max(frame_count / fps, 0.0)
        finally:
            capture.release()

    return 0.0


def crop_face_region(frame_bgr: np.ndarray, box: list[float]) -> np.ndarray:
    frame_height, frame_width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)

    margin_x = int(round(width * config.CROP_MARGIN_RATIO))
    margin_y = int(round(height * config.CROP_MARGIN_RATIO))

    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(frame_width, x2 + margin_x)
    crop_y2 = min(frame_height, y2 + margin_y)

    return frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()


def score_face_crop(
    crop_bgr: np.ndarray,
    landmarks: list[list[float]],
    detector_score: float,
) -> dict[str, Any]:
    if crop_bgr.size == 0:
        return rejected_metrics("empty_crop")

    height, width = crop_bgr.shape[:2]
    if width < config.MIN_FACE_WIDTH or height < config.MIN_FACE_HEIGHT:
        return rejected_metrics("face_too_small")

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    lower_half = gray[int(height * 0.55): int(height * 0.95), :]
    lower_std = float(np.std(lower_half)) if lower_half.size else 0.0
    lower_texture = float(cv2.Laplacian(lower_half, cv2.CV_64F).var()) if lower_half.size else 0.0

    frontal_score, frontal_metrics = compute_frontal_score(landmarks)
    eye_angle = float(frontal_metrics.get("eye_angle", 0.0))
    nose_offset_ratio = float(frontal_metrics.get("nose_offset_ratio", 0.0))
    mouth_offset_ratio = float(frontal_metrics.get("mouth_offset_ratio", 0.0))
    vertical_ratio = float(frontal_metrics.get("vertical_ratio", 0.0))
    reasons = []

    if (
        eye_angle > config.HARD_MAX_EYE_ANGLE_DEGREES
        or nose_offset_ratio > config.HARD_MAX_NOSE_OFFSET_RATIO
        or mouth_offset_ratio > config.HARD_MAX_MOUTH_OFFSET_RATIO
        or vertical_ratio > config.HARD_MAX_VERTICAL_RATIO
    ):
        reasons.append("side_face")
    if detector_score < config.MIN_DETECTION_PROBABILITY:
        reasons.append("low_detector_score")
    if blur_score < config.MIN_BLUR_SCORE:
        reasons.append("blurry")
    if brightness < config.MIN_BRIGHTNESS:
        reasons.append("too_dark")
    if lower_std < config.MIN_LOWER_FACE_STD or lower_texture < config.MIN_LOWER_FACE_TEXTURE:
        reasons.append("masked_or_occluded")
    if frontal_score < config.MIN_FRONT_SCORE:
        reasons.append("side_face")

    reasons = list(dict.fromkeys(reasons))

    quality_score = (
        min(blur_score / (config.MIN_BLUR_SCORE * 2.2), 1.0) * 0.34
        + min(max((brightness - config.MIN_BRIGHTNESS) / 85.0, 0.0), 1.0) * 0.14
        + min(lower_texture / (config.MIN_LOWER_FACE_TEXTURE * 3.0), 1.0) * 0.12
        + frontal_score * 0.24
        + min(max((detector_score - config.MIN_DETECTION_PROBABILITY) / 0.03, 0.0), 1.0) * 0.16
    )

    if reasons:
        return {
            "accepted": False,
            "reason": reasons[0],
            "reasons": reasons,
            "detector_score": detector_score,
            "blur_score": blur_score,
            "frontal_score": frontal_score,
            "quality_score": quality_score,
            "brightness": brightness,
            "lower_std": lower_std,
            "lower_texture": lower_texture,
            "width": width,
            "height": height,
            "eye_angle": eye_angle,
            "nose_offset_ratio": nose_offset_ratio,
            "mouth_offset_ratio": mouth_offset_ratio,
            "vertical_ratio": vertical_ratio,
        }

    return {
        "accepted": True,
        "reason": None,
        "reasons": [],
        "detector_score": detector_score,
        "blur_score": blur_score,
        "frontal_score": frontal_score,
        "quality_score": quality_score,
        "brightness": brightness,
        "lower_std": lower_std,
        "lower_texture": lower_texture,
        "width": width,
        "height": height,
        "eye_angle": eye_angle,
        "nose_offset_ratio": nose_offset_ratio,
        "mouth_offset_ratio": mouth_offset_ratio,
        "vertical_ratio": vertical_ratio,
    }


def compute_frontal_score(landmarks: list[list[float]]) -> tuple[float, dict[str, float]]:
    if len(landmarks) != 5:
        return 0.0, {}

    left_eye = np.array(landmarks[0], dtype=np.float32)
    right_eye = np.array(landmarks[1], dtype=np.float32)
    nose = np.array(landmarks[2], dtype=np.float32)
    mouth_left = np.array(landmarks[3], dtype=np.float32)
    mouth_right = np.array(landmarks[4], dtype=np.float32)

    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance <= 1.0:
        return 0.0, {}

    eye_angle = abs(math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0]))))
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (mouth_left + mouth_right) / 2.0
    nose_offset_ratio = abs(float(nose[0] - eye_center[0])) / eye_distance
    mouth_offset_ratio = abs(float(mouth_center[0] - eye_center[0])) / eye_distance
    vertical_ratio = float(mouth_center[1] - eye_center[1]) / eye_distance

    penalties = [
        min(eye_angle / config.MAX_EYE_ANGLE_DEGREES, 1.0) * 0.24,
        min(nose_offset_ratio / config.MAX_NOSE_OFFSET_RATIO, 1.0) * 0.34,
        min(mouth_offset_ratio / config.MAX_MOUTH_OFFSET_RATIO, 1.0) * 0.24,
    ]

    if vertical_ratio < config.MIN_VERTICAL_RATIO or vertical_ratio > config.MAX_VERTICAL_RATIO:
        penalties.append(0.18)

    frontal_score = max(0.0, 1.0 - sum(penalties))

    return frontal_score, {
        "eye_angle": eye_angle,
        "nose_offset_ratio": nose_offset_ratio,
        "mouth_offset_ratio": mouth_offset_ratio,
        "vertical_ratio": vertical_ratio,
    }


def rejected_metrics(reason: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "reason": reason,
        "reasons": [reason],
        "detector_score": 0.0,
        "blur_score": 0.0,
        "frontal_score": 0.0,
        "quality_score": 0.0,
        "brightness": 0.0,
        "lower_std": 0.0,
        "lower_texture": 0.0,
        "width": 0,
        "height": 0,
        "eye_angle": 0.0,
        "nose_offset_ratio": 0.0,
        "mouth_offset_ratio": 0.0,
        "vertical_ratio": 0.0,
    }


def append_debug_event(events: list[dict[str, Any]], event: dict[str, Any]) -> None:
    if len(events) >= config.DEBUG_METADATA_EVENT_LIMIT:
        return

    events.append(event)


def format_frame_event(event: dict[str, Any]) -> str:
    status = str(event.get("status") or "unknown")
    second = format_metric(event.get("second"), precision=3)
    faces_detected = int(event.get("faces_detected") or 0)
    parts = [
        f"t={second}s",
        f"status={status}",
        f"faces={faces_detected}",
    ]

    reason = event.get("reason")
    if reason:
        parts.append(f"reason={reason}")

    if event.get("detector_score") is not None:
        parts.append(f"det={format_metric(event.get('detector_score'), precision=4)}")
    if event.get("blur_score") is not None:
        parts.append(f"blur={format_metric(event.get('blur_score'))}")
    if event.get("frontal_score") is not None:
        parts.append(f"front={format_metric(event.get('frontal_score'), precision=3)}")
    if event.get("quality_score") is not None:
        parts.append(f"quality={format_metric(event.get('quality_score'), precision=3)}")
    if event.get("brightness") is not None:
        parts.append(f"bright={format_metric(event.get('brightness'))}")
    if event.get("lower_texture") is not None:
        parts.append(f"tex={format_metric(event.get('lower_texture'))}")
    if event.get("lower_std") is not None:
        parts.append(f"std={format_metric(event.get('lower_std'))}")

    width = int(event.get("width") or 0)
    height = int(event.get("height") or 0)
    if width > 0 and height > 0:
        parts.append(f"size={width}x{height}")

    if event.get("eye_angle") is not None:
        parts.append(f"eye={format_metric(event.get('eye_angle'))}")
    if event.get("nose_offset_ratio") is not None:
        parts.append(f"nose={format_metric(event.get('nose_offset_ratio'), precision=3)}")
    if event.get("mouth_offset_ratio") is not None:
        parts.append(f"mouth={format_metric(event.get('mouth_offset_ratio'), precision=3)}")
    if event.get("vertical_ratio") is not None:
        parts.append(f"vertical={format_metric(event.get('vertical_ratio'), precision=3)}")

    cluster_size = event.get("cluster_size")
    if cluster_size is not None:
        parts.append(f"cluster={cluster_size}")

    reasons = event.get("reasons") or []
    if isinstance(reasons, list) and reasons:
        parts.append(f"all_reasons={','.join(str(reason) for reason in reasons)}")

    return " | ".join(parts)


def format_metric(value: Any, precision: int = 2) -> str:
    if value is None:
        return "-"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    return f"{numeric:.{precision}f}"


def pick_best_cluster(clusters: list[CandidateCluster]) -> Optional[CandidateCluster]:
    if not clusters:
        return None

    return max(
        clusters,
        key=lambda cluster: (
            len(cluster.samples),
            round(sum(sample.quality_score for sample in cluster.samples) / max(len(cluster.samples), 1), 6),
        ),
    )


def save_sample_images(path_sha1: str, samples: list[FaceCandidate]) -> list[str]:
    timestamp_token = datetime.now().strftime("%Y%m%d%H%M%S")
    relative_paths: list[str] = []

    for index, sample in enumerate(samples, start=1):
        relative_path = (
            Path(config.SAMPLE_IMAGE_SUBDIR)
            / datetime.now().strftime("%Y%m%d")
            / path_sha1[:16]
            / timestamp_token
            / f"sample_{index:02d}.jpg"
        )
        absolute_path = config.BLOG_PUBLIC_STORAGE_ROOT / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(str(absolute_path), sample.crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"無法寫入樣本圖：{absolute_path}")

        relative_paths.append(str(relative_path).replace("\\", "/"))

    return relative_paths


def delete_relative_paths(paths: Iterable[str]) -> None:
    for relative_path in paths:
        if not relative_path:
            continue
        absolute_path = config.BLOG_PUBLIC_STORAGE_ROOT / relative_path
        try:
            if absolute_path.exists():
                absolute_path.unlink()
        except Exception:  # noqa: BLE001
            continue


def normalize_relative_path(root_path: Path, absolute_path: Path) -> str:
    try:
        relative = absolute_path.relative_to(root_path)
        return str(relative).replace("\\", "/")
    except ValueError:
        return absolute_path.name


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return -1.0
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-9))


def json_to_vector(payload: Any) -> Optional[np.ndarray]:
    if payload in (None, ""):
        return None

    if isinstance(payload, str):
        try:
            values = json.loads(payload)
        except json.JSONDecodeError:
            return None
    else:
        values = payload

    if not isinstance(values, list) or not values:
        return None

    try:
        array = np.array([float(value) for value in values], dtype=np.float32)
    except (TypeError, ValueError):
        return None

    return normalize_vector(array)


def vector_to_json(vector: np.ndarray) -> str:
    return json.dumps([round(float(value), 8) for value in vector.tolist()], separators=(",", ":"))


def vector_sha1(vector: np.ndarray) -> str:
    payload = vector_to_json(vector)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def sha1_path(path: Path) -> str:
    normalized = str(path).replace("\\", "/").lower().strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan videos and group same-person works into face_identity_* tables.")
    parser.add_argument("--force", action="store_true", help="Re-scan videos even if the DB already has completed rows.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N discovered videos.")
    parser.add_argument("--path", action="append", default=None, help="Scan a custom file or directory instead of config roots.")
    args = parser.parse_args()

    scanner = FaceIdentityScanner(force=args.force, limit=args.limit, paths=args.path)
    try:
        return scanner.run()
    finally:
        scanner.close()


if __name__ == "__main__":
    raise SystemExit(main())
