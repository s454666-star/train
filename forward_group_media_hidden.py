import argparse
import asyncio
import logging
import os
import shutil
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from telethon import TelegramClient, utils
from telethon.errors import FloodWaitError
from telethon.tl.types import DocumentAttributeSticker

from activate_telegram_session import API_HASH, API_ID, BASE_DIR, resolve_session_base


DEFAULT_DB_PATH = os.path.join(BASE_DIR, "logs", "hidden_media_forward.sqlite3")
DEFAULT_LOG_PATH = os.path.join(BASE_DIR, "logs", "hidden_media_forward.log")
DEFAULT_SOURCE_SESSION = os.path.join(BASE_DIR, "session", "main_account2.session")
DEFAULT_WORK_SESSION = os.path.join(BASE_DIR, "session", "codex_hidden_media_forward.session")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_title(value: str) -> str:
    text = str(value or "").casefold().strip()
    for suffix in ("group", "groups", "chat", "chats", "群組", "群组", "群", "聊天室"):
        text = text.replace(suffix.casefold(), "")
    for ch in (" ", "\t", "\r", "\n", "-", "_", "—", "–", "(", ")", "[", "]"):
        text = text.replace(ch, "")
    return text


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def setup_logging(log_path: str) -> None:
    ensure_parent(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def init_db(db_path: str) -> sqlite3.Connection:
    ensure_parent(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS media_forward_jobs (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            source_query TEXT NOT NULL,
            target_query TEXT NOT NULL,
            source_peer_id INTEGER,
            source_peer_title TEXT,
            target_peer_id INTEGER,
            target_peer_title TEXT,
            hide_sender INTEGER NOT NULL DEFAULT 1,
            drop_media_captions INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_scanned_message_id INTEGER NOT NULL DEFAULT 0,
            last_forwarded_source_message_id INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'running'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS media_forward_items (
            source_peer_id INTEGER NOT NULL,
            source_message_id INTEGER NOT NULL,
            source_date TEXT,
            grouped_id INTEGER,
            media_file_id TEXT,
            media_kind TEXT NOT NULL,
            caption_present INTEGER NOT NULL DEFAULT 0,
            file_name TEXT,
            mime_type TEXT,
            file_size INTEGER,
            status TEXT NOT NULL,
            target_peer_id INTEGER,
            target_message_id INTEGER,
            duplicate_of_message_id INTEGER,
            attempts INTEGER NOT NULL DEFAULT 0,
            first_seen_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            forwarded_at TEXT,
            error TEXT,
            PRIMARY KEY (source_peer_id, source_message_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS media_forward_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            level TEXT NOT NULL,
            source_peer_id INTEGER,
            source_message_id INTEGER,
            action TEXT NOT NULL,
            detail TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_media_forward_items_status
        ON media_forward_items (status, source_peer_id, source_message_id)
        """
    )
    ensure_column(conn, "media_forward_items", "media_file_id", "TEXT")
    ensure_column(conn, "media_forward_items", "duplicate_of_message_id", "INTEGER")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_media_forward_items_media_file
        ON media_forward_items (source_peer_id, media_file_id, status, source_message_id)
        """
    )
    conn.commit()
    return conn


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()


def db_event(
    conn: sqlite3.Connection,
    level: str,
    action: str,
    detail: str = "",
    source_peer_id: Optional[int] = None,
    source_message_id: Optional[int] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO media_forward_events
            (ts, level, source_peer_id, source_message_id, action, detail)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (utc_now(), level, source_peer_id, source_message_id, action, detail[:2000]),
    )
    conn.commit()


def update_job(
    conn: sqlite3.Connection,
    source_query: str,
    target_query: str,
    source_peer_id: Optional[int] = None,
    source_peer_title: Optional[str] = None,
    target_peer_id: Optional[int] = None,
    target_peer_title: Optional[str] = None,
    drop_media_captions: bool = False,
    status: str = "running",
    last_scanned_message_id: Optional[int] = None,
    last_forwarded_source_message_id: Optional[int] = None,
) -> None:
    existing = conn.execute("SELECT id FROM media_forward_jobs WHERE id = 1").fetchone()
    if existing:
        fields = [
            "source_query = ?",
            "target_query = ?",
            "updated_at = ?",
            "drop_media_captions = ?",
            "status = ?",
        ]
        values: List[Any] = [source_query, target_query, utc_now(), int(drop_media_captions), status]
        optional_pairs = [
            ("source_peer_id", source_peer_id),
            ("source_peer_title", source_peer_title),
            ("target_peer_id", target_peer_id),
            ("target_peer_title", target_peer_title),
            ("last_scanned_message_id", last_scanned_message_id),
            ("last_forwarded_source_message_id", last_forwarded_source_message_id),
        ]
        for name, value in optional_pairs:
            if value is not None:
                fields.append(f"{name} = ?")
                values.append(value)
        values.append(1)
        conn.execute(f"UPDATE media_forward_jobs SET {', '.join(fields)} WHERE id = ?", values)
    else:
        now = utc_now()
        conn.execute(
            """
            INSERT INTO media_forward_jobs (
                id, source_query, target_query, source_peer_id, source_peer_title,
                target_peer_id, target_peer_title, hide_sender, drop_media_captions,
                started_at, updated_at, status
            )
            VALUES (1, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            """,
            (
                source_query,
                target_query,
                source_peer_id,
                source_peer_title,
                target_peer_id,
                target_peer_title,
                int(drop_media_captions),
                now,
                now,
                status,
            ),
        )
    conn.commit()


def media_file_id(msg: Any) -> Optional[str]:
    photo = getattr(msg, "photo", None)
    if photo is not None:
        value = getattr(photo, "id", None)
        return f"photo:{value}" if value is not None else None

    document = getattr(msg, "document", None)
    if document is not None:
        value = getattr(document, "id", None)
        return f"document:{value}" if value is not None else None

    return None


def record_seen_item(conn: sqlite3.Connection, source_peer_id: int, msg: Any, media_kind: str) -> None:
    document = getattr(msg, "document", None)
    file_info = getattr(msg, "file", None)
    grouped_id = getattr(msg, "grouped_id", None)
    date_value = getattr(msg, "date", None)
    source_date = date_value.isoformat() if date_value else None
    file_name = getattr(file_info, "name", None) if file_info else None
    mime_type = getattr(document, "mime_type", None) if document else getattr(file_info, "mime_type", None)
    file_size = getattr(file_info, "size", None) if file_info else None
    file_id = media_file_id(msg)
    now = utc_now()
    conn.execute(
        """
        INSERT INTO media_forward_items (
            source_peer_id, source_message_id, source_date, grouped_id, media_file_id, media_kind,
            caption_present, file_name, mime_type, file_size, status,
            first_seen_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
        ON CONFLICT(source_peer_id, source_message_id) DO UPDATE SET
            source_date = excluded.source_date,
            grouped_id = excluded.grouped_id,
            media_file_id = excluded.media_file_id,
            media_kind = excluded.media_kind,
            caption_present = excluded.caption_present,
            file_name = excluded.file_name,
            mime_type = excluded.mime_type,
            file_size = excluded.file_size,
            updated_at = excluded.updated_at
        """,
        (
            source_peer_id,
            int(msg.id),
            source_date,
            int(grouped_id) if grouped_id else None,
            file_id,
            media_kind,
            int(bool(getattr(msg, "message", None))),
            file_name,
            mime_type,
            file_size,
            now,
            now,
        ),
    )
    conn.commit()


def get_item(conn: sqlite3.Connection, source_peer_id: int, source_message_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM media_forward_items
        WHERE source_peer_id = ? AND source_message_id = ?
        """,
        (source_peer_id, source_message_id),
    ).fetchone()


def find_forwarded_duplicate(
    conn: sqlite3.Connection,
    source_peer_id: int,
    media_file_id_value: Optional[str],
    source_message_id: int,
) -> Optional[sqlite3.Row]:
    if not media_file_id_value:
        return None
    return conn.execute(
        """
        SELECT source_message_id, target_message_id
        FROM media_forward_items
        WHERE source_peer_id = ?
          AND media_file_id = ?
          AND source_message_id <> ?
          AND status = 'forwarded'
        ORDER BY source_message_id ASC
        LIMIT 1
        """,
        (source_peer_id, media_file_id_value, source_message_id),
    ).fetchone()


def find_any_prior_duplicate(
    conn: sqlite3.Connection,
    source_peer_id: int,
    media_file_id_value: Optional[str],
    source_message_id: int,
) -> Optional[sqlite3.Row]:
    if not media_file_id_value:
        return None
    return conn.execute(
        """
        SELECT source_message_id, status, target_message_id
        FROM media_forward_items
        WHERE source_peer_id = ?
          AND media_file_id = ?
          AND source_message_id < ?
          AND status IN ('forwarded', 'duplicate_skipped')
        ORDER BY source_message_id ASC
        LIMIT 1
        """,
        (source_peer_id, media_file_id_value, source_message_id),
    ).fetchone()


def mark_gif_skipped(conn: sqlite3.Connection, source_peer_id: int, source_message_id: int) -> None:
    conn.execute(
        """
        UPDATE media_forward_items
        SET status = 'gif_skipped',
            updated_at = ?,
            error = NULL
        WHERE source_peer_id = ? AND source_message_id = ?
        """,
        (utc_now(), source_peer_id, source_message_id),
    )
    conn.commit()


def mark_sticker_skipped(
    conn: sqlite3.Connection,
    source_peer_id: int,
    source_message_id: int,
    detail: str = "",
) -> None:
    conn.execute(
        """
        UPDATE media_forward_items
        SET status = 'sticker_skipped',
            updated_at = ?,
            error = ?
        WHERE source_peer_id = ? AND source_message_id = ?
        """,
        (utc_now(), detail[:2000] or None, source_peer_id, source_message_id),
    )
    conn.commit()


def mark_duplicate(
    conn: sqlite3.Connection,
    source_peer_id: int,
    source_message_id: int,
    duplicate_of_message_id: int,
) -> None:
    conn.execute(
        """
        UPDATE media_forward_items
        SET status = 'duplicate_skipped',
            duplicate_of_message_id = ?,
            updated_at = ?,
            error = NULL
        WHERE source_peer_id = ? AND source_message_id = ?
        """,
        (duplicate_of_message_id, utc_now(), source_peer_id, source_message_id),
    )
    conn.commit()


def mark_forwarding(conn: sqlite3.Connection, source_peer_id: int, source_message_id: int) -> int:
    row = get_item(conn, source_peer_id, source_message_id)
    attempts = int(row["attempts"] or 0) + 1 if row else 1
    conn.execute(
        """
        UPDATE media_forward_items
        SET status = 'forwarding', attempts = ?, updated_at = ?, error = NULL
        WHERE source_peer_id = ? AND source_message_id = ?
        """,
        (attempts, utc_now(), source_peer_id, source_message_id),
    )
    conn.commit()
    return attempts


def mark_forwarded(
    conn: sqlite3.Connection,
    source_peer_id: int,
    source_message_id: int,
    target_peer_id: int,
    target_message_id: int,
) -> None:
    now = utc_now()
    conn.execute(
        """
        UPDATE media_forward_items
        SET status = 'forwarded',
            target_peer_id = ?,
            target_message_id = ?,
            updated_at = ?,
            forwarded_at = ?,
            error = NULL
        WHERE source_peer_id = ? AND source_message_id = ?
        """,
        (target_peer_id, target_message_id, now, now, source_peer_id, source_message_id),
    )
    update_job(
        conn,
        source_query=get_job_value(conn, "source_query") or "",
        target_query=get_job_value(conn, "target_query") or "",
        last_forwarded_source_message_id=source_message_id,
    )
    conn.commit()


def mark_error(
    conn: sqlite3.Connection,
    source_peer_id: int,
    source_message_id: int,
    error: str,
    final: bool = False,
) -> None:
    conn.execute(
        """
        UPDATE media_forward_items
        SET status = ?, updated_at = ?, error = ?
        WHERE source_peer_id = ? AND source_message_id = ?
        """,
        ("error" if final else "pending", utc_now(), error[:2000], source_peer_id, source_message_id),
    )
    conn.commit()


def get_job_value(conn: sqlite3.Connection, column: str) -> Optional[Any]:
    row = conn.execute(f"SELECT {column} FROM media_forward_jobs WHERE id = 1").fetchone()
    return row[0] if row else None


def is_sticker_message(msg: Any) -> bool:
    if getattr(msg, "sticker", None):
        return True

    document = getattr(msg, "document", None)
    attributes = getattr(document, "attributes", None) if document is not None else None
    if attributes:
        for attr in attributes:
            if isinstance(attr, DocumentAttributeSticker) or attr.__class__.__name__ == "DocumentAttributeSticker":
                return True

    file_info = getattr(msg, "file", None)
    file_name = str(getattr(file_info, "name", "") or "").casefold() if file_info else ""
    mime_type = str(
        getattr(document, "mime_type", None)
        or getattr(file_info, "mime_type", None)
        or ""
    ).casefold()

    if mime_type == "application/x-tgsticker" or "sticker" in mime_type:
        return True
    if file_name in {"sticker.webp", "sticker.tgs", "sticker.webm"}:
        return True
    return False


def media_kind(msg: Any) -> Optional[str]:
    if is_sticker_message(msg):
        return "sticker"

    if getattr(msg, "gif", None):
        return "gif"

    file_info = getattr(msg, "file", None)
    document = getattr(msg, "document", None)
    file_name = str(getattr(file_info, "name", "") or "").casefold() if file_info else ""
    mime_type = str(
        getattr(document, "mime_type", None)
        or getattr(file_info, "mime_type", None)
        or ""
    ).casefold()

    if mime_type == "image/gif" or file_name.endswith(".gif"):
        return "gif"

    if getattr(msg, "photo", None):
        return "photo"

    if getattr(msg, "video", None) or mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("image/"):
        return "image_document"
    return None


def copy_sqlite_session(source_session: str, work_session: str, refresh: bool) -> str:
    source_path = os.path.abspath(source_session)
    work_path = os.path.abspath(work_session)
    if not source_path.endswith(".session"):
        source_path += ".session"
    if not work_path.endswith(".session"):
        work_path += ".session"

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source session not found: {source_path}")

    if os.path.exists(work_path) and not refresh:
        return os.path.splitext(work_path)[0]

    ensure_parent(work_path)
    temp_path = f"{work_path}.tmp"
    if os.path.exists(temp_path):
        os.remove(temp_path)

    try:
        with closing(sqlite3.connect(f"file:{source_path}?mode=ro", uri=True, timeout=30)) as src:
            with closing(sqlite3.connect(temp_path, timeout=30)) as dst:
                src.backup(dst)
                dst.commit()
    except sqlite3.DatabaseError:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        shutil.copy2(source_path, temp_path)

    os.replace(temp_path, work_path)
    return os.path.splitext(work_path)[0]


async def resolve_dialog(client: TelegramClient, query: str) -> Tuple[Any, str, int]:
    raw_query = str(query or "").strip()
    if not raw_query:
        raise ValueError("Dialog query is empty.")

    try:
        entity = await client.get_entity(raw_query)
        title = getattr(entity, "title", None) or getattr(entity, "username", None) or raw_query
        return entity, str(title), int(utils.get_peer_id(entity))
    except Exception:
        pass

    query_norm = normalize_title(raw_query)
    exact: List[Tuple[Any, str, int]] = []
    contains: List[Tuple[Any, str, int]] = []
    reverse_contains: List[Tuple[Any, str, int]] = []

    async for dialog in client.iter_dialogs():
        title = dialog.name or ""
        entity = dialog.entity
        peer_id = int(utils.get_peer_id(entity))
        title_norm = normalize_title(title)
        item = (entity, title, peer_id)
        if title_norm == query_norm:
            exact.append(item)
        elif query_norm and query_norm in title_norm:
            contains.append(item)
        elif title_norm and title_norm in query_norm:
            reverse_contains.append(item)

    for bucket in (exact, contains, reverse_contains):
        if len(bucket) == 1:
            return bucket[0]
        if len(bucket) > 1:
            options = ", ".join(f"{title} ({peer_id})" for _, title, peer_id in bucket[:10])
            raise ValueError(f"Ambiguous dialog query {raw_query!r}. Matches: {options}")

    raise ValueError(f"Dialog not found: {raw_query!r}")


def sent_message_id(result: Any) -> int:
    if isinstance(result, list):
        result = result[0] if result else None
    return int(getattr(result, "id", 0) or 0)


async def forward_one(
    client: TelegramClient,
    source_entity: Any,
    target_entity: Any,
    msg: Any,
    drop_media_captions: bool,
) -> int:
    result = await client.forward_messages(
        target_entity,
        [int(msg.id)],
        from_peer=source_entity,
        drop_author=True,
        drop_media_captions=drop_media_captions,
    )
    target_message_id = sent_message_id(result)
    if target_message_id <= 0:
        raise RuntimeError("Forward succeeded but target message id was not returned.")
    return target_message_id


def summarize(conn: sqlite3.Connection, source_peer_id: int) -> Dict[str, int]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS total
        FROM media_forward_items
        WHERE source_peer_id = ?
        GROUP BY status
        """,
        (source_peer_id,),
    ).fetchall()
    data = {str(row["status"]): int(row["total"]) for row in rows}
    data["total"] = sum(data.values())
    return data


def format_summary(data: Dict[str, int]) -> str:
    keys = [
        "total",
        "forwarded",
        "duplicate_skipped",
        "gif_skipped",
        "sticker_skipped",
        "pending",
        "forwarding",
        "error",
    ]
    return " ".join(f"{key}={int(data.get(key, 0))}" for key in keys)


async def run(args: argparse.Namespace) -> int:
    setup_logging(args.log_path)
    conn = init_db(args.db_path)

    work_session_base = copy_sqlite_session(args.source_session, args.work_session, args.refresh_session_copy)
    logging.info("Using work session: %s", work_session_base)
    logging.info("Progress DB: %s", os.path.abspath(args.db_path))

    client = TelegramClient(work_session_base, API_ID, API_HASH)
    await client.connect()
    try:
        if not await client.is_user_authorized():
            raise RuntimeError(f"Work session is not authorized: {work_session_base}")

        source_entity, source_title, source_peer_id = await resolve_dialog(client, args.source)
        target_entity, target_title, target_peer_id = await resolve_dialog(client, args.target)

        update_job(
            conn,
            args.source,
            args.target,
            source_peer_id=source_peer_id,
            source_peer_title=source_title,
            target_peer_id=target_peer_id,
            target_peer_title=target_title,
            drop_media_captions=args.drop_media_captions,
            status="running",
        )
        db_event(
            conn,
            "info",
            "job_start",
            f"source={source_title} ({source_peer_id}) target={target_title} ({target_peer_id})",
        )

        logging.info("Source: %s (%s)", source_title, source_peer_id)
        logging.info("Target: %s (%s)", target_title, target_peer_id)
        logging.info("Hide sender name: enabled")

        scanned = 0
        forwarded = 0
        skipped_done = 0
        skipped_duplicate = 0
        skipped_gif = 0
        skipped_sticker = 0
        errors = 0
        stopped_reason = ""
        started_at = time.monotonic()

        async for msg in client.iter_messages(source_entity, reverse=True, min_id=max(int(args.min_id or 0), 0)):
            if int(args.scan_limit or 0) > 0 and scanned >= int(args.scan_limit):
                stopped_reason = "scan_limit"
                logging.info("Scan limit reached: %s", args.scan_limit)
                break
            scanned += 1

            try:
                source_message_id = int(msg.id)
            except Exception:
                continue

            update_job(
                conn,
                args.source,
                args.target,
                last_scanned_message_id=source_message_id,
                status="running",
            )

            kind = media_kind(msg)
            if not kind:
                continue

            record_seen_item(conn, source_peer_id, msg, kind)
            row = get_item(conn, source_peer_id, source_message_id)
            status = str(row["status"] or "") if row else ""
            attempts = int(row["attempts"] or 0) if row else 0

            if status == "forwarded":
                skipped_done += 1
                continue
            if status == "duplicate_skipped":
                skipped_duplicate += 1
                continue
            if status == "gif_skipped":
                skipped_gif += 1
                continue
            if status == "sticker_skipped":
                skipped_sticker += 1
                continue

            if kind == "gif":
                mark_gif_skipped(conn, source_peer_id, source_message_id)
                db_event(conn, "info", "gif_skipped", "", source_peer_id, source_message_id)
                skipped_gif += 1
                logging.info("Skipped gif source_message_id=%s", source_message_id)
                continue
            if kind == "sticker":
                mark_sticker_skipped(conn, source_peer_id, source_message_id)
                db_event(conn, "info", "sticker_skipped", "", source_peer_id, source_message_id)
                skipped_sticker += 1
                logging.info("Skipped sticker source_message_id=%s", source_message_id)
                continue

            duplicate = find_forwarded_duplicate(
                conn,
                source_peer_id,
                row["media_file_id"] if row else None,
                source_message_id,
            )
            if duplicate is None:
                duplicate = find_any_prior_duplicate(
                    conn,
                    source_peer_id,
                    row["media_file_id"] if row else None,
                    source_message_id,
                )
            if duplicate is not None:
                duplicate_of = int(duplicate["source_message_id"])
                mark_duplicate(conn, source_peer_id, source_message_id, duplicate_of)
                db_event(
                    conn,
                    "info",
                    "duplicate_skipped",
                    f"duplicate_of_message_id={duplicate_of}",
                    source_peer_id,
                    source_message_id,
                )
                skipped_duplicate += 1
                logging.info(
                    "Skipped duplicate media source_message_id=%s duplicate_of=%s file_id=%s",
                    source_message_id,
                    duplicate_of,
                    row["media_file_id"] if row else "",
                )
                continue

            if attempts >= int(args.max_attempts):
                errors += 1
                mark_error(
                    conn,
                    source_peer_id,
                    source_message_id,
                    (row["error"] if row else "") or "max_attempts_reached",
                    final=True,
                )
                logging.warning(
                    "Skipping source message %s after %s attempts: %s",
                    source_message_id,
                    attempts,
                    (row["error"] if row else "") or "",
                )
                continue

            if args.dry_run:
                logging.info("DRY RUN would forward %s message_id=%s", kind, source_message_id)
                continue

            while True:
                attempt = mark_forwarding(conn, source_peer_id, source_message_id)
                try:
                    target_message_id = await forward_one(
                        client,
                        source_entity,
                        target_entity,
                        msg,
                        drop_media_captions=bool(args.drop_media_captions),
                    )
                    mark_forwarded(conn, source_peer_id, source_message_id, target_peer_id, target_message_id)
                    db_event(
                        conn,
                        "info",
                        "forwarded",
                        f"target_message_id={target_message_id}",
                        source_peer_id,
                        source_message_id,
                    )
                    forwarded += 1
                    logging.info(
                        "Forwarded %s source_message_id=%s -> target_message_id=%s",
                        kind,
                        source_message_id,
                        target_message_id,
                    )
                    break
                except FloodWaitError as error:
                    wait_seconds = int(error.seconds or 0) + 3
                    mark_error(conn, source_peer_id, source_message_id, f"flood_wait:{wait_seconds}", final=False)
                    db_event(
                        conn,
                        "warning",
                        "flood_wait",
                        f"wait_seconds={wait_seconds}",
                        source_peer_id,
                        source_message_id,
                    )
                    logging.warning("Flood wait %ss at source_message_id=%s", wait_seconds, source_message_id)
                    if wait_seconds > int(args.max_flood_wait):
                        mark_error(
                            conn,
                            source_peer_id,
                            source_message_id,
                            f"flood_wait_exceeded:{wait_seconds}",
                            final=True,
                        )
                        errors += 1
                        break
                    await asyncio.sleep(wait_seconds)
                except Exception as error:
                    final = attempt >= int(args.max_attempts)
                    mark_error(conn, source_peer_id, source_message_id, repr(error), final=final)
                    db_event(
                        conn,
                        "error",
                        "forward_error",
                        repr(error),
                        source_peer_id,
                        source_message_id,
                    )
                    logging.exception("Forward failed source_message_id=%s attempt=%s", source_message_id, attempt)
                    if final:
                        errors += 1
                        break
                    await asyncio.sleep(float(args.retry_sleep))

            if int(args.forward_limit or 0) > 0 and forwarded >= int(args.forward_limit):
                stopped_reason = "forward_limit"
                logging.info("Forward limit reached: %s", args.forward_limit)
                break

            if float(args.sleep_between) > 0:
                await asyncio.sleep(float(args.sleep_between))

            if scanned % 200 == 0:
                logging.info("Progress %s", format_summary(summarize(conn, source_peer_id)))

        summary = summarize(conn, source_peer_id)
        if args.dry_run:
            status = "dry_run"
        elif stopped_reason:
            status = f"partial_{stopped_reason}"
        elif int(summary.get("error", 0)) > 0:
            status = "completed_with_errors"
        elif int(summary.get("pending", 0)) > 0 or int(summary.get("forwarding", 0)) > 0:
            status = "incomplete_pending"
        else:
            status = "completed"
        update_job(conn, args.source, args.target, status=status)
        db_event(conn, "info", status, format_summary(summary), source_peer_id, None)
        elapsed = time.monotonic() - started_at
        logging.info(
            "Finished status=%s scanned=%s forwarded_now=%s skipped_done=%s skipped_duplicate=%s skipped_gif=%s skipped_sticker=%s errors_now=%s elapsed=%.1fs %s",
            status,
            scanned,
            forwarded,
            skipped_done,
            skipped_duplicate,
            skipped_gif,
            skipped_sticker,
            errors,
            elapsed,
            format_summary(summary),
        )
        return 0 if status in {"completed", "dry_run"} or status.startswith("partial_") else 2
    finally:
        await client.disconnect()
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward all photos/videos from one Telegram dialog to another with sender hidden."
    )
    parser.add_argument("--source", required=True, help="Source dialog title, username, invite alias, or peer id.")
    parser.add_argument("--target", required=True, help="Target dialog title, username, invite alias, or peer id.")
    parser.add_argument("--source-session", default=DEFAULT_SOURCE_SESSION)
    parser.add_argument("--work-session", default=DEFAULT_WORK_SESSION)
    parser.add_argument("--refresh-session-copy", action="store_true")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--drop-media-captions", action="store_true")
    parser.add_argument("--min-id", type=int, default=0)
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--forward-limit", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--sleep-between", type=float, default=0.7)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--max-flood-wait", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    return parser


async def async_main() -> int:
    args = build_parser().parse_args()
    return await run(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
