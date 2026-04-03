import argparse
import json
import os
import secrets
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PREFIX = "[TG-IDM][helper]"
TOKEN_TTL_SECONDS = 8 * 60 * 60
CHUNK_SIZE = 64 * 1024
MAX_EVENTS = 200
EVENTS = []
EVENTS_LOCK = threading.Lock()
TOKENS = {}
TOKENS_LOCK = threading.Lock()
CONFIG = {}


def cache_root():
    path = os.path.join(BASE_DIR, "cache")
    os.makedirs(path, exist_ok=True)
    return path


def log_file_path():
    return os.path.join(BASE_DIR, "helper.log")


def log(message, **fields):
    payload = {"message": message, **fields}
    line = f"{LOG_PREFIX} {json.dumps(payload, ensure_ascii=False)}"

    try:
        with open(log_file_path(), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass

    with EVENTS_LOCK:
        EVENTS.append({"time": int(time.time()), **payload})
        if len(EVENTS) > MAX_EVENTS:
            del EVENTS[: len(EVENTS) - MAX_EVENTS]

    if fields:
        print(LOG_PREFIX, message, json.dumps(fields, ensure_ascii=False), flush=True)
    else:
        print(LOG_PREFIX, message, flush=True)


def sanitize_filename(value):
    fallback = "telegram-video.mp4"
    if not isinstance(value, str):
        return fallback

    cleaned = "".join(" " if ord(ch) < 32 or ch in '<>:"/\\|?*' else ch for ch in value)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return fallback
    return cleaned[:120]


def ascii_filename(value):
    ascii_only = value.encode("ascii", "ignore").decode("ascii").strip()
    ascii_only = ascii_only.replace('"', "")
    return ascii_only or "telegram-video.mp4"


def find_idm_path(cli_value):
    if cli_value:
        return cli_value

    candidates = [
        r"C:\Program Files (x86)\Internet Download Manager1\IDMan.exe",
        r"C:\Program Files\Internet Download Manager\IDMan.exe",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("找不到 IDMan.exe，請確認 IDM 已安裝。")


def build_attachment_header(filename):
    ascii_name = ascii_filename(filename)
    utf8_name = urllib.parse.quote(filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"


def build_file_url(token, filename):
    quoted_name = urllib.parse.quote(filename)
    return f"http://{CONFIG['host']}:{CONFIG['port']}/file/{token}/{quoted_name}"


def launch_idm(file_url, filename):
    command = [CONFIG["idm_path"], "/n", "/d", file_url, "/f", filename]
    log("launching idm", file_url=file_url, filename=filename, idm_path=CONFIG["idm_path"])
    subprocess.Popen(command)


def cleanup_tokens():
    now = time.time()
    expired = []

    with TOKENS_LOCK:
        for token, entry in list(TOKENS.items()):
            if entry["expires_at"] <= now:
                expired.append((token, entry.get("temp_path")))
                TOKENS.pop(token, None)

    for token, temp_path in expired:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if expired:
        log("expired tokens cleaned", count=len(expired))


def parse_range_header(range_header, file_size):
    if not range_header or not range_header.startswith("bytes="):
        return None

    try:
        value = range_header.split("=", 1)[1]
        start_text, end_text = value.split("-", 1)
        if start_text == "":
            suffix = int(end_text)
            start = max(0, file_size - suffix)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1

        if start < 0 or end < start or start >= file_size:
            return None

        return start, min(end, file_size - 1)
    except Exception:
        return None


class HelperHandler(BaseHTTPRequestHandler):
    server_version = "TelegramIDMHelper/0.3"

    def log_message(self, format_string, *args):
        log("http", client=self.address_string(), path=self.path, detail=format_string % args)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_common_headers()
        self.send_header("Access-Control-Allow-Methods", "GET,HEAD,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_GET(self):
        self.route_request(head_only=False)

    def do_HEAD(self):
        self.route_request(head_only=True)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/upload-init":
            self.handle_upload_init()
            return

        if parsed.path.startswith("/api/upload/"):
            self.handle_upload_data(parsed.path.rsplit("/", 1)[-1])
            return

        self.respond_json(404, {"ok": False, "error": "Not found"})

    def route_request(self, head_only):
        parsed = urllib.parse.urlparse(self.path)
        cleanup_tokens()

        if parsed.path == "/health":
            self.respond_json(
                200,
                {
                    "ok": True,
                    "host": CONFIG["host"],
                    "port": CONFIG["port"],
                    "idmPath": CONFIG["idm_path"],
                    "tokenCount": len(TOKENS),
                    "cacheRoot": cache_root(),
                },
            )
            return

        if parsed.path == "/debug/state":
            with TOKENS_LOCK:
                tokens = list(TOKENS.values())
            with EVENTS_LOCK:
                events = list(EVENTS)
            self.respond_json(
                200,
                {
                    "ok": True,
                    "tokens": tokens,
                    "events": events,
                    "logFile": log_file_path(),
                },
            )
            return

        if parsed.path.startswith("/file/"):
            self.handle_file_download(parsed.path, head_only=head_only)
            return

        self.respond_json(404, {"ok": False, "error": "Not found"})

    def handle_upload_init(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as error:
            self.respond_json(400, {"ok": False, "error": f"無法解析 JSON：{error}"})
            return

        filename = sanitize_filename(payload.get("filename") or "telegram-video.mp4")
        token = secrets.token_urlsafe(18)
        temp_path = os.path.join(cache_root(), f"{token}.bin")
        entry = {
            "token": token,
            "filename": filename,
            "temp_path": temp_path,
            "created_at": time.time(),
            "expires_at": time.time() + TOKEN_TTL_SECONDS,
            "status": "initialized",
            "page_url": str(payload.get("pageUrl", "")).strip(),
            "download_url": str(payload.get("downloadUrl", "")).strip(),
            "content_type": "",
            "size": 0,
            "idm_launched": False,
            "file_url": build_file_url(token, filename),
        }

        with TOKENS_LOCK:
            TOKENS[token] = entry

        log("upload initialized", token=token, filename=filename, download_url=entry["download_url"])
        self.respond_json(
            200,
            {
                "ok": True,
                "token": token,
                "uploadUrl": f"http://{CONFIG['host']}:{CONFIG['port']}/api/upload/{token}",
                "fileUrl": entry["file_url"],
                "filename": filename,
            },
        )

    def handle_upload_data(self, token):
        with TOKENS_LOCK:
            entry = TOKENS.get(token)

        if not entry:
            self.respond_json(404, {"ok": False, "error": "upload token 不存在或已過期"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        written = 0
        with open(entry["temp_path"], "wb") as handle:
            while written < length:
                chunk = self.rfile.read(min(CHUNK_SIZE, length - written))
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)

        content_type = self.headers.get("Content-Type", "application/octet-stream")
        entry["content_type"] = content_type
        entry["size"] = written
        entry["status"] = "uploaded"

        try:
            launch_idm(entry["file_url"], entry["filename"])
            entry["idm_launched"] = True
        except Exception as error:
            self.respond_json(500, {"ok": False, "error": f"啟動 IDM 失敗：{error}"})
            return

        log(
            "upload stored",
            token=token,
            filename=entry["filename"],
            size=written,
            content_type=content_type,
            file_url=entry["file_url"],
        )
        self.respond_json(
            200,
            {
                "ok": True,
                "token": token,
                "fileUrl": entry["file_url"],
                "filename": entry["filename"],
                "size": written,
                "idmLaunched": True,
            },
        )

    def handle_file_download(self, path, head_only):
        parts = path.split("/")
        if len(parts) < 4:
            self.respond_json(404, {"ok": False, "error": "Invalid file path"})
            return

        token = parts[2]
        with TOKENS_LOCK:
            entry = TOKENS.get(token)

        if not entry or entry.get("status") != "uploaded":
            self.respond_json(404, {"ok": False, "error": "檔案尚未準備完成"})
            return

        temp_path = entry["temp_path"]
        if not os.path.exists(temp_path):
            self.respond_json(404, {"ok": False, "error": "暫存檔不存在"})
            return

        file_size = os.path.getsize(temp_path)
        range_value = parse_range_header(self.headers.get("Range"), file_size)
        status = 200
        start = 0
        end = file_size - 1

        if range_value:
            start, end = range_value
            status = 206

        content_length = max(0, end - start + 1)

        log(
            "file request",
            token=token,
            filename=entry["filename"],
            method=self.command,
            status=status,
            range=self.headers.get("Range", ""),
            size=file_size,
        )

        self.send_response(status)
        self.send_common_headers()
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", entry.get("content_type") or "application/octet-stream")
        self.send_header("Content-Disposition", build_attachment_header(entry["filename"]))
        self.send_header("Content-Length", str(content_length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        if head_only:
            return

        with open(temp_path, "rb") as handle:
            handle.seek(start)
            remaining = content_length
            try:
                while remaining > 0:
                    chunk = handle.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                log("client disconnected during file download", token=token, filename=entry["filename"])

    def send_common_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Cache-Control", "no-store")

    def respond_json(self, status, payload):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)


def parse_args():
    parser = argparse.ArgumentParser(description="Telegram IDM localhost helper")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--idm-path", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    CONFIG["host"] = args.host
    CONFIG["port"] = args.port
    CONFIG["idm_path"] = find_idm_path(args.idm_path)
    cache_root()

    server = ThreadingHTTPServer((CONFIG["host"], CONFIG["port"]), HelperHandler)
    log("helper started", host=CONFIG["host"], port=CONFIG["port"], idm_path=CONFIG["idm_path"])
    log("health endpoint", url=f"http://{CONFIG['host']}:{CONFIG['port']}/health")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("helper stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("fatal error", error=str(exc))
        raise
