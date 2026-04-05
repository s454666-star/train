import argparse
import asyncio
import json
import os
import time

from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
    SessionPasswordNeededError,
)

API_ID = 27946982
API_HASH = "2324efd7bed05b02a63e3809fa93048c"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize_phone(raw_value: str) -> str:
    raw_value = str(raw_value or "").strip()
    digits = "".join(ch for ch in raw_value if ch.isdigit())
    if not digits:
        raise ValueError("Phone number is empty.")
    return f"+{digits}"


def resolve_session_base(session_name: str) -> str:
    session_name = str(session_name or "").strip()
    if not session_name:
        raise ValueError("Session name is required.")

    if os.path.isabs(session_name):
        session_base = session_name
    else:
        session_base = os.path.join(BASE_DIR, session_name)

    session_base = os.path.splitext(session_base)[0]
    session_dir = os.path.dirname(session_base)
    if session_dir:
        os.makedirs(session_dir, exist_ok=True)
    return session_base


def pending_login_path(session_base: str) -> str:
    return f"{session_base}.login.json"


def save_pending_login(session_base: str, payload: dict) -> str:
    file_path = pending_login_path(session_base)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return file_path


def load_pending_login(session_base: str) -> dict:
    file_path = pending_login_path(session_base)
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def clear_pending_login(session_base: str) -> None:
    file_path = pending_login_path(session_base)
    if os.path.exists(file_path):
        os.remove(file_path)


async def get_authorized_summary(client: TelegramClient) -> dict:
    me = await client.get_me()
    return {
        "status": "authorized",
        "user_id": getattr(me, "id", None),
        "phone": getattr(me, "phone", None),
        "username": getattr(me, "username", None),
        "display_name": " ".join(
            part for part in [getattr(me, "first_name", None), getattr(me, "last_name", None)] if part
        ).strip(),
    }


async def request_code(session_base: str, phone: str) -> int:
    client = TelegramClient(session_base, API_ID, API_HASH)
    await client.connect()
    try:
        if await client.is_user_authorized():
            print(json.dumps(await get_authorized_summary(client), ensure_ascii=False))
            return 0

        sent = await client.send_code_request(phone)
        payload = {
            "phone": phone,
            "phone_code_hash": sent.phone_code_hash,
            "requested_at": int(time.time()),
        }
        login_path = save_pending_login(session_base, payload)
        print(
            json.dumps(
                {
                    "status": "code_sent",
                    "phone": phone,
                    "session": session_base,
                    "pending_login_file": login_path,
                    "type": sent.type.__class__.__name__,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except PhoneNumberInvalidError:
        print(json.dumps({"status": "error", "reason": "phone_number_invalid", "phone": phone}, ensure_ascii=False))
        return 1
    except FloodWaitError as error:
        print(
            json.dumps(
                {"status": "error", "reason": "flood_wait", "wait_seconds": int(error.seconds or 0), "phone": phone},
                ensure_ascii=False,
            )
        )
        return 1
    finally:
        await client.disconnect()


async def complete_login(session_base: str, code: str, password: str) -> int:
    try:
        pending = load_pending_login(session_base)
    except FileNotFoundError:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason": "pending_login_not_found",
                    "session": session_base,
                },
                ensure_ascii=False,
            )
        )
        return 1

    phone = normalize_phone(pending.get("phone"))
    phone_code_hash = str(pending.get("phone_code_hash") or "").strip()
    if not phone_code_hash:
        print(json.dumps({"status": "error", "reason": "missing_phone_code_hash", "phone": phone}, ensure_ascii=False))
        return 1

    client = TelegramClient(session_base, API_ID, API_HASH)
    await client.connect()
    try:
        if await client.is_user_authorized():
            clear_pending_login(session_base)
            print(json.dumps(await get_authorized_summary(client), ensure_ascii=False))
            return 0

        try:
            await client.sign_in(phone=phone, code=str(code or "").strip(), phone_code_hash=phone_code_hash)
        except SessionPasswordNeededError:
            if not password:
                print(json.dumps({"status": "password_required", "phone": phone}, ensure_ascii=False))
                return 2
            await client.sign_in(password=password)

        clear_pending_login(session_base)
        print(json.dumps(await get_authorized_summary(client), ensure_ascii=False))
        return 0
    except PhoneCodeInvalidError:
        print(json.dumps({"status": "error", "reason": "phone_code_invalid", "phone": phone}, ensure_ascii=False))
        return 1
    except PhoneCodeExpiredError:
        print(json.dumps({"status": "error", "reason": "phone_code_expired", "phone": phone}, ensure_ascii=False))
        return 1
    except PasswordHashInvalidError:
        print(json.dumps({"status": "error", "reason": "password_invalid", "phone": phone}, ensure_ascii=False))
        return 1
    except FloodWaitError as error:
        print(
            json.dumps(
                {"status": "error", "reason": "flood_wait", "wait_seconds": int(error.seconds or 0), "phone": phone},
                ensure_ascii=False,
            )
        )
        return 1
    finally:
        await client.disconnect()


async def show_status(session_base: str) -> int:
    client = TelegramClient(session_base, API_ID, API_HASH)
    await client.connect()
    try:
        if await client.is_user_authorized():
            print(json.dumps(await get_authorized_summary(client), ensure_ascii=False))
            return 0

        payload = {"status": "not_authorized", "session": session_base}
        pending_file = pending_login_path(session_base)
        if os.path.exists(pending_file):
            payload["pending_login_file"] = pending_file
        print(json.dumps(payload, ensure_ascii=False))
        return 1
    finally:
        await client.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Activate a Telegram Telethon session in two steps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    request_parser = subparsers.add_parser("request-code")
    request_parser.add_argument("--session", required=True)
    request_parser.add_argument("--phone", required=True)

    complete_parser = subparsers.add_parser("complete-login")
    complete_parser.add_argument("--session", required=True)
    complete_parser.add_argument("--code", required=True)
    complete_parser.add_argument("--password", default="")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--session", required=True)

    return parser


async def async_main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    session_base = resolve_session_base(args.session)

    if args.command == "request-code":
        return await request_code(session_base, normalize_phone(args.phone))
    if args.command == "complete-login":
        return await complete_login(session_base, args.code, args.password)
    if args.command == "status":
        return await show_status(session_base)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
