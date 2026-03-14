import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv


load_dotenv()


def _get_env(prefix: Optional[str], key: str, default: Optional[str] = None) -> Optional[str]:
    if prefix:
        prefixed_key = f"{prefix}_{key}"
        value = os.getenv(prefixed_key)
        if value not in (None, ""):
            return value
    return os.getenv(key, default)


def load_db_config(prefix: Optional[str] = None, *, charset: Optional[str] = None) -> Dict[str, Any]:
    env_prefix = prefix.upper() if prefix else None
    config: Dict[str, Any] = {
        "host": _get_env(env_prefix, "DB_HOST"),
        "user": _get_env(env_prefix, "DB_USER"),
        "password": _get_env(env_prefix, "DB_PASSWORD"),
        "database": _get_env(env_prefix, "DB_NAME"),
    }

    missing = [key for key, value in config.items() if value in (None, "")]
    if missing:
        missing_vars = [f"{env_prefix}_DB_{key.upper()}" if env_prefix else f"DB_{key.upper()}" for key in missing]
        raise ValueError(f"Missing database environment variables: {', '.join(missing_vars)}")

    port_value = _get_env(env_prefix, "DB_PORT", "3306")
    try:
        config["port"] = int(str(port_value).strip())
    except (TypeError, ValueError) as exc:
        target = f"{env_prefix}_DB_PORT" if env_prefix else "DB_PORT"
        raise ValueError(f"Invalid database port in {target}: {port_value}") from exc

    charset_value = _get_env(env_prefix, "DB_CHARSET", charset) if charset else _get_env(env_prefix, "DB_CHARSET")
    if charset_value:
        config["charset"] = charset_value

    return config
