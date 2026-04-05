import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PYTHON_DIR = r"C:\www\blog\python"

if REPO_PYTHON_DIR not in sys.path:
    sys.path.insert(0, REPO_PYTHON_DIR)

os.environ.setdefault("TELEGRAM_SERVICE_HOME", BASE_DIR)
os.environ.setdefault("TELEGRAM_SERVICE_SESSION", "session/main_account4")

from telegram_service_shared import app
