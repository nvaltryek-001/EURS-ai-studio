"""EURS training / evaluation toolkit."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))  # shared inference core lives in backend/eurs_core
