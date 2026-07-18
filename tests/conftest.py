import sys
from pathlib import Path

# Ensure project root is on sys.path so imports like `from src...` work when running tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
