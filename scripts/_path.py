# Lets the scripts run from a checkout without `pip install -e .`
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
