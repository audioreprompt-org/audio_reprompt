import sys
from pathlib import Path

# Make the repo root importable so `import models.ir_eval...` works under pytest.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
