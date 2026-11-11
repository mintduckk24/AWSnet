"""Root CLI entry point for AWSnet evaluation, forwarding to evaluate.py:main()."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import parse_args, evaluate, main

if __name__ == "__main__":
    main()
