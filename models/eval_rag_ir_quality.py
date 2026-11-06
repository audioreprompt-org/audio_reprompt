"""IR Quality Evaluation for RAG Retrieval Functions.

Legacy entry point, forwards to models.ir_eval.cli.main.

Usage:
    python -m models.eval_rag_ir_quality [--output-dir data/ablations/ir_eval]
"""

from __future__ import annotations

from models.ir_eval.cli import main

if __name__ == "__main__":
    main()
