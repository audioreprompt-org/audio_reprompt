from typing import List, Dict, Optional
from pathlib import Path


class _UnavailableFAD(Exception):
    """Raised when the FAD metric is requested but not configured."""
    pass


def compute_fad_to_references(
        gen_files: List[Path],
        flavor: Optional[str],
        backend: Optional[str] = None,
        device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Not implemented on purpose in this phase. We expose the function with the correct
    signature so call sites fail fast with a helpful message if invoked.
    """
    raise _UnavailableFAD(
        "FAD metric is not implemented in this phase (CLAP-only). "
        "Enable later by adding references, embeddings backend, and implementation."
    )
