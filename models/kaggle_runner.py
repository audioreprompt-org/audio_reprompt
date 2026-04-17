from __future__ import annotations

import re

"""
Automate audio generation on Kaggle for ablation experiments.

Workflow:
    1. Upload reprompt CSVs as a Kaggle dataset
    2. Push a generated notebook that runs TTS on GPU
    3. Poll until the kernel finishes
    4. Download generated .wav files

Usage:
    python -m models.kaggle_runner upload --csvs data/ablations/reprompts/*.csv
    python -m models.kaggle_runner status
    python -m models.kaggle_runner download --output-dir data/ablations/audio
    python -m models.kaggle_runner run --csvs data/ablations/reprompts/*.csv  # upload + push + poll + download
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_USER = "mfreyeso"
DATASET_SLUG = f"{KAGGLE_USER}/ablation-reprompts"
KERNEL_SLUG = f"{KAGGLE_USER}/ablation-audio-gen"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = PROJECT_ROOT / ".kaggle_staging"
DATASET_DIR = STAGING_DIR / "dataset"
KERNEL_DIR = STAGING_DIR / "kernel"
NB_FILENAME = "ablation-audio-gen.ipynb"

ABLATIONS_AUDIO_PATH = PROJECT_ROOT / "data" / "ablations" / "audio"


def _run_kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a kaggle CLI command."""
    cmd = ["kaggle", *args]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if result.stderr:
            print(f"  ✗ stderr: {result.stderr.strip()}", file=sys.stderr)
        if result.stdout:
            print(f"  ✗ stdout: {result.stdout.strip()}", file=sys.stderr)
        if check:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
    return result


def _prepare_staging(csv_paths: list[str]) -> None:
    """Prepare the staging directory with CSVs and metadata."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    DATASET_DIR.mkdir(parents=True)
    KERNEL_DIR.mkdir(parents=True)

    for csv_path in csv_paths:
        src = Path(csv_path)
        if not src.exists():
            print(f"  ⚠ CSV not found: {csv_path}", file=sys.stderr)
            continue
        shutil.copy2(src, DATASET_DIR / src.name)

    csv_count = len(list(DATASET_DIR.glob("*.csv")))
    print(f"  Staged {csv_count} CSV(s) in {DATASET_DIR}")


def _write_dataset_metadata() -> None:
    """Write dataset-metadata.json for Kaggle dataset API."""
    metadata = {
        "title": "ablation-reprompts",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = DATASET_DIR / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Dataset metadata written to {meta_path}")


def _upload_dataset() -> None:
    """Create or update the Kaggle dataset with reprompt CSVs.

    Always creates a new version when the dataset already exists to ensure
    stale files are replaced.  Falls back to ``datasets create`` only when the
    dataset does not yet exist on Kaggle.
    """
    _write_dataset_metadata()

    local_csvs = sorted(p.name for p in DATASET_DIR.glob("*.csv"))

    # Try creating a new version first (common case — dataset already exists)
    result = _run_kaggle(
        "datasets",
        "version",
        "-p",
        str(DATASET_DIR),
        "-m",
        "Updated ablation reprompt CSVs",
        "--delete-old-versions",
        "-q",
        check=False,
    )

    if result.returncode != 0:
        combined = (result.stdout + result.stderr).lower()
        if "not found" in combined or "does not exist" in combined or "404" in combined:
            print("  Dataset does not exist, creating...")
            _run_kaggle("datasets", "create", "-p", str(DATASET_DIR), "-q")
        else:
            print(f"  Error updating dataset: {result.stderr}", file=sys.stderr)
            sys.exit(1)

    print(f"  ✓ Dataset uploaded: {DATASET_SLUG}")
    print(f"  ℹ Local CSVs: {local_csvs}")

    # Verify remote files match what we uploaded
    verify = _run_kaggle("datasets", "files", DATASET_SLUG, check=False)
    if verify.returncode == 0:
        print(f"  ℹ Remote dataset files:\n{verify.stdout.strip()}")
        # Quick sanity check: make sure all local CSV names appear in remote listing
        missing = [f for f in local_csvs if f not in verify.stdout]
        if missing:
            print(
                f"  ⚠ WARNING: these CSVs are missing from remote dataset: {missing}",
                file=sys.stderr,
            )

    # Wait for new dataset version to propagate to kernels
    print("  ⏳ Waiting 60s for dataset propagation...")
    time.sleep(60)


def _generate_notebook() -> Path:
    """Generate a notebook .ipynb that reads CSVs and produces audio."""
    cells = [
        # Cell 1: GPU diagnostics — detect architecture early
        _code_cell(
            "import torch\n"
            "TORCH_VER = torch.__version__\n"
            "gpu_name = torch.cuda.get_device_name(0)\n"
            "cap = torch.cuda.get_device_capability(0)\n"
            "print(f'torch={TORCH_VER}  cuda={torch.version.cuda}')\n"
            "print(f'GPU: {gpu_name}  compute_capability={cap[0]}.{cap[1]}')\n"
            "# P100 (6.0) is known to fail with MusicGen; T4 (7.5) works\n"
            "if cap < (7, 0):\n"
            "    raise RuntimeError(\n"
            "        f'GPU {gpu_name} (compute {cap[0]}.{cap[1]}) is too old. '"
            "        f'Select GPU T4 x2 in Kaggle Settings.'\n"
            "    )"
        ),
        # Cell 2: install deps, pinning torch so pip never replaces it
        _code_cell(
            "!pip install -q transformers soundfile accelerate "
            "'huggingface-hub>=1.5.0' torch=={TORCH_VER}"
        ),
        # Cell 3: imports
        _code_cell(
            "import csv\n"
            "import os\n"
            "from pathlib import Path\n"
            "from tqdm import tqdm\n"
            "from transformers import pipeline\n"
            "import soundfile as sf\n"
            "import shutil"
        ),
        # Cell 4: load model
        _code_cell(
            'synthesiser = pipeline("text-to-audio", "csc-unipd/tasty-musicgen-small")'
        ),
        # Cell 5: debug — list dataset input directory so we can see actual structure
        _code_cell(
            "import os\n"
            "print('Dataset input directory tree:')\n"
            "for root, dirs, files in os.walk('/kaggle/input'):\n"
            "    level = root.replace('/kaggle/input', '').count(os.sep)\n"
            "    indent = '  ' * level\n"
            "    print(f'{indent}{os.path.basename(root)}/')\n"
            "    sub_indent = '  ' * (level + 1)\n"
            "    for f in files:\n"
            "        print(f'{sub_indent}{f}')"
        ),
        # Cell 6: find CSVs (search recursively) and generate audio
        _code_cell(
            'INPUT_DIR = Path("/kaggle/input/datasets/mfreyeso/ablation-reprompts")\n'
            'OUTPUT_DIR = Path("/kaggle/working")\n'
            "\n"
            "# Search recursively — Kaggle may nest files in subdirectories\n"
            'csv_files = sorted(INPUT_DIR.rglob("*.csv"))\n'
            'print(f"Found {len(csv_files)} CSV files:")\n'
            "for f in csv_files:\n"
            '    print(f"  {f}")\n'
            "\n"
            "for csv_file in csv_files:\n"
            "    # use CSV stem as subdirectory name\n"
            "    sub_dir = OUTPUT_DIR / csv_file.stem\n"
            "    sub_dir.mkdir(parents=True, exist_ok=True)\n"
            '    print(f"\\nProcessing: {csv_file.name} -> {sub_dir.name}/")\n'
            "\n"
            '    with open(csv_file, "r") as f:\n'
            "        reader = csv.DictReader(f)\n"
            "        rows = list(reader)\n"
            "\n"
            "    for row in tqdm(rows, desc=csv_file.stem):\n"
            '        id_prompt = row["id_prompt"]\n'
            '        reprompt = row["reprompt"]\n'
            '        output_path = sub_dir / f"{id_prompt}.wav"\n'
            "\n"
            "        if output_path.exists():\n"
            '            print(f"  Skipping {id_prompt} (exists)")\n'
            "            continue\n"
            "\n"
            '        music = synthesiser(reprompt, forward_params={"do_sample": True})\n'
            '        sf.write(str(output_path), music["audio"].squeeze(), music["sampling_rate"])\n'
            "\n"
            "    # zip each subdirectory for easier download\n"
            "    archive = OUTPUT_DIR / csv_file.stem\n"
            '    shutil.make_archive(str(archive), "zip", str(sub_dir))\n'
            '    print(f"  ✓ Archived: {csv_file.stem}.zip")\n'
        ),
        _code_cell(
            'wav_count = len(list(OUTPUT_DIR.rglob("*.wav")))\n'
            'zip_count = len(list(OUTPUT_DIR.glob("*.zip")))\n'
            'print(f"\\nDone! {wav_count} WAV files in {zip_count} archives")'
        ),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

    nb_path = KERNEL_DIR / NB_FILENAME
    nb_path.write_text(json.dumps(notebook, indent=2))
    print(f"  Notebook generated: {nb_path}")
    return nb_path


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "source": source,
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
    }


def _write_kernel_metadata() -> None:
    """Write kernel-metadata.json for Kaggle kernel API."""
    metadata = {
        "id": KERNEL_SLUG,
        "id_no": 115695468,
        "title": "Ablation Audio Gen",
        "code_file": NB_FILENAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [DATASET_SLUG],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    meta_path = KERNEL_DIR / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Kernel metadata written to {meta_path}")


def _push_kernel() -> None:
    """Push the notebook to Kaggle for execution."""
    _generate_notebook()
    _write_kernel_metadata()
    _run_kaggle("kernels", "push", "-p", str(KERNEL_DIR))
    print(f"  ✓ Kernel pushed: {KERNEL_SLUG}")


def _check_status() -> str:
    """Check the kernel execution status."""
    result = _run_kaggle("kernels", "status", KERNEL_SLUG, check=False)
    output = result.stdout.strip()
    print(f"  Status: {output}")
    return output


def _poll_until_complete(interval: int = 60, timeout: int = 7200) -> bool:
    """Poll kernel status until complete or timeout."""
    elapsed = 0
    print(f"  Polling every {interval}s (timeout: {timeout}s)...")

    while elapsed < timeout:
        status = _check_status()
        status_lower = status.lower()

        if "complete" in status_lower:
            print("  ✓ Kernel execution completed!")
            return True
        if "error" in status_lower or "cancel" in status_lower:
            print(f"  ✗ Kernel failed: {status}", file=sys.stderr)
            return False

        time.sleep(interval)
        elapsed += interval
        print(f"  [{elapsed}s elapsed]")

    print(f"  ✗ Timeout after {timeout}s", file=sys.stderr)
    return False


def _download_output(output_dir: str | None = None, run_id: str = "") -> None:
    """Download kernel output (zip files with .wav audio).

    If *run_id* is provided, the extracted directories are suffixed with it
    so that different ablation runs don't overwrite each other.
    """
    dest = Path(output_dir) if output_dir else ABLATIONS_AUDIO_PATH
    dest.mkdir(parents=True, exist_ok=True)

    download_tmp = STAGING_DIR / "output"
    download_tmp.mkdir(parents=True, exist_ok=True)

    _run_kaggle("kernels", "output", KERNEL_SLUG, "-p", str(download_tmp))

    # Extract zip files into audio subdirectories
    import zipfile

    _RUN_ID_RE = re.compile(r"_R\d{8}_\d{6}$")

    zip_files = list(download_tmp.glob("*.zip"))
    print(f"  Found {len(zip_files)} zip archive(s)")

    for zf in zip_files:
        dir_name = zf.stem
        # Append run_id if provided and not already present
        if run_id and not _RUN_ID_RE.search(dir_name):
            dir_name = f"{dir_name}_{run_id}"
        extract_dir = dest / dir_name
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(extract_dir)
        wav_count = len(list(extract_dir.glob("*.wav")))
        print(f"  ✓ Extracted {wav_count} WAV(s) to {extract_dir}")

    print(f"  Audio files downloaded to: {dest}")


# ── Public API (for use by ablation_runner) ──────────────────


def run_pipeline(
    csv_paths: list[str],
    output_dir: str | None = None,
    poll_interval: int = 60,
    timeout: int = 7200,
    run_id: str = "",
) -> None:
    """Run the full Kaggle pipeline: upload → push → poll → download.

    Raises RuntimeError if the Kaggle kernel fails or times out.
    """
    print("\n═══ Kaggle: Upload dataset + push kernel ═══")
    _prepare_staging(csv_paths)
    _upload_dataset()
    _push_kernel()

    print("\n═══ Kaggle: Wait for GPU execution ═══")
    success = _poll_until_complete(interval=poll_interval, timeout=timeout)
    if not success:
        raise RuntimeError("Kaggle kernel execution failed or timed out")

    print("\n═══ Kaggle: Download audio files ═══")
    _download_output(output_dir, run_id=run_id)


# ── CLI ──────────────────────────────────────────────────────


def cmd_upload(args):
    csv_paths = args.csvs
    if not csv_paths:
        print("Error: --csvs is required", file=sys.stderr)
        sys.exit(1)
    _prepare_staging(csv_paths)
    _upload_dataset()
    _push_kernel()


def cmd_status(_args):
    _check_status()


def cmd_download(args):
    _download_output(args.output_dir, run_id=getattr(args, 'run_id', ''))


def cmd_run(args):
    """Full pipeline: upload → push → poll → download."""
    csv_paths = args.csvs
    if not csv_paths:
        print("Error: --csvs is required", file=sys.stderr)
        sys.exit(1)

    print("\n═══ Phase 1: Upload dataset + push kernel ═══")
    _prepare_staging(csv_paths)
    _upload_dataset()
    _push_kernel()

    print("\n═══ Phase 2: Wait for GPU execution ═══")
    success = _poll_until_complete(interval=args.poll_interval, timeout=args.timeout)
    if not success:
        print("Aborting download due to execution failure.", file=sys.stderr)
        sys.exit(1)

    print("\n═══ Phase 3: Download audio files ═══")
    _download_output(args.output_dir, run_id=getattr(args, 'run_id', ''))


def main():
    parser = argparse.ArgumentParser(
        description="Automate Kaggle GPU audio generation for ablation experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # upload
    p_upload = sub.add_parser("upload", help="Upload CSVs + push notebook to Kaggle")
    p_upload.add_argument("--csvs", nargs="+", required=True, help="Reprompt CSV paths")
    p_upload.set_defaults(func=cmd_upload)

    # status
    p_status = sub.add_parser("status", help="Check kernel execution status")
    p_status.set_defaults(func=cmd_status)

    # download
    p_download = sub.add_parser("download", help="Download generated audio files")
    p_download.add_argument(
        "--output-dir",
        default=str(ABLATIONS_AUDIO_PATH),
        help=f"Output directory (default: {ABLATIONS_AUDIO_PATH})",
    )
    p_download.add_argument(
        "--run-id", default="", help="Run ID to append to audio directory names",
    )
    p_download.set_defaults(func=cmd_download)

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Full pipeline: upload → push → poll → download")
    p_run.add_argument("--csvs", nargs="+", required=True, help="Reprompt CSV paths")
    p_run.add_argument(
        "--output-dir",
        default=str(ABLATIONS_AUDIO_PATH),
        help=f"Output directory (default: {ABLATIONS_AUDIO_PATH})",
    )
    p_run.add_argument(
        "--run-id", default="", help="Run ID to append to audio directory names",
    )
    p_run.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status checks (default: 60)",
    )
    p_run.add_argument(
        "--timeout", type=int, default=7200, help="Max seconds to wait (default: 7200)"
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
