import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

from config import setup_project_paths, load_config, PROJECT_ROOT
from tqdm import tqdm
import pandas as pd

from models.music_curator.kimi_mcu import (
    KIMI_K2_THINKING_MODEL,
    OPENAI_GPT_5_NANO_MODEL,
)

setup_project_paths()
config = load_config()

FOOD_PROMPTS_DIR = (
    PROJECT_ROOT / config.data.raw_data_path / "user"
)

AUDIOS_PATH = (PROJECT_ROOT / config.data.tracks_data_path).parent

CLAP_RESULTS_PATH = PROJECT_ROOT / config.data.data_clap_path

REPROMPTS_PATH = PROJECT_ROOT / config.data.reprompts_csv_path

ABLATIONS_PATH = PROJECT_ROOT / "data" / "ablations"
ABLATIONS_REPROMPTS_PATH = ABLATIONS_PATH / "reprompts"
ABLATIONS_SCORES_PATH = ABLATIONS_PATH / "scores"

RAW_ABLATIONS_AUDIO_PATH = AUDIOS_PATH / "raw_ablations"
ABLATIONS_RAW_BASELINE_SCORE_CSV = (
    ABLATIONS_SCORES_PATH / "clap_score_results_raw_prompts_baseline.csv"
)

STRATIFY_COLUMN = "taste"

clap_model = None


def _get_clap_model():
    """Lazily initialize the CLAP model on first scoring call."""
    global clap_model
    if clap_model is None:
        from models.clap_score.model import ClapModel

        clap_model = ClapModel(device="auto", enable_fusion=True)
    return clap_model


def _id_sort_key(pid: object) -> tuple[int, int, str]:
    """Mixed int/str safe sort key for prompt IDs."""
    s = str(pid)
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def compute_sample_hash(ids: Iterable[object]) -> str:
    """Return a deterministic SHA-1 hex digest for a set of prompt IDs."""
    sorted_ids = sorted((str(i) for i in ids), key=_id_sort_key)
    return hashlib.sha1(",".join(sorted_ids).encode("utf-8")).hexdigest()


def load_all_raw_prompts() -> list[dict]:
    """Load all raw prompts across data/raw/user/*.csv in deterministic order."""
    all_rows: list[dict] = []
    for csv_file in sorted(FOOD_PROMPTS_DIR.glob("*.csv")):
        with open(csv_file, "r", encoding="utf-8") as file_:
            reader = csv.DictReader(file_)
            all_rows.extend(list(reader))
    return all_rows


def _stratified_sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Stratified random sample ensuring equal count per `taste` category.

    Rows within each category are sorted by `id_prompt` before sampling so the
    output is independent of input file names or row order.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[STRATIFY_COLUMN], []).append(row)

    n_groups = len(groups)
    if n_groups == 0:
        return []
    per_group = sample_size // n_groups

    if any(len(v) < per_group for v in groups.values()):
        smallest = min(len(v) for v in groups.values())
        per_group = min(per_group, smallest)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for category in sorted(groups.keys()):
        sorted_group = sorted(
            groups[category],
            key=lambda r: _id_sort_key(r["id_prompt"]),
        )
        sampled.extend(rng.sample(sorted_group, per_group))

    return sampled


def compute_sample_ids(
    sample_size: int,
    seed: int,
    rows: list[dict] | None = None,
) -> list[str]:
    """Compute deterministic stratified sample IDs sorted by `id_prompt`."""
    all_rows = load_all_raw_prompts() if rows is None else rows
    sampled = _stratified_sample(all_rows, sample_size, seed)
    return sorted((str(r["id_prompt"]) for r in sampled), key=_id_sort_key)


def write_sample_manifest(
    output_dir: Path | str,
    sample_size: int,
    seed: int,
    ids: Iterable[object],
    run_id: str = "",
    group: str = "",
) -> tuple[Path, str]:
    """Write `sample_manifest.json` and return `(manifest_path, sha1)`."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted((str(i) for i in ids), key=_id_sort_key)
    sha1 = compute_sample_hash(sorted_ids)
    manifest = {
        "run_id": run_id,
        "group": group,
        "seed": seed,
        "requested_size": sample_size,
        "size": len(sorted_ids),
        "ids": sorted_ids,
        "sha1": sha1,
        "sample_hash": sha1,
    }
    manifest_path = out_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path, sha1


def load_raw_baseline_cache(
    cache_path: Path | None = None,
) -> dict[str, dict]:
    """Load the baseline score cache CSV without float64 precision loss on mtime."""
    path = cache_path or ABLATIONS_RAW_BASELINE_SCORE_CSV
    cache_by_id: dict[str, dict] = {}
    if not path.exists():
        return cache_by_id
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get("id_prompt", "")).strip()
                if not pid:
                    continue
                raw_score = row.get("clap_score", "")
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if math.isnan(score):
                    continue
                raw_size = str(row.get("wav_size", "")).strip()
                raw_mtime = str(row.get("wav_mtime", "")).strip()
                try:
                    wav_size = int(raw_size) if raw_size else None
                except ValueError:
                    wav_size = None
                try:
                    wav_mtime = int(raw_mtime) if raw_mtime else None
                except ValueError:
                    wav_mtime = None
                cache_by_id[pid] = {
                    "id_prompt": pid,
                    "text": str(row.get("text", "")),
                    "audio": str(row.get("audio", "")),
                    "clap_score": score,
                    "source": str(row.get("source") or "raw_ablations"),
                    "wav_size": wav_size,
                    "wav_mtime": wav_mtime,
                }
    except Exception:
        return {}
    return cache_by_id


def is_valid_baseline_cache_entry(
    cached: dict | None,
    prompt_text: str,
    wav_path: Path,
) -> bool:
    """Validate a cached baseline score entry against prompt text and WAV file metadata."""
    if cached is None:
        return False
    if cached.get("text") != prompt_text:
        return False
    score = cached.get("clap_score")
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return False
    if wav_path.exists():
        c_size = cached.get("wav_size")
        c_mtime = cached.get("wav_mtime")
        if c_size is None or c_mtime is None:
            return False
        st = wav_path.stat()
        if int(c_size) != st.st_size or int(c_mtime) != int(st.st_mtime_ns):
            return False
    return True


def generate_reprompts(
    model: str,
    prompt_version: str,
    sample_size: int = 30,
    seed: int = 42,
    cut_results: bool = True,
    k: int = 10,
    filter_dimensions: tuple[str, ...] | None = None,
    system_prompt_modifier: str | None = None,
    reduce_random_effort: bool = False,
    reasoning_effort: str | None = None,
    tag: str = "",
    output_dir: str | None = None,
):
    from models.pipeline import transform

    all_rows = load_all_raw_prompts()

    sampled_rows = _stratified_sample(all_rows, sample_size, seed)
    sample_hash = compute_sample_hash(r["id_prompt"] for r in sampled_rows)
    print(f"  Muestra estratificada: {len(sampled_rows)} prompts (seed={seed}, sha1={sample_hash})")

    results = []
    for row in tqdm(sampled_rows, desc="prompts progress"):
        results.append(
            {
                "id_prompt": row["id_prompt"],
                "prompt": row["sentence"],
                "taste": row[STRATIFY_COLUMN],
                "reprompt": transform(
                    row["sentence"],
                    model=model,
                    prompt_version=prompt_version,
                    cut_results=cut_results,
                    k=k,
                    filter_dimensions=filter_dimensions,
                    system_prompt_modifier=system_prompt_modifier,
                    reduce_random_effort=reduce_random_effort,
                    reasoning_effort=reasoning_effort,
                ),
            }
        )

    dest = Path(output_dir) if output_dir else REPROMPTS_PATH
    dest.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    output_path = (
        f"{dest}/pipeline_results_"
        f"{model.replace('-', '_')}_{len(sampled_rows)}_prompt_{prompt_version}{suffix}.csv"
    )
    pd.DataFrame(results).to_csv(output_path, index=False)
    return output_path


def calculate_clap_score_alignment(
    result_filepath: str,
    using_raw_prompts: bool = False,
    optional_suffix: str = "",
    cross_evaluation: bool = False,
    output_dir: str | None = None,
    audio_dir: str | None = None,
) -> None:
    from models.clap_score.typedefs import CLAPItem
    if using_raw_prompts:
        print("calculando scores de clap para prompts de entrada (audio baseline raw_ablations)...")
        field = "prompt"
        relative_audio_path = (
            Path(audio_dir) if audio_dir else RAW_ABLATIONS_AUDIO_PATH
        )
    elif cross_evaluation:
        print("calculando scores de clap usando prompt original - reprompt audio...")
        field = "prompt"
        relative_audio_path = (
            Path(audio_dir) if audio_dir else AUDIOS_PATH / "reprompt_audios"
        )
        optional_suffix = optional_suffix or "cross_evaluation"
    else:
        print("calculando scores de clap para reprompts generados...")
        field = "reprompt"
        relative_audio_path = (
            Path(audio_dir) if audio_dir else AUDIOS_PATH / "reprompt_audios"
        )

    with open(result_filepath, "r", encoding="utf-8") as file_:
        reader = csv.DictReader(file_)
        items = [row for row in reader]

    dest = Path(output_dir) if output_dir else CLAP_RESULTS_PATH
    dest.mkdir(parents=True, exist_ok=True)
    out_csv_path = dest / f"clap_score_results_{field}_outputs_{optional_suffix}.csv"

    if using_raw_prompts:
        # Check cache in ABLATIONS_RAW_BASELINE_SCORE_CSV so we compute CLAP(prompt, audio_prompt) only once
        cache_by_id = load_raw_baseline_cache(ABLATIONS_RAW_BASELINE_SCORE_CSV)

        uncached_items: list[CLAPItem] = []
        missing_wavs: list[str] = []
        for row in items:
            pid = str(row["id_prompt"])
            prompt_text = row.get(field) or row.get("sentence") or ""
            wav_path = relative_audio_path / f"{pid}.wav"
            cached = cache_by_id.get(pid)
            if is_valid_baseline_cache_entry(cached, prompt_text, wav_path):
                continue
            if not wav_path.exists():
                missing_wavs.append(pid)
            else:
                uncached_items.append(
                    CLAPItem(
                        id=pid,
                        prompt=prompt_text,
                        audio_path=str(wav_path),
                    )
                )

        if missing_wavs:
            raise FileNotFoundError(
                f"Faltan {len(missing_wavs)} audio(s) baseline en {relative_audio_path} "
                f"para calcular _raw.csv ({', '.join(missing_wavs[:10])}). "
                "Ejecuta con --generate-raw-audio."
            )

        if uncached_items:
            print(
                f"  Calculando CLAP baseline para {len(uncached_items)} prompt(s) nuevos "
                f"(reutilizando {len(items) - len(uncached_items)} de caché)..."
            )
            new_results = _get_clap_model().calculate_scores(uncached_items)
            for res in new_results:
                wav_path = Path(res.item.audio_path)
                st = wav_path.stat() if wav_path.exists() else None
                cache_by_id[str(res.item.id)] = {
                    "id_prompt": str(res.item.id),
                    "text": res.item.prompt,
                    "audio": res.item.audio_path,
                    "clap_score": float(res.clap_score),
                    "source": "raw_ablations",
                    "wav_size": st.st_size if st is not None else None,
                    "wav_mtime": int(st.st_mtime_ns) if st is not None else None,
                }
            ABLATIONS_RAW_BASELINE_SCORE_CSV.parent.mkdir(parents=True, exist_ok=True)
            sorted_cache = sorted(
                cache_by_id.values(),
                key=lambda x: _id_sort_key(x["id_prompt"]),
            )
            pd.DataFrame(sorted_cache).to_csv(ABLATIONS_RAW_BASELINE_SCORE_CSV, index=False)
        else:
            print(
                f"  ✓ Reutilizando {len(items)} scores baseline CLAP(prompt, audio_prompt) "
                f"desde {ABLATIONS_RAW_BASELINE_SCORE_CSV.name}"
            )

        out_rows = []
        missing_scores = []
        for row in items:
            pid = str(row["id_prompt"])
            entry = cache_by_id.get(pid)
            if entry is None or math.isnan(float(entry["clap_score"])):
                missing_scores.append(pid)
                continue
            out_rows.append(
                {
                    "id_prompt": entry["id_prompt"],
                    "text": entry["text"],
                    "audio": entry["audio"],
                    "clap_score": entry["clap_score"],
                    "source": "raw_ablations",
                }
            )
        if missing_scores:
            raise RuntimeError(
                f"Scores baseline faltantes tras calcular CLAP para IDs: {', '.join(missing_scores[:10])}"
            )
        pd.DataFrame(out_rows).to_csv(out_csv_path, index=False)
        return

    clap_items = [
        CLAPItem(
            id=row["id_prompt"],
            prompt=row[field],
            audio_path=str(relative_audio_path / f"{row['id_prompt']}.wav"),
        )
        for row in items
    ]

    results = _get_clap_model().calculate_scores(clap_items)
    pd.DataFrame(
        [
            {
                "id_prompt": res.item.id,
                "text": res.item.prompt,
                "audio": res.item.audio_path,
                "clap_score": res.clap_score,
            }
            for res in results
        ]
    ).to_csv(
        out_csv_path,
        index=False,
    )


def calculate_clap_score_reprompts() -> None:
    calculate_clap_score_alignment(
        f"{REPROMPTS_PATH}/pipeline_results_kimi_k2_thinking_turbo_80_prompt_V4.csv",
    )

    calculate_clap_score_alignment(
        f"{REPROMPTS_PATH}/pipeline_results_kimi_k2_thinking_turbo_80_prompt_V4.csv",
        using_raw_prompts=True,
    )

    calculate_clap_score_alignment(
        f"{REPROMPTS_PATH}/pipeline_results_kimi_k2_thinking_turbo_80_prompt_V4.csv",
        cross_evaluation=True,
    )


if __name__ == "__main__":
    generate_reprompts(OPENAI_GPT_5_NANO_MODEL, "V4")
    # calculate_clap_score_reprompts()
