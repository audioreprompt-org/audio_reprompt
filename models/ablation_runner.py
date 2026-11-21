"""
CLI para ejecutar experimentos de ablación definidos en Ablations.md.

Generation (B-series) — flujo por fases (audio en Kaggle GPU):
    Fase 1: python -m models.ablation_runner --experiment B1a --phase reprompt
    Fase 2: python -m models.kaggle_runner run --csvs data/ablations/reprompts/*.csv
    Fase 3: python -m models.ablation_runner --experiment B1a --phase score
             python -m models.ablation_runner --phase score --reprompt-csv <path>

Evaluación NLP de reprompts (diversidad léxica, entropía, legibilidad):
    python -m models.ablation_runner --phase nlp --reprompt-csv <path>

Pipeline completo (reprompt → Kaggle GPU → score → análisis):
    python -m models.ablation_runner --experiment B2 --phase full
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from config import PROJECT_ROOT, setup_project_paths

setup_project_paths()

from models.music_curator.kimi_mcu import (
    OPENAI_GPT_LUNA_MODEL,
)
from models.pipeline import FILTER_DIMENSIONS_DEFAULT
from models.validate import (
    ABLATIONS_PATH,
    ABLATIONS_RAW_BASELINE_SCORE_CSV,
    ABLATIONS_REPROMPTS_PATH,
    ABLATIONS_SCORES_PATH,
    RAW_ABLATIONS_AUDIO_PATH,
    calculate_clap_score_alignment,
    compute_sample_ids,
    generate_reprompts,
    is_valid_baseline_cache_entry,
    load_all_raw_prompts,
    load_raw_baseline_cache,
    write_sample_manifest,
)

ABLATIONS_AUDIO_PATH = ABLATIONS_PATH / "audio"
ABLATION_STATE_DIR = PROJECT_ROOT / ".ablation_state"

# Exit codes for `--phase full` (consumed by scripts/ablation_job.sh)
EXIT_OK = 0
EXIT_REPROMPT_FAILED = 2
EXIT_KAGGLE_FAILED = 3
EXIT_SCORE_FAILED = 4
EXIT_ANALYSIS_FAILED = 5
EXIT_BASELINE_MISSING = 6


@dataclass
class AblationConfig:
    name: str
    tag: str
    models: list[str] = field(default_factory=lambda: [OPENAI_GPT_LUNA_MODEL])
    prompt_version: str = "V3"
    cut_results: bool = True
    k: int = 10
    filter_dimensions: tuple[str, ...] | None = FILTER_DIMENSIONS_DEFAULT
    system_prompt_modifier: str | None = None
    reduce_random_effort: bool = False
    reasoning_effort: str | None = None
    requires_audio: bool = True
    in_group: bool = True  # False: solo se ejecuta si se pide en --variants


# ── Generation ─────────────────────────────────────────────

GENERATION_ABLATIONS: dict[str, AblationConfig] = {
    # B1 — Versión de prompt
    "B1a": AblationConfig(name="B1a — V1", tag="B1a_V1", prompt_version="V1", requires_audio=True),
    "B1b": AblationConfig(name="B1b — V2", tag="B1b_V2", prompt_version="V2", requires_audio=True),
    "B1c": AblationConfig(name="B1c — V3", tag="B1c_V3", prompt_version="V3", requires_audio=True),
    "B1d": AblationConfig(name="B1d — V4", tag="B1d_V4", prompt_version="V4", requires_audio=True),
    # B2 — Filtro de dimensiones (impacto en generación de audio)
    "B2a": AblationConfig(
        name="B2a — emotion/taste/texture",
        tag="B2a_filter_default",
        filter_dimensions=FILTER_DIMENSIONS_DEFAULT,
        requires_audio=True,
    ),
    "B2b": AblationConfig(
        name="B2b — sin filtro",
        tag="B2b_nofilter",
        filter_dimensions=None,
        requires_audio=True,
    ),
    # B3 — Nivel de Creatividad Inducida por Prompt y Reasoning Effort
    "B3a": AblationConfig(
        name="B3a — prompt determinista conservador", tag="B3a_prompt_determinista", system_prompt_modifier="Be deterministic and conservative", requires_audio=True,
    ),
    "B3b": AblationConfig(
        name="B3b — prompt default", tag="B3b_default_pen0", requires_audio=True,
    ),
    "B3c": AblationConfig(
        name="B3c — prompt creativo e imaginativo", tag="B3c_prompt_creativo", system_prompt_modifier="Be highly creative and imaginative", requires_audio=True,
    ),
    "B3d": AblationConfig(
        name="B3d — prompt creativo + reasoning effort high", tag="B3d_creativo_effort_high", system_prompt_modifier="Be highly creative and imaginative", reasoning_effort="high", requires_audio=True,
    ),
    "B3e": AblationConfig(
        name="B3e — prompt determinista + reasoning effort low", tag="B3e_determinista_effort_low", system_prompt_modifier="Be deterministic and conservative", reasoning_effort="low", requires_audio=True,
    ),
    "B3f": AblationConfig(
            name="B3f — random effort none", tag="B3f_random_effort_none", reduce_random_effort=True, requires_audio=True,
        ),
    # Réplica idéntica de B3b: B3b − B3g estima el ruido de generación dentro de un run (control A/A).
    "B3g": AblationConfig(
        name="B3g — control A/A (réplica de B3b)", tag="B3g_default_replica", requires_audio=True, in_group=False,
    ),
}

ALL_ABLATIONS = GENERATION_ABLATIONS
GROUPS = ("B1", "B2", "B3")

# Subconjuntos nombrados para --variants (B3d y B3e se omiten: efecto nulo en el run R20261117_021911).
VARIANT_PRESETS: dict[str, dict[str, list[str]]] = {
    "B3": {
        "core": ["B3a", "B3b", "B3c", "B3f"],
        "core+aa": ["B3a", "B3b", "B3c", "B3f", "B3g"],
    },
}


def resolve_targets(experiment: str, variants: str | None = None) -> dict[str, AblationConfig]:
    """Resolve --experiment (and optional --variants) into the ablations to run.

    ``variants`` is a comma-separated list of IDs or a preset name, restricted to the
    experiment's group. Raises ValueError before any costly work starts.
    """
    if experiment in ("all", "generation"):
        pool = ALL_ABLATIONS
    elif experiment in GROUPS:
        pool = {k: v for k, v in ALL_ABLATIONS.items() if k.startswith(experiment)}
    elif experiment in ALL_ABLATIONS:
        pool = {experiment: ALL_ABLATIONS[experiment]}
    else:
        raise ValueError(
            f"experimento '{experiment}' no encontrado. Disponibles: {', '.join(ALL_ABLATIONS)}"
        )
    if not variants:
        return {k: v for k, v in pool.items() if v.in_group}

    ids = VARIANT_PRESETS.get(experiment, {}).get(variants)
    if ids is None:
        ids = [v.strip() for v in variants.split(",") if v.strip()]
    ids = list(dict.fromkeys(ids))
    unknown = [i for i in ids if i not in pool]
    if not ids or unknown:
        raise ValueError(
            f"--variants inválido {unknown or '(vacío)'} para '{experiment}'. "
            f"Disponibles: {', '.join(pool)}; presets: {', '.join(VARIANT_PRESETS.get(experiment, {})) or '—'}"
        )
    return {i: pool[i] for i in ids}


def _generate_run_id() -> str:
    """Generate a unique run ID from the current timestamp."""
    return f"R{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_tag(cfg: AblationConfig, model: str, run_id: str = "") -> str:
    base = f"{cfg.tag}_{model.replace('-', '_')}" if len(cfg.models) > 1 else cfg.tag
    return f"{base}_{run_id}" if run_id else base


def _find_reprompt_csvs(tag: str, run_id: str = "") -> list[str]:
    """Find previously generated reprompt CSVs matching a tag and optional run ID."""
    if run_id:
        pattern = f"{ABLATIONS_REPROMPTS_PATH}/pipeline_results_*_{tag}.csv"
    else:
        # match any run ID suffix
        pattern = f"{ABLATIONS_REPROMPTS_PATH}/pipeline_results_*_{tag}*.csv"
    return sorted(glob.glob(pattern))


def prepare_raw_ablations_csv(only_ids: Iterable[str] | None = None) -> str | None:
    """Prepare a staging CSV (raw_ablations.csv) for raw prompts missing
    from data/tracks/raw_ablations/.

    If ``only_ids`` is given, restrict staging to those prompt IDs.
    Returns the staging CSV path if generation is needed, or None if all target
    raw baseline audios already exist.
    """
    import pandas as pd

    RAW_ABLATIONS_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
    all_raw = load_all_raw_prompts()
    allowed = {str(i) for i in only_ids} if only_ids is not None else None
    rows_to_check = (
        [r for r in all_raw if str(r["id_prompt"]) in allowed]
        if allowed is not None
        else all_raw
    )
    missing_rows = []
    for row in rows_to_check:
        pid = str(row["id_prompt"])
        if not (RAW_ABLATIONS_AUDIO_PATH / f"{pid}.wav").exists():
            missing_rows.append(
                {
                    "id_prompt": pid,
                    "prompt": row["sentence"],
                    "taste": row.get("taste", ""),
                    "reprompt": row["sentence"],
                }
            )

    if not missing_rows:
        print(
            f"  ✓ Todos los {len(rows_to_check)} audios baseline ya existen en "
            f"{RAW_ABLATIONS_AUDIO_PATH}; omitiendo generación raw."
        )
        return None

    ABLATIONS_REPROMPTS_PATH.mkdir(parents=True, exist_ok=True)
    staging_csv = ABLATIONS_REPROMPTS_PATH / "raw_ablations.csv"
    pd.DataFrame(missing_rows).to_csv(staging_csv, index=False)
    print(
        f"  ✓ Preparado {staging_csv} con {len(missing_rows)}/{len(rows_to_check)} "
        f"prompts crudos pendientes para generar en {RAW_ABLATIONS_AUDIO_PATH}"
    )
    return str(staging_csv)


def ensure_raw_baseline(
    sample_ids: list[str],
    generate: bool = False,
    scope: str = "sample",
    dvc_cmd: str = "dvc",
) -> tuple[bool, str | None, int]:
    """Preflight check for raw baseline scores/audio before costly phases.

    Checks in order for each required ID:
      1. Valid cached score in ``clap_score_results_raw_prompts_baseline.csv``
      2. Local ``data/tracks/raw_ablations/<id>.wav``
      3. Granular ``dvc pull data/tracks/raw_ablations/<id>.wav ...`` (non-fatal)

    Returns ``(ok, staging_csv_path, n_raw_to_generate)``.
    """
    all_raw = load_all_raw_prompts()
    raw_by_id = {str(r["id_prompt"]): r for r in all_raw}
    if generate and scope == "all":
        target_ids = [str(r["id_prompt"]) for r in all_raw]
    else:
        target_ids = [str(i) for i in sample_ids]

    cache_by_id = load_raw_baseline_cache(ABLATIONS_RAW_BASELINE_SCORE_CSV)

    needs_pull: list[str] = []
    cached_hits = 0
    local_wavs = 0
    for pid in target_ids:
        wav_path = RAW_ABLATIONS_AUDIO_PATH / f"{pid}.wav"
        prompt_text = raw_by_id[pid]["sentence"] if pid in raw_by_id else (cache_by_id.get(pid, {}).get("text", ""))
        if is_valid_baseline_cache_entry(cache_by_id.get(pid), prompt_text, wav_path):
            cached_hits += 1
            continue
        if wav_path.exists():
            local_wavs += 1
            continue
        needs_pull.append(pid)

    if not needs_pull:
        print(
            f"  ✓ Baseline raw completo para {len(target_ids)} ID(s) "
            f"(caché={cached_hits}, wav local={local_wavs}; 0 pulls DVC)."
        )
        return True, None, 0

    rel_paths = [f"data/tracks/raw_ablations/{pid}.wav" for pid in needs_pull]
    print(
        f"  ↓ Baseline incompleto: {len(needs_pull)} ID(s) sin caché ni WAV local; "
        f"ejecutando dvc pull granular..."
    )
    try:
        proc = subprocess.run(
            [dvc_cmd, "pull", *rel_paths],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            err_msg = (proc.stderr or proc.stdout or "").strip()
            print(
                f"  ⚠ dvc pull granular devolvió código {proc.returncode}"
                + (f": {err_msg}" if err_msg else ""),
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"  ⚠ No se pudo ejecutar dvc pull granular: {exc}", file=sys.stderr)

    still_missing = [
        pid for pid in needs_pull if not (RAW_ABLATIONS_AUDIO_PATH / f"{pid}.wav").exists()
    ]
    if not still_missing:
        print(f"  ✓ Recuperados {len(needs_pull)} audio(s) baseline vía dvc pull granular.")
        return True, None, 0

    if not generate:
        preview = ", ".join(still_missing[:10])
        suffix = "..." if len(still_missing) > 10 else ""
        print(
            f"  ✗ Faltan {len(still_missing)} audio(s)/score(s) baseline en "
            f"{RAW_ABLATIONS_AUDIO_PATH} (IDs: {preview}{suffix}). "
            "Re-ejecuta con GENERATE_RAW_AUDIO=1 (--generate-raw-audio).",
            file=sys.stderr,
        )
        return False, None, 0

    staging_csv = prepare_raw_ablations_csv(only_ids=still_missing)
    n_gen = len(still_missing) if staging_csv else 0
    return True, staging_csv, n_gen


def _write_ablation_state(run_id: str, raw_audio_generated: int) -> Path:
    """Persist per-run state (.ablation_state/<RUN_ID>.json) for publish gating."""
    ABLATION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = ABLATION_STATE_DIR / f"{run_id}.json"
    state_path.write_text(
        json.dumps(
            {"run_id": run_id, "raw_audio_generated": int(raw_audio_generated)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return state_path


def run_nlp_phase(csv_paths: list[str], run_id: str = "") -> dict[str, dict]:
    """Run non-fatal NLP diversity evaluation for reprompt CSVs and write nlp_status.json."""
    status_by_out_dir: dict[Path, dict[str, dict]] = {}
    all_statuses: dict[str, dict] = {}

    def _out_dir_for(csv_path: str) -> Path:
        effective_rid = run_id
        if not effective_rid:
            m = re.search(r"(R\d{8}_\d{6})", Path(csv_path).stem)
            if m:
                effective_rid = m.group(1)
        return (
            ABLATIONS_PATH / "analysis" / f"nlp_{effective_rid}"
            if effective_rid
            else ABLATIONS_PATH / "analysis" / "nlp"
        )

    def _tag_for(csv_path: str) -> str:
        return Path(csv_path).stem.replace("pipeline_results_", "")

    try:
        from models.scripts.eval_reprompt_nlp import evaluate_csv
    except ImportError as exc:
        print(
            f"  ⚠ Faltan dependencias (textstat, nltk, sentence-transformers) para evaluar NLP: {exc}",
            file=sys.stderr,
        )
        for csv_path in csv_paths:
            out_dir = _out_dir_for(csv_path)
            tag = _tag_for(csv_path)
            entry = {
                "status": "skipped_missing_deps",
                "error": str(exc),
                "backends": {},
            }
            status_by_out_dir.setdefault(out_dir, {})[tag] = entry
            all_statuses[tag] = entry
    else:
        for csv_path in csv_paths:
            out_dir = _out_dir_for(csv_path)
            tag = _tag_for(csv_path)
            try:
                res = evaluate_csv(str(csv_path), output_dir=str(out_dir))
                backends = res.get("backends", {}) if isinstance(res, dict) else {}
                entry = {"status": "ok", "error": None, "backends": backends}
            except ImportError as exc:
                print(
                    f"  ⚠ Dependencias NLP faltantes para '{tag}': {exc}",
                    file=sys.stderr,
                )
                entry = {
                    "status": "skipped_missing_deps",
                    "error": str(exc),
                    "backends": {},
                }
            except Exception as exc:  # noqa: BLE001 — NLP phase must never fail the pipeline
                print(
                    f"  ⚠ Evaluación NLP falló para '{tag}': {exc}",
                    file=sys.stderr,
                )
                entry = {
                    "status": "failed",
                    "error": str(exc),
                    "backends": {},
                }
            status_by_out_dir.setdefault(out_dir, {})[tag] = entry
            all_statuses[tag] = entry

    for out_dir, entries in status_by_out_dir.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        status_path = out_dir / "nlp_status.json"
        existing: dict[str, dict] = {}
        if status_path.exists():
            try:
                loaded = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}
        existing.update(entries)
        status_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

    return all_statuses


def run_reprompt(cfg: AblationConfig, sample_size: int, seed: int, run_id: str = "") -> list[str]:
    """Phase 1: generate reprompts only."""
    generated: list[str] = []
    for model in cfg.models:
        run_tag = _resolve_tag(cfg, model, run_id)

        print(f"\n{'=' * 60}")
        print(f"  [reprompt] {cfg.name} | modelo: {model}")
        print(f"  seed={seed}  sample_size={sample_size}  run_id={run_id}")
        print(f"{'=' * 60}")

        result_csv = generate_reprompts(
            model=model,
            prompt_version=cfg.prompt_version,
            sample_size=sample_size,
            seed=seed,
            cut_results=cfg.cut_results,
            k=cfg.k,
            filter_dimensions=cfg.filter_dimensions,
            system_prompt_modifier=cfg.system_prompt_modifier,
            reduce_random_effort=cfg.reduce_random_effort,
            reasoning_effort=cfg.reasoning_effort,
            tag=run_tag,
            output_dir=str(ABLATIONS_REPROMPTS_PATH),
        )

        generated.append(result_csv)
        print(f"  ✓ Reprompts guardados en: {result_csv}")

    print(f"\n  ⏸  Genera los audios con los CSVs anteriores y ejecuta --phase score")
    return generated


def run_score(csv_paths: list[str], audio_dir: str | None = None) -> None:
    """Phase 3: calculate CLAP scores for existing reprompt CSVs.

    Args:
        csv_paths: Reprompt CSV files to score.
        audio_dir: Base directory containing per-experiment audio subdirs.
                   Each CSV looks for audio in <audio_dir>/<csv_stem>/.
                   If None, falls back to the default reprompt_audios path.
    """
    for result_csv in csv_paths:
        # extract tag from filename: pipeline_results_..._<tag>.csv
        csv_name = Path(result_csv).name
        tag = csv_name.replace("pipeline_results_", "").rsplit(".csv", 1)[0]

        # Resolve per-experiment audio directory
        exp_audio_dir: str | None = None
        if audio_dir:
            csv_stem = Path(result_csv).stem
            candidate = Path(audio_dir) / csv_stem
            if candidate.is_dir():
                exp_audio_dir = str(candidate)
            else:
                # Strip run_id suffix (_RYYYYMMDD_HHMMSS) and retry
                stem_no_rid = re.sub(r"_R\d{8}_\d{6}$", "", csv_stem)
                candidate = Path(audio_dir) / stem_no_rid
                if candidate.is_dir():
                    exp_audio_dir = str(candidate)
                    print(f"  ℹ Audio dir matched (without run_id): {candidate.name}")
                else:
                    print(
                        f"  ⚠ No matching audio subdir for '{csv_stem}' in {audio_dir}",
                        file=sys.stderr,
                    )
                    exp_audio_dir = audio_dir  # fallback to flat dir

        print(f"\n{'=' * 60}")
        print(f"  [score] {result_csv}")
        if exp_audio_dir:
            print(f"  [audio] {exp_audio_dir}")
        print(f"{'=' * 60}")

        calculate_clap_score_alignment(
            result_csv,
            optional_suffix=tag,
            output_dir=str(ABLATIONS_SCORES_PATH),
            audio_dir=exp_audio_dir,
        )
        calculate_clap_score_alignment(
            result_csv,
            using_raw_prompts=True,
            optional_suffix=f"{tag}_raw",
            output_dir=str(ABLATIONS_SCORES_PATH),
            audio_dir=str(RAW_ABLATIONS_AUDIO_PATH),
        )
        calculate_clap_score_alignment(
            result_csv,
            cross_evaluation=True,
            optional_suffix=f"{tag}_cross",
            output_dir=str(ABLATIONS_SCORES_PATH),
            audio_dir=exp_audio_dir,
        )

        print(f"  ✓ CLAP scores completados para: {result_csv}\n")


def run_ablation(cfg: AblationConfig, sample_size: int, seed: int, phase: str, run_id: str = "") -> None:
    csvs: list[str] = []
    if phase in ("reprompt", "all"):
        csvs = run_reprompt(cfg, sample_size, seed, run_id)

    if phase in ("score", "nlp"):
        # find previously generated CSVs for this experiment
        for model in cfg.models:
            run_tag = _resolve_tag(cfg, model, run_id)
            found = _find_reprompt_csvs(run_tag, run_id)
            if not found:
                print(
                    f"  ⚠ No se encontraron CSVs para tag '{run_tag}' en {ABLATIONS_REPROMPTS_PATH}",
                    file=sys.stderr,
                )
                continue
            csvs.extend(found)

        if not csvs:
            print("  Error: no hay CSVs de reprompts para calcular scores.", file=sys.stderr)
            return

    if phase in ("score", "all"):
        run_score(csvs, audio_dir=str(ABLATIONS_AUDIO_PATH))
        print(f"\n{'═' * 60}")
        print("  FASE NLP: Evaluando métricas de diversidad y consistencia")
        print(f"{'═' * 60}")
        run_nlp_phase(csvs, run_id=run_id)
    elif phase == "nlp":
        print(f"\n{'═' * 60}")
        print("  FASE NLP: Evaluando métricas ligeras")
        print(f"{'═' * 60}")
        run_nlp_phase(csvs, run_id=run_id)


def run_full_pipeline(
    targets: dict[str, AblationConfig],
    sample_size: int,
    seed: int,
    group_prefix: str,
    run_id: str = "",
    generate_raw_audio: bool = False,
    raw_audio_scope: str = "sample",
) -> int:
    """Run the full pipeline for generation ablations:
    preflight → reprompt → Kaggle GPU audio → CLAP score → NLP → statistical analysis.

    All variant CSVs (plus optional raw_ablations.csv for missing baseline IDs) are
    batched into a single Kaggle kernel push.
    Returns an EXIT_* code so callers (make/systemd) can detect failures.
    """
    from models.ablation_analysis import analyse_group
    from models.kaggle_runner import run_pipeline as kaggle_run

    if not run_id:
        run_id = _generate_run_id()
    print(f"  Run ID: {run_id}")

    # ── Preflight: deterministic sample manifest + raw baseline check ──
    sample_ids = compute_sample_ids(sample_size, seed)
    analysis_dir = ABLATIONS_PATH / "analysis" / f"{group_prefix}_{run_id}"
    manifest_path, sample_hash = write_sample_manifest(
        analysis_dir,
        sample_size=sample_size,
        seed=seed,
        ids=sample_ids,
        run_id=run_id,
        group=group_prefix,
    )
    print(f"  Sample manifest: {manifest_path} ({len(sample_ids)} IDs, seed={seed}, sha1={sample_hash})")

    ok, raw_staging_csv, n_raw_to_generate = ensure_raw_baseline(
        sample_ids,
        generate=generate_raw_audio,
        scope=raw_audio_scope,
    )
    if not ok:
        return EXIT_BASELINE_MISSING

    _write_ablation_state(run_id, raw_audio_generated=0)

    # ── Phase 1: Generate reprompts ───────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  FASE 1: Generar reprompts ({len(targets)} variante(s))")
    print(f"{'═' * 60}")
    all_csvs: list[str] = []
    for cfg in targets.values():
        csvs = run_reprompt(cfg, sample_size, seed, run_id)
        all_csvs.extend(csvs)

    if not all_csvs:
        print("  Error: no se generaron CSVs de reprompts.", file=sys.stderr)
        return EXIT_REPROMPT_FAILED

    print(f"\n  Total CSVs generados: {len(all_csvs)}")

    kaggle_csvs = [raw_staging_csv, *all_csvs] if raw_staging_csv else list(all_csvs)

    # ── Phase 2: Kaggle GPU audio generation ──────────────────
    print(f"\n{'═' * 60}")
    print(f"  FASE 2: Generar audio en Kaggle GPU ({len(kaggle_csvs)} CSV(s))")
    print(f"{'═' * 60}")
    try:
        kaggle_run(
            csv_paths=kaggle_csvs,
            output_dir=str(ABLATIONS_AUDIO_PATH),
            run_id=run_id,
        )
    except (RuntimeError, SystemExit) as e:
        if raw_staging_csv and Path(raw_staging_csv).exists():
            Path(raw_staging_csv).unlink(missing_ok=True)
        print(f"  ✗ Kaggle falló: {e}", file=sys.stderr)
        print(
            "  Puedes reintentar la fase de score manualmente:\n"
            f"    python -m models.ablation_runner --experiment {group_prefix} --phase score",
            file=sys.stderr,
        )
        return EXIT_KAGGLE_FAILED
    finally:
        if raw_staging_csv and Path(raw_staging_csv).exists():
            Path(raw_staging_csv).unlink(missing_ok=True)

    _write_ablation_state(run_id, raw_audio_generated=n_raw_to_generate)

    # ── Phase 3: CLAP scoring ────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  FASE 3: Calcular CLAP scores")
    print(f"{'═' * 60}")
    try:
        run_score(all_csvs, audio_dir=str(ABLATIONS_AUDIO_PATH))
    except Exception as e:  # noqa: BLE001 — report and map to an exit code
        print(f"  ✗ CLAP scoring falló: {e!r}", file=sys.stderr)
        return EXIT_SCORE_FAILED

    # ── Phase 4: NLP diversity evaluation (non-fatal) ────────
    print(f"\n{'═' * 60}")
    print("  FASE 4: Evaluación NLP de reprompts (no fatal)")
    print(f"{'═' * 60}")
    run_nlp_phase(all_csvs, run_id=run_id)

    # ── Phase 5: Statistical analysis ────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  FASE 5: Análisis estadístico — {group_prefix}")
    print(f"{'═' * 60}")
    try:
        analyse_group(group_prefix, run_id=run_id)
    except SystemExit:
        print(
            f"  ⚠ Análisis falló. Ejecutar manualmente:\n"
            f"    python -m models.ablation_analysis --experiment {group_prefix}",
            file=sys.stderr,
        )
        return EXIT_ANALYSIS_FAILED

    print(f"\n  ✓ Pipeline completo para {group_prefix}.")
    return EXIT_OK


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar ablaciones del pipeline audio-reprompt"
    )
    parser.add_argument(
        "--experiment",
        help=(
            "ID del experimento (e.g. B1a, B3c) o grupo: "
            "'all', 'generation', 'B1', 'B2', 'B3'"
        ),
    )
    parser.add_argument(
        "--phase",
        choices=["reprompt", "score", "nlp", "all", "full"],
        default="all",
        help=(
            "Fase a ejecutar: reprompt (solo genera), score (solo CLAP), "
            "nlp (métricas ligeras), all (reprompt+score sin Kaggle), "
            "full (reprompt+Kaggle+score+NLP+análisis). Default: all"
        ),
    )
    parser.add_argument(
        "--reprompt-csv",
        nargs="+",
        help="Ruta(s) a CSV(s) de reprompts existentes para calcular CLAP scores directamente (solo con --phase score)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Tamaño de muestra estratificada por taste (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para muestreo aleatorio reproducible (default: 42)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_experiments",
        help="Listar experimentos disponibles",
    )
    parser.add_argument(
        "--variants",
        default="",
        help=(
            "Subconjunto de variantes del grupo, separadas por coma (e.g. B3a,B3b,B3c,B3f) "
            "o un preset (e.g. core, core+aa para B3). Default: todas"
        ),
    )
    parser.add_argument(
        "--run-id",
        default="",
        help=(
            "Run ID (timestamp) para identificar esta ejecución. "
            "Se genera automáticamente si no se proporciona. "
            "Formato: RYYYYMMDD_HHMMSS"
        ),
    )
    parser.add_argument(
        "--generate-raw-audio",
        action="store_true",
        help=(
            "Genera los audios baseline faltantes de los prompts originales "
            "(data/tracks/raw_ablations) en el batch de Kaggle."
        ),
    )
    parser.add_argument(
        "--raw-audio-scope",
        choices=["sample", "all"],
        default=os.getenv("RAW_AUDIO_SCOPE", "sample"),
        help=(
            "Alcance de generación cuando se usa --generate-raw-audio: "
            "'sample' (solo IDs muestreados faltantes, default) o 'all' (todos los prompts crudos faltantes)."
        ),
    )
    args = parser.parse_args()

    if args.list_experiments:
        print("\nExperimentos de Generación (requieren audio/Kaggle + CLAP):")
        for key, cfg in GENERATION_ABLATIONS.items():
            models_str = ", ".join(cfg.models)
            print(f"  {key:6s}  {cfg.name}  [{models_str}]")
        return

    # Direct CSV scoring without experiment config
    if args.reprompt_csv:
        if args.phase == "score":
            run_score(args.reprompt_csv)
            run_nlp_phase(args.reprompt_csv, run_id=args.run_id)
            return
        elif args.phase == "nlp":
            run_nlp_phase(args.reprompt_csv, run_id=args.run_id)
            return
        else:
            print("Error: --reprompt-csv solo se puede usar con --phase score o --phase nlp", file=sys.stderr)
            sys.exit(1)

    if not args.experiment:
        print("Error: se requiere --experiment o --reprompt-csv", file=sys.stderr)
        sys.exit(1)

    experiment = args.experiment
    try:
        targets = resolve_targets(experiment, args.variants)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    skipped = [k for k in resolve_targets(experiment) if k not in targets]
    print(f"Variantes a ejecutar: {', '.join(targets)}" + (f" | omitidas: {', '.join(skipped)}" if skipped else ""))

    # Full pipeline: preflight → reprompt → Kaggle → score → NLP → analysis
    if args.phase == "full":
        # Determine group prefix for analysis
        group_prefix = experiment if len(experiment) == 2 else experiment[:2]
        sys.exit(
            run_full_pipeline(
                targets,
                args.sample_size,
                args.seed,
                group_prefix,
                args.run_id,
                generate_raw_audio=args.generate_raw_audio,
                raw_audio_scope=args.raw_audio_scope,
            )
        )

    # For non-full phases, generate a run_id if not provided
    run_id = args.run_id or _generate_run_id()
    print(
        f"Ejecutando {len(targets)} experimento(s) | phase={args.phase} | "
        f"sample_size={args.sample_size} | seed={args.seed} | run_id={run_id}"
    )
    for cfg in targets.values():
        run_ablation(cfg, args.sample_size, args.seed, args.phase, run_id)

    print("Ablaciones completadas.")


if __name__ == "__main__":
    main()
