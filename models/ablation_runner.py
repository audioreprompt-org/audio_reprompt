"""
CLI para ejecutar experimentos de ablación definidos en Ablations.md.

Retrieval (A-series) — solo genera reprompts (métricas textuales):
    python -m models.ablation_runner --experiment A1a

Generation (B-series) — flujo por fases (audio en Kaggle GPU):
    Fase 1: python -m models.ablation_runner --experiment B1a --phase reprompt
    Fase 2: python -m models.kaggle_runner run --csvs data/ablations/reprompts/*.csv
    Fase 3: python -m models.ablation_runner --experiment B1a --phase score
             python -m models.ablation_runner --phase score --reprompt-csv <path>

Pipeline completo (reprompt → Kaggle GPU → score → análisis):
    python -m models.ablation_runner --experiment B2 --phase full
"""

import argparse
import re
import glob
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from config import setup_project_paths

setup_project_paths()

from models.music_curator.kimi_mcu import (
    KIMI_K2_THINKING_MODEL,
    OPENAI_GPT_5_NANO_MODEL,
)
from models.pipeline import FILTER_DIMENSIONS_DEFAULT
from models.validate import (
    generate_reprompts,
    calculate_clap_score_alignment,
    ABLATIONS_REPROMPTS_PATH,
    ABLATIONS_SCORES_PATH,
    ABLATIONS_PATH,
)

ABLATIONS_AUDIO_PATH = ABLATIONS_PATH / "audio"

BOTH_MODELS = [KIMI_K2_THINKING_MODEL, OPENAI_GPT_5_NANO_MODEL]


@dataclass
class AblationConfig:
    name: str
    tag: str
    models: list[str] = field(default_factory=lambda: [KIMI_K2_THINKING_MODEL])
    prompt_version: str = "V4"
    cut_results: bool = True
    k: int = 10
    filter_dimensions: tuple[str, ...] | None = FILTER_DIMENSIONS_DEFAULT
    system_prompt_modifier: str | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    requires_audio: bool = False


# ── Retrieval (se ejecutan con ambos modelos) ──────────────

RETRIEVAL_ABLATIONS: dict[str, AblationConfig] = {
    # A1 — Heurística de corte
    "A1a": AblationConfig(
        name="A1a — cut_results=True", tag="A1a", models=BOTH_MODELS, cut_results=True
    ),
    "A1b": AblationConfig(
        name="A1b — cut_results=False", tag="A1b", models=BOTH_MODELS, cut_results=False
    ),
    # A2 — Top-K captions
    "A2a": AblationConfig(name="A2a — k=5", tag="A2a_k5", models=BOTH_MODELS, k=5),
    "A2b": AblationConfig(name="A2b — k=10", tag="A2b_k10", models=BOTH_MODELS, k=10),
    "A2c": AblationConfig(name="A2c — k=50", tag="A2c_k50", models=BOTH_MODELS, k=50),
    # A3 — Filtro de dimensiones
    "A3a": AblationConfig(
        name="A3a — emotion/taste/texture",
        tag="A3a",
        models=BOTH_MODELS,
        filter_dimensions=FILTER_DIMENSIONS_DEFAULT,
    ),
    "A3b": AblationConfig(
        name="A3b — sin filtro",
        tag="A3b_nofilter",
        models=BOTH_MODELS,
        filter_dimensions=None,
    ),
}

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
    # B3 — Nivel de Creatividad Inducida por Prompt y Penalizaciones
    "B3a": AblationConfig(
        name="B3a — prompt determinista conservador", tag="B3a_prompt_determinista", system_prompt_modifier="Sé determinista y conservador", presence_penalty=0.0, frequency_penalty=0.0, requires_audio=True,
    ),
    "B3b": AblationConfig(
        name="B3b — prompt default pen=0.0", tag="B3b_default_pen0", presence_penalty=0.0, frequency_penalty=0.0, requires_audio=True,
    ),
    "B3c": AblationConfig(
        name="B3c — prompt creativo e imaginativo", tag="B3c_prompt_creativo", system_prompt_modifier="Sé altamente creativo e imaginativo", presence_penalty=0.0, frequency_penalty=0.0, requires_audio=True,
    ),
    "B3d": AblationConfig(
        name="B3d — presence penalty 0.6", tag="B3d_presence_06", presence_penalty=0.6, requires_audio=True,
    ),
    "B3e": AblationConfig(
        name="B3e — frequency penalty 0.6", tag="B3e_frequency_06", frequency_penalty=0.6, requires_audio=True,
    ),
}

ALL_ABLATIONS = {**RETRIEVAL_ABLATIONS, **GENERATION_ABLATIONS}


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
            presence_penalty=cfg.presence_penalty,
            frequency_penalty=cfg.frequency_penalty,
            tag=run_tag,
            output_dir=str(ABLATIONS_REPROMPTS_PATH),
        )

        generated.append(result_csv)
        print(f"  ✓ Reprompts guardados en: {result_csv}")

    if cfg.requires_audio:
        print(f"\n  ⏸  Genera los audios con los CSVs anteriores y ejecuta --phase score")
    else:
        print(f"\n  ✓ Retrieval completado (no requiere generación de audio)")
    return generated


def run_score(csv_paths: list[str], audio_dir: str | None = None) -> None:
    """Phase 2: calculate CLAP scores for existing reprompt CSVs.

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
            audio_dir=exp_audio_dir,
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
    # Retrieval ablations only generate reprompts (text-level metrics)
    if not cfg.requires_audio and phase in ("score", "all"):
        effective_phase = "reprompt"
        if phase == "score":
            print(
                f"  ⚠ '{cfg.name}' es retrieval — no requiere audio/CLAP. Saltando.",
                file=sys.stderr,
            )
            return
        # phase == "all" → just run reprompt
        print(f"  ℹ '{cfg.name}' es retrieval — ejecutando solo reprompts.")
    elif phase == "nlp":
        effective_phase = "score" # Lo usamos como alias para recolectar CSVs
    else:
        effective_phase = phase

    if effective_phase in ("reprompt", "all"):
        csvs = run_reprompt(cfg, sample_size, seed, run_id)
        if not cfg.requires_audio or phase == "nlp":
            print(f"\n{'═' * 60}")
            print(f"  FASE NLP: Evaluando métricas ligeras")
            print(f"{'═' * 60}")
            try:
                from models.scripts.eval_reprompt_nlp import evaluate_csv
                from models.validate import ABLATIONS_PATH
                out_dir = str(ABLATIONS_PATH / "analysis" / f"A_{run_id}")
                for csv_path in csvs:
                    evaluate_csv(csv_path, output_dir=out_dir)
            except ImportError as e:
                print(f"  ⚠ Faltan dependencias (textstat, nltk, sentence-transformers) para evaluar NLP: {e}", file=sys.stderr)

    if effective_phase == "score":
        # find previously generated CSVs for this experiment
        csvs = []
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

    if effective_phase in ("score", "all") and phase != "nlp":
        run_score(csvs, audio_dir=str(ABLATIONS_AUDIO_PATH))
    elif phase == "nlp":
        print(f"\n{'═' * 60}")
        print(f"  FASE NLP: Evaluando métricas ligeras")
        print(f"{'═' * 60}")
        try:
            from models.scripts.eval_reprompt_nlp import evaluate_csv
            from models.validate import ABLATIONS_PATH
            out_dir = str(ABLATIONS_PATH / "analysis" / f"A_{run_id}")
            for csv_path in csvs:
                evaluate_csv(csv_path, output_dir=out_dir)
        except ImportError as e:
            print(f"  ⚠ Faltan dependencias (textstat, nltk, sentence-transformers) para evaluar NLP: {e}", file=sys.stderr)


def run_full_pipeline(
    targets: dict[str, AblationConfig],
    sample_size: int,
    seed: int,
    group_prefix: str,
    run_id: str = "",
) -> None:
    """Run the full pipeline for generation ablations:
    reprompt → Kaggle GPU audio → CLAP score → statistical analysis.

    All variant CSVs are batched into a single Kaggle kernel push.
    """
    from models.kaggle_runner import run_pipeline as kaggle_run
    from models.ablation_analysis import analyse_group

    if not run_id:
        run_id = _generate_run_id()
    print(f"  Run ID: {run_id}")

    # Verify all targets require audio
    non_audio = [k for k, cfg in targets.items() if not cfg.requires_audio]
    if non_audio:
        print(
            f"  ⚠ Retrieval experiments no soportan --phase full: {non_audio}",
            file=sys.stderr,
        )
        return

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
        return

    print(f"\n  Total CSVs generados: {len(all_csvs)}")

    # ── Phase 2: Kaggle GPU audio generation ──────────────────
    print(f"\n{'═' * 60}")
    print(f"  FASE 2: Generar audio en Kaggle GPU ({len(all_csvs)} CSV(s))")
    print(f"{'═' * 60}")
    try:
        kaggle_run(
            csv_paths=all_csvs,
            output_dir=str(ABLATIONS_AUDIO_PATH),
            run_id=run_id,
        )
    except RuntimeError as e:
        print(f"  ✗ Kaggle falló: {e}", file=sys.stderr)
        print(
            "  Puedes reintentar la fase de score manualmente:\n"
            f"    python -m models.ablation_runner --experiment {group_prefix} --phase score",
            file=sys.stderr,
        )
        return

    # ── Phase 3: CLAP scoring ────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  FASE 3: Calcular CLAP scores")
    print(f"{'═' * 60}")
    run_score(all_csvs, audio_dir=str(ABLATIONS_AUDIO_PATH))

    # ── Phase 4: Statistical analysis ────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  FASE 4: Análisis estadístico — {group_prefix}")
    print(f"{'═' * 60}")
    try:
        analyse_group(group_prefix, run_id=run_id)
    except SystemExit:
        print(
            f"  ⚠ Análisis falló. Ejecutar manualmente:\n"
            f"    python -m models.ablation_analysis --experiment {group_prefix}",
            file=sys.stderr,
        )

    print(f"\n  ✓ Pipeline completo para {group_prefix}.")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar ablaciones del pipeline audio-reprompt"
    )
    parser.add_argument(
        "--experiment",
        help=(
            "ID del experimento (e.g. A1a, B3c) o grupo: "
            "'all', 'retrieval', 'generation', 'A1', 'A2', 'A3', 'B1', 'B3'"
        ),
    )
    parser.add_argument(
        "--phase",
        choices=["reprompt", "score", "nlp", "all", "full"],
        default="all",
        help=(
            "Fase a ejecutar: reprompt (solo genera), score (solo CLAP), "
            "nlp (métricas ligeras), all (reprompt+score sin Kaggle), "
            "full (reprompt+Kaggle+score+análisis). Default: all"
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
        "--run-id",
        default="",
        help=(
            "Run ID (timestamp) para identificar esta ejecución. "
            "Se genera automáticamente si no se proporciona. "
            "Formato: RYYYYMMDD_HHMMSS"
        ),
    )
    args = parser.parse_args()

    if args.list_experiments:
        print("\nExperimentos de Retrieval (solo reprompts — métricas textuales):")
        for key, cfg in RETRIEVAL_ABLATIONS.items():
            models_str = ", ".join(cfg.models)
            print(f"  {key:6s}  {cfg.name}  [{models_str}]")
        print("\nExperimentos de Generación (requieren audio/Kaggle + CLAP):")
        for key, cfg in GENERATION_ABLATIONS.items():
            models_str = ", ".join(cfg.models)
            print(f"  {key:6s}  {cfg.name}  [{models_str}]")
        return

    # Direct CSV scoring without experiment config
    if args.reprompt_csv:
        if args.phase == "score":
            run_score(args.reprompt_csv)
            return
        elif args.phase == "nlp":
            try:
                from models.scripts.eval_reprompt_nlp import evaluate_csv
                from models.validate import ABLATIONS_PATH
                run_id = args.run_id or _generate_run_id()
                out_dir = str(ABLATIONS_PATH / "analysis" / f"A_{run_id}")
                for path in args.reprompt_csv:
                    evaluate_csv(path, output_dir=out_dir)
            except ImportError as e:
                print(f"  ⚠ Faltan dependencias para evaluar NLP: {e}", file=sys.stderr)
            return
        else:
            print("Error: --reprompt-csv solo se puede usar con --phase score o --phase nlp", file=sys.stderr)
            sys.exit(1)

    if not args.experiment:
        print("Error: se requiere --experiment o --reprompt-csv", file=sys.stderr)
        sys.exit(1)

    experiment = args.experiment

    if experiment == "all":
        targets = ALL_ABLATIONS
    elif experiment == "retrieval":
        targets = RETRIEVAL_ABLATIONS
    elif experiment == "generation":
        targets = GENERATION_ABLATIONS
    elif len(experiment) == 2 and experiment in ("A1", "A2", "A3", "B1", "B2", "B3"):
        targets = {k: v for k, v in ALL_ABLATIONS.items() if k.startswith(experiment)}
    elif experiment in ALL_ABLATIONS:
        targets = {experiment: ALL_ABLATIONS[experiment]}
    else:
        print(f"Error: experimento '{experiment}' no encontrado.", file=sys.stderr)
        print(f"Disponibles: {', '.join(ALL_ABLATIONS.keys())}", file=sys.stderr)
        sys.exit(1)

    # Full pipeline: reprompt → Kaggle → score → analysis
    if args.phase == "full":
        # Determine group prefix for analysis
        group_prefix = experiment if len(experiment) == 2 else experiment[:2]
        run_full_pipeline(targets, args.sample_size, args.seed, group_prefix, args.run_id)
        return

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
