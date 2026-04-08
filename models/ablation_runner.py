"""
CLI para ejecutar experimentos de ablación definidos en Ablations.md.

Uso:
    python -m models.ablation_runner --experiment A1a
    python -m models.ablation_runner --experiment B3c --limit 20
    python -m models.ablation_runner --experiment all
"""

import argparse
import sys
from dataclasses import dataclass

from config import setup_project_paths

setup_project_paths()

from models.music_curator.kimi_mcu import KIMI_K2_THINKING_MODEL, OPENAI_GPT_5_NANO_MODEL
from models.pipeline import FILTER_DIMENSIONS_DEFAULT
from models.validate import generate_reprompts, calculate_clap_score_alignment


@dataclass
class AblationConfig:
    name: str
    tag: str
    model: str = KIMI_K2_THINKING_MODEL
    prompt_version: str = "V4"
    cut_results: bool = True
    k: int = 10
    filter_dimensions: tuple[str, ...] | None = FILTER_DIMENSIONS_DEFAULT
    temperature: float | None = None
    top_p: float | None = None


# ── Retrieval ──────────────────────────────────────────────

RETRIEVAL_ABLATIONS: dict[str, AblationConfig] = {
    # A1 — Heurística de corte
    "A1a": AblationConfig(name="A1a — cut_results=True", tag="A1a", cut_results=True),
    "A1b": AblationConfig(name="A1b — cut_results=False", tag="A1b", cut_results=False),
    # A2 — Top-K captions
    "A2a": AblationConfig(name="A2a — k=5", tag="A2a_k5", k=5),
    "A2b": AblationConfig(name="A2b — k=10", tag="A2b_k10", k=10),
    "A2c": AblationConfig(name="A2c — k=20", tag="A2c_k20", k=20),
    # A3 — Filtro de dimensiones
    "A3a": AblationConfig(
        name="A3a — emotion/taste/texture",
        tag="A3a",
        filter_dimensions=FILTER_DIMENSIONS_DEFAULT,
    ),
    "A3b": AblationConfig(
        name="A3b — sin filtro", tag="A3b_nofilter", filter_dimensions=None
    ),
}

# ── Generation ─────────────────────────────────────────────

GENERATION_ABLATIONS: dict[str, AblationConfig] = {
    # B1 — Versión de prompt
    "B1a": AblationConfig(name="B1a — V1", tag="B1a_V1", prompt_version="V1"),
    "B1b": AblationConfig(name="B1b — V2", tag="B1b_V2", prompt_version="V2"),
    "B1c": AblationConfig(name="B1c — V3", tag="B1c_V3", prompt_version="V3"),
    "B1d": AblationConfig(name="B1d — V4", tag="B1d_V4", prompt_version="V4"),
    # B2 — Modelo LLM
    "B2a": AblationConfig(
        name="B2a — kimi-k2", tag="B2a_kimi", model=KIMI_K2_THINKING_MODEL
    ),
    "B2b": AblationConfig(
        name="B2b — gpt-5-nano", tag="B2b_gpt5nano", model=OPENAI_GPT_5_NANO_MODEL
    ),
    # B3 — Sampling
    "B3a": AblationConfig(name="B3a — t=0.3 p=0.9", tag="B3a_t03_p09", temperature=0.3, top_p=0.9),
    "B3b": AblationConfig(name="B3b — t=0.7 p=0.9", tag="B3b_t07_p09", temperature=0.7, top_p=0.9),
    "B3c": AblationConfig(name="B3c — t=1.0 p=0.9", tag="B3c_t10_p09", temperature=1.0, top_p=0.9),
    "B3d": AblationConfig(name="B3d — t=0.7 p=0.5", tag="B3d_t07_p05", temperature=0.7, top_p=0.5),
    "B3e": AblationConfig(name="B3e — t=0.7 p=1.0", tag="B3e_t07_p10", temperature=0.7, top_p=1.0),
}

ALL_ABLATIONS = {**RETRIEVAL_ABLATIONS, **GENERATION_ABLATIONS}


def run_ablation(cfg: AblationConfig, limit: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Ejecutando: {cfg.name}")
    print(f"{'='*60}")

    result_csv = generate_reprompts(
        model=cfg.model,
        prompt_version=cfg.prompt_version,
        limit=limit,
        cut_results=cfg.cut_results,
        k=cfg.k,
        filter_dimensions=cfg.filter_dimensions,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        tag=cfg.tag,
    )

    print(f"  Reprompts guardados en: {result_csv}")
    print(f"  Calculando CLAP scores...")

    calculate_clap_score_alignment(result_csv, optional_suffix=cfg.tag)
    calculate_clap_score_alignment(result_csv, using_raw_prompts=True, optional_suffix=f"{cfg.tag}_raw")
    calculate_clap_score_alignment(result_csv, cross_evaluation=True, optional_suffix=f"{cfg.tag}_cross")

    print(f"  ✓ {cfg.name} completado\n")


def main():
    parser = argparse.ArgumentParser(description="Ejecutar ablaciones del pipeline audio-reprompt")
    parser.add_argument(
        "--experiment",
        required=True,
        help=(
            "ID del experimento (e.g. A1a, B3c) o grupo: "
            "'all', 'retrieval', 'generation', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3'"
        ),
    )
    parser.add_argument("--limit", type=int, default=100_000, help="Límite de prompts a procesar")
    parser.add_argument("--list", action="store_true", dest="list_experiments", help="Listar experimentos disponibles")
    args = parser.parse_args()

    if args.list_experiments:
        print("\nExperimentos de Retrieval:")
        for key, cfg in RETRIEVAL_ABLATIONS.items():
            print(f"  {key:6s}  {cfg.name}")
        print("\nExperimentos de Generación:")
        for key, cfg in GENERATION_ABLATIONS.items():
            print(f"  {key:6s}  {cfg.name}")
        return

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

    print(f"Ejecutando {len(targets)} experimento(s) con limit={args.limit}")
    for cfg in targets.values():
        run_ablation(cfg, args.limit)

    print("Todas las ablaciones completadas.")


if __name__ == "__main__":
    main()
