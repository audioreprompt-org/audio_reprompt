import csv
import random
from pathlib import Path

from config import setup_project_paths, load_config, PROJECT_ROOT
from tqdm import tqdm
import pandas as pd

from models.clap_score import CLAPItem, ClapModel
from models.music_curator.kimi_mcu import (
    KIMI_K2_THINKING_MODEL,
    OPENAI_GPT_5_NANO_MODEL,
)
from models.pipeline import transform

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

STRATIFY_COLUMN = "taste"

clap_model = ClapModel(device="auto", enable_fusion=True)


def _stratified_sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Stratified random sample ensuring equal count per `taste` category."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[STRATIFY_COLUMN], []).append(row)

    n_groups = len(groups)
    per_group = sample_size // n_groups

    if any(len(v) < per_group for v in groups.values()):
        smallest = min(len(v) for v in groups.values())
        per_group = min(per_group, smallest)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for category in sorted(groups.keys()):
        sampled.extend(rng.sample(groups[category], per_group))

    return sampled


def generate_reprompts(
    model: str,
    prompt_version: str,
    sample_size: int = 30,
    seed: int = 42,
    cut_results: bool = True,
    k: int = 10,
    filter_dimensions: tuple[str, ...] | None = None,
    system_prompt_modifier: str | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    tag: str = "",
    output_dir: str | None = None,
):
    all_rows = []
    for csv_file in FOOD_PROMPTS_DIR.glob("*.csv"):
        with open(csv_file, "r") as file_:
            reader = csv.DictReader(file_)
            all_rows.extend(list(reader))

    sampled_rows = _stratified_sample(all_rows, sample_size, seed)
    print(f"  Muestra estratificada: {len(sampled_rows)} prompts (seed={seed})")

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
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
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
    if using_raw_prompts:
        print("calculando scores de clap para prompts de entrada...")
        field = "prompt"
        relative_audio_path = (
            Path(audio_dir) if audio_dir else AUDIOS_PATH / "raw_prompts_audios"
        )
    elif cross_evaluation:
        print("calculando scores de clap usando prompt original - reprompt audio...")
        field = "prompt"
        relative_audio_path = (
            Path(audio_dir) if audio_dir else AUDIOS_PATH / "reprompt_audios"
        )
        optional_suffix = "cross_evaluation"
    else:
        print("calculando scores de clap para reprompts generados...")
        field = "reprompt"
        relative_audio_path = (
            Path(audio_dir) if audio_dir else AUDIOS_PATH / "reprompt_audios"
        )

    with open(result_filepath, "r") as file_:
        reader = csv.DictReader(file_)
        items = [row for row in reader]

    clap_items = [
        CLAPItem(
            id=row["id_prompt"],
            prompt=row[field],
            audio_path=str(relative_audio_path / f"{row['id_prompt']}.wav"),
        )
        for row in items
    ]

    dest = Path(output_dir) if output_dir else CLAP_RESULTS_PATH
    dest.mkdir(parents=True, exist_ok=True)

    results = clap_model.calculate_scores(clap_items)
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
        f"{dest}/clap_score_results_{field}_outputs_{optional_suffix}.csv",
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
