import csv

from config import setup_project_paths, load_config, PROJECT_ROOT
from tqdm import tqdm
import pandas as pd

from models.clap_score import CLAPItem, ClapModel
from models.music_curator.kimi_mcu import KIMI_K2_THINKING_MODEL, OPENAI_GPT_5_NANO_MODEL
from models.pipeline import transform

setup_project_paths()
config = load_config()

FOOD_PROMPTS_PATH = (
    PROJECT_ROOT / config.data.raw_data_path / "user" / "raw_prompts.csv"
)

AUDIOS_PATH = (PROJECT_ROOT / config.data.tracks_data_path).parent

CLAP_RESULTS_PATH = PROJECT_ROOT / config.data.data_clap_path

REPROMPTS_PATH = PROJECT_ROOT / config.data.reprompts_csv_path


clap_model = ClapModel(device="auto", enable_fusion=True)


def generate_reprompts(model: str, prompt_version: str, limit: int = 100_000):
    with open(FOOD_PROMPTS_PATH, "r") as file_:
        reader = csv.DictReader(file_)
        prompts = [prompt["sentence"] for prompt in reader][:limit]

    results = []
    for pos, user_prompt in enumerate(tqdm(prompts, desc="prompts progress"), start=1):
        results.append(
            {
                "id_prompt": pos,
                "prompt": user_prompt,
                "reprompt": transform(user_prompt, model=model, prompt_version=prompt_version),
            }
        )

    pd.DataFrame(results).to_csv(
        f"pipeline_results_{model.replace('-', '_')}_{len(prompts)}_prompt_{prompt_version}.csv",
        index=False,
    )

def calculate_clap_score_alignment(result_filepath: str, using_raw_prompts: bool = False, optional_suffix: str = ""):
    if using_raw_prompts:
        print("calculando scores de clap para prompts de entrada...")
        field = 'prompt'
        relative_audio_path = AUDIOS_PATH / 'raw_prompts_audios'
    else:
        print("calculando scores de clap para reprompts generados...")
        field = 'reprompt'
        relative_audio_path = AUDIOS_PATH / 'reprompt_audios'

    with open(result_filepath, "r") as file_:
        reader = csv.DictReader(file_)
        items = [row for row in reader]

    clap_items = [
        CLAPItem(
            id=row['id_prompt'],
            prompt=row[field],
            audio_path=str(relative_audio_path / f"{row['id_prompt']}.wav")
        )
        for row in items
    ]

    results = clap_model.calculate_scores(clap_items)

    pd.DataFrame([
        {
            "id_prompt": res.item.id,
            "text": res.item.prompt,
            "audio": res.item.audio_path,
            "clap_score": res.clap_score,
         }
        for res in results
        ]
    ).to_csv(f"{CLAP_RESULTS_PATH}/clap_score_results_{field}_outputs_{optional_suffix}.csv", index=False)


def calculate_clap_score_reprompts() -> None:
    calculate_clap_score_alignment(
        f"{REPROMPTS_PATH}/pipeline_results_kimi_k2_thinking_50_prompt_v3.csv",
    )

    calculate_clap_score_alignment(
        f"{REPROMPTS_PATH}/pipeline_results_kimi_k2_thinking_50_prompt_v3.csv",
        using_raw_prompts=True
    )

if __name__ == "__main__":
    generate_reprompts(OPENAI_GPT_5_NANO_MODEL, "V4")
    # calculate_clap_score_reprompts()
