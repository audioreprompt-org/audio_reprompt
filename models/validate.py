import csv

from config import setup_project_paths, load_config, PROJECT_ROOT
from tqdm import tqdm
import pandas as pd

from models.music_curator.kimi_mcu import KIMI_K2_THINKING_MODEL
from models.pipeline import transform

setup_project_paths()
config = load_config()

FOOD_PROMPTS_PATH = (
    PROJECT_ROOT / config.data.raw_data_path / "user" / "raw_prompts.csv"
)


def generate_reprompts(model: str):
    with open(FOOD_PROMPTS_PATH, "r") as file_:
        reader = csv.DictReader(file_)
        prompts = [prompt["sentence"] for prompt in reader]

    results = []
    for pos, prompt in enumerate(tqdm(prompts, desc="prompts progress"), start=1):
        results.append(
            {
                "id_prompt": pos,
                "prompt": prompt,
                "reprompt": transform(prompt, model=model),
            }
        )

    pd.DataFrame(results).to_csv(
        f"pipeline_results_{model.replace('-', '_')}_{len(prompts)}_prompt_v3.csv",
        index=False,
    )


if __name__ == "__main__":
    generate_reprompts(KIMI_K2_THINKING_MODEL)
    # generate_reprompts(OPENAI_GPT_5_NANO_MODEL)
