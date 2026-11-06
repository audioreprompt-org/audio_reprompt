import argparse
import glob
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

def calculate_cosine_distance(embeds1: torch.Tensor, embeds2: torch.Tensor) -> np.ndarray:
    # embeds: shape (N, D)
    # Cosine distance = 1 - cosine similarity
    # similarity = (u * v).sum() / (|u| |v|)
    embeds1 = torch.nn.functional.normalize(embeds1, dim=-1)
    embeds2 = torch.nn.functional.normalize(embeds2, dim=-1)
    sims = (embeds1 * embeds2).sum(dim=-1)
    distances = 1.0 - sims
    return distances.detach().cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Analyze prompt to reprompt CLAP text embedding distance.")
    parser.add_argument("suffix", type=str, help="Ablation identifier suffix, e.g. R20260917_113439")
    args = parser.parse_args()

    # Resolve root directory relative to this script
    root_dir = Path(__file__).resolve().parent.parent.parent
    reprompts_dir = root_dir / "data" / "ablations" / "reprompts"
    if not reprompts_dir.exists():
        print(f"Directory {reprompts_dir} not found.")
        return

    # Find the 4 files
    versions = ["V1", "V2", "V3", "V4"]
    files = {}
    for v in versions:
        # Looking for a file containing _V1_ (or similar) and ending with args.suffix.csv
        pattern = f"*{v}*{args.suffix}.csv"
        matched = list(reprompts_dir.glob(pattern))
        if not matched:
            print(f"Warning: No file found for {v} with suffix {args.suffix}")
        elif len(matched) > 1:
            print(f"Warning: Multiple files found for {v} with suffix {args.suffix}. Taking the first one.")
            files[v] = matched[0]
        else:
            files[v] = matched[0]

    if not files:
        print("No files found.")
        return

    print("Loading SentenceTransformer all-MiniLM-L6-v2 Model...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {}
    for v, file_path in files.items():
        print(f"\nProcessing {v} from {file_path.name}...")
        df = pd.read_csv(file_path)
        
        if "prompt" not in df.columns or "reprompt" not in df.columns:
            print(f"Skipping {v}: missing 'prompt' or 'reprompt' column.")
            continue

        # Drop rows with NaN in prompts or reprompts just in case
        df = df.dropna(subset=["prompt", "reprompt"])
        
        prompts = df["prompt"].tolist()
        reprompts = df["reprompt"].tolist()

        # Generate embeddings
        with torch.no_grad():
            prompt_embeds = st_model.encode(prompts, convert_to_tensor=True)
            reprompt_embeds = st_model.encode(reprompts, convert_to_tensor=True)

        distances = calculate_cosine_distance(prompt_embeds, reprompt_embeds)
        results[v] = distances

        # Print stats
        print(f"Stats for {v}:")
        print(f"  Count:  {len(distances)}")
        print(f"  Mean:   {distances.mean():.4f}")
        print(f"  Std:    {distances.std():.4f}")
        print(f"  Min:    {distances.min():.4f}")
        print(f"  Median: {np.median(distances):.4f}")
        print(f"  Max:    {distances.max():.4f}")

    print("\nSummary of Means:")
    for v in versions:
        if v in results:
            print(f"{v}: {results[v].mean():.4f}")

if __name__ == "__main__":
    main()
