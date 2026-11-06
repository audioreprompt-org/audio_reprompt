import argparse
import json
import os
import pandas as pd
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

def get_system_prompt():
    return """You are an expert music curator and evaluator.
Your task is to evaluate a music generation reprompt text using the following baseline music features:
- pitch, contour, range, tesitura, phrasing, interval, root, consonance, progression, tonality, beat, pulse, tempo, meter, accent

You must evaluate the reprompt using the following three metrics and output your evaluation as a strict JSON object:

1. "Cohesion Text Level": Score 0 to 5 for the correctness of the existing music descriptors. Use 1-5 to indicate how much the descriptor is aligned in the entire text and its level of detail. Put 0 if there is no descriptor related to the baseline music descriptors.
2. "Actionability Level": Score 1 to 5 for the clarity and actionability of the existing music descriptors for an audio-generation model.
3. "Coherence Music Level": Score 1 to 5 for how congruent the music descriptors define a melody, harmony, and rhythm.

Output format:
{
  "Cohesion Text Level": <int>,
  "Actionability Level": <int>,
  "Coherence Music Level": <int>
}
"""

def evaluate_reprompt(client: OpenAI, model_name: str, reprompt_text: str):
    if pd.isna(reprompt_text) or not str(reprompt_text).strip():
        return {"Cohesion Text Level": 0, "Actionability Level": 1, "Coherence Music Level": 1, "Overall Score": 0.67}

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": f"Reprompt text to evaluate:\n\n{reprompt_text}"}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        cohesion = int(data.get("Cohesion Text Level", 0))
        actionability = int(data.get("Actionability Level", 1))
        coherence = int(data.get("Coherence Music Level", 1))
        
        overall = (cohesion + actionability + coherence) / 3.0
        
        return {
            "Cohesion Text Level": cohesion,
            "Actionability Level": actionability,
            "Coherence Music Level": coherence,
            "Overall Score": round(overall, 2)
        }
    except Exception as e:
        print(f"Error evaluating reprompt: {e}")
        # Return default worst scores on failure
        return {"Cohesion Text Level": 0, "Actionability Level": 1, "Coherence Music Level": 1, "Overall Score": 0.67}

def main():
    parser = argparse.ArgumentParser(description="Evaluate reprompts using LLM Judge.")
    parser.add_argument("suffix", type=str, help="Ablation identifier suffix, e.g. R20260917_113439")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of items to sample, stratified by taste.")
    parser.add_argument("--model", type=str, default="gpt-5.6-luna", help="OpenAI model to use as the judge.")
    args = parser.parse_args()

    reprompts_dir = Path("data/ablations/reprompts")
    if not reprompts_dir.exists():
        print(f"Directory {reprompts_dir} not found.")
        return

    versions = ["V1", "V2", "V3", "V4"]
    files = {}
    for v in versions:
        pattern = f"*{v}*{args.suffix}.csv"
        matched = list(reprompts_dir.glob(pattern))
        if matched:
            files[v] = matched[0]
        else:
            print(f"Warning: No file found for {v}")

    if "V1" not in files:
        print("Error: V1 file is required to create the stratified sample.")
        return

    print("Loading datasets...")
    dfs = {v: pd.read_csv(f) for v, f in files.items()}

    # Check for taste column
    if "taste" not in dfs["V1"].columns:
        print("Error: 'taste' column not found in V1 file. Cannot stratify.")
        return
    if "id_prompt" not in dfs["V1"].columns:
        print("Error: 'id_prompt' column not found in V1 file.")
        return

    # Sample based on V1
    v1_df = dfs["V1"].dropna(subset=["prompt", "reprompt", "taste"])
    
    if args.sample_size and args.sample_size < len(v1_df):
        print(f"Stratified sampling {args.sample_size} items based on 'taste'...")
        # Try to stratify
        try:
            sampled_v1 = v1_df.groupby("taste", group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(len(x) / len(v1_df) * args.sample_size))), random_state=42),
                include_groups=False
            )
            # If rounding issues cause it to have fewer/more items, just adjust
            if len(sampled_v1) > args.sample_size:
                sampled_v1 = sampled_v1.sample(args.sample_size, random_state=42)
            elif len(sampled_v1) < args.sample_size:
                shortfall = args.sample_size - len(sampled_v1)
                remaining = v1_df[~v1_df.index.isin(sampled_v1.index)]
                if not remaining.empty:
                    sampled_v1 = pd.concat([sampled_v1, remaining.sample(min(shortfall, len(remaining)), random_state=42)])
        except ValueError:
            print("Stratified sampling failed (possibly due to small groups), falling back to random sampling.")
            sampled_v1 = v1_df.sample(args.sample_size, random_state=42)
    else:
        sampled_v1 = v1_df

    target_ids = sampled_v1["id_prompt"].tolist()
    print(f"Selected {len(target_ids)} items for evaluation.")

    # Filter all dataframes to use the same id_prompts
    for v in versions:
        if v in dfs:
            dfs[v] = dfs[v][dfs[v]["id_prompt"].isin(target_ids)].copy()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set. Please set it or load your .env file.")
        return

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    all_stats = {}

    for v in versions:
        if v not in dfs:
            continue
        df = dfs[v]
        print(f"\nEvaluating {v} ({len(df)} items)...")
        
        cohesion_scores = []
        actionability_scores = []
        coherence_scores = []
        overall_scores = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            reprompt = row["reprompt"]
            res = evaluate_reprompt(client, args.model, reprompt)
            
            cohesion_scores.append(res["Cohesion Text Level"])
            actionability_scores.append(res["Actionability Level"])
            coherence_scores.append(res["Coherence Music Level"])
            overall_scores.append(res["Overall Score"])

        df["eval_cohesion"] = cohesion_scores
        df["eval_actionability"] = actionability_scores
        df["eval_coherence"] = coherence_scores
        df["eval_overall"] = overall_scores

        # Save eval CSV
        out_path = files[v].with_name(files[v].stem + "_eval.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {v} evaluation to {out_path.name}")

        all_stats[v] = {
            "Cohesion Mean": round(df["eval_cohesion"].mean(), 2),
            "Actionability Mean": round(df["eval_actionability"].mean(), 2),
            "Coherence Mean": round(df["eval_coherence"].mean(), 2),
            "Overall Mean": round(df["eval_overall"].mean(), 2)
        }

    # Generate Markdown Report
    report_path = reprompts_dir / f"evaluation_report_{args.suffix}.md"
    with open(report_path, "w") as f:
        f.write(f"# Reprompt Judge Evaluation Report\n")
        f.write(f"- **Suffix**: `{args.suffix}`\n")
        f.write(f"- **Model**: `{args.model}`\n")
        f.write(f"- **Sample Size**: {len(target_ids)}\n\n")
        
        f.write("## Aggregate Statistics\n\n")
        f.write("| Version | Cohesion (0-5) | Actionability (1-5) | Coherence (1-5) | Overall (Avg) |\n")
        f.write("|---------|----------------|---------------------|-----------------|---------------|\n")
        for v in versions:
            if v in all_stats:
                stats = all_stats[v]
                f.write(f"| **{v}** | {stats['Cohesion Mean']:.2f} | {stats['Actionability Mean']:.2f} | {stats['Coherence Mean']:.2f} | {stats['Overall Mean']:.2f} |\n")
    
    print(f"\nEvaluation complete. Markdown report saved to {report_path.name}")

if __name__ == "__main__":
    main()
