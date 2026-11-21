"""Rank crossmodal dimensions by alignment quality to audio descriptors.

For each dimension, samples its descriptors, retrieves audio captions,
and measures how "musical" the results are.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from psycopg import sql
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models.descriptors.connection import get_conn
from models.descriptors.rag import get_top_k_audio_captions

# Music-related reference terms to measure alignment
MUSIC_REFERENCE_TERMS = [
    "melody", "rhythm", "harmony", "beat", "tempo", "bass", "guitar",
    "piano", "drums", "vocal", "song", "music", "acoustic", "tone",
    "chord", "orchestral", "instrumental", "genre", "jazz", "pop",
    "rock", "folk", "classical", "electronic", "ambient", "upbeat",
    "mellow", "soft", "loud", "cheerful", "nostalgic", "romantic",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
music_ref_embs = model.encode(MUSIC_REFERENCE_TERMS)

def musicality_score(captions: list[str]) -> float:
    """How close are retrieved captions to music vocabulary (0-1 scale)."""
    if not captions:
        return 0.0
    caption_embs = model.encode(captions)
    # Max similarity of each caption to any music reference term
    sim_matrix = cosine_similarity(caption_embs, music_ref_embs)
    max_sims = sim_matrix.max(axis=1)  # best music match per caption
    return float(np.mean(max_sims))


# Get distinct descriptors per dimension
conn = get_conn()
with conn.cursor() as cur:
    cur.execute("""
        SELECT dimension, descriptor, count(*) as cnt
        FROM crossmodal_food_embeddings
        GROUP BY dimension, descriptor
        ORDER BY dimension, cnt DESC
    """)
    dim_descriptors = defaultdict(list)
    for dim, desc, cnt in cur.fetchall():
        dim_descriptors[dim].append((desc, cnt))

print(f" CROSSMODAL DIMENSION → AUDIO DESCRIPTOR ALIGNMENT RANKING")
print(f"\nMusic reference vocabulary: {len(MUSIC_REFERENCE_TERMS)} terms")
print(f"Scoring: avg max-cosine-similarity of retrieved captions to music terms\n")

dimension_scores = {}

for dim, descriptors in sorted(dim_descriptors.items()):
    # Sample top-20 most common descriptors per dimension
    sample = [desc for desc, cnt in descriptors[:20]]
    
    all_musicality = []
    all_captions_retrieved = []
    example_good = None
    example_bad = None
    
    for desc in sample:
        emb = model.encode([desc])[0].tolist()
        results = get_top_k_audio_captions(emb, k=5, using_clap=False)
        captions = list(results.keys())
        sims = list(results.values())
        
        score = musicality_score(captions)
        all_musicality.append(score)
        all_captions_retrieved.append((desc, captions, sims, score))
        
        if example_good is None or score > example_good[1]:
            example_good = (desc, score, captions[:3])
        if example_bad is None or score < example_bad[1]:
            example_bad = (desc, score, captions[:3])
    
    avg_score = float(np.mean(all_musicality))
    std_score = float(np.std(all_musicality))
    dimension_scores[dim] = {
        "avg_musicality": avg_score,
        "std": std_score,
        "n_descriptors_tested": len(sample),
        "n_total_descriptors": len(descriptors),
        "best_example": example_good,
        "worst_example": example_bad,
    }

# Rank and print
ranked = sorted(dimension_scores.items(), key=lambda x: x[1]["avg_musicality"], reverse=True)

print(f"{'Rank':<5} {'Dimension':<20} {'Musicality':<12} {'Std':<8} {'Tested':<8} {'Total':<8}")

for rank, (dim, data) in enumerate(ranked, 1):
    print(f"{rank:<5} {dim:<20} {data['avg_musicality']:<12.4f} {data['std']:<8.4f} "
          f"{data['n_descriptors_tested']:<8} {data['n_total_descriptors']:<8}")

print(f"\nDetailed examples per dimension:\n")

for rank, (dim, data) in enumerate(ranked, 1):
    best = data["best_example"]
    worst = data["worst_example"]
    print(f"  {rank}. {dim} (avg={data['avg_musicality']:.4f})")
    print(f"     Best:  \"{best[0]}\" → {best[2]} (musicality={best[1]:.4f})")
    print(f"     Worst: \"{worst[0]}\" → {worst[2]} (musicality={worst[1]:.4f})")
    print()

# Save full dimension_scores dictionary to JSON
output_path = Path("data/ablations/ir_eval/dimension_scores.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dimension_scores, f, indent=2)
print(f"Saved dimension_scores to {output_path}")
