from pathlib import Path
from typing import Dict, Any, List
import csv, json
import numpy as np

from metrics.core.base import Metric, PromptRow, AudioItem
from metrics import clap as clap_lib
from config import PROJECT_ROOT


class CLAPMetric(Metric):
    name = "clap"

    def run(
        self,
        prompts: List[PromptRow],
        audio_items: List[AudioItem],
        metric_cfg: Dict[str, Any],
        device: str,
        scores_dir: Path
    ) -> Dict[str, Any]:
        out_csv_name = metric_cfg.get("output_csv_name", "results_with_clap.csv")

        # Filter to items that actually exist on disk (engine should ensure this already)
        items = [
            clap_lib.CLAPItem(
                id=it.id, description=it.description, audio_path=it.audio_path, instrument=it.instrument
            )
            for it in audio_items
        ]

        scored = clap_lib.score(items, device=device)

        # Persist CSV (flat, DVC-friendly)
        out_csv = scores_dir / out_csv_name
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "instrument", "description", "audio_path", "clap_score"])
            for s in scored:
                w.writerow([s.id, s.instrument, s.description, s.audio_path, f"{s.clap_score:.6f}"])

        # Simple stats by flavor
        by_flavor = {}
        for s in scored:
            by_flavor.setdefault(s.instrument or "unknown", []).append(s.clap_score)
        per_flavor = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": int(len(v))}
            for k, v in by_flavor.items()
        }

        # Summary JSON
        summary = {
            "metric": self.name,
            "device": device,
            "total_scored": len(scored),
            "csv_path": str(out_csv.relative_to(PROJECT_ROOT)),
            "per_flavor": per_flavor,
        }

        out_json = scores_dir / f"{self.name}_summary.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary
