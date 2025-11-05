from pathlib import Path
from typing import Any, Optional
import csv
import json
import numpy as np

from metrics.core.base import Metric, PromptRow, AudioItem
from metrics import clap as clap_lib
from config import PROJECT_ROOT


class CLAPMetric(Metric):
    name = "clap"

    def run(
            self,
            prompts: list[PromptRow],
            audio_items: list[AudioItem],
            metric_cfg: dict[str, Any],
            device: str,
            scores_dir: Path
    ) -> dict[str, Any]:

        csv_name = metric_cfg.get("output_csv_name", "results_with_clap.csv")
        backend: str = metric_cfg.get("backend", "laion_module")
        backend_cfg: Optional[dict[str, Any]] = metric_cfg.get("backend_cfg")

        items = [
            clap_lib.CLAPItem(
                id=it.id, description=it.description, audio_path=it.audio_path, instrument=it.instrument
            )
            for it in audio_items
        ]

        scored = clap_lib.calculate_scores(items, device=device, backend=backend, backend_cfg=backend_cfg)

        out_csv = scores_dir / csv_name
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "instrument", "description", "audio_path", "clap_score"])
            for s in scored:
                w.writerow([s.item.id, s.item.instrument, s.item.description, s.item.audio_path, f"{s.clap_score:.6f}"])

        by_flavor: dict[str, list[float]] = {}
        for s in scored:
            by_flavor.setdefault(s.item.instrument or "unknown", []).append(s.clap_score)
        per_flavor = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": int(len(v))}
            for k, v in by_flavor.items()
        }

        summary = {
            "metric": self.name,
            "device": device,
            "backend": backend,
            "backend_cfg": backend_cfg or {},
            "total_scored": len(scored),
            "csv_path": str(out_csv.relative_to(PROJECT_ROOT)),
            "per_flavor": per_flavor,
        }

        out_json = scores_dir / f"{self.name}_summary.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

