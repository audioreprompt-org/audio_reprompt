from pathlib import Path
from typing import Any
import csv, json

from config import PROJECT_ROOT, load_config, get_data_config, get_evaluation_config
from metrics.core.base import PromptRow, AudioItem, Metric
from metrics.implementation import CLAPMetric


def _read_prompts_csv(csv_path: Path) -> list[PromptRow]:
    rows: list[PromptRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("id"):
                continue
            rows.append(PromptRow(
                id=str(r["id"]).strip(),
                instrument=(r.get("instrument") or "").strip(),
                description=(r.get("description") or "").strip()
            ))
    return rows


def _match_audio_items(prompts: list[PromptRow], tracks_dir: Path) -> list[AudioItem]:
    out: list[AudioItem] = []
    for p in prompts:
        f = tracks_dir / f"{p.id}.wav"
        if f.exists():
            out.append(AudioItem(
                id=p.id, instrument=p.instrument, description=p.description, audio_path=str(f)
            ))
    return out


def run_from_config() -> dict[str, Any]:
    cfg = load_config()
    data_cfg = get_data_config()
    eval_cfg = get_evaluation_config()
    device = cfg.environment.device

    prompts_csv = PROJECT_ROOT / data_cfg.data_prompts_path / "spanio_prompts.csv"
    tracks_dir = PROJECT_ROOT / data_cfg.tracks_data_path
    scores_dir = PROJECT_ROOT / data_cfg.data_clap_path

    prompts = _read_prompts_csv(prompts_csv)
    audio_items = _match_audio_items(prompts, tracks_dir)

    registry: dict[str, Metric] = {
        "clap": CLAPMetric(),
    }
    results: dict[str, Any] = {}
    for name, metric_cfg in (eval_cfg.metrics or {}).items():
        metric = registry.get(name)
        if not metric:
            # Produce a small “skipped” stub so the combined JSON is consistent
            stub = {"metric": name, "skipped": True, "reason": "not implemented"}
            (scores_dir / f"{name}_summary.json").parent.mkdir(parents=True, exist_ok=True)
            with open(scores_dir / f"{name}_summary.json", "w", encoding="utf-8") as f:
                json.dump(stub, f, indent=2, ensure_ascii=False)
            results[name] = stub
            continue

        res = metric.run(
            prompts=prompts,
            audio_items=audio_items,
            metric_cfg=metric_cfg or {},
            device=device,
            scores_dir=scores_dir
        )
        results[name] = res

    # Combined
    combined = {"device": device, "metrics": results}
    scores_dir.mkdir(parents=True, exist_ok=True)
    with open(scores_dir / "combined_summary.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    return combined
