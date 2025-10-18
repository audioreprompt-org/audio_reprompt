import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from config import PROJECT_ROOT, get_data_config
from metrics.clap import CLAPItem, calculate_scores, available_backends


def _load_real_items(max_items: int = 24) -> list[CLAPItem]:
    data_cfg = get_data_config()
    prompts_csv = PROJECT_ROOT / data_cfg.data_prompts_path / "spanio_prompts.csv"
    tracks_dir = PROJECT_ROOT / data_cfg.tracks_data_path
    if not prompts_csv.exists() or not tracks_dir.exists():
        return []
    items: list[CLAPItem] = []
    with open(prompts_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rid = str(r.get("id") or "").strip()
            if not rid:
                continue
            wav_path = tracks_dir / f"{rid}.wav"
            if not wav_path.exists():
                continue
            desc = str(r.get("description") or "").strip()
            instr = str(r.get("instrument") or "").strip() or None
            items.append(CLAPItem(id=rid, description=desc, audio_path=str(wav_path), instrument=instr))
            if len(items) >= max_items:
                break
    return items


def _score_backend_multiple(
    backend: str,
    items: list[CLAPItem],
    device: str,
    n_runs: int,
) -> np.ndarray:
    runs: list[np.ndarray] = []
    for _ in range(n_runs):
        scored = calculate_scores(items, device=device, backend=backend)
        vals = np.array([s.clap_score for s in scored], dtype=np.float64)
        runs.append(vals)
    return np.stack(runs, axis=0)  # (n_runs, n_items)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return _pearson(ra.astype(np.float64), rb.astype(np.float64))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, header: list[str], rows: Iterable[Iterable[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


@pytest.mark.integration
def test_compare_backends_with_real_audio_exports() -> None:
    max_items = int(os.getenv("CLAP_TEST_MAX_ITEMS", "24"))
    n_runs = int(os.getenv("CLAP_TEST_RUNS", "3"))
    device = os.getenv("CLAP_TEST_DEVICE", "cpu")

    try:
        data_cfg = get_data_config()
        base_scores = PROJECT_ROOT / data_cfg.data_clap_path
    except Exception:
        base_scores = PROJECT_ROOT / "data" / "scores"
    out_dir = base_scores / "method_comparison"
    _ensure_dir(out_dir)

    items = _load_real_items(max_items=max_items)
    if len(items) < 3:
        print("SKIP: Not enough real items with matching .wav files found via config.")
        return

    preferred = ["hf_processor", "lass", "laion_module"]
    avail = [b for b in preferred if b in set(available_backends())]
    if len(avail) < 2:
        print(f"SKIP: Need at least 2 CLAP backends available. Found: {avail}")
        return

    print("\n=== CLAP Backend Comparison (real audio) ===")
    print(f"Items: {len(items)} | Runs per backend: {n_runs} | Device: {device}")
    print(f"Backends considered: {avail}")
    print(f"Outputs -> {out_dir}")

    # Save items metadata
    _write_csv(
        out_dir / "items.csv",
        ["idx", "id", "instrument", "description", "audio_path"],
        ((i, it.id, it.instrument or "unknown", it.description, it.audio_path) for i, it in enumerate(items)),
    )

    # Scores (long format) and per-backend summaries
    scores_long_rows: list[tuple[str, int, str, str, float]] = []
    per_backend_means: dict[str, np.ndarray] = {}
    per_backend_stds: dict[str, np.ndarray] = {}
    per_backend_summary_rows: list[tuple[str, int, float, float, float, float, float]] = []

    matrices: dict[str, np.ndarray] = {}
    for be in avail:
        try:
            mat = _score_backend_multiple(be, items, device=device, n_runs=n_runs)  # (n_runs, n_items)
        except Exception as e:
            print(f"[{be}] failed to compute scores: {e}")
            continue

        matrices[be] = mat
        # Long rows
        for run_idx in range(mat.shape[0]):
            for item_idx, it in enumerate(items):
                val = float(mat[run_idx, item_idx])
                scores_long_rows.append((be, run_idx, it.id, it.instrument or "unknown", val))

        # Per-item mean/std across runs
        per_item_mean = mat.mean(axis=0)
        per_item_std = mat.std(axis=0)
        per_backend_means[be] = per_item_mean
        per_backend_stds[be] = per_item_std

        # Overall summaries of per-item means
        overall_mean = float(per_item_mean.mean())
        overall_std = float(per_item_mean.std())
        avg_run_std = float(per_item_std.mean())
        min_v = float(per_item_mean.min())
        max_v = float(per_item_mean.max())

        per_backend_summary_rows.append(
            (be, len(items), overall_mean, overall_std, avg_run_std, min_v, max_v)
        )

    # If fewer than 2 succeeded, finish with what we produced
    if len(matrices) < 2:
        print(f"Only {len(matrices)} backend(s) produced scores. Exporting partial results.")
        if scores_long_rows:
            _write_csv(out_dir / "scores_long.csv", ["backend", "run", "item_id", "instrument", "score"], scores_long_rows)
        if per_backend_summary_rows:
            _write_csv(
                out_dir / "per_backend_summary.csv",
                ["backend", "n_items", "mean_of_means", "std_of_means", "avg_run_std", "min_mean", "max_mean"],
                per_backend_summary_rows,
            )
        return

    # Write combined scores and backend summaries
    _write_csv(out_dir / "scores_long.csv", ["backend", "run", "item_id", "instrument", "score"], scores_long_rows)
    _write_csv(
        out_dir / "per_backend_summary.csv",
        ["backend", "n_items", "mean_of_means", "std_of_means", "avg_run_std", "min_mean", "max_mean"],
        per_backend_summary_rows,
    )

    # Pairwise comparisons on per-item means
    pair_rows: list[tuple[str, str, float, float, float, float]] = []
    names = list(matrices.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a = per_backend_means[a_name]
            b = per_backend_means[b_name]
            pear = _pearson(a, b)
            spear = _spearman(a, b)
            mae = float(np.mean(np.abs(a - b)))
            rmse = float(np.sqrt(np.mean((a - b) ** 2)))
            pair_rows.append((a_name, b_name, pear, spear, mae, rmse))
    _write_csv(out_dir / "pairwise_summary.csv", ["backend_a", "backend_b", "pearson", "spearman", "mae", "rmse"], pair_rows)

    # Group differences (first pair for brevity)
    group_rows: list[tuple[str, str, str, int, float, float]] = []
    if len(names) >= 2:
        a_name, b_name = names[0], names[1]
        a_mean = per_backend_means[a_name]
        b_mean = per_backend_means[b_name]
        groups: dict[str, list[int]] = {}
        for idx, it in enumerate(items):
            groups.setdefault(it.instrument or "unknown", []).append(idx)
        for g, idxs in groups.items():
            idxs_arr = np.array(idxs, dtype=int)
            if idxs_arr.size == 0:
                continue
            diff = a_mean[idxs_arr] - b_mean[idxs_arr]
            g_mae = float(np.mean(np.abs(diff)))
            g_rmse = float(np.sqrt(np.mean(diff**2)))
            g_n = int(idxs_arr.size)
            group_rows.append((a_name, b_name, g, g_n, g_mae, g_rmse))
        _write_csv(
            out_dir / f"group_diffs_{a_name}_vs_{b_name}.csv",
            ["backend_a", "backend_b", "group", "n", "mae", "rmse"],
            group_rows,
        )

    # Plots (optional)
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        # Box plot of per-item means per backend
        box_fig = go.Figure()
        for be, means in per_backend_means.items():
            box_fig.add_trace(go.Box(y=means.tolist(), name=be, boxmean=True))
        box_fig.update_layout(
            title="CLAP per-item mean score distribution by backend",
            xaxis_title="Backend",
            yaxis_title="CLAP score (mean across runs)",
            template="plotly_white",
        )
        pio.write_html(box_fig, file=str(out_dir / "box_backend_means.html"), include_plotlyjs="cdn", auto_open=False)

        # Scatter plots per pair (per-item means)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_name, b_name = names[i], names[j]
                a = per_backend_means[a_name]
                b = per_backend_means[b_name]
                ids = [it.id for it in items]

                min_v = float(min(a.min(), b.min()))
                max_v = float(max(a.max(), b.max()))
                pad = 0.05 * (max_v - min_v + 1e-6)
                lo, hi = min_v - pad, max_v + pad

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=a.tolist(), y=b.tolist(), mode="markers", text=ids, name="items"))
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="x=y"))
                fig.update_layout(
                    title=f"Per-item mean scores: {a_name} vs {b_name}",
                    xaxis_title=f"{a_name} mean score",
                    yaxis_title=f"{b_name} mean score",
                    template="plotly_white",
                )
                pio.write_html(
                    fig,
                    file=str(out_dir / f"scatter_{a_name}_vs_{b_name}.html"),
                    include_plotlyjs="cdn",
                    auto_open=False,
                )
    except Exception as e:
        print(f"(plots skipped) plotly error: {e}")

    print("Done. CSVs and plots (if available) written to:", out_dir)
