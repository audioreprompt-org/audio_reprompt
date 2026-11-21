import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import models.ablation_analysis as ablation_analysis
import models.ablation_runner as runner
import models.validate as validate
from models.ablation_runner import (
    ALL_ABLATIONS,
    EXIT_BASELINE_MISSING,
    EXIT_OK,
    ensure_raw_baseline,
    prepare_raw_ablations_csv,
    resolve_targets,
    run_full_pipeline,
    run_nlp_phase,
)
from models.scripts.eval_reprompt_nlp import (
    calculate_mattr,
    calculate_musical_vocab,
    compute_corpus_distinct_n,
    compute_self_bleu,
    evaluate_csv,
)


def test_group_default_excludes_aa_control():
    assert list(resolve_targets("B3")) == ["B3a", "B3b", "B3c", "B3d", "B3e", "B3f"]


def test_subset_keeps_requested_order_and_dedupes():
    assert list(resolve_targets("B3", "B3f, B3a,B3f")) == ["B3f", "B3a"]


def test_preset_core():
    assert list(resolve_targets("B3", "core")) == ["B3a", "B3b", "B3c", "B3f"]


def test_preset_with_aa_control_is_identical_to_b3b():
    targets = resolve_targets("B3", "core+aa")
    assert "B3g" in targets
    a, b = targets["B3b"], targets["B3g"]
    assert (a.prompt_version, a.system_prompt_modifier, a.reasoning_effort, a.filter_dimensions) == (
        b.prompt_version, b.system_prompt_modifier, b.reasoning_effort, b.filter_dimensions
    )
    assert a.tag != b.tag


@pytest.mark.parametrize(
    "experiment, variants",
    [("B3", "B2a"), ("B3", "B3z"), ("B3", ","), ("B2", "core"), ("B9", None)],
)
def test_invalid_selection_raises(experiment, variants):
    with pytest.raises(ValueError):
        resolve_targets(experiment, variants)


def test_single_variant_experiment_unchanged():
    assert list(resolve_targets("B1c")) == ["B1c"]
    assert set(resolve_targets("all")) == {k for k, v in ALL_ABLATIONS.items() if v.in_group}


def test_prepare_raw_ablations_csv_skips_existing_and_stages_missing(tmp_path, monkeypatch):
    import models.ablation_runner as runner
    import pandas as pd

    raw_audio_dir = tmp_path / "raw_ablations"
    reprompts_dir = tmp_path / "reprompts"
    raw_audio_dir.mkdir()
    reprompts_dir.mkdir()

    # Prompt 1 already has a wav, Prompt 2 does not
    (raw_audio_dir / "1.wav").write_bytes(b"RIFF")

    monkeypatch.setattr(runner, "RAW_ABLATIONS_AUDIO_PATH", raw_audio_dir)
    monkeypatch.setattr(runner, "ABLATIONS_REPROMPTS_PATH", reprompts_dir)
    monkeypatch.setattr(
        runner,
        "load_all_raw_prompts",
        lambda: [
            {"id_prompt": "1", "sentence": "Prompt one", "taste": "sweet"},
            {"id_prompt": "2", "sentence": "Prompt two", "taste": "bitter"},
        ],
    )

    csv_path = prepare_raw_ablations_csv()
    assert csv_path is not None
    df = pd.read_csv(csv_path)
    assert list(df["id_prompt"].astype(str)) == ["2"]
    assert list(df["reprompt"]) == ["Prompt two"]

    # Once 2.wav also exists, prepare_raw_ablations_csv returns None (no-op)
    (raw_audio_dir / "2.wav").write_bytes(b"RIFF")
    assert prepare_raw_ablations_csv() is None


def test_nlp_diversity_metrics():
    text = (
        "A bright acoustic guitar leads a syncopated melody in major key with warm timbre "
        "and legato strings harmony over a steady piano rhythm."
    )
    mattr = calculate_mattr(text, window=10)
    assert 0.8 <= mattr <= 1.0
    mus_count, mus_density = calculate_musical_vocab(text)
    assert mus_count >= 8
    assert mus_density > 0.3

    diverse_corpus = [
        "Acoustic guitar plays a fast upbeat major melody with bright percussion and flute.",
        "Dark cello and ambient synth pads sustain a slow minor harmony with deep reverb.",
    ]
    collapsed_corpus = [
        "Acoustic guitar plays a fast upbeat major melody with bright percussion and flute.",
        "Acoustic guitar plays a fast upbeat major melody with bright percussion and flute.",
    ]
    assert compute_corpus_distinct_n(diverse_corpus, n=2) > compute_corpus_distinct_n(collapsed_corpus, n=2)
    assert compute_self_bleu(diverse_corpus) < compute_self_bleu(collapsed_corpus)


def test_stratified_sample_deterministic_across_file_and_row_order(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    tastes = ["sweet", "sour", "salty", "bitter"]
    rows_a = [
        {"id_prompt": str(i + 1), "taste": tastes[i % 4], "sentence": f"Prompt {i + 1}"}
        for i in range(12)
    ]
    rows_b = [
        {"id_prompt": str(i + 13), "taste": tastes[i % 4], "sentence": f"Prompt {i + 13}"}
        for i in range(12)
    ]

    pd.DataFrame(rows_a).to_csv(prompts_dir / "a.csv", index=False)
    pd.DataFrame(rows_b).to_csv(prompts_dir / "b.csv", index=False)
    monkeypatch.setattr(validate, "FOOD_PROMPTS_DIR", prompts_dir)

    ids_1 = validate.compute_sample_ids(sample_size=8, seed=42)
    hash_1 = validate.compute_sample_hash(ids_1)

    # Shuffle file names and reverse row order inside each CSV
    (prompts_dir / "a.csv").unlink()
    (prompts_dir / "b.csv").unlink()
    pd.DataFrame(list(reversed(rows_b))).to_csv(prompts_dir / "z_part1.csv", index=False)
    pd.DataFrame(list(reversed(rows_a))).to_csv(prompts_dir / "m_part2.csv", index=False)

    ids_2 = validate.compute_sample_ids(sample_size=8, seed=42)
    hash_2 = validate.compute_sample_hash(ids_2)
    assert ids_1 == ids_2
    assert hash_1 == hash_2

    ids_other_seed = validate.compute_sample_ids(sample_size=8, seed=99)
    assert ids_other_seed != ids_1

    manifest_path, manifest_hash = validate.write_sample_manifest(
        tmp_path / "analysis",
        sample_size=8,
        seed=42,
        ids=list(reversed(ids_1)),
        run_id="R20261119_000001",
        group="B2",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["size"] == len(ids_1)
    assert manifest["ids"] == ids_1
    assert manifest["sha1"] == manifest_hash == hash_1


def test_eval_reprompt_nlp_returns_backends(tmp_path, monkeypatch):
    import models.scripts.eval_reprompt_nlp as nlp_mod

    monkeypatch.setattr(nlp_mod, "_get_embedding_model", lambda: None)

    csv_path = tmp_path / "pipeline_results_B2a_filter_default_R20261119_000001.csv"
    pd.DataFrame(
        [
            {
                "id_prompt": "1",
                "prompt": "Sweet chocolate cake at a sunny cafe.",
                "taste": "sweet",
                "reprompt": "Bright acoustic guitar melody in major key with warm piano harmony.",
            },
            {
                "id_prompt": "2",
                "prompt": "Sour lemon tart in a quiet kitchen.",
                "taste": "sour",
                "reprompt": "Crisp flute arpeggios and staccato strings rhythm at upbeat tempo.",
            },
        ]
    ).to_csv(csv_path, index=False)

    out_dir = tmp_path / "nlp_out"
    res = evaluate_csv(str(csv_path), output_dir=str(out_dir))
    assert res["backends"]["encoder"] == "TF-IDF"
    assert res["backends"]["tokenizer"] in ("punkt", "regex")
    assert res["backends"]["flesch"] in ("cmudict", "heuristic")

    eval_df = pd.read_csv(out_dir / "nlp_eval_B2a_filter_default_R20261119_000001.csv")
    assert eval_df.loc[0, "backend_encoder"] == "TF-IDF"


def test_run_nlp_phase_non_fatal_and_full_pipeline_exit_ok(tmp_path, monkeypatch):
    import models.kaggle_runner as kaggle_runner
    import models.scripts.eval_reprompt_nlp as nlp_mod

    run_id = "R20261119_000010"
    ablations_dir = tmp_path / "ablations"
    reprompts_dir = ablations_dir / "reprompts"
    scores_dir = ablations_dir / "scores"
    analysis_dir = ablations_dir / "analysis"
    audio_dir = ablations_dir / "audio"
    raw_audio_dir = tmp_path / "raw_ablations"
    for d in (reprompts_dir, scores_dir, analysis_dir, audio_dir, raw_audio_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(runner, "ABLATIONS_PATH", ablations_dir)
    monkeypatch.setattr(runner, "ABLATIONS_REPROMPTS_PATH", reprompts_dir)
    monkeypatch.setattr(runner, "ABLATIONS_SCORES_PATH", scores_dir)
    monkeypatch.setattr(runner, "ABLATIONS_AUDIO_PATH", audio_dir)
    monkeypatch.setattr(runner, "RAW_ABLATIONS_AUDIO_PATH", raw_audio_dir)
    monkeypatch.setattr(runner, "ABLATION_STATE_DIR", tmp_path / ".ablation_state")

    monkeypatch.setattr(ablation_analysis, "ABLATIONS_SCORES", scores_dir)
    monkeypatch.setattr(ablation_analysis, "ABLATIONS_REPROMPTS", reprompts_dir)
    monkeypatch.setattr(ablation_analysis, "ABLATIONS_ANALYSIS", analysis_dir)

    sample_ids = ["1", "2", "3", "4"]
    monkeypatch.setattr(runner, "compute_sample_ids", lambda size, seed: sample_ids)
    monkeypatch.setattr(runner, "ensure_raw_baseline", lambda ids, generate=False, scope="sample": (True, None, 0))

    def fake_run_reprompt(cfg, sample_size, seed, run_id=""):
        tag = runner._resolve_tag(cfg, cfg.models[0], run_id)
        p = reprompts_dir / f"pipeline_results_gpt_5.6_luna_4_prompt_V3_{tag}.csv"
        pd.DataFrame(
            [
                {
                    "id_prompt": pid,
                    "prompt": f"Raw prompt {pid}",
                    "taste": ["sweet", "sour", "salty", "bitter"][idx],
                    "reprompt": f"Melodic guitar and piano rhythm in major key {pid}",
                }
                for idx, pid in enumerate(sample_ids)
            ]
        ).to_csv(p, index=False)
        return [str(p)]

    def fake_run_score(csv_paths, audio_dir=None):
        for p in csv_paths:
            tag = Path(p).name.replace("pipeline_results_", "").rsplit(".csv", 1)[0]
            pd.DataFrame(
                [{"id_prompt": pid, "text": f"Rep {pid}", "audio": f"{pid}.wav", "clap_score": 0.30 + idx * 0.02} for idx, pid in enumerate(sample_ids)]
            ).to_csv(scores_dir / f"clap_score_results_reprompt_outputs_{tag}.csv", index=False)
            pd.DataFrame(
                [{"id_prompt": pid, "text": f"Raw {pid}", "audio": f"raw_ablations/{pid}.wav", "clap_score": 0.15 + idx * 0.01, "source": "raw_ablations"} for idx, pid in enumerate(sample_ids)]
            ).to_csv(scores_dir / f"clap_score_results_prompt_outputs_{tag}_raw.csv", index=False)
            pd.DataFrame(
                [{"id_prompt": pid, "text": f"Raw {pid}", "audio": f"{pid}.wav", "clap_score": 0.18 + idx * 0.01} for idx, pid in enumerate(sample_ids)]
            ).to_csv(scores_dir / f"clap_score_results_prompt_outputs_{tag}_cross.csv", index=False)

    monkeypatch.setattr(runner, "run_reprompt", fake_run_reprompt)
    monkeypatch.setattr(kaggle_runner, "run_pipeline", lambda **kwargs: None)
    monkeypatch.setattr(runner, "run_score", fake_run_score)
    monkeypatch.setattr(nlp_mod, "_get_embedding_model", lambda: None)

    orig_eval = nlp_mod.evaluate_csv

    def flaky_eval(csv_path, output_dir=None, **kwargs):
        if "B2b" in Path(csv_path).name:
            raise RuntimeError("forced NLP error")
        return orig_eval(csv_path, output_dir=output_dir, **kwargs)

    monkeypatch.setattr(nlp_mod, "evaluate_csv", flaky_eval)

    targets = resolve_targets("B2")
    rc = run_full_pipeline(targets, sample_size=4, seed=42, group_prefix="B2", run_id=run_id)
    assert rc == EXIT_OK

    status_path = analysis_dir / f"nlp_{run_id}" / "nlp_status.json"
    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    b2a_key = next(k for k in status_data if "B2a" in k)
    b2b_key = next(k for k in status_data if "B2b" in k)
    assert status_data[b2a_key]["status"] == "ok"
    assert status_data[b2b_key]["status"] == "failed"
    assert "forced NLP error" in status_data[b2b_key]["error"]

    report_md = (analysis_dir / f"B2_{run_id}" / "hallazgos_B2.md").read_text(encoding="utf-8")
    assert "> [!WARNING]" in report_md
    assert "— (NLP falló: forced NLP error)" in report_md


def test_ensure_raw_baseline_preflight_cache_and_granular_pull(tmp_path, monkeypatch):
    raw_audio_dir = tmp_path / "raw_ablations"
    reprompts_dir = tmp_path / "reprompts"
    scores_dir = tmp_path / "scores"
    for d in (raw_audio_dir, reprompts_dir, scores_dir):
        d.mkdir(parents=True, exist_ok=True)
    cache_csv = scores_dir / "clap_score_results_raw_prompts_baseline.csv"

    all_rows = [
        {"id_prompt": "1", "sentence": "Prompt 1", "taste": "sweet"},
        {"id_prompt": "2", "sentence": "Prompt 2", "taste": "sour"},
        {"id_prompt": "3", "sentence": "Prompt 3", "taste": "salty"},
    ]
    monkeypatch.setattr(runner, "RAW_ABLATIONS_AUDIO_PATH", raw_audio_dir)
    monkeypatch.setattr(runner, "ABLATIONS_REPROMPTS_PATH", reprompts_dir)
    monkeypatch.setattr(runner, "ABLATIONS_RAW_BASELINE_SCORE_CSV", cache_csv)
    monkeypatch.setattr(runner, "load_all_raw_prompts", lambda: all_rows)
    monkeypatch.setattr(validate, "RAW_ABLATIONS_AUDIO_PATH", raw_audio_dir)
    monkeypatch.setattr(validate, "ABLATIONS_REPROMPTS_PATH", reprompts_dir)
    monkeypatch.setattr(validate, "load_all_raw_prompts", lambda: all_rows)

    pull_calls = []

    def fake_run(cmd, **kwargs):
        pull_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="not found")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    # 1. Missing baseline without generate -> fail-fast
    ok, staging, n_gen = ensure_raw_baseline(["1", "2"], generate=False, scope="sample")
    assert (ok, staging, n_gen) == (False, None, 0)
    assert pull_calls == [[
        "dvc", "pull", "data/tracks/raw_ablations/1.wav", "data/tracks/raw_ablations/2.wav"
    ]]

    # 2. Cached baseline scores cover sample -> 0 DVC pulls, no staging
    pull_calls.clear()
    pd.DataFrame(
        [
            {"id_prompt": "1", "text": "Prompt 1", "audio": "1.wav", "clap_score": 0.21, "source": "raw_ablations", "wav_size": "", "wav_mtime": ""},
            {"id_prompt": "2", "text": "Prompt 2", "audio": "2.wav", "clap_score": 0.22, "source": "raw_ablations", "wav_size": "", "wav_mtime": ""},
        ]
    ).to_csv(cache_csv, index=False)
    ok, staging, n_gen = ensure_raw_baseline(["1", "2"], generate=False, scope="sample")
    assert (ok, staging, n_gen) == (True, None, 0)
    assert pull_calls == []

    # 3. Partial cache with generate=True: scope=sample stages only missing sampled ID ("3" not in sample -> only uncached sample ID)
    ok, staging, n_gen = ensure_raw_baseline(["1", "3"], generate=True, scope="sample")
    assert ok is True
    assert staging is not None
    assert n_gen == 1
    assert pull_calls[-1] == ["dvc", "pull", "data/tracks/raw_ablations/3.wav"]
    staged_df = pd.read_csv(staging)
    assert list(staged_df["id_prompt"].astype(str)) == ["3"]

    # 4. scope=all checks all raw prompts and stages only uncached/missing ones
    ok_all, staging_all, n_gen_all = ensure_raw_baseline(["1"], generate=True, scope="all")
    assert ok_all is True
    assert staging_all is not None
    assert n_gen_all == 1
    assert list(pd.read_csv(staging_all)["id_prompt"].astype(str)) == ["3"]


def test_run_full_pipeline_fails_fast_when_baseline_missing(tmp_path, monkeypatch):
    ablations_dir = tmp_path / "ablations"
    monkeypatch.setattr(runner, "ABLATIONS_PATH", ablations_dir)
    monkeypatch.setattr(runner, "compute_sample_ids", lambda size, seed: ["1", "2"])
    monkeypatch.setattr(runner, "ensure_raw_baseline", lambda ids, generate=False, scope="sample": (False, None, 0))
    reprompt_mock = MagicMock()
    monkeypatch.setattr(runner, "run_reprompt", reprompt_mock)

    rc = run_full_pipeline(resolve_targets("B2"), sample_size=2, seed=42, group_prefix="B2", run_id="R20261119_000099")
    assert rc == EXIT_BASELINE_MISSING
    reprompt_mock.assert_not_called()


def test_validate_raw_baseline_cache_invalidation_and_mixed_ids(tmp_path, monkeypatch):
    raw_audio_dir = tmp_path / "raw_ablations"
    scores_dir = tmp_path / "scores"
    raw_audio_dir.mkdir()
    scores_dir.mkdir()
    cache_csv = scores_dir / "clap_score_results_raw_prompts_baseline.csv"

    (raw_audio_dir / "10.wav").write_bytes(b"RIFF_10")
    (raw_audio_dir / "custom_a.wav").write_bytes(b"RIFF_A")

    reprompt_csv = tmp_path / "pipeline_results_test.csv"
    pd.DataFrame(
        [
            {"id_prompt": "10", "prompt": "P10", "taste": "sweet", "reprompt": "R10"},
            {"id_prompt": "custom_a", "prompt": "PA", "taste": "sour", "reprompt": "RA"},
        ]
    ).to_csv(reprompt_csv, index=False)

    calls = []

    class FakeClap:
        def calculate_scores(self, items):
            calls.append([it.id for it in items])
            return [SimpleNamespace(item=it, clap_score=0.25) for it in items]

    monkeypatch.setattr(validate, "clap_model", FakeClap())
    monkeypatch.setattr(validate, "ABLATIONS_RAW_BASELINE_SCORE_CSV", cache_csv)

    # First run scores both mixed int/str IDs without TypeError and sets source="raw_ablations"
    validate.calculate_clap_score_alignment(
        str(reprompt_csv),
        using_raw_prompts=True,
        optional_suffix="tag1_raw",
        output_dir=str(scores_dir),
        audio_dir=str(raw_audio_dir),
    )
    assert calls == [["10", "custom_a"]]
    out_raw = pd.read_csv(scores_dir / "clap_score_results_prompt_outputs_tag1_raw.csv")
    assert list(out_raw["source"]) == ["raw_ablations", "raw_ablations"]

    # Second run hits cache (0 CLAP calls)
    validate.calculate_clap_score_alignment(
        str(reprompt_csv),
        using_raw_prompts=True,
        optional_suffix="tag1_raw",
        output_dir=str(scores_dir),
        audio_dir=str(raw_audio_dir),
    )
    assert len(calls) == 1

    # Modify custom_a.wav size/mtime -> invalidates only custom_a
    (raw_audio_dir / "custom_a.wav").write_bytes(b"RIFF_A_REGENERATED")
    validate.calculate_clap_score_alignment(
        str(reprompt_csv),
        using_raw_prompts=True,
        optional_suffix="tag1_raw",
        output_dir=str(scores_dir),
        audio_dir=str(raw_audio_dir),
    )
    assert calls[-1] == ["custom_a"]

    # Missing wav and no valid cache -> raises FileNotFoundError
    (raw_audio_dir / "10.wav").unlink()
    cache_csv.unlink()
    with pytest.raises(FileNotFoundError):
        validate.calculate_clap_score_alignment(
            str(reprompt_csv),
            using_raw_prompts=True,
            optional_suffix="tag1_raw",
            output_dir=str(scores_dir),
            audio_dir=str(raw_audio_dir),
        )


def test_ablation_analysis_enforcement_new_schema_and_nan_holm(tmp_path, monkeypatch):
    scores_dir = tmp_path / "scores"
    reprompts_dir = tmp_path / "reprompts"
    analysis_dir = tmp_path / "analysis"
    for d in (scores_dir, reprompts_dir, analysis_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ablation_analysis, "ABLATIONS_SCORES", scores_dir)
    monkeypatch.setattr(ablation_analysis, "ABLATIONS_REPROMPTS", reprompts_dir)
    monkeypatch.setattr(ablation_analysis, "ABLATIONS_ANALYSIS", analysis_dir)

    run_id = "R20261119_000002"
    nlp_dir = analysis_dir / f"nlp_{run_id}"
    nlp_dir.mkdir(parents=True, exist_ok=True)

    ids = ["1", "2", "3", "4"]
    tastes = ["sweet", "sour", "salty", "bitter"]

    # Valid variants B2a and B2b with all 3 CSVs and NLP results
    for v_idx, tag_base in enumerate(("B2a_filter_default", "B2b_nofilter")):
        tag = f"gpt_5.6_luna_4_prompt_V3_{tag_base}_{run_id}"
        pd.DataFrame(
            [
                {"id_prompt": pid, "prompt": f"P{pid}", "taste": t, "reprompt": f"Rep {pid}"}
                for pid, t in zip(ids, tastes)
            ]
        ).to_csv(reprompts_dir / f"pipeline_results_{tag}.csv", index=False)
        pd.DataFrame(
            [
                {"id_prompt": pid, "text": f"Rep {pid}", "audio": f"{pid}.wav", "clap_score": 0.30 + 0.02 * i + 0.03 * v_idx}
                for i, pid in enumerate(ids)
            ]
        ).to_csv(scores_dir / f"clap_score_results_reprompt_outputs_{tag}.csv", index=False)
        pd.DataFrame(
            [
                {"id_prompt": pid, "text": f"P{pid}", "audio": f"raw_ablations/{pid}.wav", "clap_score": 0.12 + 0.01 * i, "source": "raw_ablations"}
                for i, pid in enumerate(ids)
            ]
        ).to_csv(scores_dir / f"clap_score_results_prompt_outputs_{tag}_raw.csv", index=False)
        pd.DataFrame(
            [
                {"id_prompt": pid, "text": f"P{pid}", "audio": f"{pid}.wav", "clap_score": 0.16 + 0.01 * i + 0.01 * v_idx}
                for i, pid in enumerate(ids)
            ]
        ).to_csv(scores_dir / f"clap_score_results_prompt_outputs_{tag}_cross.csv", index=False)
        pd.DataFrame(
            [
                {
                    "mattr": 0.8 + 0.02 * v_idx,
                    "corpus_distinct_1": 0.7,
                    "corpus_distinct_2": 0.9,
                    "self_bleu": 0.2,
                    "inter_prompt_cos_dist": 0.5,
                    "musical_vocab_density": 0.4,
                    "coseno": 0.3,
                    "long_rep": 20.0,
                    "ttr": 0.85,
                    "entropia": 4.0,
                    "flesch": 60.0,
                    "backend_encoder": "MiniLM",
                    "backend_tokenizer": "punkt",
                    "backend_flesch": "cmudict",
                }
            ]
        ).to_csv(nlp_dir / f"nlp_eval_{tag}.csv", index=False)
        pd.DataFrame(
            [
                {
                    "id_prompt": pid,
                    "taste": t,
                    "ttr": 0.85,
                    "mattr": 0.8 + 0.02 * v_idx + 0.01 * i,
                    "bigram_entropy": 4.0,
                    "flesch": 60.0,
                    "musical_vocab_count": 5,
                    "musical_vocab_density": 0.4 + 0.01 * i,
                    "semantic_sim_raw_rep": 0.3 + 0.01 * i,
                    "distinctiveness_cos_dist": 0.5 + 0.01 * i,
                }
                for i, (pid, t) in enumerate(zip(ids, tastes))
            ]
        ).to_csv(nlp_dir / f"nlp_rows_{tag}.csv", index=False)

    # Invalid variant B2c missing source=="raw_ablations" (legacy fallback must NOT be used)
    bad_tag = f"gpt_5.6_luna_4_prompt_V3_B2c_legacy_{run_id}"
    pd.DataFrame(
        [{"id_prompt": pid, "text": f"Rep {pid}", "audio": f"{pid}.wav", "clap_score": 0.25} for pid in ids]
    ).to_csv(scores_dir / f"clap_score_results_reprompt_outputs_{bad_tag}.csv", index=False)
    pd.DataFrame(
        [{"id_prompt": pid, "text": f"P{pid}", "audio": f"{pid}.wav", "clap_score": 0.15} for pid in ids]
    ).to_csv(scores_dir / f"clap_score_results_prompt_outputs_{bad_tag}_raw.csv", index=False)
    pd.DataFrame(
        [{"id_prompt": pid, "text": f"P{pid}", "audio": f"{pid}.wav", "clap_score": 0.15} for pid in ids]
    ).to_csv(scores_dir / f"clap_score_results_prompt_outputs_{bad_tag}_cross.csv", index=False)

    ablation_analysis.analyse_group("B2", run_id=run_id)

    out_dir = analysis_dir / f"B2_{run_id}"
    summary_df = pd.read_csv(out_dir / "summary_B2.csv")
    assert len(summary_df) == 2
    for col in (
        "mean_raw_baseline",
        "mean_reprompt",
        "mean_cross",
        "delta_rep_vs_raw",
        "p_rep_vs_raw",
        "dz_rep_vs_raw",
        "delta_cross_vs_raw",
        "p_cross_vs_raw",
        "dz_cross_vs_raw",
        "sample_hash",
    ):
        assert col in summary_df.columns
    assert summary_df["sample_hash"].iloc[0] == validate.compute_sample_hash(ids)

    report = (out_dir / "hallazgos_B2.md").read_text(encoding="utf-8")
    assert "## Variantes excluidas" in report
    assert bad_tag in report
    assert "## 0. Matriz de Decisión Conjunta" in report
    assert "### 2A. reprompt vs baseline (texto y audio distintos)" in report
    assert "### 2B. cross vs baseline (mismo texto)" in report
    assert "sobre el mismo audio" not in report

    # NaN-safe Holm adjustment
    adjusted = ablation_analysis._holm_adjust(pd.Series([0.01, np.nan, 0.04]))
    assert np.isclose(adjusted.iloc[0], 0.02)
    assert np.isnan(adjusted.iloc[1])
    assert np.isclose(adjusted.iloc[2], 0.04)
