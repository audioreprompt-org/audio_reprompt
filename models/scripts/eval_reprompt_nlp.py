import argparse
from collections import Counter
import math
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError as exc:
    raise ImportError("Falta el paquete 'nltk'. Por favor instálalo con: pip install nltk") from exc

try:
    import textstat
except ImportError as exc:
    raise ImportError("Falta el paquete 'textstat'. Por favor instálalo con: pip install textstat") from exc

try:
    nltk.data.find("corpora/cmudict")
    _HAS_CMUDICT = True
except LookupError:
    _HAS_CMUDICT = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as exc:
    raise ImportError(
        "Faltan paquetes de embeddings. Por favor instala: "
        "pip install sentence-transformers scikit-learn"
    ) from exc


_EMBEDDING_MODEL: SentenceTransformer | None = None

# Curated lexicon of musical instructions expected by text-to-music models (MusicGen / CLAP)
MUSICAL_LEXICON: frozenset[str] = frozenset(
    {
        # Tempo / rhythm / articulation
        "tempo", "bpm", "rhythm", "rhythmic", "beat", "beats", "pulse", "groove", "syncopated",
        "syncopation", "staccato", "legato", "portato", "pizzicato", "arpeggio", "arpeggios",
        "arpeggiated", "ostinato", "swing", "upbeat", "downtempo", "midtempo", "fast", "slow",
        "andante", "allegro", "adagio", "moderato", "presto", "lento", "rubato", "cadence",
        # Pitch / harmony / tonality
        "melody", "melodic", "harmony", "harmonic", "chord", "chords", "major", "minor",
        "scale", "modal", "dissonant", "dissonance", "consonant", "consonance", "tonal",
        "tonality", "key", "octave", "pitch", "pitched", "register", "treble", "bass",
        "baritone", "alto", "soprano", "counterpoint", "progression", "voicing", "voicings",
        "triad", "seventh", "pentatonic", "chromatic", "modulation", "interval",
        # Dynamics / timbre / texture
        "timbre", "timbral", "dynamics", "dynamic", "crescendo", "decrescendo", "forte",
        "piano", "mezzo", "dolce", "swell", "swells", "sustained", "sustain", "resonance",
        "resonant", "reverb", "reverberant", "acoustic", "electric", "electronic", "synth",
        "synthesizer", "pad", "pads", "ambient", "orchestral", "orchestration", "instrumental",
        "polyphonic", "monophonic", "homophonic", "overtones", "vibrato", "tremolo",
        # Instruments
        "guitar", "piano", "violin", "viola", "cello", "strings", "flute", "clarinet",
        "oboe", "saxophone", "trumpet", "horn", "trombone", "brass", "woodwind", "woodwinds",
        "percussion", "drums", "drum", "marimba", "xylophone", "vibraphone", "glockenspiel",
        "kalimba", "harp", "mandolin", "ukulele", "harmonium", "accordion", "organ", "chimes",
        "bell", "bells", "rhodes", "celesta", "harpsichord", "sitar", "banjo", "bongo",
        "conga", "shaker", "cymbal", "hihat", "snare", "kick", "tambourine", "bassline",
    }
)


def _get_embedding_model() -> SentenceTransformer | None:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print("\nCargando mini-modelo para similitud Coseno (Solo CPU)...")
        try:
            _EMBEDDING_MODEL = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu", local_files_only=True
            )
        except Exception:
            try:
                _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            except Exception as e:
                print(
                    f"  ⚠ No se pudo cargar SentenceTransformer('all-MiniLM-L6-v2') ({e}); "
                    "usando vectorizador TF-IDF de caracteres/palabras como fallback local.",
                    file=sys.stderr,
                )
                _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def _encode_texts_with_backend(texts: list[str]) -> tuple[np.ndarray, str]:
    """Encode texts with SentenceTransformer if available, else L2-normalized TF-IDF."""
    model = _get_embedding_model()
    if model is not None:
        return np.asarray(model.encode(texts, show_progress_bar=False)), "MiniLM"
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    return vec.fit_transform(texts).toarray(), "TF-IDF"


def _encode_texts(texts: list[str]) -> np.ndarray:
    """Encode texts with SentenceTransformer if available, else L2-normalized TF-IDF."""
    embeddings, _ = _encode_texts_with_backend(texts)
    return embeddings


def _detect_backends(encoder_name: str | None = None) -> dict[str, str]:
    """Return the active backends for encoder, tokenizer, and Flesch readability."""
    if encoder_name is None:
        encoder_name = "MiniLM" if _get_embedding_model() is not None else "TF-IDF"
    try:
        word_tokenize("test")
        tokenizer_name = "punkt"
    except LookupError:
        tokenizer_name = "regex"

    flesch_name = "heuristic"
    if _HAS_CMUDICT:
        try:
            textstat.flesch_reading_ease("test")
            flesch_name = "cmudict"
        except LookupError:
            flesch_name = "heuristic"

    return {
        "encoder": encoder_name,
        "tokenizer": tokenizer_name,
        "flesch": flesch_name,
    }


def _count_syllables_en(word: str) -> int:
    """Lightweight heuristic English syllable counter (offline fallback when cmudict is absent)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    groups = re.findall(r"[aeiouy]{1,2}", w)
    return max(1, len(groups))


def _safe_flesch_reading_ease(text: str) -> float:
    """Compute Flesch Reading Ease using textstat if cmudict is present, else offline formula."""
    if _HAS_CMUDICT:
        try:
            return float(textstat.flesch_reading_ease(text))
        except LookupError:
            pass
    words = _alpha_tokens(text)
    if not words:
        return 0.0
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sents = max(1, len(sentences))
    n_words = len(words)
    n_syllables = sum(_count_syllables_en(w) for w in words)
    return float(206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_syllables / n_words))


def _alpha_tokens(text: str) -> list[str]:
    """Tokenize into lowercase alphabetic words with regex fallback if punkt is unavailable."""
    try:
        return [w.lower() for w in word_tokenize(text) if w.isalpha()]
    except LookupError:
        return re.findall(r"[a-zA-Z]+", text.lower())


def calculate_ttr(text: str) -> float:
    """Calcula el Type-Token Ratio (Diversidad Léxica intra-prompt)."""
    words = _alpha_tokens(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def calculate_mattr(text: str, window: int = 15) -> float:
    """Calcula el Moving-Average Type-Token Ratio (MATTR), invariante a la longitud."""
    words = _alpha_tokens(text)
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    n_windows = len(words) - window + 1
    ttrs = [len(set(words[i : i + window])) / window for i in range(n_windows)]
    return float(np.mean(ttrs))


def compute_entropy(text: str, n: int = 2) -> float:
    """Calcula la Entropía empírica de Shannon sobre n-gramas, sustituto de Perplexity."""
    words = _alpha_tokens(text)
    if len(words) < n:
        return 0.0

    ngrams = list(zip(*[words[i:] for i in range(n)]))
    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())

    entropy = 0.0
    for count in ngram_counts.values():
        prob = count / total_ngrams
        entropy -= prob * math.log2(prob)

    return entropy


def calculate_musical_vocab(text: str) -> tuple[int, float]:
    """Calcula el conteo y densidad de vocabulario musical en un reprompt."""
    words = _alpha_tokens(text)
    if not words:
        return 0, 0.0
    count = sum(1 for w in words if w in MUSICAL_LEXICON)
    return count, count / len(words)


def compute_corpus_distinct_n(texts: list[str], n: int = 2) -> float:
    """Calcula Distinct-N a nivel de corpus (proporción de n-gramas únicos entre todos los reprompts)."""
    all_ngrams: list[tuple[str, ...]] = []
    for t in texts:
        words = _alpha_tokens(t)
        if len(words) >= n:
            all_ngrams.extend(zip(*[words[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_self_bleu(texts: list[str]) -> float:
    """Calcula Self-BLEU-4 promedio del corpus (menor valor = mayor diversidad inter-prompt)."""
    tokenized = [_alpha_tokens(t) for t in texts]
    valid = [toks for toks in tokenized if len(toks) >= 4]
    if len(valid) < 2:
        return 0.0
    smoother = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)
    scores: list[float] = []
    for i, hyp in enumerate(valid):
        refs = [valid[j] for j in range(len(valid)) if j != i]
        scores.append(
            float(sentence_bleu(refs, hyp, weights=weights, smoothing_function=smoother))
        )
    return float(np.mean(scores))


def evaluate_csv(
    csv_path: str,
    source_col: str | None = None,
    target_col: str = "reprompt",
    sample_n: int | None = None,
    output_dir: str | None = None,
) -> dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo {csv_path}")

    df = pd.read_csv(csv_path)

    possible_sources = ["prompt", "raw_prompt", "sentence", "text"]
    possible_targets = ["reprompt", "reprompt_sentence", "generated_text", "output", "text"]

    if not source_col or source_col not in df.columns:
        source_col = next((c for c in possible_sources if c in df.columns), None)
    if not target_col or target_col not in df.columns:
        target_col = next(
            (c for c in possible_targets if c in df.columns and c != source_col), None
        )

    if not source_col or not target_col:
        raise ValueError(
            f"No se pudieron identificar las columnas de texto en {csv_path}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    print("Evaluando métricas NLP...")
    print(f"CSV: {csv_path}")
    print(f"Comparando: '{source_col}' -> '{target_col}'")

    if sample_n and sample_n < len(df):
        df = df.sample(sample_n, random_state=42).copy()

    raw_texts = df[source_col].fillna("").astype(str).tolist()
    reprompt_texts = df[target_col].fillna("").astype(str).tolist()

    # Row-level lexical & readability metrics
    word_counts_raw = [len(str(x).split()) for x in raw_texts]
    word_counts_rep = [len(str(x).split()) for x in reprompt_texts]
    ttr_list = [calculate_ttr(t) for t in reprompt_texts]
    mattr_list = [calculate_mattr(t, window=15) for t in reprompt_texts]
    entropy_list = [compute_entropy(t, n=2) for t in reprompt_texts]
    flesch_list = [_safe_flesch_reading_ease(t) for t in reprompt_texts]
    mus_tuples = [calculate_musical_vocab(t) for t in reprompt_texts]
    mus_count_list = [c for c, _ in mus_tuples]
    mus_density_list = [d for _, d in mus_tuples]

    # Embeddings & semantic similarity / inter-prompt distinctiveness
    combined_embeddings, encoder_backend = _encode_texts_with_backend(raw_texts + reprompt_texts)
    backends = _detect_backends(encoder_name=encoder_backend)
    n_rows = len(reprompt_texts)
    raw_embeddings = combined_embeddings[:n_rows]
    rep_embeddings = combined_embeddings[n_rows:]

    # Row-level cosine similarity (raw <-> reprompt)
    similarities = [
        float(cosine_similarity([r], [g])[0][0])
        for r, g in zip(raw_embeddings, rep_embeddings)
    ]

    # Inter-prompt pairwise cosine distance matrix (1 - cosine_similarity)
    rep_sim_matrix = cosine_similarity(rep_embeddings)
    n_rows = len(reprompt_texts)
    if n_rows > 1:
        # For each prompt i, mean cosine distance (1 - cos) to all other N-1 reprompts
        distinctiveness_list = [
            float(1.0 - (np.sum(rep_sim_matrix[i]) - rep_sim_matrix[i, i]) / (n_rows - 1))
            for i in range(n_rows)
        ]
        triu_idx = np.triu_indices(n_rows, k=1)
        inter_prompt_cos_dist = float(np.mean(1.0 - rep_sim_matrix[triu_idx]))
    else:
        distinctiveness_list = [0.0] * n_rows
        inter_prompt_cos_dist = 0.0

    # Corpus-level diversity (Distinct-1, Distinct-2, Self-BLEU-4)
    corpus_distinct_1 = compute_corpus_distinct_n(reprompt_texts, n=1)
    corpus_distinct_2 = compute_corpus_distinct_n(reprompt_texts, n=2)
    self_bleu = compute_self_bleu(reprompt_texts)

    # Aggregate means
    avg_len_raw = float(np.mean(word_counts_raw))
    avg_len_rep = float(np.mean(word_counts_rep))
    ttr_rep = float(np.mean(ttr_list))
    mattr_rep = float(np.mean(mattr_list))
    entropy_rep = float(np.mean(entropy_list))
    flesch_rep = float(np.mean(flesch_list))
    mus_count_rep = float(np.mean(mus_count_list))
    mus_density_rep = float(np.mean(mus_density_list))
    avg_cosine = float(np.mean(similarities))

    print("\n" + "=" * 60)
    print(" RESULTADOS NLP (Intra-Prompt + Diversidad Inter-Prompt)")
    print("=" * 60)
    print(f"Total evaluados: {len(df)} pares")
    print(f"1. Longitud promedio Prompts Crudos: {avg_len_raw:.1f} palabras")
    print(f"   Longitud promedio Reprompts:      {avg_len_rep:.1f} palabras")
    print(f"2. Diversidad Léxica (TTR / MATTR):  {ttr_rep:.3f} / {mattr_rep:.3f}")
    print(f"3. Entropía de Shannon (Bigramas):   {entropy_rep:.2f} bits")
    print(f"4. Legibilidad (Flesch Score):       {flesch_rep:.1f}")
    print(
        f"5. Vocabulario Musical:              {mus_count_rep:.1f} términos "
        f"({mus_density_rep * 100:.1f}% densidad)"
    )
    print(f"6. Consistencia Semántica (Cos):     {avg_cosine:.3f}")
    print(
        f"7. Diversidad Corpus (Distinct-1/2): {corpus_distinct_1:.3f} / {corpus_distinct_2:.3f}"
    )
    print(f"8. Distancia Coseno Inter-Prompt:    {inter_prompt_cos_dist:.3f} (mayor = menos colapso)")
    print(f"9. Self-BLEU-4 (Anti-colapso):       {self_bleu:.3f} (menor = mayor diversidad)")

    # Build row-level DataFrame for paired tests with CLAP
    rows_df = pd.DataFrame(
        {
            "id_prompt": (
                df["id_prompt"].astype(str).tolist()
                if "id_prompt" in df.columns
                else [str(i) for i in range(len(df))]
            ),
            "taste": (
                df["taste"].astype(str).tolist()
                if "taste" in df.columns
                else [""] * len(df)
            ),
            "word_count_raw": word_counts_raw,
            "word_count_reprompt": word_counts_rep,
            "ttr": ttr_list,
            "mattr": mattr_list,
            "bigram_entropy": entropy_list,
            "flesch": flesch_list,
            "musical_vocab_count": mus_count_list,
            "musical_vocab_density": mus_density_list,
            "semantic_sim_raw_rep": similarities,
            "distinctiveness_cos_dist": distinctiveness_list,
        }
    )

    taste_breakdown = {}
    if "taste" in df.columns:
        print("\n--- Desglose por Taste ---")
        for taste_val, group in rows_df.groupby("taste"):
            grp_ttr = float(group["ttr"].mean())
            grp_mattr = float(group["mattr"].mean())
            grp_cos = float(group["semantic_sim_raw_rep"].mean())
            grp_dist = float(group["distinctiveness_cos_dist"].mean())
            grp_mus = float(group["musical_vocab_density"].mean())
            taste_breakdown[taste_val] = {
                "count": len(group),
                "ttr": grp_ttr,
                "mattr": grp_mattr,
                "coseno": grp_cos,
                "distinctiveness": grp_dist,
                "musical_density": grp_mus,
            }
            print(
                f"  [{taste_val:7s}] n={len(group):2d} | TTR={grp_ttr:.3f} | "
                f"MATTR={grp_mattr:.3f} | Mus={grp_mus * 100:.1f}% | "
                f"Dist={grp_dist:.3f} | Cos={grp_cos:.3f}"
            )

    print("=" * 60)

    results = {
        "total_pares": len(df),
        "long_raw": avg_len_raw,
        "long_rep": avg_len_rep,
        "ttr": ttr_rep,
        "mattr": mattr_rep,
        "entropia": entropy_rep,
        "flesch": flesch_rep,
        "musical_vocab_count": mus_count_rep,
        "musical_vocab_density": mus_density_rep,
        "coseno": avg_cosine,
        "corpus_distinct_1": corpus_distinct_1,
        "corpus_distinct_2": corpus_distinct_2,
        "inter_prompt_cos_dist": inter_prompt_cos_dist,
        "self_bleu": self_bleu,
        "backend_encoder": backends["encoder"],
        "backend_tokenizer": backends["tokenizer"],
        "backend_flesch": backends["flesch"],
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_name = os.path.basename(csv_path)
        tag = csv_name.replace("pipeline_results_", "").replace(".csv", "")

        md_file = os.path.join(output_dir, f"nlp_eval_{tag}.md")
        csv_file = os.path.join(output_dir, f"nlp_eval_{tag}.csv")
        rows_csv_file = os.path.join(output_dir, f"nlp_rows_{tag}.csv")

        pd.DataFrame([results]).to_csv(csv_file, index=False)
        rows_df.to_csv(rows_csv_file, index=False)

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# Evaluación NLP para: {tag}\n\n")
            f.write(f"- **Archivo Origen**: `{csv_name}`\n")
            f.write(f"- **Pares Evaluados**: `{len(df)}`\n")
            f.write(
                f"- **Backends**: encoder=`{backends['encoder']}`, "
                f"tokenizer=`{backends['tokenizer']}`, flesch=`{backends['flesch']}`\n\n"
            )
            f.write("## Métricas Calculadas (Intra-Prompt & Diversidad de Corpus)\n\n")
            f.write(
                f"- **Promedio de Palabras**: `{avg_len_raw:.1f}` (Crudo) "
                f"$\\rightarrow$ `{avg_len_rep:.1f}` (Reprompt)\n"
            )
            f.write(
                f"- **Diversidad Léxica (TTR / MATTR-15)**: `{ttr_rep:.3f}` / `{mattr_rep:.3f}` "
                "*(MATTR corrige el sesgo por longitud)*\n"
            )
            f.write(
                f"- **Entropía de Bigramas**: `{entropy_rep:.2f} bits` "
                "*(Variedad/sorpresa sintáctica intra-prompt)*\n"
            )
            f.write(
                f"- **Legibilidad Flesch**: `{flesch_rep:.1f}` "
                "*(Modelos TTM como MusicGen prefieren valores $>50$)*\n"
            )
            f.write(
                f"- **Vocabulario Musical**: `{mus_count_rep:.1f}` términos/prompt "
                f"(`{mus_density_rep * 100:.1f}%` densidad)\n"
            )
            f.write(
                f"- **Similitud Coseno (MiniLM Crudo $\\leftrightarrow$ Reprompt)**: `{avg_cosine:.3f}`\n"
            )
            f.write(
                f"- **Diversidad de Corpus (Distinct-1 / Distinct-2)**: "
                f"`{corpus_distinct_1:.3f}` / `{corpus_distinct_2:.3f}`\n"
            )
            f.write(
                f"- **Distancia Coseno Inter-Prompt**: `{inter_prompt_cos_dist:.3f}` "
                "*(Mayor = menor colapso entre reprompts de distintos prompts)*\n"
            )
            f.write(
                f"- **Self-BLEU-4 (Colapso de Plantilla)**: `{self_bleu:.3f}` "
                "*(Menor = mayor diversidad estructural entre los $N$ reprompts)*\n\n"
            )

            if taste_breakdown:
                f.write("## Desglose por Taste\n\n")
                f.write(
                    "| Taste | Muestras | TTR | MATTR-15 | Densidad Musical | "
                    "Dist. Inter-Prompt | Coseno MiniLM |\n"
                )
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for taste_val, stats_t in taste_breakdown.items():
                    f.write(
                        f"| {taste_val} | {stats_t['count']} | {stats_t['ttr']:.3f} | "
                        f"{stats_t['mattr']:.3f} | {stats_t['musical_density'] * 100:.1f}% | "
                        f"{stats_t['distinctiveness']:.3f} | {stats_t['coseno']:.3f} |\n"
                    )
                f.write("\n")

        print(f"\n  ✓ Reporte guardado en: {md_file}")
        print(f"  ✓ Métricas por fila guardadas en: {rows_csv_file}")

    return {**results, "backends": backends}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa calidad y diversidad NLP de los reprompts "
            "(TTR, MATTR, entropía, vocabulario musical, Distinct-N, Self-BLEU, similitud)."
        )
    )
    parser.add_argument(
        "csv_path",
        help="Ruta al archivo CSV con reprompts (ej. data/ablations/reprompts/pipeline_results_XXX.csv)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Columna del prompt original (default: auto-detecta 'prompt' o 'sentence')",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="reprompt",
        help="Columna del prompt generado/modificado",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluar solo un sampleo N aleatorio para test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio para guardar los reportes",
    )

    args = parser.parse_args()
    try:
        evaluate_csv(
            args.csv_path,
            source_col=args.source,
            target_col=args.target,
            sample_n=args.sample,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
