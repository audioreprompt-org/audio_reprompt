import argparse
import pandas as pd
import numpy as np
from collections import Counter
import math
import sys
import os

try:
    import nltk
    from nltk.tokenize import word_tokenize
    # Asegurar que se tengan las utilidades de tokenización
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    print("ERROR: Falta el paquete 'nltk'. Por favor instálalo con: pip install nltk")
    sys.exit(1)

try:
    import textstat
except ImportError:
    print("ERROR: Falta el paquete 'textstat'. Por favor instálalo con: pip install textstat")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("ERROR: Faltan paquetes de embeddings. Por favor instala: pip install sentence-transformers scikit-learn")
    sys.exit(1)


def calculate_ttr(text):
    """Calcula el Type-Token Ratio (Diversidad Léxica)."""
    words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def compute_entropy(text, n=2):
    """Calcula la Entropía empírica de Shannon sobre n-gramas, sustituto de Perplexity."""
    words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    if len(words) < n:
        return 0.0
    
    ngrams = zip(*[words[i:] for i in range(n)])
    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())
    
    entropy = 0.0
    for count in ngram_counts.values():
        prob = count / total_ngrams
        entropy -= prob * math.log2(prob)
        
    return entropy

def evaluate_csv(csv_path, source_col=None, target_col='reprompt', sample_n=None, output_dir=None):
    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró el archivo {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Auto-detección de columnas si no se especifican con precisión
    possible_sources = ['prompt', 'raw_prompt', 'sentence', 'text']
    possible_targets = ['reprompt', 'reprompt_sentence', 'generated_text', 'output', 'text']
    
    if not source_col or source_col not in df.columns:
        source_col = next((c for c in possible_sources if c in df.columns), None)
    if not target_col or target_col not in df.columns:
        target_col = next((c for c in possible_targets if c in df.columns and c != source_col), None)
        
    if not source_col or not target_col:
        print(f"ERROR: No se pudieron identificar las columnas de texto en el CSV.")
        print(f"Columnas disponibles: {list(df.columns)}")
        return None
        
    print(f"Evaluando métricas NLP...")
    print(f"CSV: {csv_path}")
    print(f"Comparando: '{source_col}' -> '{target_col}'")
    
    if sample_n and sample_n < len(df):
        df = df.sample(sample_n, random_state=42)
        
    raw_texts = df[source_col].fillna("").astype(str).tolist()
    reprompt_texts = df[target_col].fillna("").astype(str).tolist()
    
    # 1. Longitud promedio
    avg_len_raw = np.mean([len(str(x).split()) for x in raw_texts])
    avg_len_rep = np.mean([len(str(x).split()) for x in reprompt_texts])
    
    # 2. Diversidad Léxica (TTR) Mide Si el LLM "colapsó" creativamente
    ttr_rep = np.mean([calculate_ttr(t) for t in reprompt_texts])
    
    # 3. Entropía (Sustituto de perplexity)
    entropy_rep = np.mean([compute_entropy(t, n=2) for t in reprompt_texts])
    
    # 4. Readability / Complejidad del Prompt (Flesch Reading Ease)
    flesch_rep = np.mean([textstat.flesch_reading_ease(t) for t in reprompt_texts])
    
    # 5. Semantic Similarity (all-MiniLM-L6-v2 es ultra rápido y se carga de local cache)
    print("\nCargando mini-modelo para similitud Coseno (Solo CPU)...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    raw_embeddings = model.encode(raw_texts, show_progress_bar=False)
    rep_embeddings = model.encode(reprompt_texts, show_progress_bar=False)
    
    # Coseno individual y luego promedio
    similarities = [cosine_similarity([r], [g])[0][0] for r, g in zip(raw_embeddings, rep_embeddings)]
    avg_cosine = np.mean(similarities)
    
    print("\n" + "="*50)
    print(" RESULTADOS NLP (Ligeros - CPU-only)")
    print("="*50)
    print(f"Total evaluados: {len(df)} pares")
    print(f"1. Longitud promedio Prompts Crudos: {avg_len_raw:.1f} palabras")
    print(f"   Longitud promedio Reprompts:    {avg_len_rep:.1f} palabras")
    print(f"2. Diversidad Léxica (TTR):        {ttr_rep:.3f} (Más alto = Mayor variedad de vocabulario)")
    print(f"3. Entropía de Shannon (Bigramas): {entropy_rep:.2f} bits (Equivale a 'Perplexity' empírica)")
    print(f"4. Legibilidad (Flesch Score):     {flesch_rep:.1f} (Ideal > 50 para MusicGen)")
    print(f"5. Consistencia Semántica (Cos):   {avg_cosine:.3f} (Coseno texto crudo vs reprompt)")

    # 6. Desglose opcional por taste si está presente en el CSV (típico en B1, B2, B3)
    taste_breakdown = {}
    if 'taste' in df.columns:
        print("\n--- Desglose por Taste ---")
        df['_sim'] = similarities
        for taste_val, group in df.groupby('taste'):
            grp_reps = group[target_col].fillna("").astype(str).tolist()
            grp_ttr = np.mean([calculate_ttr(t) for t in grp_reps]) if grp_reps else 0.0
            grp_cos = float(group['_sim'].mean())
            taste_breakdown[taste_val] = {
                "count": len(group),
                "ttr": grp_ttr,
                "coseno": grp_cos,
            }
            print(f"  [{taste_val:7s}] n={len(group):2d} | TTR={grp_ttr:.3f} | Cos={grp_cos:.3f}")
        df.drop(columns=['_sim'], inplace=True)

    print("="*50)

    results = {
        "total_pares": len(df),
        "long_raw": avg_len_raw,
        "long_rep": avg_len_rep,
        "ttr": ttr_rep,
        "entropia": entropy_rep,
        "flesch": flesch_rep,
        "coseno": avg_cosine,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_name = os.path.basename(csv_path)
        tag = csv_name.replace("pipeline_results_", "").replace(".csv", "")
        
        md_file = os.path.join(output_dir, f"nlp_eval_{tag}.md")
        csv_file = os.path.join(output_dir, f"nlp_eval_{tag}.csv")
        
        # Guardar en CSV
        pd.DataFrame([results]).to_csv(csv_file, index=False)
        
        # Escribir reporte en markdown
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# Evaluación NLP para: {tag}\n\n")
            f.write(f"- **Archivo Origen**: `{csv_name}`\n")
            f.write(f"- **Pares Evaluados**: `{len(df)}`\n\n")
            f.write("## Métricas Calculadas\n\n")
            f.write(f"- **Promedio de Palabras**: `{avg_len_raw:.1f}` (Crudo) $\\rightarrow$ `{avg_len_rep:.1f}` (Reprompt)\n")
            f.write(f"- **Diversidad Léxica (TTR)**: `{ttr_rep:.3f}` *(Mayor indica vocabulario más rico)*\n")
            f.write(f"- **Entropía de Bigramas**: `{entropy_rep:.2f} bits` *(Mide variedad/sorpresa sintáctica)*\n")
            f.write(f"- **Legibilidad Flesch**: `{flesch_rep:.1f}` *(Modelos TTS como MusicGen prefieren valores $>50$)*\n")
            f.write(f"- **Similitud Coseno (MiniLM)**: `{avg_cosine:.3f}` *(Cuánto retuvo de la intención original)*\n\n")

            if taste_breakdown:
                f.write("## Desglose por Taste\n\n")
                f.write("| Taste | Muestras | TTR | Coseno MiniLM |\n")
                f.write("| :--- | :--- | :--- | :--- |\n")
                for taste_val, stats_t in taste_breakdown.items():
                    f.write(f"| {taste_val} | {stats_t['count']} | {stats_t['ttr']:.3f} | {stats_t['coseno']:.3f} |\n")
                f.write("\n")
            
        print(f"\n  ✓ Reporte guardado en: {md_file}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalúa calidad y consistencia NLP de los reprompts (diversidad léxica, entropía, legibilidad, similitud)."
    )
    parser.add_argument("csv_path", help="Ruta al archivo CSV con reprompts (ej. data/ablations/reprompts/pipeline_results_XXX.csv)")
    parser.add_argument("--source", type=str, default=None, help="Columna del prompt original (default: auto-detecta 'prompt' o 'sentence')")
    parser.add_argument("--target", type=str, default="reprompt", help="Columna del prompt generado/modificado")
    parser.add_argument("--sample", type=int, default=None, help="Evaluar solo un sampleo N aleatorio para test")
    parser.add_argument("--output-dir", type=str, default=None, help="Directorio para guardar los reportes")
    
    args = parser.parse_args()
    evaluate_csv(args.csv_path, source_col=args.source, target_col=args.target, sample_n=args.sample, output_dir=args.output_dir)
