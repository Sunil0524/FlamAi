#!/usr/bin/env python3
"""
corrected_fertility.py — Fixed tokenizer fertility benchmark (v1)

Fixes from the audit of fertility.py (v0):
  1. Removed line.lower() — casing affects tokenization asymmetrically
  2. Changed split(" ") → split() — handles multiple spaces correctly
  3. Added multiple denominators: per-word, per-grapheme-cluster, per-UTF8-byte,
     per-sentence (for parallel corpora)
  4. Supports multiple tokenizers for comparison

Usage:
    python corrected_fertility.py \\
        --corpus eng=partA/corpus/eng.txt \\
        --corpus hin=partA/corpus/hin.txt \\
        --corpus kan=partA/corpus/kan.txt \\
        --corpus tam=partA/corpus/tam.txt \\
        --tokenizer gpt2 \\
        --tokenizer hf:xlm-roberta-base

Author: [your name] (v1, audited and corrected)
"""

import argparse
import csv
import os
import sys
import unicodedata
import regex  # pip install regex — supports \X for grapheme clusters


def load_tokenizer(spec: str):
    """Load a tokenizer by spec string."""
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken
        enc = tiktoken.get_encoding(spec)
        return enc.encode


def read_lines(path: str):
    """Read and NFC-normalize lines from a text file."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def count_grapheme_clusters(text: str) -> int:
    """Count extended grapheme clusters using Unicode \\X."""
    return len(regex.findall(r"\X", text))


def analyze_line(line: str, encode) -> dict:
    """Compute all metrics for a single line.

    Returns dict with:
      - n_tokens: number of tokens
      - n_words: number of whitespace words (via .split())
      - n_graphemes: number of extended grapheme clusters
      - n_utf8_bytes: length in UTF-8 bytes
      - n_chars: number of Unicode codepoints
    """
    # DO NOT lowercase — casing affects tokenization asymmetrically
    tokens = encode(line)
    words = line.split()  # FIX: use .split() not .split(" ")

    return {
        "n_tokens": len(tokens),
        "n_words": len(words),
        "n_graphemes": count_grapheme_clusters(line),
        "n_utf8_bytes": len(line.encode("utf-8")),
        "n_chars": len(line),
    }


def compute_corpus_metrics(lines, encode):
    """Compute aggregate metrics over all lines in a corpus."""
    per_line = [analyze_line(line, encode) for line in lines]

    totals = {k: sum(d[k] for d in per_line) for k in per_line[0]}

    # Fertility metrics (corpus-level: total tokens / total denominator)
    return {
        "tok_per_word": totals["n_tokens"] / totals["n_words"],
        "tok_per_grapheme": totals["n_tokens"] / totals["n_graphemes"],
        "tok_per_utf8_byte": totals["n_tokens"] / totals["n_utf8_bytes"],
        "tok_per_char": totals["n_tokens"] / totals["n_chars"],
        "tok_per_sentence": totals["n_tokens"] / len(lines),
        "avg_words_per_sentence": totals["n_words"] / len(lines),
        "total_tokens": totals["n_tokens"],
        "n_sentences": len(lines),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Corrected tokenizer fertility benchmark (v1)"
    )
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="language code and path, e.g. eng=corpus/eng.txt (repeatable)",
    )
    ap.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        metavar="SPEC",
        help="tokenizer spec: gpt2, hf:xlm-roberta-base, etc. (repeatable)",
    )
    ap.add_argument(
        "--csv-out",
        default=None,
        help="optional path to write results CSV",
    )
    args = ap.parse_args()

    # Parse corpus specs
    corpora = {}
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        corpora[lang] = read_lines(path)
        print(f"Loaded {lang}: {len(corpora[lang])} sentences from {path}")

    all_results = []

    for tok_spec in args.tokenizer:
        print(f"\n{'='*70}")
        print(f"Tokenizer: {tok_spec}")
        print(f"{'='*70}")

        encode = load_tokenizer(tok_spec)
        results = {}

        # Header
        print(
            f"{'lang':<6} {'tok/word':>10} {'tok/grph':>10} "
            f"{'tok/byte':>10} {'tok/char':>10} {'tok/sent':>10} "
            f"{'words/sent':>11}"
        )
        print("-" * 75)

        for lang in corpora:
            metrics = compute_corpus_metrics(corpora[lang], encode)
            results[lang] = metrics

            print(
                f"{lang:<6} {metrics['tok_per_word']:>10.3f} "
                f"{metrics['tok_per_grapheme']:>10.3f} "
                f"{metrics['tok_per_utf8_byte']:>10.3f} "
                f"{metrics['tok_per_char']:>10.3f} "
                f"{metrics['tok_per_sentence']:>10.1f} "
                f"{metrics['avg_words_per_sentence']:>11.1f}"
            )

            all_results.append({
                "tokenizer": tok_spec,
                "lang": lang,
                **metrics,
            })

        # Cross-language ratios (relative to first language)
        langs = list(results.keys())
        if len(langs) >= 2:
            base = langs[0]
            print(f"\nRatios relative to {base}:")
            print(
                f"{'lang':<6} {'tok/word':>10} {'tok/grph':>10} "
                f"{'tok/byte':>10} {'tok/sent':>10}"
            )
            print("-" * 50)
            for lang in langs:
                r_word = results[lang]["tok_per_word"] / results[base]["tok_per_word"]
                r_grph = (
                    results[lang]["tok_per_grapheme"]
                    / results[base]["tok_per_grapheme"]
                )
                r_byte = (
                    results[lang]["tok_per_utf8_byte"]
                    / results[base]["tok_per_utf8_byte"]
                )
                r_sent = (
                    results[lang]["tok_per_sentence"]
                    / results[base]["tok_per_sentence"]
                )
                print(
                    f"{lang:<6} {r_word:>10.2f}× "
                    f"{r_grph:>10.2f}× "
                    f"{r_byte:>10.2f}× "
                    f"{r_sent:>10.2f}×"
                )

            print(f"\n  KEY INSIGHT: tok/sentence holds meaning constant (parallel text).")
            print(f"  This is the metric that should drive routing-and-cost decisions,")
            print(f"  because it answers: 'how many more tokens does the same user")
            print(f"  request cost in language X vs. English?'")

    # Write CSV if requested
    if args.csv_out and all_results:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {args.csv_out}")


if __name__ == "__main__":
    main()
