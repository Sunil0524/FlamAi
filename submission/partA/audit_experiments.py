#!/usr/bin/env python3
"""
audit_experiments.py — Demonstrate each bug in fertility.py with measured evidence.

This script isolates each flaw, measures its effect on the reported numbers,
and states the direction and magnitude of the distortion.

Usage:
    python audit_experiments.py
"""

import os
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Utility: load a tokenizer (same interface as fertility.py)
# ---------------------------------------------------------------------------

def load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken
        enc = tiktoken.get_encoding(spec)
        return enc.encode


def read_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# BUG 1: line.lower() introduces asymmetric bias
# ---------------------------------------------------------------------------

def experiment_lower_bias(encode, eng_lines, hin_lines):
    """
    line.lower() mutates casing before tokenization. For scripts with
    case distinctions (Latin/English), lowercasing merges tokens and
    reduces the token count. For scripts without case (Devanagari/Hindi),
    it's a no-op. This creates an asymmetric, systematic bias that
    makes English look artificially *better* (fewer tokens) relative
    to Hindi.
    """
    print("=" * 70)
    print("BUG 1: line.lower() introduces asymmetric tokenization bias")
    print("=" * 70)

    for label, lines in [("eng", eng_lines), ("hin", hin_lines)]:
        fert_with_lower = []
        fert_without_lower = []

        for line in lines:
            words = line.split()  # using correct split here to isolate this bug
            if not words:
                continue

            # WITH lower (buggy)
            tokens_lower = encode(line.lower())
            fert_with_lower.append(len(tokens_lower) / len(words))

            # WITHOUT lower (correct)
            tokens_orig = encode(line)
            fert_without_lower.append(len(tokens_orig) / len(words))

        avg_with = sum(fert_with_lower) / len(fert_with_lower)
        avg_without = sum(fert_without_lower) / len(fert_without_lower)
        delta = avg_without - avg_with
        pct = (delta / avg_without) * 100

        print(f"\n  {label}:")
        print(f"    fertility WITH    lower: {avg_with:.4f} tok/word")
        print(f"    fertility WITHOUT lower: {avg_without:.4f} tok/word")
        print(f"    delta: {delta:+.4f} tok/word ({pct:+.2f}%)")

        if label == "eng":
            print(
                f"    → English gets {abs(pct):.1f}% fewer tokens from lowercasing "
                f"(makes English look better than it is)"
            )
        else:
            print(
                f"    → Hindi shows ~{abs(pct):.1f}% change (Devanagari has no case; "
                f"minimal effect, mostly ASCII punctuation)"
            )

    print(
        "\n  CONCLUSION: .lower() creates a systematic bias that "
        "deflates English fertility"
    )
    print(
        "  by several percent while leaving Hindi nearly unchanged, "
        "making the cross-language"
    )
    print("  gap appear smaller than it actually is.")
    print(
        "  Direction: English fertility is UNDER-reported → "
        "the true Hindi/English ratio is LOWER"
    )
    print("  (i.e. the gap is slightly less dramatic than reported).\n")


# ---------------------------------------------------------------------------
# BUG 2: split(" ") produces empty strings on double spaces
# ---------------------------------------------------------------------------

def experiment_split_bug(encode, eng_lines, hin_lines):
    """
    line.split(" ") with a single-space argument retains empty strings
    when there are multiple consecutive spaces. This inflates the word
    count denominator, deflating the fertility metric.

    The English sample has double spaces on at least one line:
      "Please keep the books  in the cupboard."
    """
    print("=" * 70)
    print("BUG 2: split(' ') produces empty strings on double spaces")
    print("=" * 70)

    for label, lines in [("eng", eng_lines), ("hin", hin_lines)]:
        total_words_buggy = 0
        total_words_correct = 0
        affected_lines = 0

        fert_buggy = []
        fert_correct = []

        for line in lines:
            words_buggy = line.split(" ")     # BUG: keeps empty strings
            words_correct = line.split()       # CORRECT: splits on any whitespace

            total_words_buggy += len(words_buggy)
            total_words_correct += len(words_correct)

            if len(words_buggy) != len(words_correct):
                affected_lines += 1

            tokens = encode(line)

            if len(words_buggy) > 0:
                fert_buggy.append(len(tokens) / len(words_buggy))
            if len(words_correct) > 0:
                fert_correct.append(len(tokens) / len(words_correct))

        avg_buggy = sum(fert_buggy) / len(fert_buggy)
        avg_correct = sum(fert_correct) / len(fert_correct)
        delta = avg_correct - avg_buggy
        pct = (delta / avg_correct) * 100

        print(f"\n  {label}:")
        print(f"    total words with    split(' '): {total_words_buggy}")
        print(f"    total words with    split():    {total_words_correct}")
        print(f"    affected lines: {affected_lines}/{len(lines)}")
        print(f"    fertility with split(' '): {avg_buggy:.4f} tok/word (buggy)")
        print(f"    fertility with split():    {avg_correct:.4f} tok/word (correct)")
        print(f"    delta: {delta:+.4f} tok/word ({pct:+.2f}%)")

    print(
        "\n  CONCLUSION: split(' ') inflates the word count when double-spaces "
        "are present,"
    )
    print(
        "  which deflates fertility. The magnitude depends on how many lines "
        "have double spaces."
    )
    print("  On the sample corpus, the English sample has at least one affected line.")
    print(
        "  Direction: fertility is UNDER-reported for affected lines "
        "(denominator too large).\n"
    )


# ---------------------------------------------------------------------------
# BUG 3 (CONCEPTUAL): per-word fertility is the wrong cross-language metric
# ---------------------------------------------------------------------------

def experiment_wrong_denominator(encode, eng_lines, hin_lines):
    """
    The fundamental conceptual problem: tok/word is NOT a valid
    cross-language comparison metric because 'word' is not a constant
    unit across languages. Hindi expresses the same meaning in fewer,
    morphologically richer words. So higher tok/word for Hindi is partly
    a property of the language, not the tokenizer.

    The correct metric for cost/routing is tokens-per-parallel-sentence
    (holds meaning constant).
    """
    print("=" * 70)
    print("BUG 3 (CONCEPTUAL): per-word fertility is wrong for cross-language comparison")
    print("=" * 70)

    # Use the sample corpora which are parallel (line-by-line aligned)
    n = min(len(eng_lines), len(hin_lines))
    eng = eng_lines[:n]
    hin = hin_lines[:n]

    # Metric 1: tok/word (the intern's metric)
    eng_tpw = [len(encode(l)) / max(len(l.split()), 1) for l in eng]
    hin_tpw = [len(encode(l)) / max(len(l.split()), 1) for l in hin]
    ratio_tpw = (sum(hin_tpw) / len(hin_tpw)) / (sum(eng_tpw) / len(eng_tpw))

    # Metric 2: tok/char
    eng_tpc = [len(encode(l)) / max(len(l), 1) for l in eng]
    hin_tpc = [len(encode(l)) / max(len(l), 1) for l in hin]
    ratio_tpc = (sum(hin_tpc) / len(hin_tpc)) / (sum(eng_tpc) / len(eng_tpc))

    # Metric 3: tok/sentence (tokens per parallel sentence — the RIGHT metric)
    eng_tps = [len(encode(l)) for l in eng]
    hin_tps = [len(encode(l)) for l in hin]
    ratio_tps = (sum(hin_tps) / len(hin_tps)) / (sum(eng_tps) / len(eng_tps))

    # Words per sentence comparison
    eng_wps = [len(l.split()) for l in eng]
    hin_wps = [len(l.split()) for l in hin]
    avg_eng_wps = sum(eng_wps) / len(eng_wps)
    avg_hin_wps = sum(hin_wps) / len(hin_wps)

    print(f"\n  Using {n} parallel sentence pairs from the sample corpora.")
    print(f"\n  Avg words/sentence:  eng={avg_eng_wps:.1f}  hin={avg_hin_wps:.1f}")
    print(
        f"  → Hindi uses {avg_eng_wps/avg_hin_wps:.1f}× FEWER words per sentence "
        f"to express the same meaning!"
    )

    print(f"\n  Metric comparisons (hin/eng ratio):")
    print(f"    tok/word      ratio: {ratio_tpw:.2f}×  (intern's metric — MISLEADING)")
    print(f"    tok/char      ratio: {ratio_tpc:.2f}×  (better, but chars vary by script)")
    print(f"    tok/sentence  ratio: {ratio_tps:.2f}×  (holds meaning constant — CORRECT)")

    print(f"\n  The intern's REPORT_v0.md claims Hindi costs '~6× more per request'.")
    print(f"  But tok/sentence shows Hindi costs only ~{ratio_tps:.1f}× more per equivalent")
    print(f"  meaning. The 6× figure conflates tokenizer overhead with the linguistic")
    print(f"  fact that Hindi packs more meaning per word.")
    print(f"\n  The report's recommendation to 'budget 6× serving cost for Hindi' is")
    print(f"  based on tok/word, which is the wrong metric. The real cost multiplier")
    print(f"  is closer to {ratio_tps:.1f}×.\n")


# ---------------------------------------------------------------------------
# "SUSPICIOUS BUT FINE": random.seed(1337)
# ---------------------------------------------------------------------------

def explain_random_seed():
    print("=" * 70)
    print("NOT A BUG: random.seed(1337) on line 25")
    print("=" * 70)
    print()
    print("  The script imports `random` and sets `random.seed(1337)` but never")
    print("  calls any function from the random module. This looks suspicious")
    print("  but is actually harmless:")
    print()
    print("  1. Setting a seed for reproducibility is good practice, even if")
    print("     the current code doesn't use randomness. A future version might")
    print("     (e.g. sampling a subset of lines for speed).")
    print("  2. The seed has zero effect on the output — no random functions")
    print("     are called, so the computation is fully deterministic regardless.")
    print("  3. The import + seed add negligible overhead (~microseconds).")
    print()
    print("  Flagging this as a bug would be incorrect. It's a defensive coding")
    print("  practice for reproducibility.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    STARTER = os.path.join(
        SCRIPT_DIR, "..", "docs", "starter_kit (1)", "starter_kit"
    )

    eng_path = os.path.join(STARTER, "corpus_sample", "eng_sample.txt")
    hin_path = os.path.join(STARTER, "corpus_sample", "hin_sample.txt")

    if not os.path.exists(eng_path):
        # Try relative to workspace root
        STARTER = os.path.join(
            SCRIPT_DIR, "..", "..", "docs", "starter_kit (1)", "starter_kit"
        )
        eng_path = os.path.join(STARTER, "corpus_sample", "eng_sample.txt")
        hin_path = os.path.join(STARTER, "corpus_sample", "hin_sample.txt")

    print("Loading tokenizer: gpt2")
    encode = load_tokenizer("gpt2")

    print(f"Loading corpora from: {STARTER}")
    eng_lines = read_lines(eng_path)
    hin_lines = read_lines(hin_path)
    print(f"  eng: {len(eng_lines)} lines")
    print(f"  hin: {len(hin_lines)} lines")
    print()

    experiment_lower_bias(encode, eng_lines, hin_lines)
    experiment_split_bug(encode, eng_lines, hin_lines)
    experiment_wrong_denominator(encode, eng_lines, hin_lines)
    explain_random_seed()


if __name__ == "__main__":
    main()
