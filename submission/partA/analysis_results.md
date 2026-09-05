# A3 — Corrected Analysis Results

## Setup

- **Corpus**: 30 parallel sentences per language (topically aligned)
  - English (eng), Hindi (hin), Kannada (kan), Tamil (tam)
- **Tokenizers**:
  1. `gpt2` — English-centric BPE (50,257 vocab)
  2. `xlm-roberta-base` — Multilingual SentencePiece (250,002 vocab)
- **Denominators**: tok/word, tok/grapheme, tok/byte, tok/char, tok/sentence
- **Bugs fixed**: No `.lower()`, use `.split()`, grapheme-cluster counting via
  Unicode `\X`

Full reproduction: `python corrected_fertility.py --corpus eng=... --tokenizer gpt2 --tokenizer hf:xlm-roberta-base`

---

## Results: GPT-2 tokenizer (English-centric)

| lang | tok/word | tok/grapheme | tok/byte | tok/char | tok/sentence | words/sent |
|------|---------|-------------|---------|---------|-------------|-----------|
| eng  | 1.135   | 0.172       | 0.172   | 0.172   | 10.6        | 9.4       |
| hin  | 8.856   | 2.593       | 0.600   | 1.602   | 90.3        | 10.2      |
| kan  | 25.412  | 4.189       | 0.980   | 2.730   | 164.3       | 6.5       |
| tam  | 27.025  | 4.361       | 1.000   | 2.793   | 182.9       | 6.8       |

**Ratios relative to English:**

| lang | tok/word | tok/grapheme | tok/byte | tok/sentence |
|------|---------|-------------|---------|-------------|
| eng  | 1.00×   | 1.00×       | 1.00×   | 1.00×       |
| hin  | 7.80×   | 15.08×      | 3.49×   | **8.50×**   |
| kan  | 22.39×  | 24.36×      | 5.70×   | **15.45×**  |
| tam  | 23.81×  | 25.36×      | 5.82×   | **17.20×**  |

GPT-2 is catastrophically inefficient for Indic languages. Hindi costs 8.5× per
equivalent sentence, and Dravidian languages cost 15–17×. The tok/word metric
(7.8×) actually *understates* the true cost for Hindi because Hindi uses fewer
words per sentence.

---

## Results: XLM-RoBERTa tokenizer (multilingual)

| lang | tok/word | tok/grapheme | tok/byte | tok/char | tok/sentence | words/sent |
|------|---------|-------------|---------|---------|-------------|-----------|
| eng  | 1.381   | 0.209       | 0.209   | 0.209   | 12.9        | 9.4       |
| hin  | 1.444   | 0.423       | 0.098   | 0.261   | 14.7        | 10.2      |
| kan  | 2.577   | 0.425       | 0.099   | 0.277   | 16.7        | 6.5       |
| tam  | 2.399   | 0.387       | 0.089   | 0.248   | 16.2        | 6.8       |

**Ratios relative to English:**

| lang | tok/word | tok/grapheme | tok/byte | tok/sentence |
|------|---------|-------------|---------|-------------|
| eng  | 1.00×   | 1.00×       | 1.00×   | 1.00×       |
| hin  | 1.05×   | 2.02×       | 0.47×   | **1.14×**   |
| kan  | 1.87×   | 2.03×       | 0.48×   | **1.29×**   |
| tam  | 1.74×   | 1.85×       | 0.42×   | **1.26×**   |

With a multilingual tokenizer, the cost gap nearly vanishes:
- Hindi: 1.14× per sentence (vs. 8.50× with GPT-2)
- Kannada: 1.29× (vs. 15.45×)
- Tamil: 1.26× (vs. 17.20×)

---

## Which single number should drive routing-and-cost decisions?

**`tok/sentence` on parallel text** (tokens per equivalent meaning) is the
correct metric. Here's why:

1. **It holds meaning constant.** When a user asks "What's the weather today?"
   in Hindi vs. English, the *meaning* is the same. The cost driver is how many
   tokens the tokenizer produces for that equivalent request. tok/sentence
   directly measures this.

2. **tok/word is misleading.** Hindi packs more meaning per word (fewer words
   per sentence). Higher tok/word is partly a linguistic property, not a
   tokenizer deficiency. The intern's 5.89× ratio conflates both effects.

3. **tok/byte is interesting but misleading too.** Devanagari and Dravidian
   scripts use more bytes per character (3 bytes/char in UTF-8 vs. 1 for ASCII).
   XLM-RoBERTa actually achieves *lower* tok/byte for Hindi (0.47×) than
   English, but this doesn't mean Hindi is "cheaper" — it just means the
   tokenizer is efficient at compressing multi-byte characters.

4. **tok/sentence maps directly to API cost.** LLM pricing is per-token.
   If Hindi costs 1.14× per sentence with XLM-RoBERTa, then serving Hindi
   costs ~14% more than English for equivalent requests. This is the number
   that drives budgeting.

---

## Key takeaways

1. **The tokenizer choice matters far more than the language.** Switching from
   GPT-2 to XLM-RoBERTa reduces the Hindi cost gap from 8.5× to 1.14× — a
   **7.5× improvement** from tokenizer choice alone.

2. **The intern's report drastically overstated the problem** by using the
   wrong metric (tok/word) with the wrong tokenizer (GPT-2). The recommendation
   to "budget 6× serving cost for Hindi" was wrong on both counts.

3. **The REPORT_v0 claim that "this is a property of the script, not the
   tokenizer" is exactly backwards.** Most of the cost gap is a property of the
   tokenizer, and can be eliminated by choosing a multilingual one.
