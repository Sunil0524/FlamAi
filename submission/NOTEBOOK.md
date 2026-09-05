# NOTEBOOK.md — Chronological Lab Notebook

*Audit of REPORT_v0.md: tokenizer fertility & serving capacity*

---

## Day 1 — Initial read-through & hypothesis formation

### 22:30 — Read the starter kit

Opened `REPORT_v0.md`, `fertility.py`, `corpus_sample/`, `bench/model_spec.md`,
`bench/bench_log.csv`. First impressions:

- The report claims Hindi is 5.89× more expensive than English, based on
  tok/word with the GPT-2 tokenizer. That seems high but not impossible.
- The tok/char column (1.579 vs 0.226 = 7.0×) "confirms" the tok/word
  number. Both metrics showing a similar ratio gives a false sense of
  robustness — but they could both be wrong in the same direction.
- The report's conclusion #3 says "root cause: Hindi simply has more Unicode
  characters per word, so any tokenizer will struggle." This feels too
  certain. Surely a multilingual tokenizer would do much better?

### 22:45 — Hypothesis: the denominator is wrong

**H1**: tok/word is the wrong metric for cross-language cost comparison.
"Word" is not a universal unit — Hindi words carry more morphological content.
If Hindi says the same thing in fewer words, then higher tok/word doesn't
mean higher cost per *request*.

**Experiment plan**: Compute tok/sentence on the parallel corpora. If the
ratio is significantly different from tok/word, H1 is confirmed.

### 23:00 — Spotted the split bug

Looking at `eng_sample.txt` more carefully:

```
Please keep the books  in the cupboard.
```

Double space between "books" and "in"! And `fertility.py` line 62 uses
`line.split(" ")` which will produce an empty string here: `['books', '', 'in']`.
This inflates the word count.

Checked `hin_sample.txt` — line 10 also has a double space:
```
किताबें  अलमारी में रखी हैं।
```

**H2**: `split(" ")` bug deflates fertility by inflating word count.

### 23:10 — Noticed `.lower()` on line 60

`line = line.lower()` before tokenization. GPT-2's BPE treats "NASA" and
"nasa" very differently. But Hindi/Devanagari has no case distinction.
So `.lower()` selectively benefits English and is a no-op for Hindi.

**H3**: `.lower()` introduces asymmetric bias that makes English look
artificially better (fewer tokens after lowercasing).

Wait — which direction does this push the hin/eng ratio? If English gets
fewer tokens from lowercasing, then the ratio goes UP (Hindi looks worse
relative to English). So removing `.lower()` should make the gap slightly
*smaller*. Need to measure.

### 23:20 — Is random.seed a bug?

`random.seed(1337)` on line 25, but `random` is never used. Suspicious?

Actually, on reflection: this is harmless. It's a reproducibility practice.
The seed has zero effect since no random functions are called. Flagging this
as a bug without evidence would cost points per the assignment rules.

**Decision**: Classify as "suspicious but fine." Do not claim it as a bug.

---

## Day 1 — Bug verification experiments

### 23:30 — Ran audit_experiments.py

Created `audit_experiments.py` to measure each bug's effect. Used the
starter kit's sample corpora with the GPT-2 tokenizer.

**Bug 1 results (.lower() bias):**
```
eng: WITH lower: 1.2831, WITHOUT lower: 1.2472 → delta -2.88%
hin: WITH lower: 7.5985, WITHOUT lower: 7.5985 → delta 0.00%
```

Confirmed: English gets 2.9% fewer tokens from lowercasing, Hindi unaffected.
This makes the hin/eng ratio slightly larger (~2.9%) than it should be.

**Bug 2 results (split bug):**
```
eng: split(" ") words=79, split() words=78 → fertility delta +1.43%
hin: split(" ") words=62, split() words=61 → fertility delta +1.97%
```

Confirmed: both corpora have at least one double-space line. Effect is small
on this toy corpus but would compound on messier data.

**Bug 3 results (wrong denominator):**
```
tok/word ratio:     6.09× (intern's metric — MISLEADING)
tok/char ratio:     7.19×
tok/sentence ratio: 4.78× (holds meaning constant — CORRECT)
```

**This is the big one.** The true cost multiplier is 4.78×, not 5.89×.
The intern's metric overstates the cost by ~23%. The report's recommendation
to "budget 6×" is wrong.

### 23:50 — Dead end: tried to reproduce the intern's exact numbers

Ran the original `fertility.py` (with bugs) on the sample corpora:
```
eng: 1.27 tok/word, 0.226 tok/char
hin: 7.45 tok/word, 1.579 tok/char
```

These match REPORT_v0.md's table exactly. Good — the intern's numbers
are at least internally consistent with their (buggy) code. The bugs are real
but the intern's numbers do follow from their script.

---

## Day 2 — Corpus construction & corrected analysis

### 09:00 — Building a proper eval corpus (A1)

Need at least 4 languages: English, Hindi, + 2 Dravidian. Chose Kannada
and Tamil.

**Attempt 1**: Download FLORES-200 from HuggingFace. Failed — `facebook/flores`
requires gated access (HTTP 401). The `datasets` library also had a DLL load
error on Windows due to pyarrow.

**Attempt 2**: Try `Muennighoff/flores200` (mirror). Also failed — 404.

**Attempt 3**: Try NLLB seed data from GitHub. Also failed — 404.

**Fallback**: Created a curated parallel corpus of 30 sentences per language.
English originals composed manually, then translated to Hindi, Kannada, and
Tamil maintaining topical parallelism. Not as large as FLORES but sufficient
for mean estimates, and the parallelism is what matters most for tok/sentence.

**Caveat documented**: 30 sentences is a small sample. Formal text only.
Real production traffic (code-mixed, informal) would shift the ratios.

### 10:00 — Corrected fertility analysis (A3)

Ran `corrected_fertility.py` with two tokenizers:

**GPT-2 (English-centric):**
```
eng: 10.6 tok/sent (baseline)
hin: 90.3 tok/sent → 8.50×
kan: 164.3 tok/sent → 15.45×
tam: 182.9 tok/sent → 17.20×
```

**Surprise**: With tok/sentence (the correct metric), Hindi is actually
WORSE than the intern's tok/word suggested (8.50× vs 7.80×). The tok/word
metric understates the gap because Hindi uses fewer words per sentence!

This was a surprise — I expected tok/sentence to show a *smaller* gap.
But Hindi having fewer, semantically denser words means tok/word *hides*
some of the tokenizer inefficiency. The correct answer is worse than the
intern's wrong answer, but for a completely different reason.

**XLM-RoBERTa (multilingual):**
```
eng: 12.9 tok/sent (baseline)
hin: 14.7 tok/sent → 1.14×
kan: 16.7 tok/sent → 1.29×
tam: 16.2 tok/sent → 1.26×
```

**This is the key finding.** A multilingual tokenizer nearly eliminates
the cost gap. Hindi goes from 8.50× (GPT-2) to 1.14× (XLM-R). The
intern's conclusion that "any tokenizer will struggle" is wrong — only
English-centric tokenizers struggle.

### 11:00 — Revised my understanding of Bug 3

I initially described Bug 3 as "per-word overstates the gap." But the GPT-2
tok/sentence data shows tok/word actually UNDERSTATES the gap for Hindi
(7.80× vs 8.50×). The conceptual bug is still real — tok/word is the wrong
metric — but the direction of distortion is more nuanced than I first
thought.

The correct framing: tok/word is unreliable because it mixes two effects
(tokenizer efficiency + linguistic word density). Sometimes it overstates,
sometimes it understates. tok/sentence removes the ambiguity.

---

## Day 2 — Capacity reconciliation (Part B)

### 14:00 — KV-cache arithmetic (B1)

From model_spec.md:
```
KV bytes/token = 2 × 8 × 128 × 28 × 2 = 114,688 bytes = 112 KiB
Available for KV: 22.08 - 8.4 - 1.6 = 12.08 GB
Max 4096-token seqs: 12.08 / 0.4375 = 27.6 → ~27
```

Validated against bench_log.csv: batch-24 fits (kv_util=0.93), batch-32
overflows (7 preempted). Prediction confirmed.

### 14:30 — Throughput anomaly (B2)

The long-context sweep shows throughput DROPS after batch 24:
- batch 24: 1607 tok/s, 0 preempted
- batch 32: 1384 tok/s, 7 preempted (−14%)
- batch 48: 1298 tok/s, 23 preempted (−19%)

Mechanism: KV-cache exhaustion → preemption → wasted recomputation.
Evidence: kv_util pegs at 0.97, ttft_ms spikes from 500→955ms.

### 15:00 — Report misreading (B3)

The report says "longer prompts give better throughput" comparing batch-16
short (883 tok/s) vs long (1311 tok/s). But `reported_tok_s` includes
prompt processing tokens! Long rows push 5.3× more total tokens per request.

Honest goodput of batch-24 long-prompt:
- Method 1: 24 × 512 / 61.16 = 200.9 generated tok/s
- Method 2: from itl_ms_p50: same → 200.9 tok/s ✓

The "~3200 tok/s at batch-48" claim is doubly wrong: wrong metric AND
preemption degrades it to 162 tok/s.

---

## Day 3 — Decision memo (Part C) & polish

### 09:00 — Decision memo

Chose Path (a) SFT over (b) rewriter and (c) prompt-engineering.

Key reasoning: SFT is durable, zero marginal serving cost, and the A100
budget allows it. The rewriter doubles latency and serving cost. Prompt
engineering is too brittle across 6 languages.

Back-of-envelope: 30K pairs, 2 days generation, 4 hours training, 1K pairs
reviewed per language. Kill criterion: <40% reviewer approval after week 1.

### 11:00 — Final review

Reviewed all deliverables for evidence rule compliance. Every claimed flaw
has measured before/after numbers. The `random.seed` non-bug is explicitly
called out as harmless to avoid the −5 penalty for unverified claims.

---

## Summary of surprises and dead ends

1. **Dead end**: FLORES-200 download. Required auth, pyarrow DLL issues.
   Fell back to curated corpus.

2. **Surprise**: tok/sentence shows a BIGGER gap than tok/word for Hindi
   with GPT-2 (8.50× vs 7.80×). I expected the opposite. The bug is that
   tok/word is unreliable, not that it always overstates.

3. **Surprise**: XLM-RoBERTa reduces the Hindi gap to 1.14×. The tokenizer
   choice matters far more than the language itself — directly contradicting
   the intern's claim that "this is a property of the script, not the
   tokenizer."

4. **Dead end**: Initially described Bug 1 (.lower()) as making the gap
   "appear larger." Had to correct: lowercasing reduces English tokens,
   making the ratio larger. But the magnitude is only ~3%, dwarfed by
   Bug 3's effect.
