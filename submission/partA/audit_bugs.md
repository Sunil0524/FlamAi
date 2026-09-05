# A2 — Audit of `fertility.py` and its Metric

## Summary

`fertility.py` has **two code bugs** and **one conceptual flaw**. One element
that looks suspicious (`random.seed(1337)`) is actually harmless. Each finding
below includes measured evidence from the starter kit's sample corpora using
the `gpt2` tokenizer.

Full experimental reproduction: `python audit_experiments.py`

---

## Bug 1: `line.lower()` introduces asymmetric tokenization bias

**Location**: Line 60 of `fertility.py`  
**Type**: Data mutation / asymmetric bias  

### What happens

The line `line = line.lower()` converts all text to lowercase before
tokenization. BPE tokenizers like GPT-2 encode uppercase and lowercase text
differently — "NASA" tokenizes differently than "nasa". Since Devanagari
(Hindi) has no letter-casing distinction, `.lower()` is a no-op for Hindi
but actively reduces the token count for English.

### Measured effect

| Metric | English | Hindi |
|---|---|---|
| fertility WITH `.lower()` | 1.2831 | 7.5985 |
| fertility WITHOUT `.lower()` | 1.2472 | 7.5985 |
| Delta | −0.0359 (−2.88%) | 0.0000 (0.00%) |

### Direction of distortion

English fertility is **under-reported** by ~2.9% due to lowercasing.
Hindi is unaffected. This makes the hin/eng gap appear **slightly larger**
than it truly is. The magnitude is small (~2.9%) but the asymmetry is
systematic and would compound with larger corpora containing more
proper nouns and acronyms.

### Fix

Remove `line = line.lower()` from the `analyze()` function. Tokenizer
fertility should be measured on text as-is, since real serving traffic
will include mixed casing.

---

## Bug 2: `split(" ")` produces empty strings on multi-space input

**Location**: Line 62 of `fertility.py`  
**Type**: Code bug  

### What happens

`words = line.split(" ")` uses a single-space delimiter. When input contains
double spaces (e.g., `"books  in"`), `split(" ")` produces an empty string
between them: `['books', '', 'in']`. This inflates `len(words)`, deflating
the fertility ratio.

The English sample corpus contains double spaces on line 7:
`"Please keep the books  in the cupboard."` — and the Hindi sample on line 10:
`"किताबें  अलमारी में रखी हैं।"`

### Measured effect

| Metric | English | Hindi |
|---|---|---|
| Word count with `split(" ")` | 79 | 62 |
| Word count with `split()` | 78 | 61 |
| Fertility with `split(" ")` | 1.2293 (buggy) | 7.4485 (buggy) |
| Fertility with `split()` | 1.2472 (correct) | 7.5985 (correct) |
| Delta | +0.0179 (+1.43%) | +0.1500 (+1.97%) |

### Direction of distortion

Fertility is **under-reported** for both languages (denominator too large).
The magnitude here is small on the toy corpus (1 affected line out of 10),
but on larger corpora with inconsistent spacing (common in web-scraped text),
the error could be significantly larger.

### Fix

Change `line.split(" ")` to `line.split()` (no argument). Python's `split()`
with no argument splits on any whitespace and discards empty strings.

---

## Bug 3 (Conceptual): per-word fertility is the wrong metric for cross-language cost comparison

**Location**: Lines 64, 67 (the metric definition itself)  
**Type**: Methodological / conceptual flaw  

### What happens

The metric `tokens / words` measures how many tokens the tokenizer produces
per whitespace word. The **report then uses this ratio** to conclude that
Hindi "costs 6× more per request" and recommends budgeting 6× serving cost.

The problem: **"word" is not a constant unit across languages.** Hindi expresses
the same meaning in fewer, morphologically richer words. So higher tok/word
for Hindi is partly a property of the **language**, not the **tokenizer**.

### Measured effect (parallel sentences from sample corpus)

| Metric | eng → hin ratio | What it measures |
|---|---|---|
| tok/word | 6.09× | Misleading — mixes tokenizer overhead with word structure |
| tok/char | 7.19× | Slightly better, but char count varies by script |
| **tok/sentence** | **4.78×** | **Correct — holds meaning constant** |
| Avg words/sentence | eng: 7.8, hin: 6.1 | Hindi uses 1.3× fewer words per equivalent meaning |

### Direction of distortion

The report's headline number (5.89× from tok/word) **overstates** the true
cost multiplier. When we hold meaning constant (tok/sentence on parallel
text), the ratio is **~4.8×** — still significant, but 20% lower than claimed.

The report's recommendation to "budget 6× serving cost for Hindi" and its
claim that "this is a property of the script, not the tokenizer" are both
incorrect:
- The 6× figure is inflated by using the wrong denominator
- Part of the cost difference IS a property of the tokenizer (a multilingual
  tokenizer could compress Hindi better), not just the script

### Fix

Use **tokens per parallel sentence** (or per UTF-8 byte) as the primary metric
for routing-and-cost decisions. This holds the user's intent constant across
languages and directly maps to API cost per request.

---

## Not a Bug: `random.seed(1337)` (Line 25)

### What it looks like

The script imports `random` and sets `random.seed(1337)` on line 25, but
no function from the `random` module is ever called. This looks like
dead code or a potential sign of removed/forgotten functionality.

### Why it's harmless

1. Setting a reproducibility seed is a defensive coding practice. A future
   version might use `random.sample()` to subsample large corpora.
2. The seed has **zero effect** on the output — no random functions are
   called, so computation is fully deterministic regardless.
3. The import + seed add negligible overhead (microseconds).

**This is not a bug.** Flagging it as one without evidence of actual impact
would violate the evidence rule.
