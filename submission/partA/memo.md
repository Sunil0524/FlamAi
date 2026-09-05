# A4 — Recommendation Memo: Tokenizer Routing & Cost

## Corrected Headline Numbers

| Language | GPT-2 (tok/sent ratio) | XLM-RoBERTa (tok/sent ratio) |
|----------|----------------------|------------------------------|
| English  | 1.00× (baseline)     | 1.00× (baseline)             |
| Hindi    | 8.50×                | **1.14×**                     |
| Kannada  | 15.45×               | **1.29×**                     |
| Tamil    | 17.20×               | **1.26×**                     |

Metric: tokens per parallel sentence (holds meaning constant across languages).

## Routing Recommendation

**Deploy a multilingual tokenizer (e.g., XLM-RoBERTa-based or IndicBERT-based)
for all Indic traffic.** This reduces the per-request cost gap from 8–17× (GPT-2)
to 1.1–1.3× (multilingual). There is no need for a separate "Indic-specialized"
routing path — a single multilingual model/tokenizer serves all languages at
near-parity cost.

Do NOT budget 6× serving cost for Hindi as the original report recommended.
The correct budget multiplier is **~1.15×** with a multilingual tokenizer.

## Biggest Caveat

Our eval corpus is 30 parallel sentences of formal/informational text. Production
traffic will include:
- Code-mixed text (Hinglish: "mujhe ek meeting schedule karna hai") where
  tokenization is less predictable
- Longer, multi-turn conversations where token efficiency may diverge from
  single-sentence measurements
- Domain-specific jargon not covered by our eval set

The 1.14× number is a lower bound on the Hindi cost gap; real-world traffic
will likely be 1.2–1.5× depending on code-mixing prevalence.

## Production Monitoring Metric

Track **median tokens-per-request by detected language** in production. Alert if
any language's ratio drifts >30% above its eval-corpus baseline for 7 consecutive
days. This catches:
- Distribution shift in user traffic (e.g., more code-mixed queries)
- Tokenizer vocabulary gaps for new domains
- Regression from model/tokenizer updates
