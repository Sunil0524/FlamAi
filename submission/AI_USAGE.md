# AI Usage Disclosure

## Where AI helped

1. **PDF text extraction**: Used AI to identify the best approach for
   extracting text from the assignment PDF on a Windows system without
   pre-installed Python.

2. **Corpus generation**: AI assisted in generating the curated parallel
   corpus sentences in Hindi, Kannada, and Tamil. The English originals
   were human-composed, and AI translated them to maintain topical
   parallelism. Each sentence was reviewed for naturalness.

3. **Code scaffolding**: AI helped write boilerplate code for argument
   parsing, CSV I/O, and file handling in the Python scripts. The core
   analysis logic (metric definitions, bug identification, capacity
   arithmetic) was human-directed.

4. **LaTeX/markdown formatting**: AI helped format tables and structure
   the written deliverables consistently.

5. **Grapheme cluster counting**: AI suggested using the `regex` library's
   `\X` pattern for Unicode extended grapheme cluster counting, which I
   verified against the Unicode standard.

## Where AI misled or required correction

1. **FLORES-200 access**: AI initially assumed FLORES-200 could be
   downloaded without authentication via the HuggingFace datasets library.
   In practice, `facebook/flores` requires gated access (HTTP 401), and
   the `datasets` library hit a DLL load error on Windows due to pyarrow.
   Fell back to curated parallel sentences instead.

2. **Initial analysis of Bug 1 direction**: AI's first pass described the
   `.lower()` bug as making "the gap appear larger than it is," but on
   reflection, lowercasing *reduces* English token count (making English
   look *better*), so it makes the hin/eng ratio slightly *larger* than
   truth. The corrected direction is: removing `.lower()` makes the gap
   slightly *smaller*.

3. **B3 goodput calculation**: AI initially conflated `reported_tok_s`
   with generation-only throughput. After manual verification against the
   bench_log.csv rows, the correct interpretation is that `reported_tok_s`
   includes prompt-processing tokens, and goodput must be derived from
   `gen_len × num_requests / wall_clock_s`.

## Summary

AI was used as a research assistant and code accelerator. All numerical
claims were independently verified by running the actual scripts and
checking against the raw data. The bug identifications, capacity
arithmetic, and strategic recommendations reflect my own analysis,
with AI helping to structure and express them.
