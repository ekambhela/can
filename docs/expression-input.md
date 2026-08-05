# Design: optional gene-expression input

**Status: proposal. Nothing here is implemented.** Written for review before any
code is written, per the request that this be planned first.

## Why

`experiments/RESULTS.md` (Tier 5, reproduce with `python -m
experiments.run_expression`) measured expression as the single strongest
accuracy lever in the study. On the 577 of 988 cell lines that have expression,
split by cell line:

| Test (577 covered lines) | Spearman | R² | top-1 | top-10 |
|---|---|---|---|---|
| mutations + tissue (what ships today) | 0.346 | 0.119 | 0.052 | 0.181 |
| **+ expression** | **0.445** | **0.202** | **0.069** | **0.215** |

| Tissue-blocked (whole tissues held out) | Spearman | R² |
|---|---|---|
| mutations + tissue | 0.102 | −0.065 |
| **+ expression** | **0.262** | **+0.023** |

The second table is the one that matters. The model's stated weakness is that it
leans on tissue identity — held out a whole tissue, it collapses. Expression more
than doubles blocked Spearman and turns R² positive, because it captures cell
*state* rather than just which organ the sample came from.

**What this does not buy.** Top-1 accuracy goes 5.2% → 6.9%. This makes a better
shortlist, not a verdict, and none of the labelling work in the rest of the app
should be softened because expression is available.

## The actual hard problem: input scale

The gain above was measured on GDSC's own expression matrix
(`experiments/expr_cosmic.csv.gz`): 580 lines × 706 census genes, indexed by
COSMIC ID, columns named `g<EntrezID>`, values roughly log-scale in **0 – 13.9**
(median 3.78, ~5% exact zeros, per-gene sd ≈ 1.07). That is RMA-style
log2 intensity.

A user with RNA-seq has **TPM, FPKM or raw counts**. Those are not the same
scale, not the same dynamic range, and not even monotonically comparable
gene-to-gene after the microarray's probe-affinity effects. Feeding raw TPM into
a model trained on log2 RMA is an out-of-distribution input that will still
produce a confident-looking ranking.

This is the same failure mode the tissue and wild-type work just removed:
silently answering a different question. So the design treats normalization as
the core problem, not a detail.

**Proposed handling, in order of preference:**

1. **Rank-normalize per sample, and train the model on rank-normalized data
   too.** Convert each sample's 706 genes to within-sample ranks (or quantiles),
   and fit the shipped expression model on GDSC data put through the identical
   transform. This makes platform and units irrelevant — ranks survive
   microarray vs RNA-seq, TPM vs FPKM. Cost: it discards absolute level, so it
   must be re-measured; the 0.445 figure above does **not** yet hold for a
   rank-transformed model.
2. Accept a declared unit (`log2_tpm`, `tpm`, `rma`) and apply a per-unit
   transform. Simpler, but every unit we accept is a separate calibration we
   have to validate, and users mislabel units.
3. Require GDSC-style RMA. Honest but nearly useless — almost nobody has it.

**Recommendation: (1), gated on re-running `run_expression.py` with the rank
transform applied.** If rank-normalizing costs most of the gain, this feature is
not worth building, and that measurement should happen before any product work.
That is the first go/no-go.

## Input format

```
gene,value          # or a wide single-row CSV: one column per gene
EGFR,9.12
ERBB2,11.40
...
```

- **Identifiers:** accept HGNC symbols (what users have) and Entrez IDs (what
  the matrix uses); ship a symbol→Entrez map in the bundle. Reject ambiguous or
  unmapped symbols loudly rather than dropping them silently.
- **Coverage:** only the 706 census genes are used; extra genes are ignored (and
  the count reported). Missing genes are the interesting case — see below.
- **Size:** a 706-row CSV is a few tens of KB; a whole-transcriptome upload is
  ~500 KB–2 MB. The existing 2 MB cap covers it. The cohort row cap does not
  apply — one expression profile is one sample.
- **Not supported initially:** BAM/FASTQ, multi-sample expression matrices,
  single-cell. Out of scope.

## How the bundle carries both models

One artifact, two models, explicit about which inputs each requires:

```python
{
  "kind": "ensemble",
  ...                                  # existing keys, unchanged
  "expression": {                      # absent in bundles without it
     "genes":       [...706 Entrez ids...],   # required column order
     "symbol_to_id": {...},                   # HGNC -> Entrez
     "transform":   "rank",                   # what the model was trained on
     "model":        <multi-task estimator>,
     "per_drug_models": {...},
     "blend_w_perdrug": 0.35,
     "metrics":     {...},                    # its OWN held-out metrics
     "reliability": {...},                    # its OWN per-drug tiers
  },
}
```

Two properties this buys:

- **The existing model is untouched**, so a bundle with a broken or missing
  expression block still serves every request the app serves today.
- **Metrics and reliability tiers are per-model.** The two paths have different
  accuracy, so `/api/health`, the accuracy note under the recommendation, and
  the per-drug reliability flags must all report the numbers for the path that
  actually ran. Sharing one metrics blob would misreport both.

Bundle version bumps to 9. `model/schema.py` gains the expression feature list
and gene-ID mapping (code, not learned parameters — same rule as the existing
feature labels).

## How the API decides which model to use

Explicitly, from the presence of usable expression — never inferred, never
partial:

```
expression absent            -> mutation model   (today's behaviour, unchanged)
expression present + valid   -> expression model
expression present + invalid -> 422, naming what is wrong
```

"Valid" means: parses, maps to a known gene vocabulary, covers at least a
threshold fraction of the 706 genes (proposed: **80%**, to be set by measuring
degradation vs coverage), and passes a plausibility check on the value
distribution.

Endpoint shape: extend `/api/predict` with an optional second file part
(`expression=@profile.csv`) rather than adding a new route, so the response
schema and all the existing error handling stay in one place. The response
gains:

```json
{ "model_path": "expression" | "mutations",
  "expression": { "genes_matched": 690, "genes_expected": 706,
                  "coverage": 0.977, "transform": "rank" } }
```

**Partial expression is rejected, not imputed.** Below the coverage threshold we
fall back to the mutation model and say so in `warnings` — we do not mean-impute
300 missing genes and present the result as an expression-grade prediction.

## Graceful degradation

| Situation | Behaviour |
|---|---|
| No expression uploaded | Mutation model. Identical to today. |
| Expression block missing from bundle | Mutation model, log a warning at boot. |
| Coverage below threshold | Mutation model + a warning naming the coverage. |
| Unparseable / unmapped genes | 422 naming the field (as tissue does today). |
| Values implausible for the declared transform | 422, not a silent rescale. |
| Expression model raises at predict time | Fall back to the mutation model, log with traceback, flag `degraded: true` in the response. |

The last row is the only place a silent-ish fallback is acceptable, and even
there the response says it happened.

Tissue stays **required on both paths**. Expression reduces the tissue
dependence; it does not remove tissue from the feature set.

## What the UI shows differently

The two paths must not look identical, because they are not equally accurate.

- **Input:** an optional "I have an RNA-seq profile" affordance next to the
  existing upload, clearly marked as advanced, with a downloadable example file
  in the accepted format. Never a required field.
- **Which model ran:** a badge on the result — "ranked using expression + 
  mutations" vs "ranked using mutations + tissue" — beside the recommendation,
  not buried in the parsed-sample details.
- **Accuracy note:** the existing "#1 pick is the best drug 11% of the time"
  line reads from `model_metrics`, so it updates itself once metrics are
  per-path. It must not be dropped on the expression path; the number improves,
  it does not become a verdict.
- **Coverage:** "690 of 706 genes matched" near the badge, so a user with a
  patchy profile can see why they got the answer they got.
- **The assumed-biomarker banner still applies.** Expression does not tell us
  BRAF mutation status; unspecified mutations are still assumed wild-type and
  must still say so.

## Risks

1. **Rank normalization may eat the gain.** Measure first (see go/no-go above).
2. **Cell lines are not tumors, and this widens the gap.** Cultured-line
   expression carries strong culture-adaptation signal that a patient biopsy
   will not have; the transfer penalty is plausibly worse for expression than
   for mutation flags. The disclaimers do not get weaker here.
3. **577 of 988 lines have expression** — the expression model trains on ~58% of
   the data. Some of the per-drug models will get thin, and their reliability
   tiers (Task 7 machinery) will show it. That is the mechanism for surfacing
   it, and it should be checked before shipping.
4. **Artifact size.** A second multi-task model plus 369 more per-drug boosters
   roughly doubles `model.joblib`, which is already 11 MB and committed. This
   interacts directly with the Git LFS / release-asset question.
5. **Upload cost.** Parsing a 2 MB expression CSV is real work on a free tier;
   the existing rate limits should be tightened for this path.

## Staged plan

1. **Measure the rank-normalized model.** Rerun `run_expression.py` with the
   within-sample rank transform. Go/no-go. *(No product code.)*
2. **Coverage sensitivity.** Degrade held-out lines to 90/80/70/50% gene
   coverage; set the threshold from the curve rather than by guessing.
3. Train and serialize the expression block; bump to version 9.
4. Parsing + validation layer, with the same "fail loudly, name the field"
   contract as tissue.
5. API routing, per-path metrics and reliability.
6. UI: optional input, path badge, coverage, unchanged assumption banner.
7. Tests: format matrix, coverage-threshold behaviour, fallback paths, parity
   that an absent expression file reproduces today's answers exactly.

Steps 1–2 are measurement and answer whether 3–7 are worth doing.

## Open questions for review

- Rank-normalize (platform-independent, unmeasured) or declared-units
  (measured, brittle)? This is the main decision.
- Is the target user real? If nobody with RNA-seq uses this, the honest move is
  to leave the finding in `RESULTS.md` — where it is already the headline
  result — and not add an input modality to a demo.
- Should the expression path be gated behind an explicit "I understand these are
  cell-line-trained predictions" step, given it will look more authoritative?
