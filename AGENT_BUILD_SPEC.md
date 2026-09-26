# Business Entity Resolution — Agent Build Spec

This file is the single source of truth for building this project across multiple AI
agent sessions. Read Section 1 (rules) and the specific phase section requested — you
do not need to read the rest of this file or the whole repo to do your job.

---

## 1. Agent Operating Rules (read every session)

1. **One phase at a time.** Only build the phase the user explicitly names (e.g. "build
   Phase 2"). Do not build ahead, do not "helpfully" start the next phase. If a phase
   depends on a file that doesn't exist yet, stop and say so instead of creating it
   yourself out of order.

2. **Ground every phase in the actual repo, not the plan.** Section 2's tree is the
   *target end-state*, not a claim about what exists right now. Before writing any
   code, inspect the real current layout yourself (`ls -R`, `find . -maxdepth 4`, or
   `git ls-files`) instead of asking the user to describe it or assuming Section 2 is
   already true. Known deviations from the target tree as of this writing:
   - The challenge data actually lives under `student_resource/dataset/{train,test}/`,
     not a top-level `dataset/` — use the real path.
   - `pyproject.toml` already exists at repo root (dependency management is
     pyproject-based, not a bare `requirements.txt` — Phase 8 should update
     `pyproject.toml`, not create a separate `requirements.txt`, unless the user says
     otherwise).
   - `utils/validate_submission.py`, `README.md`, `Documentation_template.md`, and
     `.gitignore` are already present and provided by the challenge — never overwrite
     `validate_submission.py`.
   - This spec file (`AGENT_BUILD_SPEC.md`) sits at repo root, not inside `src/`.
   If the repo has moved on further since this file was last read, trust what you
   observe in the filesystem over what this section says.

3. **Modularity contract — no direct swaps without an interface.**
   - Any blocking strategy implements `BaseBlocker` (`src/blocking/base.py`).
   - Any ML classifier implements `BaseMatcher` (`src/models/base_matcher.py`).
   - Only `src/models/__init__.py`'s registry and `src/config.py`'s `MODEL_TYPE` /
     `BLOCKING_STRATEGIES` string(s) may reference a specific library (LightGBM,
     XGBoost, CatBoost, etc.) or a specific blocking algorithm.
   - **No other file** may import `lightgbm`, `xgboost`, `jellyfish`, etc. directly.
     If you find yourself doing that outside the designated module, stop and route
     through the interface instead.
   - Test of correctness: swapping `MODEL_TYPE = "lightgbm"` → `"xgboost"` in
     `config.py` alone should retrain and run inference with zero other file edits.

4. **Accuracy is not optional — every phase that touches predictions must score itself.**
   - Any phase that produces matches, candidates, or probabilities must run the
     relevant scorer (`src/evaluation/recall_scorer.py` for blocking,
     `src/evaluation/f05_scorer.py` for final/validation matches) and print the number
     before the phase is declared done.
   - A phase is not "done" on "the code ran without error" — it's done when the
     printed metric is captured in `PROGRESS.md`.
   - If a phase changes anything upstream (blocking, features, model), re-run
     `f05_scorer.py` on the validation split before touching the next phase.

5. **Token economy — spend tokens only where they buy accuracy or safety.**
   - Before starting a phase: read only `PROGRESS.md`, this spec's section for that
     phase, and the interface files (`base_matcher.py`, `base.py`, `config.py`). Do not
     re-read the full repo or prior phases' full source.
   - Prefer diffs/edits over reprinting whole files in chat. Only show full file
     contents when creating a brand-new file or when a bug requires it.
   - Cache expensive artifacts (`candidate_pairs.tsv`, `features.parquet`). Check if
     the output already exists and is newer than its inputs before recomputing.
   - Summarize test/scoring output in 2–3 lines. Don't paste full logs unless asked.
   - Phase 7 (embeddings) is the expensive phase — confirm with the user before
     running it, and only after Phase 6 shows a measured need.

6. **Documentation & readability.**
   - Every function: type hints + a one-line Google-style docstring (what it does,
     not how).
   - No function over ~40 lines; one responsibility per function.
   - Every module starts with a 2–3 line comment: what this file owns, what it must
     never do (e.g. "never call an ML library directly — see BaseMatcher").
   - Config values (paths, thresholds, hyperparams) live only in `src/config.py` —
     never hard-coded inline elsewhere.

---

## 2. Target Project Structure

```
business_entity_resolution/
├── src/
│   ├── config.py                    # all paths, thresholds, hyperparams, MODEL_TYPE
│   ├── data_loader.py                # phase 0
│   ├── eda.py                        # phase 0
│   ├── blocking/
│   │   ├── base.py                   # BaseBlocker interface
│   │   ├── name_blocking.py
│   │   ├── phonetic_blocking.py
│   │   ├── tokenset_blocking.py
│   │   └── union.py                  # combines all registered blockers
│   ├── features/
│   │   ├── string_similarity.py
│   │   ├── address_similarity.py
│   │   ├── country_features.py
│   │   └── build_features.py         # orchestrates all feature modules
│   ├── models/
│   │   ├── base_matcher.py           # BaseMatcher interface
│   │   ├── lightgbm_matcher.py
│   │   ├── xgboost_matcher.py         # alt implementation, same interface
│   │   └── __init__.py               # registry / get_matcher(model_type)
│   ├── decision/
│   │   └── threshold_decision.py     # singleton-aware margin thresholding
│   ├── evaluation/
│   │   ├── recall_scorer.py          # blocking recall
│   │   ├── f05_scorer.py             # official macro F_0.5
│   │   └── error_analysis.py
│   ├── embeddings/
│   │   └── embedding_blocking.py     # phase 7, conditional
│   └── pipeline/
│       ├── run_train_pipeline.py
│       └── run_inference_pipeline.py
├── tests/
│   └── test_phaseX.py                # one small test file per phase
├── utils/
│   └── validate_submission.py        # given by challenge, do not modify
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── PROGRESS.md
└── requirements.txt
```

---

## 3. Core Interfaces (define these first, in Phase 1/3 — never bypass them)

```python
# src/blocking/base.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseBlocker(ABC):
    """Owns: generating candidate (source1_id, other_id) pairs from one strategy.
    Never: scores or classifies pairs — that's the model's job."""

    @abstractmethod
    def generate(self, source1: pd.DataFrame, other: pd.DataFrame) -> set[tuple[str, str]]:
        """Return candidate pairs as a set of (source1_entity_id, other_entity_id)."""
```

```python
# src/models/base_matcher.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseMatcher(ABC):
    """Owns: fitting on labeled feature pairs and scoring new pairs.
    Never: touches raw text, blocking, or thresholding — features come in pre-built."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def feature_importance(self) -> pd.Series: ...
```

```python
# src/models/__init__.py
from .lightgbm_matcher import LightGBMMatcher
from .xgboost_matcher import XGBoostMatcher

REGISTRY = {"lightgbm": LightGBMMatcher, "xgboost": XGBoostMatcher}

def get_matcher(model_type: str):
    """Single point of model selection. Add new models only by registering here."""
    return REGISTRY[model_type]()
```

```python
# src/config.py  (excerpt — the only file allowed to name a specific library)
MODEL_TYPE = "lightgbm"          # change this one line to swap models
BLOCKING_STRATEGIES = ["name", "phonetic", "tokenset"]
THRESHOLD_TAU = 0.5
MARGIN = 0.1
```

---

## 4. Phases

Each phase lists: **Goal**, **Build**, **Read this phase / Don't re-read**, **Test
command**, **Accuracy check**, **Done when**, **PROGRESS.md line to append**.

### Phase 0 — Skeleton + EDA
- **Goal:** Understand the data before writing any logic.
- **First action:** run `ls -R student_resource` (or `find student_resource -maxdepth 4`)
  to confirm real file paths before hard-coding any in `config.py`.
- **Build:** `src/config.py` (real paths — data lives under
  `student_resource/dataset/{train,test}/`, not a top-level `dataset/`),
  `src/data_loader.py` (load all 6 tsvs with `sep="\t"`), `src/eda.py` (schema, null
  rates, country distribution, ground-truth singleton rate, name-length stats).
- **Read:** nothing but the README — this phase has no dependencies.
- **Test:** `python -m src.eda` runs clean, prints summary.
- **Accuracy check:** N/A (no predictions yet).
- **Done when:** you can state row counts, singleton %, and confirm France is absent
  from train/present in test.
- **PROGRESS.md:** `Phase 0 DONE — N rows/source, singleton rate X%, France confirmed test-only.`

### Phase 1 — Blocking interface + strategies
- **Build:** `src/blocking/base.py`, `name_blocking.py`, `phonetic_blocking.py`,
  `tokenset_blocking.py`, `union.py` → writes `candidate_pairs.tsv` (train split).
- **Read:** `config.py`, `data_loader.py` output shape only.
- **Test:** `python -m src.blocking.union --split train`
- **Accuracy check:** `src/evaluation/recall_scorer.py` — recall =
  `matched_in_candidates / total_true_matches` against `train_ground_truth.tsv`.
  Print it. Target ≥95%.
- **Done when:** recall printed and ≥95%, candidate count / reduction ratio printed.
- **PROGRESS.md:** `Phase 1 DONE — blocking recall X%, N candidate pairs (train).`

### Phase 2 — Feature engineering
- **Build:** `src/features/string_similarity.py`, `address_similarity.py`,
  `country_features.py`, `build_features.py` → `features.parquet`.
- **Read:** `candidate_pairs.tsv` schema + `config.py`. Do not re-read blocking source.
- **Test:** `python -m src.features.build_features` + assert no NaNs, print feature list.
- **Accuracy check:** print correlation of each feature with the label (sanity check,
  not a model score yet).
- **Done when:** feature table exists for 100% of candidates, no ID-leakage columns.
- **PROGRESS.md:** `Phase 2 DONE — N features built, no NaNs, no leakage.`

### Phase 3 — Baseline matcher (LightGBM via BaseMatcher)
- **Build:** `src/models/base_matcher.py`, `lightgbm_matcher.py`, `__init__.py`
  registry, `src/pipeline/run_train_pipeline.py` (entity-level train/val split —
  split by `source1_entity_id`, never by row, to avoid leakage).
- **Read:** `features.parquet` schema, `base_matcher.py` interface only.
- **Test:** `python -m src.pipeline.run_train_pipeline` — trains, saves `model.pkl`,
  reloads it, scores a batch without error.
- **Accuracy check:** print validation AUC / PR-AUC.
- **Done when:** model round-trips (save→load→predict) correctly.
- **PROGRESS.md:** `Phase 3 DONE — LightGBM, val PR-AUC X.`

### Phase 4 — Singleton-aware thresholding + official scorer
- **Build:** `src/decision/threshold_decision.py` (top-candidate + margin rule),
  `src/evaluation/f05_scorer.py` (exact macro F_0.5 from the README formula).
- **Read:** `base_matcher.py` output shape, `config.py` (`THRESHOLD_TAU`, `MARGIN`).
- **Test:** run on validation split → `matching_results_val.tsv`.
- **Accuracy check:** `f05_scorer.py` prints macro F_0.5. **This is the first real
  checkpoint number** — everything after this phase is optimization against it.
- **Done when:** F_0.5 printed and captured.
- **PROGRESS.md:** `Phase 4 DONE — val macro F_0.5 = X (tau=Y, margin=Z).`

### Phase 5 — Full test inference + validator
- **Build:** `src/pipeline/run_inference_pipeline.py` — runs Phases 1–4's logic on
  `dataset/test/*` → `output/candidate_pairs.tsv`, `output/matching_results.tsv`.
- **Read:** `run_train_pipeline.py` interface only (what it saved, not its body).
- **Test:**
  ```
  python3 utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test
  ```
- **Accuracy check:** validator must print `PASS`. Manually confirm France entities
  appear in output (country wasn't hard-filtered).
- **Done when:** `PASS`, submittable file exists.
- **PROGRESS.md:** `Phase 5 DONE — validator PASS, submission-ready.`

### Phase 6 — Error analysis
- **Build:** `src/evaluation/error_analysis.py` — splits validation misses into
  blocking-misses (never a candidate) vs classifier-misses (was a candidate, scored
  wrong).
- **Read:** validation predictions + ground truth only.
- **Test:** prints counts + a handful of examples per category.
- **Accuracy check:** the counts *are* the accuracy artifact — no separate metric.
- **Done when:** you have a numeric answer to "is blocking or the model the
  bottleneck?"
- **PROGRESS.md:** `Phase 6 DONE — blocking misses: N, classifier misses: M. Verdict: <embeddings needed / not needed>.`

### Phase 7 — Embeddings (conditional — only if Phase 6 shows a real blocking gap)
- **Build:** `src/embeddings/embedding_blocking.py` implementing `BaseBlocker`,
  registered as an additional strategy in `config.BLOCKING_STRATEGIES`.
- **Read:** `base.py` interface + Phase 6's specific miss examples only.
- **Test:** re-run Phase 1's recall scorer with the new strategy added.
- **Accuracy check:** recall and final F_0.5 must both improve over Phase 5's
  baseline, or revert this phase.
- **Done when:** improvement is measured and documented, or reverted.
- **PROGRESS.md:** `Phase 7 DONE/REVERTED — recall X→Y, F_0.5 X→Y.`

### Phase 8 — Packaging
- **Build:** pin dependency versions in the existing `pyproject.toml` (don't create a
  separate `requirements.txt` unless the challenge submission format requires one —
  check the README first), `code/business_entity_resolution/README.md` (exact run
  instructions data→output), fill `Documentation_template.md`.
- **Read:** nothing — this is assembly, not logic.
- **Test:** fresh venv + clone reproduces both output files from raw data alone.
- **Accuracy check:** final F_0.5 from Phase 4/5/7 restated in the doc, matches the
  submitted file.
- **Done when:** zip structure matches the README's required layout exactly.
- **PROGRESS.md:** `Phase 8 DONE — packaged, final val F_0.5 = X.`

---

## 5. PROGRESS.md Template (create this file in the repo root; update after every phase)

```
# Progress Log

Phase 0 DONE — ...
Phase 1 DONE — ...
(next: Phase N — <one line on what's next>)
```

Paste this file's contents at the start of any new agent session — combined with
Section 1 of this spec, that's enough context to resume work on any single phase
without re-reading the codebase.
