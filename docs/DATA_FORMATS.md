# Frozen data formats

Written by agents C and B, read by the TUI and the trainer. Treat these keys as an
append-only contract: add fields, never rename or remove one. Every record is one JSON
object per line, written with `autorlhf.storage.runs.append_jsonl`.

All records carry `run_id`, a `dataset_version` string, and `created_at`
(ISO-8601 UTC, seconds).

## `dataset.jsonl` — one record per candidate response

```json
{
  "record_type": "candidate",
  "dataset_version": "1",
  "run_id": "20260101-120000-ab12",
  "prompt_id": "p0001",
  "candidate_id": "p0001-c0",
  "split": "train",
  "category": "ordinary",
  "prompt": "Will the fog lift today?",
  "prompt_context": [{ "role": "user", "content": "..." }],
  "response": "Not before the afternoon tide...",
  "source": "teacher",
  "model_id": "openai/gpt-4o-mini",
  "status": "accepted",
  "error": null,
  "truncated": false,
  "tokens": 84,
  "latency_s": 1.42,
  "cached": false,
  "seed": 0,
  "temperature": 0.8,
  "edits": [],
  "edited": false,
  "overall": 3.66,
  "criterion_scores": { "character_consistency": 4.0, "voice_style": 3.5 },
  "judgement": { "...": "the full Judgement model_dump(mode='json')" },
  "reviewer": "",
  "created_at": "2026-01-01T12:00:00"
}
```

- `status`: `pending` | `accepted` | `rejected` | `edited` | `failed` (`SampleStatus`).
- `split`: `train` | `validation` | `eval` (`SplitName`). The split is assigned at the
  **prompt** level: every candidate of a prompt always carries that prompt's split.
- `judgement` is `null` when judging failed; `criterion_scores` then holds only the
  criteria that returned a score, and `overall` is `null`. A failed judgement must never
  be represented as a score of zero.

## `preferences.jsonl` — one record per accepted preference pair

```json
{
  "record_type": "preference",
  "dataset_version": "1",
  "run_id": "...",
  "pair_id": "p0001-a-vs-b",
  "prompt_id": "p0001",
  "split": "train",
  "prompt": "...",
  "prompt_context": [],
  "chosen_id": "p0001-c1",
  "rejected_id": "p0001-c0",
  "chosen_text": "...",
  "rejected_text": "...",
  "margin": 0.9,
  "judge_confidence": 0.81,
  "source": "jev",
  "decision": "b",
  "probabilities": { "a": 0.19, "b": 0.81 },
  "criterion_deltas": { "character_consistency": 1.0 },
  "audit": {
    "thresholds": { "tie_epsilon": 0.35, "min_confidence": 0.6 },
    "reasons": [
      "overall margin 0.90 > tie_epsilon 0.35",
      "judge confidence 0.81 >= 0.60"
    ],
    "rejected_pairs": []
  },
  "pair_decision": { "...": "the full PairDecision model_dump(mode='json')" },
  "reviewer": "",
  "created_at": "..."
}
```

- `source`: `jev` | `human`.
- Only pairs that pass the reproducibility thresholds are written. Ties, failed
  generations, malformed results, and sub-threshold margins are recorded in the
  `audit.rejected_pairs` list of the nearest accepted pair's prompt **and** in
  `judgments.jsonl`, so the reason is auditable and never silently dropped.

## `sft.jsonl` — one record per supervised example

```json
{
  "record_type": "sft",
  "dataset_version": "1",
  "run_id": "...",
  "example_id": "s0001",
  "prompt_id": "seed-1",
  "split": "train",
  "messages": [
    { "role": "system", "content": "You are Aria..." },
    { "role": "user", "content": "Will the fog lift today?" },
    { "role": "assistant", "content": "Not before the afternoon tide..." }
  ],
  "source": "seed",
  "status": "accepted",
  "created_at": "..."
}
```

- `source`: `seed` | `accepted` | `edited` | `human`.
- The first message **is** the character system prompt, so the trained model sees the
  same framing it will see at inference.

## `eval_prompts.jsonl` — one record per held-out prompt

```json
{
  "record_type": "eval_prompt",
  "run_id": "...",
  "prompt_id": "e0001",
  "text": "...",
  "context": [],
  "category": "boundary",
  "origin": "synthetic"
}
```

Held-out prompts are never used to generate training examples, choose preferences, or tune
hyperparameters. `eval_prompts_hash` (the file's content hash) is stored in both evaluation
reports so the base and fine-tuned evaluations provably used the same prompts.

## `judgments.jsonl` — audit trail of every judge call

```json
{
  "record_type": "judgement",
  "run_id": "...",
  "judgement_id": "...",
  "prompt_id": "p0001",
  "candidate_id": "p0001-c0",
  "phase": "dataset_review",
  "kind": "score",
  "judge_model": "typesafe/jev-1.13",
  "judge_build": "typesafe/jev-1.13-20260917",
  "cached": false,
  "error": null,
  "retries": 0,
  "usage": { "input_tokens": 476, "cost": 0.00002 },
  "human_override": null,
  "created_at": "...",
  "judgement": { "...": "full Judgement JSON" }
}
```

`kind` is `score` | `choice` | `noul`. Pairwise comparisons are written with
`"kind": "choice"` and the `PairDecision` JSON under `"pair_decision"` instead of
`"judgement"`.

## `baseline_eval.json` / `final_eval.json`

`EvaluationReport.model_dump(mode="json")`.

## `comparison.json`

`ComparisonReport.model_dump(mode="json")`.

## `splits.json` — the persisted split

```json
{
  "run_id": "...",
  "version": "1",
  "dataset_hash": "the dataset_fingerprint this split belongs to",
  "split_hash": "sha256 of the assignment below",
  "seed": 0,
  "fractions": { "eval": 0.2, "validation": 0.15 },
  "counts": { "train": 12, "validation": 4, "eval": 5 },
  "assignments": { "p0001": "train", "p0002": "eval" },
  "created_at": "..."
}
```

Resuming re-reads this file and reuses the assignment whenever `dataset_hash` still
matches, so a run can never re-split its own data and leak held-out prompts.
