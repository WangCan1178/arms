# ARMS Original-Pool Routing Supervision

This directory contains compact query-level supervision for the original
four-candidate experiments in the ARMS paper. Questions, reference answers,
images, source URLs, and raw model generations remain with their upstream
benchmark sources and are not redistributed here.

## Files

| File | Split | Queries | Compressed size |
|---|---|---:|---:|
| `original_pool/original_pool_id_train.jsonl.gz` | ID train | 29,626 | about 203 KB |
| `original_pool/original_pool_id_test.jsonl.gz` | ID test | 3,000 | about 21 KB |
| `original_pool/original_pool_ood_test.jsonl.gz` | OOD test | 7,084 | about 148 KB |
| `original_pool/manifest.json` | schema, task counts, and SHA-256 checksums | — | — |

Together, the files contain 39,710 unique queries and 158,840 binary
query-candidate labels.

## Candidate Pool

Every record contains labels for exactly these four model identifiers:

```text
Qwen/Qwen2.5-VL-7B-Instruct
HuggingFaceM4/idefics2-8b
HuggingFaceTB/SmolVLM2-2.2B-Instruct
meta-llama/Llama-3.2-11B-Vision-Instruct
```

## Schema

The files use gzip-compressed UTF-8 JSON Lines. Each line has exactly seven
fields:

- `query_id`: stable query identifier;
- `benchmark`, `task`, and `language`: routing metadata;
- `answer_type`: evaluation-answer type;
- `image_key`: portable key for the locally reconstructed image;
- `candidate_labels`: one Boolean correctness label for each candidate.

Load a file without extracting it:

```python
import gzip
import json

path = "data/original_pool/original_pool_id_train.jsonl.gz"
with gzip.open(path, "rt", encoding="utf-8") as handle:
    first_record = json.loads(next(handle))

print(first_record["query_id"])
print(first_record["candidate_labels"])
```

## Materializing Training Records

First reconstruct the public benchmark query files and images as described in
[`docs/DATA_REPRODUCTION.md`](../docs/DATA_REPRODUCTION.md). Then join a compact
label split with one or more standardized query files:

```bash
python scripts/materialize_router_records.py \
  --labels data/original_pool/original_pool_id_train.jsonl.gz \
  --queries data/processed/id_queries_standardized.jsonl \
  --project-root . \
  --output data/materialized/original_pool_id_train.jsonl
```

The materialized output contains the reconstructed question and image path plus
the four candidate labels. Use `--allow-missing-images` only when checking the
text-side join before image acquisition is complete.

## Split Construction

The ID training split contains ALM-Bench, MMT-Bench, and MME-RealWorld records.
The ID test split contains 3,000 held-out MMT-Bench records. All labels for a
query remain together, and no query identifier overlaps across released
splits.

The OOD release contains the six VHELM variants used for the paper's
7,084-query original-pool evaluation:

```text
a_okvqa_base
a_okvqa_chinese
a_okvqa_dialect
a_okvqa_hindi
a_okvqa_robustness
a_okvqa_spanish
```

`a_okvqa_swahili`, `crossmodal_3600`, and `bingo` are not part of this paper
split. OOD labels are not used for router training or threshold selection.

## Validation and Deterministic Export

Validate checksums, counts, schema, candidate identities, split disjointness,
OOD task selection, answerability, and path portability with:

```bash
python3 scripts/validate_public_dataset.py data/original_pool
```

`scripts/export_public_dataset.py` deterministically regenerates all three
compressed files and `manifest.json` from the full query-level outputs produced
by `scripts/merge_parse_label_filter.py`.

## Upstream Sources and Terms

- [ALM-Bench](https://huggingface.co/datasets/MBZUAI/ALM-Bench)
- [MMT-Bench](https://github.com/OpenGVLab/MMT-Bench)
- [MME-RealWorld-Base64](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Base64)
- [VHELM/HELM](https://github.com/stanford-crfm/helm)

The upstream datasets have separate terms and licensing conditions. Review
those terms before downloading or redistributing any underlying benchmark
content. The compact ARMS files do not include the underlying questions,
answers, images, or raw model responses.
