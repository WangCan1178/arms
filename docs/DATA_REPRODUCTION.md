# ARMS Data Reconstruction Guide

This guide reconstructs the public-source queries used by the original
four-model pool, regenerates their correctness labels, and exports the compact
paper splits. Commands are run from the repository root.

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
```

If the Llama checkpoint is required, accept its upstream access terms and set a
Hugging Face token in the environment. Do not write credentials into source
files.

## 2. Acquire Public Benchmark Sources

```bash
python scripts/download_open_datasets.py
python scripts/download_vhelm_v201_instances.py
```

The scripts create `data/raw/` and retain upstream provenance information.

## 3. Standardize Queries and Materialize Images

```bash
python scripts/standardize_id_benchmarks.py
python scripts/patch_mmt_answers.py
python scripts/standardize_vhelm.py --download-images
```

The default outputs are:

```text
data/processed/id_queries_standardized.jsonl
data/processed/vhelm_10k_queries.jsonl
```

## 4. Download the Four Open Candidates

```bash
python scripts/download_vlm_models.py
```

The model identifiers are fixed in the script. Model downloads are much larger
than this repository and are not committed.

## 5. Generate Candidate Responses

Run the inference entry point once for each model and split. For example:

```bash
python scripts/run_vlm_inference.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --input data/processed/id_queries_standardized.jsonl \
  --output outputs/id/id_qwen.jsonl \
  --max_new_tokens 16
```

Repeat with:

```text
HuggingFaceM4/idefics2-8b
HuggingFaceTB/SmolVLM2-2.2B-Instruct
meta-llama/Llama-3.2-11B-Vision-Instruct
```

Use the VHELM standardized file for OOD output collection. The inference script
uses each processor's chat template, bfloat16 weights, greedy decoding, and
`do_sample=False`.

## 6. Parse, Label, and Split

After all four output files are complete:

```bash
python scripts/merge_parse_label_filter.py \
  --mode id \
  --inputs outputs/id/id_qwen.jsonl outputs/id/id_idefics2.jsonl \
           outputs/id/id_smolvlm2.jsonl outputs/id/id_llama.jsonl \
  --canonical-input data/processed/id_queries_standardized.jsonl

python scripts/merge_parse_label_filter.py \
  --mode ood \
  --inputs outputs/ood/ood_qwen.jsonl outputs/ood/ood_idefics2.jsonl \
           outputs/ood/ood_smolvlm2.jsonl outputs/ood/ood_llama.jsonl
```

The merge script normalizes free-form answers, parses multiple-choice letters,
computes one binary correctness label per candidate, and retains queries for
which at least one candidate is correct.

## 7. Export the Public Paper Splits

```bash
python scripts/export_public_dataset.py \
  --id-train data/processed/m2_id_free_train.jsonl \
  --id-test data/processed/m2_id_free_test.jsonl \
  --ood-test data/processed/m2_ood_free_test.jsonl \
  --output-dir data/original_pool
```

This command converts image references to portable paths, removes
machine-specific fields, keeps only stable routing metadata and Boolean
candidate labels, selects the 7,084-query OOD split, writes deterministic gzip
files, and generates SHA-256 checksums. Questions, answers, source URLs, and raw
model generations are deliberately excluded from the compact export. The
`data/original_pool` directory is generated here and is not stored in the
repository; the same labels are already present on the four open-source models
in `dataset/result`.

## 8. Validate

```bash
python3 scripts/validate_public_dataset.py data/original_pool
```

Expected counts are 29,626 ID train, 3,000 ID test, and 7,084 OOD test.

## 9. Materialize Records for Training

To train from the compact release, join one label split with the corresponding
standardized queries reconstructed in Steps 2–3:

```bash
python scripts/materialize_router_records.py \
  --labels data/original_pool/original_pool_id_train.jsonl.gz \
  --queries data/processed/id_queries_standardized.jsonl \
  --project-root . \
  --output data/materialized/original_pool_id_train.jsonl
```

For the OOD split, pass the standardized VHELM query file instead. The join is
keyed by `query_id`; it fails on missing queries or images unless
`--allow-missing-images` is explicitly supplied.

## Upstream Data

The acquisition scripts retrieve [ALM-Bench](https://huggingface.co/datasets/MBZUAI/ALM-Bench),
[MMT-Bench](https://github.com/OpenGVLab/MMT-Bench),
[MME-RealWorld-Base64](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Base64),
and [VHELM/HELM](https://github.com/stanford-crfm/helm). These sources remain
subject to their own terms. The repository's compact release does not include
their questions, answers, images, or raw model outputs.
