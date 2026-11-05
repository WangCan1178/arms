#!/usr/bin/env python3
"""Run local open-source VLM inference over standardized M2 query JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def completed_query_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(path):
        query_id = row.get("query_id")
        if query_id:
            done.add(str(query_id))
    return done


def load_image(path: str) -> Image.Image:
    image = Image.open(path)
    image.load()
    return image.convert("RGB")


def load_model_and_processor(model_name: str, cache_dir: Optional[Path]):
    import torch
    from transformers import AutoProcessor

    kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    processor_kwargs = {"trust_remote_code": True}
    if cache_dir:
        processor_kwargs["cache_dir"] = str(cache_dir)

    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    model = None
    errors: List[str] = []
    class_names = []
    if "Qwen2.5-VL" in model_name:
        class_names.append("Qwen2_5_VLForConditionalGeneration")
    if "Llama-3.2" in model_name:
        class_names.append("MllamaForConditionalGeneration")
    class_names.extend(["AutoModelForVision2Seq", "AutoModelForImageTextToText"])

    import transformers

    for class_name in class_names:
        cls = getattr(transformers, class_name, None)
        if cls is None:
            continue
        for attn in ("sdpa", None):
            try:
                attempt_kwargs = dict(kwargs)
                if attn is not None:
                    attempt_kwargs["attn_implementation"] = attn
                model = cls.from_pretrained(model_name, **attempt_kwargs)
                model.eval()
                return model, processor
            except TypeError as exc:
                errors.append(f"{class_name}(attn={attn}) TypeError: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{class_name}(attn={attn}) {type(exc).__name__}: {exc}")

    raise RuntimeError("Could not load model. Tried: " + " | ".join(errors[-6:]))


def apply_chat_template(processor: Any, prompt: str, image: Image.Image) -> Any:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        return processor.apply_chat_template(messages, add_generation_prompt=True)
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def generate_one(model: Any, processor: Any, query: Dict[str, Any], max_new_tokens: int) -> str:
    import torch

    prompt = query.get("prompt") or query.get("question") or ""
    image = load_image(query["image_path"])
    text = apply_chat_template(processor, prompt, image)
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    new_tokens = generated[:, input_len:] if input_len else generated
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return str(new_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        print("This implementation currently processes one image at a time; batch_size is accepted for CLI compatibility.")

    done = completed_query_ids(args.output)
    print(f"Loaded {len(done)} completed query_id values from {args.output}")
    model, processor = load_model_and_processor(args.model, args.cache_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for query in tqdm(iter_jsonl(args.input), desc=f"inference {args.model}", unit="query"):
            query_id = str(query.get("query_id"))
            if query_id in done:
                skipped += 1
                continue
            if args.limit is not None and processed >= args.limit:
                break
            output = ""
            error = ""
            try:
                output = generate_one(model, processor, query, args.max_new_tokens)
            except Exception as exc:  # noqa: BLE001 - record per-query failures and continue.
                error = f"{type(exc).__name__}: {exc}"
            row = dict(query)
            row.update({"vlm_name": args.model, "vlm_output": output, "error": error})
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
    print(f"processed={processed} skipped={skipped} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
