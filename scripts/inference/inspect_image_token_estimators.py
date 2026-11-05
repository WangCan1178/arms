#!/usr/bin/env python3
"""Inspect processor outputs for estimating VLM image tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image
from transformers import AutoConfig, AutoProcessor


MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
]


def resolve_snapshot_path(model_name: str, hf_home: Path) -> Path | str:
    snapshots = hf_home / "hub" / ("models--" + model_name.replace("/", "--")) / "snapshots"
    candidates = [path for path in snapshots.iterdir() if path.is_dir()] if snapshots.exists() else []
    if not candidates:
        return model_name
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def apply_chat_template(processor: Any, prompt: str) -> Any:
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


def shape_of(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, list):
        if not value:
            return [0]
        return [len(value), shape_of(value[0])]
    return type(value).__name__


def main() -> int:
    root = Path("/root/autodl-tmp/m2-free-datasets")
    hf_home = root / "hf_cache"
    with (root / "outputs/id/id_qwen.jsonl").open(encoding="utf-8") as handle:
        row = json.loads(next(handle))
    image = Image.open(row["image_path"]).convert("RGB")
    prompt = row["prompt"]
    print("sample", row["query_id"], image.size, prompt)
    for model in MODELS:
        print("\nMODEL", model)
        location = resolve_snapshot_path(model, hf_home)
        processor = AutoProcessor.from_pretrained(
            location,
            cache_dir=str(hf_home),
            local_files_only=True,
            trust_remote_code=True,
        )
        try:
            config = AutoConfig.from_pretrained(
                location,
                cache_dir=str(hf_home),
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            print("config_error", type(exc).__name__, exc)
            config = None
        text = apply_chat_template(processor, prompt)
        print("chat_text_type", type(text).__name__, str(text)[:200].replace("\n", "\\n"))
        try:
            encoded = processor(text=text, images=image, return_tensors="pt")
        except Exception as exc:  # noqa: BLE001
            print("processor_call_error", type(exc).__name__, exc)
            encoded = {}
        print("processor_attrs", {
            name: getattr(processor, name)
            for name in ["image_token_id", "image_seq_len", "fake_image_token", "image_token"]
            if hasattr(processor, name)
        })
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            attrs = {}
            for name in [
                "patch_size",
                "merge_size",
                "temporal_patch_size",
                "image_seq_len",
                "size",
                "crop_size",
                "do_resize",
                "max_image_size",
                "max_num_images",
                "max_num_tiles",
            ]:
                if hasattr(image_processor, name):
                    attrs[name] = getattr(image_processor, name)
            print("image_processor_attrs", attrs)
        if config is not None:
            cfg = {}
            for name in ["image_token_index", "vision_config", "vision_model_name", "image_seq_len"]:
                if hasattr(config, name):
                    value = getattr(config, name)
                    if name == "vision_config":
                        value = {
                            key: getattr(value, key)
                            for key in [
                                "image_size",
                                "patch_size",
                                "num_channels",
                                "num_hidden_layers",
                                "vision_output_dim",
                                "num_global_layers",
                            ]
                            if hasattr(value, key)
                        }
                    cfg[name] = value
            print("config_attrs", cfg)
        print("encoded_keys", sorted(encoded.keys()))
        for key, value in encoded.items():
            print(key, shape_of(value))
            if key in {"image_grid_thw", "aspect_ratio_mask", "input_ids"}:
                try:
                    print(key + "_values", value.tolist() if hasattr(value, "tolist") else value)
                except Exception as exc:  # noqa: BLE001
                    print(key + "_values_error", type(exc).__name__, exc)
        ids = encoded.get("input_ids")
        if ids is not None:
            flat = ids.reshape(-1).tolist() if hasattr(ids, "reshape") else []
            counts: Dict[str, int] = {}
            for special in ["<image>", "<|image|>", "<|image_pad|>", "<fake_token_around_image>"]:
                try:
                    token_id = processor.tokenizer.convert_tokens_to_ids(special)
                    counts[special] = flat.count(token_id)
                except Exception:  # noqa: BLE001
                    pass
            print("special_counts", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
