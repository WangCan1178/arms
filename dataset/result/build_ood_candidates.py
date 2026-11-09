#!/usr/bin/env python3
"""Export full official splits into the local query format. No item subsampling."""

import base64
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/Users/wangcan/Desktop/m2/dataset/result")
SUFFIX = "Answer the multiple choice question by just giving the letter of the correct answer."


def magic_suffix(raw):
    if raw.startswith(b"\xff\xd8"):
        return ".jpg"
    if raw.startswith(b"\x89PNG"):
        return ".png"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return ".webp"
    if raw.startswith(b"GIF8"):
        return ".gif"
    return ".bin"


def image_bytes(value):
    if isinstance(value, dict):
        raw = value.get("bytes") or b""
        if raw:
            return raw
        path = value.get("path") or ""
        if path:
            return Path(path).read_bytes()
        return b""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("data:"):
            text = text.split(",", 1)[-1]
        return base64.b64decode(text)
    return b""


def save_image(folder, raw):
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(raw).hexdigest()
    path = folder / f"{digest}{magic_suffix(raw)}"
    if not path.exists():
        path.write_bytes(raw)
    return str(path)


def choice(letter, text):
    return {"letter": letter, "text": str(text).strip()}


def mc_row(query_id, grade, question_type, question, prompt, letter, choices, image_path):
    letter = str(letter).strip().upper()[:1]
    return {
        "query_id": query_id,
        "grade": grade,
        "question_type": question_type,
        "answer_type": "multiple_choice",
        "question": question,
        "prompt": prompt,
        "ground_truth": letter,
        "correct_answers": [letter],
        "correct_letters": [letter],
        "choices": choices,
        "image_path": image_path,
    }


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {path} n={len(rows)}", flush=True)


def export_hr():
    folder = ROOT / "hrbench" / "images"
    rows = []
    pf = pq.ParquetFile(ROOT / "hrbench" / "hr_bench_4k.parquet")
    for batch in pf.iter_batches(batch_size=4):
        data = batch.to_pydict()
        for i in range(len(data["index"])):
            raw = image_bytes(data["image"][i])
            letters = []
            choices = []
            lines = [str(data["question"][i]).strip()]
            for letter in "ABCD":
                text = str(data[letter][i]).strip()
                letters.append(letter)
                choices.append(choice(letter, text))
                lines.append(f"({letter}) {text}")
            lines.append(SUFFIX)
            rows.append(mc_row(
                f"hrbench4k:{data['index'][i]}",
                "hrbench_4k",
                str(data["category"][i]),
                str(data["question"][i]).strip(),
                "\n".join(lines),
                data["answer"][i],
                choices,
                save_image(folder, raw),
            ))
    write_rows(ROOT / "hrbench" / "queries.jsonl", rows)


def export_vstar():
    folder = ROOT / "vstar" / "images"
    table = pq.read_table(ROOT / "vstar" / "test.parquet")
    rows = []
    data = table.to_pydict()
    for i in range(table.num_rows):
        text = str(data["text"][i]).strip()
        found = re.findall(r"\(([A-D])\)\s*([^\n(]+)", text)
        choices = [choice(letter, body.strip()) for letter, body in found]
        rows.append(mc_row(
            f"vstar:{data['question_id'][i]}",
            "vstar",
            str(data["category"][i]),
            text,
            text,
            data["label"][i],
            choices,
            save_image(folder, image_bytes(data["image"][i])),
        ))
    write_rows(ROOT / "vstar" / "queries.jsonl", rows)


def export_mmvp():
    folder = ROOT / "mmvp" / "images"
    table = pq.read_table(ROOT / "mmvp" / "train.parquet")
    rows = []
    data = table.to_pydict()
    for i in range(table.num_rows):
        options = str(data["Options"][i])
        found = re.findall(r"\(([A-Za-z])\)\s*([^()]+)", options)
        choices = []
        lines = [str(data["Question"][i]).strip()]
        for letter, body in found:
            letter = letter.upper()
            choices.append(choice(letter, body))
            lines.append(f"({letter}) {body.strip()}")
        lines.append(SUFFIX)
        answer = str(data["Correct Answer"][i]).strip().upper().replace("(", "").replace(")", "")
        rows.append(mc_row(
            f"mmvp:{data['Index'][i]}",
            "mmvp",
            "visual_pattern",
            str(data["Question"][i]).strip(),
            "\n".join(lines),
            answer,
            choices,
            save_image(folder, image_bytes(data["image"][i])),
        ))
    write_rows(ROOT / "mmvp" / "queries.jsonl", rows)


def export_realworld():
    folder = ROOT / "realworldqa" / "images"
    rows = []
    files = sorted((ROOT / "realworldqa").glob("test-*.parquet"))
    serial = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=8):
            data = batch.to_pydict()
            for i in range(len(data["question"])):
                question = str(data["question"][i]).strip()
                answer = str(data["answer"][i]).strip()
                image_path = save_image(folder, image_bytes(data["image"][i]))
                query_id = f"realworldqa:{serial:04d}"
                serial += 1
                if re.fullmatch(r"[A-H]", answer):
                    found = re.findall(r"([A-H])[\.\)]\s*(.*?)(?=(?:[A-H][\.\)])|Please answer|$)", question)
                    choices = [choice(letter, body) for letter, body in found]
                    rows.append(mc_row(
                        query_id, "realworldqa", "multiple_choice", question, question,
                        answer, choices, image_path,
                    ))
                else:
                    rows.append({
                        "query_id": query_id,
                        "grade": "realworldqa",
                        "question_type": "short_answer",
                        "answer_type": "free_form",
                        "question": question,
                        "prompt": question,
                        "ground_truth": answer,
                        "correct_answers": [answer],
                        "correct_letters": [],
                        "choices": [],
                        "image_path": image_path,
                    })
    write_rows(ROOT / "realworldqa" / "queries.jsonl", rows)


def export_hallusion():
    folder = ROOT / "hallusion" / "images"
    pf = pq.ParquetFile(ROOT / "hallusion" / "image.parquet")
    rows = []
    serial = 0
    for batch in pf.iter_batches(batch_size=16):
        data = batch.to_pydict()
        for i in range(len(data["question"])):
            question = str(data["question"][i]).strip()
            gt = str(data["gt_answer"][i]).strip()
            letter = "A" if gt in ("1", "yes", "Yes") else "B"
            choices = [choice("A", "Yes"), choice("B", "No")]
            prompt = f"{question}\n(A) Yes\n(B) No\n{SUFFIX}"
            rows.append(mc_row(
                f"hallusion:{serial:04d}",
                "hallusion_image",
                f"{data['category'][i]}:{data['subcategory'][i]}",
                question,
                prompt,
                letter,
                choices,
                save_image(folder, image_bytes(data["image"][i])),
            ))
            serial += 1
    write_rows(ROOT / "hallusion" / "queries.jsonl", rows)


def main():
    export_vstar()
    export_mmvp()
    export_hr()
    export_realworld()
    export_hallusion()


if __name__ == "__main__":
    main()
