#!/usr/bin/env python3
"""Score one gpt.ge vision model on the ID test split. Resume-safe.

Writes a sidecar only. It does not rewrite the main JSONL, so it can run
while another scorer is merging that file.
"""

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_URL = "https://api.gpt.ge/v1/chat/completions"
MODEL = os.environ.get("GPTGE_MODEL", "gpt-4.1-nano")
WORKERS = int(os.environ.get("GPTGE_WORKERS", "16"))
MAX_ATTEMPTS = 6
ROOT = Path("/Users/wangcan/Desktop/m2/dataset/result")
SPLITS = {
    "id": (
        ROOT / "m2_id_free_test.jsonl",
        ROOT / "api_runs" / "m2_id_free_test" / f"{MODEL}.jsonl",
        ROOT / "api_runs" / "m2_id_free_test" / f"{MODEL}.score.json",
        "id_test",
    ),
    "train": (
        ROOT / "m2_id_free_train.jsonl",
        ROOT / "api_runs" / "m2_id_free_train" / f"{MODEL}.jsonl",
        ROOT / "api_runs" / "m2_id_free_train" / f"{MODEL}.score.json",
        "id_train",
    ),
}
REMOTE_IMAGE_PREFIX = "/root/autodl-tmp/m2-free-datasets/data/processed/images/"
MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def norm(text):
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def letters_of(row):
    found = "".join(choice.get("letter") or "" for choice in row.get("choices") or [])
    found += str(row.get("ground_truth") or "")
    return "".join(dict.fromkeys(found.upper()))


def parse_mc(text, letters):
    upper = str(text).upper()
    match = re.search(r"\b([" + letters + r"])\b", upper)
    if match:
        return match.group(1)
    match = re.search(r"ANSWER\s*(?:IS|:)?\s*([" + letters + r"])", upper)
    if match:
        return match.group(1)
    if upper[:1] in letters:
        return upper[:1]
    return ""


def local_image(remote):
    if remote and os.path.isfile(remote):
        return remote
    if remote.startswith(REMOTE_IMAGE_PREFIX):
        mapped = ROOT / "images" / remote[len(REMOTE_IMAGE_PREFIX):]
        if mapped.is_file():
            return str(mapped)
    return ""


def image_size(path):
    try:
        text = subprocess.check_output(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", path],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0, 0
    width = height = 0
    for line in text.splitlines():
        if "pixelWidth" in line:
            width = int(line.split()[-1])
        elif "pixelHeight" in line:
            height = int(line.split()[-1])
    return width, height


def prepare_image(path):
    width, height = image_size(path)
    size = os.path.getsize(path)
    long_side = max(width, height)
    if not (width > 0 and height > 0 and (width <= 10 or height <= 10 or size > 2000000 or long_side > 1024)):
        return path, None
    handle = tempfile.NamedTemporaryFile(prefix="armsimg", suffix=".jpg", delete=False)
    handle.close()
    scaled = handle.name
    if width <= 10 or height <= 10:
        if width <= height and width < 16:
            new_h = max(1, height * 16 // width)
            new_w = 16
        else:
            new_w = max(1, width * 16 // max(height, 1))
            new_h = 16
        command = ["sips", "-z", str(new_h), str(new_w), "-s", "format", "jpeg", "-s", "formatOptions", "85", path, "--out", scaled]
    else:
        command = ["sips", "-Z", "1024", "-s", "format", "jpeg", "-s", "formatOptions", "75", path, "--out", scaled]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if os.path.getsize(scaled) > 1500000:
        subprocess.run(
            ["sips", "-Z", "768", "-s", "format", "jpeg", "-s", "formatOptions", "60", scaled, "--out", scaled],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    if os.path.getsize(scaled) > 0:
        return scaled, scaled
    os.unlink(scaled)
    return path, None


def judge(row, output, error):
    if error or not str(output).strip():
        parsed = ""
        if row.get("answer_type") == "multiple_choice" and output and not error:
            parsed = parse_mc(output, letters_of(row))
        return parsed, False
    if row.get("answer_type") == "multiple_choice":
        parsed = parse_mc(output, letters_of(row))
        ok = parsed != "" and parsed.upper() == str(row.get("ground_truth") or "").upper()
        return parsed, ok
    parsed = norm(output)
    return parsed, parsed != "" and parsed == norm(row.get("ground_truth"))


def output_record(body, wall, width, height, output, error, label, parsed):
    usage = body.get("usage") or {} if isinstance(body, dict) else {}
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    total = usage.get("total_tokens") or (prompt + completion)
    details = usage.get("prompt_tokens_details") or {}
    text_in = details.get("text_tokens")
    input_text = prompt if text_in is None else text_in
    image_tokens = 0 if text_in is None else max(prompt - text_in, 0)
    text_total = input_text + completion
    return {
        "vlm_output": output,
        "error": error,
        "parsed_answer": parsed,
        "label": label,
        "token_usage": {
            "input_text_tokens": input_text,
            "image_tokens_estimated": image_tokens,
            "output_tokens": completion,
            "total_tokens": total,
        },
        "cost_tokens": total,
        "latency_s_estimated": wall,
        "latency_allocation_method": "per-request gpt.ge HTTP wall clock; latency_s_estimated_text_only = wall_clock * text_tokens / total_tokens",
        "text_token_usage": {
            "input_tokens": input_text,
            "output_tokens": completion,
            "total_tokens": text_total,
        },
        "text_cost_tokens": text_total,
        "latency_s_estimated_text_only": (wall * text_total / total) if total else wall,
        "image_token_estimation_method": "gpt.ge usage.prompt_tokens minus prompt_tokens_details.text_tokens when text_tokens is reported; otherwise image tokens are 0 and prompt_tokens are counted as text",
        "image_size": {"width": width, "height": height},
    }


def request_once(image_path, prompt):
    suffix = Path(image_path).suffix.lower()
    mime = MIME.get(suffix, "image/jpeg")
    payload = {
        "model": MODEL,
        "max_tokens": 64,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(Path(image_path).read_bytes()).decode()}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    if MODEL.startswith("gpt-5"):
        payload["reasoning_effort"] = "minimal"
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + os.environ["GPTGE_API_KEY"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            body = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        status = exc.code
    return status, body, time.time() - started


def call_one(row):
    image = local_image(row.get("image_path") or "")
    if not image:
        return None
    width, height = image_size(image)
    send_path, temp_path = prepare_image(image)
    error = ""
    output = ""
    status = 0
    wall = 0.0
    body = {}
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                status, raw, wall = request_once(send_path, row.get("prompt") or row.get("question") or "")
            except Exception as exc:
                status, raw, wall = 0, "", 0.0
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 * attempt, 30))
                continue
            if status == 200:
                body = json.loads(raw)
                output = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                error = ""
                break
            try:
                data = json.loads(raw)
                err = data.get("error") or {}
                error = err.get("message") or raw[:300]
            except Exception:
                error = raw[:300] or f"http {status}"
            if status in (429, 500, 502, 503, 0) or any(token in error.lower() for token in ("rate limit", "too many", "throttl")):
                time.sleep(min(2 ** (attempt - 1), 30))
                continue
            break
        else:
            return None
        if status in (429, 500, 502, 503, 0):
            return None
        parsed, label = judge(row, output, error if status != 200 else "")
        return {
            "query_id": row["query_id"],
            "output": output_record(body if status == 200 else {}, wall, width, height, output, error if status != 200 else "", label, parsed),
        }
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def load_done(sidecar):
    done = set()
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["query_id"])
    return done


def summarize(sidecar, score, split_name):
    rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
    correct = sum(1 for row in rows if row["output"]["label"])
    errors = sum(1 for row in rows if row["output"]["error"])
    report = {
        "model": MODEL,
        "split": split_name,
        "correct": correct,
        "queries": len(rows),
        "em": round(correct / len(rows), 6) if rows else None,
        "errors": errors,
    }
    score.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def score_split(split_key):
    source, sidecar, score, split_name = SPLITS[split_key]
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(sidecar)
    pending = []
    with source.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["query_id"] not in done:
                pending.append(row)
    print(f"{split_name} {MODEL} done={len(done)} pending={len(pending)} workers={WORKERS}", flush=True)
    if pending:
        with sidecar.open("a") as handle:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(call_one, row) for row in pending]
                finished = 0
                for future in as_completed(futures):
                    result = future.result()
                    finished += 1
                    if result is None:
                        if finished % 25 == 0:
                            print(f"{split_name} progress {finished}/{len(pending)} retry_skip", flush=True)
                        continue
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    if finished % 25 == 0 or finished == len(pending):
                        print(f"{split_name} progress {finished}/{len(pending)}", flush=True)
    return summarize(sidecar, score, split_name)


def merge_split(split_key):
    source, sidecar, _score, split_name = SPLITS[split_key]
    indexed = {}
    for line in sidecar.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        indexed[row["query_id"]] = row["output"]
    temporary = source.with_suffix(".jsonl.tmp")
    written = 0
    with source.open() as src, temporary.open("w") as dst:
        for line in src:
            row = json.loads(line)
            output = indexed.get(row["query_id"])
            if output is not None:
                row.setdefault("model_outputs", {})[MODEL] = output
                written += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, source)
    print(f"{split_name} merged {written} {MODEL} rows into {source}", flush=True)


def main():
    if len(os.environ.get("GPTGE_API_KEY", "")) < 20:
        raise SystemExit("GPTGE_API_KEY is not set")
    split_key = sys.argv[1] if len(sys.argv) > 1 else "id"
    if split_key not in SPLITS:
        raise SystemExit("usage: run_gptge_id.py [id|train]")
    if len(sys.argv) > 2 and sys.argv[2] == "--merge":
        merge_split(split_key)
        return
    score_split(split_key)


if __name__ == "__main__":
    main()
