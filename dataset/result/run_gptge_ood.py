#!/usr/bin/env python3
"""Score one gpt.ge vision model on a local OOD query file. Resume-safe.

The record format matches run_dashscope_ood.py. gpt-5 models send
reasoning_effort=minimal so the 64-token budget is not spent on hidden reasoning.
"""

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_URL = "https://api.gpt.ge/v1/chat/completions"
WORKERS = int(os.environ.get("GPTGE_WORKERS", "16"))
MAX_ATTEMPTS = 6
MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
RETRY_TEXT = ("rate limit", "too many", "throttl", "validate api key", "try again later")


def norm(text):
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def parse_mc(text, letters):
    upper = str(text).upper()
    allowed = "".join(letters)
    match = re.search(r"\b([" + allowed + r"])\b", upper)
    if match:
        return match.group(1)
    match = re.search(r"ANSWER\s*(?:IS|:)?\s*([" + allowed + r"])", upper)
    if match:
        return match.group(1)
    if upper[:1] in allowed:
        return upper[:1]
    return ""


def number_of(text):
    match = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def judge(row, output, error):
    if error:
        return "", False, False
    if row["answer_type"] == "multiple_choice":
        letters = [c["letter"] for c in row["choices"]] or list(row["correct_letters"])
        parsed = parse_mc(output, letters)
        ok = parsed in set(row["correct_letters"])
        return parsed, ok, ok
    parsed = norm(output)
    accepted = {norm(x) for x in row["correct_answers"]}
    strict = parsed in accepted and parsed != ""
    got = number_of(output)
    numeric_ok = False
    if got is not None:
        for answer in row["correct_answers"]:
            target = number_of(answer)
            if target is not None and abs(got - target) <= 1e-6 * max(1.0, abs(target)):
                numeric_ok = True
                break
    return parsed, strict, strict or numeric_ok


def request_once(model, image_path, prompt):
    suffix = Path(image_path).suffix.lower()
    mime = MIME.get(suffix, "application/octet-stream")
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {
        "model": model,
        "max_tokens": 64,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    if model.startswith("gpt-5"):
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


def retryable(status, error):
    lowered = error.lower()
    return status in (429, 500, 502, 503, 0) or any(token in lowered for token in RETRY_TEXT)


def call_one(model, row):
    error = ""
    output = ""
    status = 0
    wall = 0.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            status, body, wall = request_once(model, row["image_path"], row["prompt"])
        except Exception as exc:
            status, body, wall = 0, "", 0.0
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** (attempt - 1), 30))
            continue
        if status == 200:
            data = json.loads(body)
            output = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            error = ""
            break
        try:
            data = json.loads(body)
            err = data.get("error") or {}
            error = err.get("message") or body[:300]
        except Exception:
            error = body[:300] or f"http {status}"
        if retryable(status, error):
            time.sleep(min(2 ** (attempt - 1), 30))
            continue
        break
    else:
        return None
    if retryable(status, error):
        return None
    parsed, strict, loose = judge(row, output, error if status != 200 else "")
    return {
        "query_id": row["query_id"],
        "grade": row["grade"],
        "question_type": row["question_type"],
        "answer_type": row["answer_type"],
        "ground_truth": row["ground_truth"],
        "correct_answers": row["correct_answers"],
        "vlm_output": output,
        "error": error if status != 200 else "",
        "http": status,
        "parsed_answer": parsed,
        "label": strict,
        "label_numeric": loose,
        "latency_s": round(wall, 3),
    }


def load_done(sidecar):
    done = {}
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["query_id"]] = row
    return done


def summarize(model, rows):
    def pack(subset):
        n = len(subset)
        strict = sum(1 for r in subset if r["label"])
        loose = sum(1 for r in subset if r["label_numeric"])
        errors = sum(1 for r in subset if r["error"])
        return {
            "n": n,
            "correct": strict,
            "em": round(strict / n, 4) if n else None,
            "correct_numeric": loose,
            "em_numeric": round(loose / n, 4) if n else None,
            "errors": errors,
        }

    groups = {}
    for row in rows:
        groups.setdefault(row["question_type"], []).append(row)
    return {
        "model": model,
        "n": len(rows),
        "overall": pack(rows),
        "by_slice": {name: pack(subset) for name, subset in sorted(groups.items())},
    }


def run(root, model):
    if len(os.environ.get("GPTGE_API_KEY", "")) < 20:
        raise SystemExit("GPTGE_API_KEY is not set")
    root = Path(root)
    queries = [json.loads(line) for line in (root / "queries.jsonl").read_text().splitlines() if line.strip()]
    sidecar = root / f"{model}.jsonl"
    score = root / f"{model}.score.json"
    done = load_done(sidecar)
    pending = [row for row in queries if row["query_id"] not in done]
    print(f"{root.name} {model} queries={len(queries)} done={len(done)} pending={len(pending)} workers={WORKERS}", flush=True)
    if pending:
        with sidecar.open("a") as handle:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(call_one, model, row) for row in pending]
                finished = 0
                for future in as_completed(futures):
                    result = future.result()
                    finished += 1
                    if result is None:
                        if finished % 25 == 0:
                            print(f"{root.name} progress {finished} retry_skip", flush=True)
                        continue
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    if finished % 25 == 0 or finished == len(pending):
                        print(f"{root.name} progress {finished}/{len(pending)} http={result['http']}", flush=True)
    report = summarize(model, list(load_done(sidecar).values()))
    score.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"root": root.name, **report}, ensure_ascii=False), flush=True)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_gptge_ood.py ROOT MODEL")
    run(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
