#!/usr/bin/env python3
"""Score deepseek-v4.1-flash on the local MathVista queries. Resume-safe."""

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Users/wangcan/Desktop/m2/dataset/result/mathvision")
QUERIES = ROOT / "queries.jsonl"
SIDECAR = ROOT / "deepseek-v4.1-flash.jsonl"
SCORE = ROOT / "deepseek-v4.1-flash.score.json"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
WORKERS = 8
MAX_ATTEMPTS = 6


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


def request_once(image_path, prompt):
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {
        "model": MODEL,
        "max_tokens": 64,
        "temperature": 0,
        "enable_thinking": False,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + os.environ["DASHSCOPE_API_KEY"],
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
    wall = time.time() - started
    return status, body, wall


def call_one(row):
    error = ""
    output = ""
    status = 0
    wall = 0.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            status, body, wall = request_once(row["image_path"], row["prompt"])
        except Exception as exc:
            status, body, wall = 0, "", 0.0
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(attempt * 2)
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
        if status in (429, 500, 502, 503, 0):
            time.sleep(attempt * 2)
            continue
        break
    else:
        return None
    if status in (429, 500, 502, 503, 0):
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


def load_done():
    done = {}
    if SIDECAR.exists():
        for line in SIDECAR.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done[row["query_id"]] = row
    return done


def summarize(rows):
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
        groups.setdefault((row["grade"], row["question_type"]), []).append(row)
    by = {
        f"{grade}:{qtype}": pack(subset)
        for (grade, qtype), subset in sorted(groups.items())
    }
    mc = [r for r in rows if r["answer_type"] == "multiple_choice"]
    ff = [r for r in rows if r["answer_type"] == "free_form"]
    hs_col = [r for r in mc if r["grade"] in ("high_school", "college")]
    return {
        "model": MODEL,
        "n": len(rows),
        "overall": pack(rows),
        "multiple_choice": pack(mc),
        "free_form": pack(ff),
        "high_school_college_multiple_choice": pack(hs_col),
        "by_slice": by,
    }


def main():
    if len(os.environ.get("DASHSCOPE_API_KEY", "")) < 20:
        raise SystemExit("DASHSCOPE_API_KEY is not set")
    queries = [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]
    done = load_done()
    pending = [row for row in queries if row["query_id"] not in done]
    print(f"queries={len(queries)} done={len(done)} pending={len(pending)}", flush=True)
    if pending:
        with SIDECAR.open("a") as handle:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(call_one, row) for row in pending]
                finished = 0
                for future in as_completed(futures):
                    result = future.result()
                    finished += 1
                    if result is None:
                        if finished % 25 == 0:
                            print(f"progress finished={finished} saved_skip", flush=True)
                        continue
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    if finished % 25 == 0 or finished == len(pending):
                        print(f"progress finished={finished}/{len(pending)} http={result['http']}", flush=True)
    rows = list(load_done().values())
    report = summarize(rows)
    SCORE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
