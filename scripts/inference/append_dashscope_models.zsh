#!/bin/zsh
# Append DashScope vision outputs onto the JSONL splits written by
# merge_parse_label_filter.py. Every original field is kept. New
# model_outputs entries use the same keys as the open-source records.
# Reads DASHSCOPE_API_KEY from the environment; do not store it in the repo.
#
# From the repository root:
#   export DASHSCOPE_API_KEY=...
#   scripts/inference/append_dashscope_models.zsh
#   scripts/inference/append_dashscope_models.zsh --split m2_id_free_test.jsonl --limit 2 --no-merge
#
# Defaults read data/processed/m2_id_free_{test,train}.jsonl and
# data/processed/m2_ood_free_test.jsonl. Sidecars go to
# data/processed/api_runs and are merged back when a split finishes.
# An image_path is used as-is when that file exists. Paths copied from the
# AutoDL prefix are remapped under data/processed/images.

set -u
set +x
setopt NO_XTRACE EXTENDED_GLOB TYPESET_SILENT

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
RESULT_DIR="$REPO_ROOT/data/processed"
IMAGE_ROOT="$RESULT_DIR/images"
RUN_ROOT="$RESULT_DIR/api_runs"
API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
REMOTE_IMAGE_PREFIX="/root/autodl-tmp/m2-free-datasets/data/processed/images/"
VHELM_REMOTE_PREFIX="/root/autodl-tmp/m2-free-datasets/data/raw/VHELM/"
VHELM_ROOT="$REPO_ROOT/data/raw/VHELM"
MODELS=(kimi-k3 deepseek-v4.1-flash qwen3.8-max)
SPLITS=(
  "m2_id_free_test.jsonl"
  "m2_id_free_train.jsonl"
  "m2_ood_free_test.jsonl"
)
LIMIT=0
DO_MERGE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --no-merge) DO_MERGE=0; shift ;;
    --split) SPLITS=("$2"); shift 2 ;;
    --result-dir) RESULT_DIR="$2"; RUN_ROOT="$RESULT_DIR/api_runs"; shift 2 ;;
    --image-root) IMAGE_ROOT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is not set" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/progress.log"
exec 3>>"$LOG"
unsetopt xtrace
set +x
setopt TYPESET_SILENT
exec >/dev/null 2>/dev/null

log() {
  print -r -- "[$(date '+%F %T')] $*" >&3
}

mime_for() {
  local lower="${1:l}"
  case "$lower" in
    *.png) print -r -- "image/png" ;;
    *.webp) print -r -- "image/webp" ;;
    *.gif) print -r -- "image/gif" ;;
    *) print -r -- "image/jpeg" ;;
  esac
}

local_image() {
  local remote="$1"
  if [[ -n "$remote" && -f "$remote" ]]; then
    print -r -- "$remote"
    return
  fi
  if [[ -n "$REMOTE_IMAGE_PREFIX" && "$remote" == "$REMOTE_IMAGE_PREFIX"* ]]; then
    print -r -- "$IMAGE_ROOT/${remote#$REMOTE_IMAGE_PREFIX}"
    return
  fi
  if [[ -n "${VHELM_REMOTE_PREFIX:-}" && "$remote" == "$VHELM_REMOTE_PREFIX"* ]]; then
    print -r -- "$VHELM_ROOT/${remote#$VHELM_REMOTE_PREFIX}"
    return
  fi
  if [[ -n "$remote" && "$remote" != /* && -f "$REPO_ROOT/$remote" ]]; then
    print -r -- "$REPO_ROOT/$remote"
    return
  fi
  print -r -- ""
}

build_output() {
  local body_file="$1" wall="$2" width="$3" height="$4" row_file="$5" output_file="$6" error="$7" dest="$8"
  jq -nc \
    --slurpfile body "$body_file" \
    --argjson wall "$wall" \
    --argjson width "$width" \
    --argjson height "$height" \
    --slurpfile row "$row_file" \
    --rawfile output "$output_file" \
    --arg error "$error" '
      def norm($text):
        $text | tostring | gsub("^\\s+|\\s+$"; "") | gsub("\\s+"; " ") | ascii_downcase;
      ($body[0]) as $body | ($row[0]) as $row |
      def letters:
        ((($row.choices // []) | map(.letter // empty) | join("") | ascii_upcase) + ($row.ground_truth | tostring | ascii_upcase))
        | explode | unique | implode;
      def parse_mc($text):
        letters as $letters
        | ($text | tostring | ascii_upcase) as $u
        | (try ($u | capture("\\b(?<l>[" + $letters + "])\\b").l) catch null)
        // (try ($u | capture("ANSWER\\s*(?:IS|:)?\\s*(?<l>[" + $letters + "])").l) catch null)
        // (if ($letters | contains($u[0:1])) and ($u[0:1] != "") then $u[0:1] else "" end);
      ($row.answer_type == "multiple_choice") as $mc
      | (if $mc then parse_mc($output) else norm($output) end) as $parsed
      | (if $error != "" or $parsed == "" then false
         elif $mc then ($parsed | ascii_upcase) == ($row.ground_truth | tostring | ascii_upcase)
         else $parsed == norm($row.ground_truth)
         end) as $label
      | ($body.usage.prompt_tokens // 0) as $prompt
      | ($body.usage.completion_tokens // 0) as $completion
      | ($body.usage.total_tokens // ($prompt + $completion)) as $total
      | ($body.usage.prompt_tokens_details.text_tokens // null) as $text_in
      | (if $text_in == null then $prompt else $text_in end) as $input_text
      | (if $text_in == null then 0 else ([$prompt - $text_in, 0] | max) end) as $image_tokens
      | ($input_text + $completion) as $text_total
      | {
          vlm_output: $output,
          error: $error,
          parsed_answer: $parsed,
          label: $label,
          token_usage: {
            input_text_tokens: $input_text,
            image_tokens_estimated: $image_tokens,
            output_tokens: $completion,
            total_tokens: $total
          },
          cost_tokens: $total,
          latency_s_estimated: $wall,
          latency_allocation_method: "per-request DashScope HTTP wall clock; latency_s_estimated_text_only = wall_clock * text_tokens / total_tokens",
          text_token_usage: {
            input_tokens: $input_text,
            output_tokens: $completion,
            total_tokens: $text_total
          },
          text_cost_tokens: $text_total,
          latency_s_estimated_text_only: (if $total > 0 then $wall * $text_total / $total else $wall end),
          image_token_estimation_method: "dashscope usage.prompt_tokens minus prompt_tokens_details.text_tokens when text_tokens is reported; otherwise image tokens are 0 and prompt_tokens are counted as text",
          image_size: {width: $width, height: $height}
        }
    ' > "$dest"
}

request_once() {
  local payload="$1" body_file="$2"
  curl -sS -m 180 -o "$body_file" -w "%{http_code} %{time_total}" \
    -X POST "$API_URL" \
    -H "Authorization: Bearer ${DASHSCOPE_API_KEY}" \
    -H "Content-Type: application/json" \
    --data-binary @"$payload"
}

call_model() {
  local model="$1" image="$2" prompt="$3" row_file="$4" dest="$5"
  local mime b64 payload body content meta http wall width height attempt error empty
  mime="$(mime_for "$image")"
  width="$(sips -g pixelWidth "$image" 2>/dev/null | awk '/pixelWidth/{print $2}')"
  height="$(sips -g pixelHeight "$image" 2>/dev/null | awk '/pixelHeight/{print $2}')"
  [[ -z "$width" ]] && width=0
  [[ -z "$height" ]] && height=0
  b64="$(mktemp)"
  payload="$(mktemp)"
  body="$(mktemp)"
  content="$(mktemp)"
  empty="$(mktemp)"
  printf '%s' '{}' > "$empty"
  : > "$content"
  base64 < "$image" | tr -d '\n' > "$b64"
  jq -n --rawfile b64 "$b64" --arg model "$model" --arg mime "$mime" --arg prompt "$prompt" '
    {model:$model, max_tokens:64, temperature:0, enable_thinking:false,
     messages:[{role:"user", content:[
       {type:"image_url", image_url:{url:("data:" + $mime + ";base64," + $b64)}},
       {type:"text", text:$prompt}
     ]}]}
  ' > "$payload"
  error=""
  http="000"
  wall="0"
  for attempt in 1 2 3 4 5; do
    meta="$(request_once "$payload" "$body" || true)"
    http="${meta%% *}"
    wall="${meta#* }"
    [[ -z "$wall" || "$wall" == "$http" ]] && wall=0
    if [[ "$http" == "200" ]]; then
      jq -r '.choices[0].message.content // ""' "$body" > "$content" 2>/dev/null || :
      build_output "$body" "$wall" "$width" "$height" "$row_file" "$content" "" "$dest"
      break
    fi
    error="$(jq -r '.error.message // .message // "request failed"' "$body" 2>/dev/null || echo "http $http")"
    if [[ "$http" == "400" && "$error" == *[Tt]emperature* && "$attempt" == 1 ]]; then
      jq 'del(.temperature)' "$payload" > "$payload.tmp" && mv "$payload.tmp" "$payload"
      continue
    fi
    if [[ "$http" == "429" || "$http" == "500" || "$http" == "502" || "$http" == "503" || "$http" == "000" ]]; then
      sleep $((attempt * 2))
      continue
    fi
    break
  done
  if [[ "$http" != "200" ]]; then
    if [[ "$http" == "429" || "$http" == "500" || "$http" == "502" || "$http" == "503" || "$http" == "000" ]]; then
      rm -f "$b64" "$payload" "$body" "$content" "$empty"
      print -r -- "RETRY"
      return
    fi
    : > "$content"
    build_output "$empty" "$wall" "$width" "$height" "$row_file" "$content" "$error" "$dest"
  fi
  rm -f "$b64" "$payload" "$body" "$content" "$empty"
  if [[ -s "$dest" ]]; then
    print -r -- "OK"
  else
    print -r -- "FAIL"
  fi
}

load_done() {
  local sidecar="$1"
  typeset -gA DONE
  DONE=()
  [[ -f "$sidecar" ]] || return
  while IFS= read -r qid; do
    [[ -n "$qid" ]] && DONE[$qid]=1
  done < <(jq -r '.query_id' "$sidecar")
}

process_model() {
  local split_name="$1" model="$2"
  local input="$RESULT_DIR/$split_name"
  local run_dir="$RUN_ROOT/${split_name%.jsonl}"
  local sidecar="$run_dir/${model}.jsonl"
  mkdir -p "$run_dir"
  load_done "$sidecar"
  local seen=0 written=0 skipped=0 missing=0
  local qid prompt remote image call_status row_file out_file
  while IFS= read -r line; do
    seen=$((seen + 1))
    if (( LIMIT > 0 && written >= LIMIT )); then
      break
    fi
    qid="$(print -r -- "$line" | jq -r '.query_id')"
    if [[ -n "${DONE[$qid]:-}" ]]; then
      skipped=$((skipped + 1))
      continue
    fi
    remote="$(print -r -- "$line" | jq -r '.image_path')"
    image="$(local_image "$remote")"
    if [[ -z "$image" || ! -f "$image" ]]; then
      missing=$((missing + 1))
      continue
    fi
    prompt="$(print -r -- "$line" | jq -r '.prompt // .question // ""')"
    row_file="$(mktemp)"
    out_file="$(mktemp)"
    print -r -- "$line" | jq -c '{answer_type, ground_truth, choices}' > "$row_file"
    call_status="$(call_model "$model" "$image" "$prompt" "$row_file" "$out_file")"
    if [[ "$call_status" == "RETRY" || "$call_status" != "OK" ]]; then
      log "$split_name $model $qid not saved ($call_status)"
      rm -f "$row_file" "$out_file"
      continue
    fi
    jq -nc --arg qid "$qid" --slurpfile output "$out_file" '{query_id:$qid, output:$output[0]}' >> "$sidecar"
    rm -f "$row_file" "$out_file"
    written=$((written + 1))
    if (( written % 25 == 0 )); then
      log "$split_name $model written=$written skipped=$skipped missing_image=$missing"
    fi
  done < "$input"
  log "$split_name $model done written=$written skipped=$skipped missing_image=$missing seen=$seen"
}

merge_split() {
  local split_name="$1"
  local input="$RESULT_DIR/$split_name"
  local run_dir="$RUN_ROOT/${split_name%.jsonl}"
  local tmp="$input.tmp"
  local model slug sidecar_path
  local -a args
  args=()
  for model in $MODELS; do
    sidecar_path="$run_dir/${model}.jsonl"
    [[ -f "$sidecar_path" ]] || : > "$sidecar_path"
    slug="${model//[-.]/_}"
    args+=(--slurpfile "$slug" "$sidecar_path")
  done
  if ! jq -c "${args[@]}" '
    def index_rows($rows):
      reduce $rows[] as $row ({}; .[$row.query_id] = $row.output);
    def with_model($name; $rows):
      (index_rows($rows)[.query_id]) as $new
      | if $new == null then . else .model_outputs[$name] = $new end;
    with_model("kimi-k3"; $kimi_k3)
    | with_model("deepseek-v4.1-flash"; $deepseek_v4_1_flash)
    | with_model("qwen3.8-max"; $qwen3_8_max)
  ' "$input" > "$tmp" 2>&3; then
    rm -f "$tmp"
    log "$split_name merge failed; original file left unchanged"
    return 1
  fi
  mv "$tmp" "$input"
  log "$split_name merged into $input"
}

log "start splits=${SPLITS[*]} limit=$LIMIT"
typeset -a pids
for split_name in $SPLITS; do
  for model in $MODELS; do
    process_model "$split_name" "$model" &
    pids+=($!)
  done
  wait $pids
  pids=()
  if (( DO_MERGE )); then
    merge_split "$split_name"
  fi
done
log "all requested splits finished"
