#!/bin/zsh
# Append gpt.ge glm-5.3-flash vision outputs onto existing M2 JSONL rows.
# Preserves every original field. Reads GPTGE_API_KEY from the environment.
# Partial results are appended under result/api_runs and merged back at the end.
# Runs beside the DashScope job. Merge writes only the glm-5.3-flash key.

set -u
set +e
set +o pipefail
set +x
setopt NO_XTRACE NO_HUP EXTENDED_GLOB TYPESET_SILENT
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin${PATH:+:$PATH}"

ROOT="/Users/wangcan/Desktop/m2/dataset"
RESULT_DIR="$ROOT/result"
IMAGE_ROOT="$RESULT_DIR/images"
RUN_ROOT="$RESULT_DIR/api_runs"
API_URL="https://api.gpt.ge/v1/chat/completions"
REMOTE_IMAGE_PREFIX="/root/autodl-tmp/m2-free-datasets/data/processed/images/"
VHELM_REMOTE_PREFIX="/root/autodl-tmp/m2-free-datasets/data/raw/VHELM/"
VHELM_ROOT="$RESULT_DIR/vhelm"
MODELS=(glm-5.3-flash)
SPLITS=(
  "m2_id_free_test.jsonl"
  "m2_id_free_train.jsonl"
  "m2_ood_free_test.jsonl"
)
LIMIT=0
DO_MERGE=1
# Concurrent requests. DashScope keeps its own workers; this job is separate.
JOBS=12
MAX_ATTEMPTS=12

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --no-merge) DO_MERGE=0; shift ;;
    --split) SPLITS=("$2"); shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if (( JOBS < 1 )); then
  echo "--jobs must be at least 1" >&2
  exit 2
fi

if [[ -z "${GPTGE_API_KEY:-}" ]]; then
  echo "GPTGE_API_KEY is not set" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/progress.glm.log"
exec 3>>"$LOG"
unsetopt xtrace
set +x
setopt TYPESET_SILENT
exec >/dev/null 2>&3
trap 'print -r -- "[$(date "+%F %T")] script exit status=$? line=$LINENO" >&3' EXIT

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
  if [[ "$remote" == "$REMOTE_IMAGE_PREFIX"* ]]; then
    print -r -- "$IMAGE_ROOT/${remote#$REMOTE_IMAGE_PREFIX}"
    return
  fi
  if [[ "$remote" == "$VHELM_REMOTE_PREFIX"* ]]; then
    print -r -- "$VHELM_ROOT/${remote#$VHELM_REMOTE_PREFIX}"
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
    -H "Authorization: Bearer ${GPTGE_API_KEY}" \
    -H "Content-Type: application/json" \
    --data-binary @"$payload"
}

call_model() {
  local model="$1" image="$2" prompt="$3" row_file="$4" dest="$5"
  local mime b64 payload body content meta http wall width height attempt error empty
  local base jitter retryable send_image scaled scaled_base bytes long_side nw nh
  mime="$(mime_for "$image")"
  width="$(sips -g pixelWidth "$image" 2>/dev/null | awk '/pixelWidth/{print $2}')"
  height="$(sips -g pixelHeight "$image" 2>/dev/null | awk '/pixelHeight/{print $2}')"
  [[ -z "$width" ]] && width=0
  [[ -z "$height" ]] && height=0
  send_image="$image"
  scaled=""
  scaled_base=""
  bytes="$(stat -f%z "$image" 2>/dev/null || echo 0)"
  long_side="$width"
  (( height > long_side )) && long_side="$height"
  # qwen rejects either side <= 10. DashScope rejects a data-uri over 20MB,
  # and the 14–28MB MME images were the ones that hung for 180s with no body.
  if (( width > 0 && height > 0 && (width <= 10 || height <= 10 || bytes > 8000000 || long_side > 4096) )); then
    scaled_base="$(mktemp -t armsimg)"
    scaled="${scaled_base}.jpg"
    if (( width <= 10 || height <= 10 )); then
      nw="$width"
      nh="$height"
      if (( width <= height && width < 16 )); then
        nh=$(( height * 16 / width ))
        (( nh < 1 )) && nh=1
        nw=16
      elif (( height < 16 )); then
        nw=$(( width * 16 / height ))
        (( nw < 1 )) && nw=1
        nh=16
      fi
      sips -z "$nh" "$nw" -s format jpeg -s formatOptions 85 "$image" --out "$scaled" >/dev/null 2>&1 || :
    else
      sips -Z 4096 -s format jpeg -s formatOptions 80 "$image" --out "$scaled" >/dev/null 2>&1 || :
    fi
    if [[ -s "$scaled" ]]; then
      send_image="$scaled"
      mime="image/jpeg"
      bytes="$(stat -f%z "$send_image" 2>/dev/null || echo 0)"
      if (( bytes > 12000000 )); then
        sips -Z 2048 -s format jpeg -s formatOptions 60 "$send_image" --out "${scaled}.tmp.jpg" >/dev/null 2>&1 || :
        if [[ -s "${scaled}.tmp.jpg" ]]; then
          mv "${scaled}.tmp.jpg" "$scaled"
        fi
      fi
    fi
    rm -f "$scaled_base"
  fi
  b64="$(mktemp)"
  payload="$(mktemp)"
  body="$(mktemp)"
  content="$(mktemp)"
  empty="$(mktemp)"
  printf '%s' '{}' > "$empty"
  : > "$content"
  base64 < "$send_image" | tr -d '\n' > "$b64"
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
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
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
    retryable=0
    case "$http" in
      429|500|502|503|000) retryable=1 ;;
    esac
    if [[ "$error" == *(429|RateQuota|[Rr]ate\ limit|[Tt]oo\ many|[Tt]hrottl|[Cc]oncurren|并发)* ]]; then
      retryable=1
    fi
    if (( retryable )); then
      if (( attempt < MAX_ATTEMPTS )); then
        base=$((1 << (attempt - 1)))
        (( base > 64 )) && base=64
        jitter=$((RANDOM % 1000))
        log "$model backoff http=$http attempt=$attempt sleep=${base}s"
        sleep "${base}.$(printf '%03d' "$jitter")"
      fi
      continue
    fi
    break
  done
  if [[ "$http" != "200" ]]; then
    retryable=0
    case "$http" in
      429|500|502|503|000) retryable=1 ;;
    esac
    if [[ "$error" == *(429|RateQuota|[Rr]ate\ limit|[Tt]oo\ many|[Tt]hrottl|[Cc]oncurren|并发)* ]]; then
      retryable=1
    fi
    if (( retryable )); then
      rm -f "$b64" "$payload" "$body" "$content" "$empty"
      [[ -n "$scaled" ]] && rm -f "$scaled" "${scaled}.tmp.jpg"
      print -r -- "RETRY"
      return
    fi
    : > "$content"
    build_output "$empty" "$wall" "$width" "$height" "$row_file" "$content" "$error" "$dest"
  fi
  rm -f "$b64" "$payload" "$body" "$content" "$empty"
  [[ -n "$scaled" ]] && rm -f "$scaled" "${scaled}.tmp.jpg"
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

append_record() {
  local sidecar="$1" lock="$2" record="$3"
  local now mtime age
  while ! mkdir "$lock" 2>/dev/null; do
    if [[ -d "$lock" ]]; then
      now=$(date +%s)
      mtime=$(stat -f %m "$lock" 2>/dev/null || echo "$now")
      age=$((now - mtime))
      if (( age > 30 )); then
        rmdir "$lock" 2>/dev/null || true
      fi
    fi
    sleep 0.05
  done
  print -r -- "$record" >> "$sidecar"
  rmdir "$lock" 2>/dev/null || true
}

process_shard() {
  local split_name="$1" model="$2" part="$3" worker="$4"
  local run_dir="$RUN_ROOT/${split_name%.jsonl}"
  local sidecar="$run_dir/${model}.jsonl"
  local lock="$run_dir/${model}.lock"
  local written=0
  local qid prompt remote image call_status row_file out_file line record
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    qid="$(print -r -- "$line" | jq -r '.query_id')"
    remote="$(print -r -- "$line" | jq -r '.image_path')"
    image="$(local_image "$remote")"
    if [[ -z "$image" || ! -f "$image" ]]; then
      continue
    fi
    prompt="$(print -r -- "$line" | jq -r '.prompt // .question // ""')"
    row_file="$(mktemp)"
    out_file="$(mktemp)"
    print -r -- "$line" | jq -c '{answer_type, ground_truth, choices}' > "$row_file"
    call_status="$(call_model "$model" "$image" "$prompt" "$row_file" "$out_file")"
    if [[ "$call_status" != "OK" ]]; then
      log "$split_name $model worker=$worker $qid not saved ($call_status)"
      rm -f "$row_file" "$out_file"
      continue
    fi
    record="$(jq -nc --arg qid "$qid" --slurpfile output "$out_file" '{query_id:$qid, output:$output[0]}')"
    rm -f "$row_file" "$out_file"
    [[ -n "$record" ]] || continue
    append_record "$sidecar" "$lock" "$record"
    written=$((written + 1))
    if (( written % 25 == 0 )); then
      log "$split_name $model worker=$worker written=$written"
    fi
  done < "$part"
  log "$split_name $model worker=$worker done written=$written"
}

process_model() {
  local split_name="$1" model="$2"
  local input="$RESULT_DIR/$split_name"
  local run_dir="$RUN_ROOT/${split_name%.jsonl}"
  local sidecar="$run_dir/${model}.jsonl"
  local lock="$run_dir/${model}.lock"
  mkdir -p "$run_dir"
  touch "$sidecar"
  rmdir "$lock" 2>/dev/null || true
  local pending part_dir done_json filtered
  pending="$(mktemp)"
  part_dir="$(mktemp -d)"
  done_json="$(mktemp)"
  filtered="$(mktemp)"
  local seen=0 skipped=0 missing=0
  local qid remote image line
  jq -s 'map({(.query_id): true}) | add // {}' "$sidecar" > "$done_json"
  jq -c --slurpfile done "$done_json" 'select(($done[0][.query_id] // false) | not)' "$input" > "$filtered"
  seen=$(wc -l < "$input" | tr -d ' ')
  skipped=$((seen - $(wc -l < "$filtered" | tr -d ' ')))
  while IFS= read -r line; do
    remote="$(print -r -- "$line" | jq -r '.image_path')"
    image="$(local_image "$remote")"
    if [[ -z "$image" || ! -f "$image" ]]; then
      missing=$((missing + 1))
      continue
    fi
    print -r -- "$line" >> "$pending"
  done < "$filtered"
  rm -f "$done_json" "$filtered"
  local npending jobs i idx
  npending=$(wc -l < "$pending" | tr -d ' ')
  [[ -z "$npending" ]] && npending=0
  if (( LIMIT > 0 && npending > LIMIT )); then
    head -n "$LIMIT" "$pending" > "$pending.lim" && mv "$pending.lim" "$pending"
    npending=$LIMIT
  fi
  njobs=$JOBS
  if (( npending < njobs )); then
    njobs=$npending
  fi
  log "$split_name $model pending=$npending jobs=$njobs skipped=$skipped missing_image=$missing"
  if (( njobs < 1 )); then
    rm -rf "$pending" "$part_dir"
    log "$split_name $model done written=0 skipped=$skipped missing_image=$missing seen=$seen"
    return
  fi
  local -a parts wpids
  for ((i = 1; i <= njobs; i++)); do
    parts[$i]="$part_dir/part.$i"
    : > "${parts[$i]}"
  done
  idx=0
  while IFS= read -r line; do
    print -r -- "$line" >> "${parts[$((idx % njobs + 1))]}"
    idx=$((idx + 1))
  done < "$pending"
  log "$split_name $model launching $njobs workers"
  for ((i = 1; i <= njobs; i++)); do
    process_shard "$split_name" "$model" "${parts[$i]}" "$i" &
    wpids+=($!)
  done
  wait ${wpids[@]}
  rm -rf "$pending" "$part_dir"
  log "$split_name $model all workers finished skipped=$skipped missing_image=$missing seen=$seen"
}

merge_split() {
  local split_name="$1"
  local input="$RESULT_DIR/$split_name"
  local run_dir="$RUN_ROOT/${split_name%.jsonl}"
  local tmp="$input.glm-5.3-flash.tmp"
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
    with_model("glm-5.3-flash"; $glm_5_3_flash)
  ' "$input" > "$tmp" 2>&3; then
    rm -f "$tmp"
    log "$split_name merge failed; original file left unchanged"
    return 1
  fi
  mv "$tmp" "$input"
  log "$split_name merged into $input"
}

log "start splits=${SPLITS[*]} limit=$LIMIT jobs=$JOBS"
typeset -a pids
for split_name in $SPLITS; do
  for model in $MODELS; do
    process_model "$split_name" "$model" &
    pids+=($!)
  done
  wait ${pids[@]}
  pids=()
  if (( DO_MERGE )); then
    merge_split "$split_name"
  fi
done
log "all requested splits finished"
