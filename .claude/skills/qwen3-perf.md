---
name: qwen3-perf
description: Benchmark a Qwen3 text or VL model via sglang. Launches a local sglang server, runs the benchmark_batch.py script, and reports latency/throughput. Use when profiling a model, comparing before/after performance, or generating a perf report for a PR.
user-invocable: true
allowed-tools: Bash, Read
argument-hint: <model-path> [--port 30000] [--tp 1] [--batch-size 8] [--num-tokens 512] [--num-requests 10] [--gen-tokens 0] [--score] [--baseline baseline.json] [--extra-args "..."]
---

# Qwen3 Performance Benchmarking

Launch an sglang server for a Qwen3 text or VL model, run the benchmark suite, and report results.

## Arguments

If `$ARGUMENTS` is provided, parse it as:
- First positional arg → `MODEL_PATH`
- `--port PORT` → sglang server port (default: `30000`)
- `--tp TP` → tensor parallel size (default: auto-detect from `nvidia-smi`, fallback `1`)
- `--batch-size N` → items per request (default: `8`)
- `--num-tokens N` → input token length per item (default: `512`)
- `--num-requests N` → total benchmark requests (default: `10`)
- `--gen-tokens N` → output tokens to generate (default: `0` — prefill-only / score mode)
- `--score` → also benchmark `/v1/score` endpoint in parallel with `/generate`
- `--baseline FILE` → if provided, compare new results against this saved baseline JSON
- `--extra-args "..."` → extra flags passed verbatim to `sglang.launch_server`

## Workflow

### 0. Setup

```bash
PORT=${PORT:-30000}
WORKDIR=/opt/tiger/workspace/inference   # adjust if running locally
LOG_FILE=/tmp/sglang_perf_$PORT.log
```

Auto-detect TP if not given:
```bash
TP=$(nvidia-smi -L 2>/dev/null | wc -l)
TP=${TP:-1}
```

Detect if model is VL (check for "vl", "vision", or "qwen2-vl" in model path, case-insensitive):
```bash
if echo "$MODEL_PATH" | grep -iqE "vl|vision|qwen2.?vl"; then
    IS_VL=true
else
    IS_VL=false
fi
```

### 1. Start sglang server

```bash
# Kill any existing server on the same port
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 1

VL_EXTRA=""
if [ "$IS_VL" = "true" ]; then
    VL_EXTRA="--chat-template qwen2-vl"
fi

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp "$TP" \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" \
    $VL_EXTRA \
    $EXTRA_ARGS \
    > "$LOG_FILE" 2>&1 &

SGLANG_PID=$!
echo "[sglang] PID=$SGLANG_PID  log=$LOG_FILE  model=$MODEL_PATH  tp=$TP  vl=$IS_VL"
```

### 2. Wait for server to be ready

Poll `/health` with exponential back-off (max 10 min):
```bash
echo "[wait] Polling http://localhost:$PORT/health ..."
for i in $(seq 1 120); do
    if curl -sf -m 3 "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[wait] Server ready after ~$((i * 5))s"
        break
    fi
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "[error] sglang process died. Last log:"
        tail -40 "$LOG_FILE"
        exit 1
    fi
    if [ $i -eq 120 ]; then
        echo "[error] Server not ready after 600s. Last log:"
        tail -40 "$LOG_FILE"
        kill $SGLANG_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done
```

### 3. Run benchmark

```bash
ENDPOINT="http://localhost:$PORT"
RESULT_JSON="/tmp/qwen3_perf_result_$PORT.json"

# Text benchmark via /generate with input_ids
python3 benchmark/benchmark_batch.py \
    --endpoint_url "$ENDPOINT" \
    --num_requests "$NUM_REQUESTS" \
    --batch_size "$BATCH_SIZE" \
    --num_tokens "$NUM_TOKENS" \
    --gen_tokens "$GEN_TOKENS" \
    --tokenizer_dir "$MODEL_PATH" \
    2>&1 | tee /tmp/qwen3_perf_stdout.txt

# Optionally also hit /v1/score
if [ "$SCORE" = "true" ]; then
    echo ""
    echo "=== /v1/score benchmark ==="
    python3 benchmark/benchmark_batch.py \
        --endpoint_url "$ENDPOINT" \
        --score_url "$ENDPOINT/v1/score" \
        --num_requests "$NUM_REQUESTS" \
        --batch_size "$BATCH_SIZE" \
        --num_tokens "$NUM_TOKENS" \
        --gen_tokens 0 \
        --tokenizer_dir "$MODEL_PATH" \
        2>&1 | tee -a /tmp/qwen3_perf_stdout.txt
fi
```

Save key metrics to JSON for comparison:
```bash
python3 - <<'PYEOF'
import re, json, sys

txt = open("/tmp/qwen3_perf_stdout.txt").read()

def extract(label, text):
    m = re.search(rf"{label}\s*:\s*([\d,\.]+)", text)
    return float(m.group(1).replace(",","")) if m else None

result = {
    "model_path": "$MODEL_PATH",
    "tp": $TP,
    "batch_size": $BATCH_SIZE,
    "num_tokens": $NUM_TOKENS,
    "num_requests": $NUM_REQUESTS,
    "gen_tokens": $GEN_TOKENS,
    "avg_latency_per_request_ms": extract("Avg latency per request", txt),
    "avg_latency_per_item_ms":    extract("Avg latency per item", txt),
    "throughput_item_per_s":      extract("Throughput", txt),
    "wall_clock_ms":              extract("Wall-clock time", txt),
}
with open("$RESULT_JSON", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
PYEOF
```

### 4. Optional: compare against baseline

If `--baseline FILE` was given:

```bash
python3 - <<'PYEOF'
import json, sys

baseline = json.load(open("$BASELINE_FILE"))
new      = json.load(open("$RESULT_JSON"))

metrics = [
    ("avg_latency_per_request_ms", "Avg req latency (ms)", True),  # True = lower is better
    ("avg_latency_per_item_ms",    "Avg item latency (ms)", True),
    ("throughput_item_per_s",      "Throughput (item/s)",   False),
    ("wall_clock_ms",              "Wall-clock (ms)",       True),
]

print("\n| Metric | Baseline | New | Delta | |")
print("|--------|----------|-----|-------|--|")
for key, label, lower_better in metrics:
    b = baseline.get(key)
    n = new.get(key)
    if b is None or n is None:
        print(f"| {label} | N/A | N/A | — | — |")
        continue
    delta_pct = (n - b) / b * 100
    if lower_better:
        symbol = "✅" if delta_pct < -1 else ("⚠️" if delta_pct > 3 else "➡️")
    else:
        symbol = "✅" if delta_pct > 1 else ("⚠️" if delta_pct < -3 else "➡️")
    print(f"| {label} | {b:,.2f} | {n:,.2f} | {delta_pct:+.1f}% | {symbol} |")
PYEOF
```

### 5. Cleanup

```bash
kill $SGLANG_PID 2>/dev/null && echo "[cleanup] sglang server stopped."
echo "[done] Result saved to $RESULT_JSON"
```

## Example invocations

```bash
# Minimal — text model, all defaults
/qwen3-perf /path/to/Qwen3-7B

# Specify TP and batch size
/qwen3-perf /path/to/Qwen3-72B --tp 4 --batch-size 16 --num-tokens 1024

# VL model (auto-detected, but also works if named explicitly)
/qwen3-perf /path/to/Qwen2-VL-7B-Instruct --tp 2

# Prefill + decode benchmark
/qwen3-perf /path/to/Qwen3-7B --gen-tokens 16 --num-requests 20

# Also benchmark /v1/score
/qwen3-perf /path/to/Qwen3-7B --score

# Compare against a saved baseline
/qwen3-perf /path/to/Qwen3-7B --baseline /tmp/baseline.json
# (baseline.json is the $RESULT_JSON from a previous run)

# Pass extra sglang flags (e.g. enable chunked prefill)
/qwen3-perf /path/to/Qwen3-7B --extra-args "--chunked-prefill-size 512"
```

## Notes

- The sglang server is killed at the end of each run; use `--port` to avoid conflicts when running multiple benchmarks simultaneously.
- For **prefill-only / score** workloads use `--gen-tokens 0` (default). This matches the production serving pattern.
- For VL models only text-path latency is measured (no images are sent); this is intentional for ranking/scoring use cases.
- `$RESULT_JSON` is always written so it can be re-used as `--baseline` for a future comparison run.
- Logs from sglang are saved to `/tmp/sglang_perf_<PORT>.log` for debugging.
