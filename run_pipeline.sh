#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# SysML → Neural Controller → Formal Verification Pipeline
#
# Usage:
#   ./run_pipeline.sh                    # run one CPU seed (42) per model
#   ./run_pipeline.sh --force-full       # force all steps from scratch
#   ./run_pipeline.sh --verify-only      # extract + verify; reuse CPU eval data
#   ./run_pipeline.sh --model mixing     # run only one model
#   ./run_pipeline.sh --num-seeds 30     # run 30 CPU training seeds per model
#   ./run_pipeline.sh --cpu-mode single  # reproducible single-core timings
#   ./run_pipeline.sh --cpu-mode aggressive  # use normal multicore scheduling
#   ./run_pipeline.sh --jobs 32          # aggressive-mode seed parallelism
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR"
SYSML_DIR="$SCRIPT_DIR/sysml-models"
SMV_DIR="$SCRIPT_DIR/SMV"
RL_DIR="$SCRIPT_DIR/rl"
MC_EXTRACT="$SYSML_DIR/mc-extract.py"
DEFAULT_VENV_PY="$HOME/git_stuff/AI_venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "$DEFAULT_VENV_PY" ]]; then
    PYTHON_BIN="$DEFAULT_VENV_PY"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
CPU_TRAINING_REPO="${CPU_TRAINING_REPO:-$(cd "$SCRIPT_DIR/../shield-pipeline-new-sysml" 2>/dev/null && pwd || true)}"
CPU_TRAINING_ENTRY="$SCRIPT_DIR/tools/run_cpu_training_seed.py"
CPU_EVAL_ENTRY="$SCRIPT_DIR/tools/run_cpu_eval_seed.py"
REPORT_ENTRY="$SCRIPT_DIR/tools/generate_pipeline_report.py"
NUXMV="${NUXMV_BIN:-$SCRIPT_DIR/nuXmv}"
if [[ ! -x "$NUXMV" ]]; then
    NUXMV="$(command -v nuXmv 2>/dev/null || echo "$HOME/tools/nuXmv-2.1.0-linux64/bin/nuXmv")"
fi
RESULTS_FILE="$SMV_DIR/verification_results.txt"
METRICS_DIR="$PIPELINE_DIR/metrics"
CPU_RESULTS_DIR="$METRICS_DIR/cpu_training"
CPU_DEFAULT_SEED=42
NUM_SEEDS=1
CPU_EXECUTION_MODE="${CPU_EXECUTION_MODE:-single}"
CPU_AFFINITY_CORE="${CPU_AFFINITY_CORE:-0}"
CPU_PARALLEL_JOBS="${CPU_PARALLEL_JOBS:-}"
CPU_BOUND_PREFIX=()

# Memory limit for nuXmv (KB). Default 10GB.
NUXMV_MEM_LIMIT_KB="${NUXMV_MEM_LIMIT_KB:-10485760}"

# --- Model registry ---
# Format: name|sysml_path|smv_dir|smv_file
MODELS=(
    "thermostat|thermostat/model.sysml|thermostat|model.smv"
    "cruise|cruise-controller-model/model.sysml|cruise-control-model|out.smv"
    "mixing|mixing-sysml-model/model.sysml|mixing-model|model.smv"
)

usage() {
    cat <<EOF
Usage: ./run_pipeline.sh [options]

Options:
  --force-full                  Regenerate requested outputs
  --verify-only                 Extract + verify; reuse CPU eval data
  --model NAME                  Run one model: thermostat, cruise, or mixing
  --num-seeds N, --seeds N      Run N deterministic seeds starting at 42
  --cpu-mode MODE               single or aggressive (default: single)
  --single-core                 Alias for --cpu-mode single
  --aggressive-multicore        Alias for --cpu-mode aggressive
  --cpu-affinity-core CORE      CPU core used in single mode (default: 0)
  --jobs N                      Parallel seed jobs in aggressive mode
  -h, --help                    Show this help

single mode caps common numeric thread pools to one thread and uses taskset
when available so timing metrics are collected on one CPU core. aggressive mode
leaves normal OS/library multicore scheduling in place and runs seed jobs in
parallel. If --jobs is omitted in aggressive mode, the pipeline uses nproc.
EOF
}

# --- Parse args ---
FORCE_FULL=false
VERIFY_ONLY=false
SINGLE_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-full) FORCE_FULL=true; shift ;;
        --verify-only) VERIFY_ONLY=true; shift ;;
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        --num-seeds|--seeds) NUM_SEEDS="$2"; shift 2 ;;
        --cpu-mode) CPU_EXECUTION_MODE="$2"; shift 2 ;;
        --single-core) CPU_EXECUTION_MODE="single"; shift ;;
        --aggressive-multicore|--multicore) CPU_EXECUTION_MODE="aggressive"; shift ;;
        --cpu-affinity-core) CPU_AFFINITY_CORE="$2"; shift 2 ;;
        --jobs) CPU_PARALLEL_JOBS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if ! [[ "$NUM_SEEDS" =~ ^[0-9]+$ ]] || (( NUM_SEEDS < 1 )); then
    echo "Invalid --num-seeds value: $NUM_SEEDS"
    exit 1
fi

if [[ "$CPU_EXECUTION_MODE" == "multicore" ]]; then
    CPU_EXECUTION_MODE="aggressive"
fi
case "$CPU_EXECUTION_MODE" in
    single|aggressive) ;;
    *) echo "Invalid --cpu-mode value: $CPU_EXECUTION_MODE"; exit 1 ;;
esac
if ! [[ "$CPU_AFFINITY_CORE" =~ ^[0-9]+$ ]]; then
    echo "Invalid --cpu-affinity-core value: $CPU_AFFINITY_CORE"
    exit 1
fi
if [[ -z "$CPU_PARALLEL_JOBS" ]]; then
    if [[ "$CPU_EXECUTION_MODE" == "aggressive" ]]; then
        CPU_PARALLEL_JOBS="$(nproc 2>/dev/null || echo 1)"
    else
        CPU_PARALLEL_JOBS=1
    fi
fi
if ! [[ "$CPU_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || (( CPU_PARALLEL_JOBS < 1 )); then
    echo "Invalid --jobs value: $CPU_PARALLEL_JOBS"
    exit 1
fi
if [[ "$CPU_EXECUTION_MODE" == "single" ]]; then
    CPU_PARALLEL_JOBS=1
fi

# --- Helpers ---
log()  { echo "=== [$(date +%H:%M:%S)] $*"; }
ok()   { echo "    ✓ $*"; }
fail() { echo "    ✗ $*"; }
sep()  { echo "────────────────────────────────────────────────────────────"; }

seed_for_index() {
    local idx="$1"
    echo $((CPU_DEFAULT_SEED + idx))
}

configure_cpu_execution() {
    CPU_BOUND_PREFIX=()
    if [[ "$CPU_EXECUTION_MODE" == "single" ]]; then
        local thread_env=(
            "OMP_NUM_THREADS=1"
            "OPENBLAS_NUM_THREADS=1"
            "MKL_NUM_THREADS=1"
            "BLIS_NUM_THREADS=1"
            "NUMEXPR_NUM_THREADS=1"
            "VECLIB_MAXIMUM_THREADS=1"
            "ACCELERATE_NUM_THREADS=1"
            "GOTO_NUM_THREADS=1"
            "OMP_DYNAMIC=FALSE"
            "MKL_DYNAMIC=FALSE"
        )
        if command -v taskset >/dev/null 2>&1; then
            CPU_BOUND_PREFIX=(env "${thread_env[@]}" taskset -c "$CPU_AFFINITY_CORE")
        else
            CPU_BOUND_PREFIX=(env "${thread_env[@]}")
        fi
    fi
}

run_cpu_bound() {
    if ((${#CPU_BOUND_PREFIX[@]})); then
        "${CPU_BOUND_PREFIX[@]}" "$@"
    else
        "$@"
    fi
}

execution_label() {
    if [[ "$CPU_EXECUTION_MODE" == "single" ]]; then
        if command -v taskset >/dev/null 2>&1; then
            echo "single-core (thread caps + taskset core $CPU_AFFINITY_CORE)"
        else
            echo "single-thread library caps (taskset unavailable)"
        fi
    else
        echo "aggressive multicore (${CPU_PARALLEL_JOBS} seed job(s))"
    fi
}

json_matches_execution_mode() {
    local json_file="$1"
    [[ -f "$json_file" ]] || return 1
    "$PYTHON_BIN" - "$json_file" "$CPU_EXECUTION_MODE" "$CPU_AFFINITY_CORE" <<'PY'
import json
import sys

path, expected_mode, expected_core = sys.argv[1:4]
try:
    with open(path) as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

execution = data.get("execution", {})
mode = execution.get("cpu_mode")
core = execution.get("cpu_affinity_core")
if mode != expected_mode:
    sys.exit(1)
if expected_mode == "single" and str(core) != str(expected_core):
    sys.exit(1)
sys.exit(0)
PY
}

seed_list_display() {
    local seeds=()
    local i
    for ((i = 0; i < NUM_SEEDS; i++)); do
        seeds+=("$(seed_for_index "$i")")
    done
    local IFS=","
    echo "${seeds[*]}"
}

model_list_display() {
    local models=()
    local entry name
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name _ _ _ <<< "$entry"
        if [[ -n "$SINGLE_MODEL" && "$name" != "$SINGLE_MODEL" ]]; then
            continue
        fi
        models+=("$name")
    done
    local IFS=","
    echo "${models[*]}"
}

require_cpu_training_program() {
    if [[ -z "$CPU_TRAINING_REPO" || ! -d "$CPU_TRAINING_REPO" ]]; then
        fail "CPU training program repo not found"
        fail "Set CPU_TRAINING_REPO=/path/to/shield-pipeline-new-sysml"
        return 1
    fi
    if [[ ! -f "$CPU_TRAINING_ENTRY" ]]; then
        fail "CPU training entry point not found: $CPU_TRAINING_ENTRY"
        return 1
    fi
    if [[ ! -f "$CPU_EVAL_ENTRY" ]]; then
        fail "CPU evaluation entry point not found: $CPU_EVAL_ENTRY"
        return 1
    fi
}

cpu_seed_complete() {
    local seed_dir="$1"
    [[ -f "$seed_dir/summary.json" && -f "$seed_dir/best.npz" ]] || return 1
    grep -q '"selected_checkpoint_source"' "$seed_dir/summary.json" || return 1
    grep -q '"final_checkpoint_safe"' "$seed_dir/summary.json" || return 1
    json_matches_execution_mode "$seed_dir/summary.json" || return 1
}

runtime_seed_complete() {
    local out_json="$1"
    [[ -f "$out_json" ]] || return 1
    grep -q '"final_checkpoint_safe"' "$out_json" || return 1
    json_matches_execution_mode "$out_json" || return 1
}

wait_for_available_job_slot() {
    while (( $(jobs -pr | wc -l) >= CPU_PARALLEL_JOBS )); do
        sleep 1
    done
}

wait_for_seed_jobs() {
    local failed=0
    local pid info name seed log_file seed_dir
    for pid in "${seed_pids[@]}"; do
        info="${seed_info[$pid]}"
        IFS='|' read -r name seed log_file seed_dir <<< "$info"
        if wait "$pid"; then
            if [[ -f "$seed_dir/summary.json" && -f "$seed_dir/best.npz" ]]; then
                ok "[$name] Seed $seed complete"
            else
                fail "[$name] Seed $seed finished without summary.json and best.npz"
                failed=1
            fi
        else
            fail "[$name] Seed $seed failed"
            if [[ -f "$log_file" ]]; then
                echo "      Last 40 log lines from $log_file:"
                tail -40 "$log_file" | sed 's/^/      /'
            fi
            failed=1
        fi
    done
    return "$failed"
}

wait_for_eval_jobs() {
    local failed=0
    local pid info name seed log_file out_json
    for pid in "${seed_pids[@]}"; do
        info="${seed_info[$pid]}"
        IFS='|' read -r name seed log_file out_json <<< "$info"
        if wait "$pid"; then
            if [[ -f "$out_json" ]]; then
                ok "[$name] Runtime eval seed $seed complete"
            else
                fail "[$name] Runtime eval seed $seed finished without output JSON"
                failed=1
            fi
        else
            fail "[$name] Runtime eval seed $seed failed"
            if [[ -f "$log_file" ]]; then
                echo "      Last 40 log lines from $log_file:"
                tail -40 "$log_file" | sed 's/^/      /'
            fi
            failed=1
        fi
    done
    return "$failed"
}

# ============================================================================
# Step 1: SMV Extraction
# ============================================================================
extract_smv() {
    local name="$1" sysml_rel="$2" smv_subdir="$3" smv_file="$4"
    local sysml_path="$SYSML_DIR/$sysml_rel"
    local smv_out_dir="$SMV_DIR/$smv_subdir"
    local smv_path="$smv_out_dir/$smv_file"

    log "[$name] Extracting SMV from SysML"

    if [[ ! -f "$sysml_path" ]]; then
        fail "SysML file not found: $sysml_path"
        return 1
    fi

    mkdir -p "$smv_out_dir"
    run_cpu_bound "$PYTHON_BIN" "$MC_EXTRACT" "$sysml_path" --dt 0.1 -o "$smv_path" 2>&1 | grep -v "ANTLR runtime" || true

    if [[ -f "$smv_path" ]]; then
        local n_spec
        n_spec=$(grep -c "INVARSPEC" "$smv_path")
        ok "Generated $smv_path ($n_spec INVARSPEC)"
    else
        fail "SMV generation failed"
        return 1
    fi
}

# ============================================================================
# Step 2: CPU Training
# ============================================================================
train_model() {
    local name="$1" sysml_rel="$2"
    local sysml_path="$SYSML_DIR/$sysml_rel"
    local model_out_dir="$CPU_RESULTS_DIR/$name"

    if [[ "$VERIFY_ONLY" == true ]]; then
        log "[$name] Skipping training (--verify-only)"
        return 0
    fi

    require_cpu_training_program || return 1

    local all_done=true
    local i seed seed_dir
    for ((i = 0; i < NUM_SEEDS; i++)); do
        seed="$(seed_for_index "$i")"
        seed_dir="$model_out_dir/seed_$seed"
        if ! cpu_seed_complete "$seed_dir"; then
            all_done=false
            break
        fi
    done

    if [[ "$FORCE_FULL" == false && "$all_done" == true ]]; then
        log "[$name] CPU training outputs exist for seeds $(seed_list_display) (skipping)"
        ok "Found $NUM_SEEDS completed seed result(s) in $model_out_dir"
        return 0
    fi

    log "[$name] Training neural controller with CPU training program"
    mkdir -p "$model_out_dir"

    local seed_pids=()
    declare -A seed_info=()
    for ((i = 0; i < NUM_SEEDS; i++)); do
        seed="$(seed_for_index "$i")"
        seed_dir="$model_out_dir/seed_$seed"

        if [[ "$FORCE_FULL" == false ]] && cpu_seed_complete "$seed_dir"; then
            ok "Seed $seed already complete"
            continue
        fi

        log "[$name] CPU training seed $seed"
        mkdir -p "$seed_dir"
        local log_file="$model_out_dir/seed_${seed}_train_log.txt"
        wait_for_available_job_slot
        (
            run_cpu_bound "$PYTHON_BIN" "$CPU_TRAINING_ENTRY" \
                --program-root "$CPU_TRAINING_REPO" \
                --model-path "$sysml_path" \
                --model-name "$name" \
                --seed "$seed" \
                --out "$seed_dir" \
                --cpu-mode "$CPU_EXECUTION_MODE" \
                --cpu-affinity-core "$CPU_AFFINITY_CORE" \
                --dt 0.1 \
                --max-steps 5000 \
                --ensure-class-coverage 200 \
                --eval-episodes 100 \
                --test-episodes 200 \
                --oracle-samples 2000 \
                --oracle-epochs 100 \
                --ppo-episodes 2000 \
                --minibatch-size 25 \
                --bptt-chunk-size 200
        ) > "$log_file" 2>&1 &
        local pid=$!
        seed_pids+=("$pid")
        seed_info[$pid]="$name|$seed|$log_file|$seed_dir"
    done

    wait_for_seed_jobs
}

# ============================================================================
# Step 3: Formal Verification (IC3 + BMC, both saved)
# ============================================================================
parse_nuxmv_output() {
    # Parse result counts and peak memory from a nuXmv /usr/bin/time output file
    # Sets: _n_true, _n_false, _peak_mem_mb
    local file="$1"
    _n_true=$(grep -c "is true" "$file" 2>/dev/null || true)
    _n_false=$(grep -c "is false" "$file" 2>/dev/null || true)
    _n_true=$(( ${_n_true:-0} + 0 ))
    _n_false=$(( ${_n_false:-0} + 0 ))

    local peak_kb
    peak_kb=$(grep "Maximum resident set size" "$file" | tail -1 | awk '{print $NF}')
    if [[ -n "$peak_kb" ]]; then
        _peak_mem_mb=$(awk "BEGIN {printf \"%.1f\", $peak_kb / 1024}")
    else
        _peak_mem_mb="?"
    fi
}

verify_model() {
    local name="$1" smv_subdir="$2" smv_file="$3"
    local smv_path="$SMV_DIR/$smv_subdir/$smv_file"
    local verify_dir="$SMV_DIR/$smv_subdir"

    log "[$name] Formal verification (IC3 + BMC)"

    if [[ ! -x "$NUXMV" ]]; then
        fail "nuXmv not found at $NUXMV"
        fail "Set NUXMV_BIN=/path/to/nuXmv or install to ~/tools/"
        return 1
    fi

    if [[ ! -f "$smv_path" ]]; then
        fail "SMV file not found: $smv_path"
        return 1
    fi

    local n_specs
    n_specs=$(grep -c "^INVARSPEC" "$smv_path" || echo 0)
    ok "$n_specs INVARSPEC properties found"

    # ── Run IC3 (unbounded) ──────────────────────────────────────────────
    local ic3_out="$verify_dir/ic3_output.txt"
    local ic3_start ic3_end ic3_elapsed ic3_rc=0

    log "[$name] Running IC3 (unbounded)..."
    ic3_start=$(date +%s%N)
    timeout 120s /usr/bin/time -v "${CPU_BOUND_PREFIX[@]}" "$NUXMV" -int "$smv_path" <<HEREDOC > "$ic3_out" 2>&1 || ic3_rc=$?
go_msat
check_invar_ic3
quit
HEREDOC
    ic3_end=$(date +%s%N)
    ic3_elapsed=$(( (ic3_end - ic3_start) / 1000000 ))

    parse_nuxmv_output "$ic3_out"
    local ic3_true=$_n_true ic3_false=$_n_false ic3_mem=$_peak_mem_mb

    if [[ $ic3_rc -eq 124 ]]; then
        ok "IC3: timeout after 120s"
    else
        ok "IC3: $ic3_true proven, $ic3_false failed (${ic3_elapsed}ms, ${ic3_mem} MB peak)"
    fi

    # ── Run BMC (een-sorensson k=200) ────────────────────────────────────
    local bmc_out="$verify_dir/bmc_output.txt"
    local bmc_start bmc_end bmc_elapsed

    log "[$name] Running BMC (een-sorensson, k=200)..."
    bmc_start=$(date +%s%N)
    /usr/bin/time -v "${CPU_BOUND_PREFIX[@]}" "$NUXMV" -int "$smv_path" <<HEREDOC > "$bmc_out" 2>&1
go_msat
msat_check_invar_bmc -a een-sorensson -k 200
quit
HEREDOC
    bmc_end=$(date +%s%N)
    bmc_elapsed=$(( (bmc_end - bmc_start) / 1000000 ))

    parse_nuxmv_output "$bmc_out"
    local bmc_true=$_n_true bmc_false=$_n_false bmc_mem=$_peak_mem_mb

    ok "BMC: $bmc_true proven, $bmc_false failed (${bmc_elapsed}ms, ${bmc_mem} MB peak)"

    # ── Report failures from either method ───────────────────────────────
    if [[ $ic3_false -gt 0 ]]; then
        fail "IC3 counterexamples:"
        grep "is false" "$ic3_out" | while read -r line; do echo "      $line"; done
    fi
    if [[ $bmc_false -gt 0 ]]; then
        fail "BMC counterexamples:"
        grep "is false" "$bmc_out" | while read -r line; do echo "      $line"; done
    fi

    # ── Write results (both methods) ─────────────────────────────────────
    write_results "$name" "$ic3_out" "$ic3_elapsed" "$ic3_mem" "$bmc_out" "$bmc_elapsed" "$bmc_mem"
}

# Write results to the verification results file
write_results() {
    local name="$1"
    local ic3_out="$2" ic3_ms="$3" ic3_mem="$4"
    local bmc_out="$5" bmc_ms="$6" bmc_mem="$7"

    sep >> "$RESULTS_FILE"
    echo "MODEL: $name" >> "$RESULTS_FILE"
    sep >> "$RESULTS_FILE"

    echo "" >> "$RESULTS_FILE"
    echo "IC3 (unbounded):" >> "$RESULTS_FILE"
    echo "  Runtime: ${ic3_ms}ms" >> "$RESULTS_FILE"
    echo "  Peak memory: ${ic3_mem} MB" >> "$RESULTS_FILE"
    if [[ -f "$ic3_out" ]]; then
        grep -E "is true|is false" "$ic3_out" >> "$RESULTS_FILE" || echo "  (no results — timeout or error)" >> "$RESULTS_FILE"
    fi

    echo "" >> "$RESULTS_FILE"
    echo "BMC (een-sorensson k=200):" >> "$RESULTS_FILE"
    echo "  Runtime: ${bmc_ms}ms" >> "$RESULTS_FILE"
    echo "  Peak memory: ${bmc_mem} MB" >> "$RESULTS_FILE"
    if [[ -f "$bmc_out" ]]; then
        grep -E "is true|is false" "$bmc_out" >> "$RESULTS_FILE" || echo "  (no results)" >> "$RESULTS_FILE"
    fi

    echo "" >> "$RESULTS_FILE"
}

# ============================================================================
# Main
# ============================================================================
main() {
    configure_cpu_execution
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  SysML → Neural Controller → Formal Verification Pipeline  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Mode:    $(if $FORCE_FULL; then echo 'FORCE FULL (all steps)'; elif $VERIFY_ONLY; then echo 'VERIFY ONLY'; else echo 'INCREMENTAL (skip existing)'; fi)"
    echo "  Models:  $(if [[ -n "$SINGLE_MODEL" ]]; then echo "$SINGLE_MODEL"; else echo "all"; fi)"
    echo "  Seeds:   $(seed_list_display)"
    echo "  CPU:     $(execution_label)"
    echo "  Python:  $PYTHON_BIN"
    echo "  nuXmv:   $NUXMV"
    echo ""

    # Initialize results file
    echo "nuXmv Formal Verification Results" > "$RESULTS_FILE"
    echo "Generated: $(date)" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    local any_failed=false

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name sysml_rel smv_subdir smv_file <<< "$entry"

        # Filter if --model specified
        if [[ -n "$SINGLE_MODEL" && "$name" != "$SINGLE_MODEL" ]]; then
            continue
        fi

        sep
        log "Processing: $name"
        sep

        # Step 1: Extract SMV
        if ! extract_smv "$name" "$sysml_rel" "$smv_subdir" "$smv_file"; then
            any_failed=true
            continue
        fi

        # Step 2: Train (if needed)
        if ! train_model "$name" "$sysml_rel"; then
            any_failed=true
            continue
        fi

        # Step 3: Verify
        if ! verify_model "$name" "$smv_subdir" "$smv_file"; then
            any_failed=true
        fi

        echo ""
    done

    sep
    log "Pipeline complete"
    echo ""
    echo "Results written to: $RESULTS_FILE"
    echo ""
    cat "$RESULTS_FILE"

    # Generate evaluation summary, automation metrics, runtime metrics, and report
    generate_evaluation_summary
    generate_automation_metrics
    generate_cpu_training_summary
    run_runtime_monitor_eval
    generate_human_report

    if $any_failed; then
        echo ""
        fail "Some steps had failures — check output above"
        return 1
    fi
}

# ============================================================================
# Evaluation Summary Table (CSV)
# ============================================================================
generate_evaluation_summary() {
    mkdir -p "$METRICS_DIR"
    local summary="$METRICS_DIR/evaluation_summary.csv"

    echo "system,ic3_proven,ic3_failed,ic3_runtime_ms,ic3_peak_mb,bmc_proven,bmc_failed,bmc_runtime_ms,bmc_peak_mb" > "$summary"

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name sysml_rel smv_subdir smv_file <<< "$entry"

        if [[ -n "$SINGLE_MODEL" && "$name" != "$SINGLE_MODEL" ]]; then
            continue
        fi

        local verify_dir="$SMV_DIR/$smv_subdir"
        local ic3_proven=0 ic3_failed=0 ic3_ms="" ic3_mem=""
        local bmc_proven=0 bmc_failed=0 bmc_ms="" bmc_mem=""

        # Parse IC3 results
        if [[ -f "$verify_dir/ic3_output.txt" ]]; then
            ic3_proven=$(grep -c "is true" "$verify_dir/ic3_output.txt" 2>/dev/null || true)
            ic3_proven=$(( ${ic3_proven:-0} + 0 ))
            ic3_failed=$(grep -c "is false" "$verify_dir/ic3_output.txt" 2>/dev/null || true)
            ic3_failed=$(( ${ic3_failed:-0} + 0 ))
            local kb
            kb=$(grep "Maximum resident set size" "$verify_dir/ic3_output.txt" | tail -1 | awk '{print $NF}')
            [[ -n "$kb" ]] && ic3_mem=$(awk "BEGIN {printf \"%.1f\", $kb / 1024}")
        fi

        # Parse BMC results
        if [[ -f "$verify_dir/bmc_output.txt" ]]; then
            bmc_proven=$(grep -c "is true" "$verify_dir/bmc_output.txt" 2>/dev/null || true)
            bmc_proven=$(( ${bmc_proven:-0} + 0 ))
            bmc_failed=$(grep -c "is false" "$verify_dir/bmc_output.txt" 2>/dev/null || true)
            bmc_failed=$(( ${bmc_failed:-0} + 0 ))
            local kb
            kb=$(grep "Maximum resident set size" "$verify_dir/bmc_output.txt" | tail -1 | awk '{print $NF}')
            [[ -n "$kb" ]] && bmc_mem=$(awk "BEGIN {printf \"%.1f\", $kb / 1024}")
        fi

        # Parse runtimes from results file
        if [[ -f "$RESULTS_FILE" ]]; then
            local in_model=false in_ic3=false in_bmc=false
            while IFS= read -r line; do
                if [[ "$line" == *"MODEL: $name"* ]]; then in_model=true; continue; fi
                if $in_model && [[ "$line" == *"MODEL:"* && "$line" != *"MODEL: $name"* ]]; then break; fi
                if $in_model; then
                    [[ "$line" == "IC3 (unbounded):" ]] && { in_ic3=true; in_bmc=false; continue; }
                    [[ "$line" == "BMC (een-sorensson k=200):" ]] && { in_bmc=true; in_ic3=false; continue; }
                    if $in_ic3 && [[ "$line" == *"Runtime:"* ]]; then ic3_ms="${line##*: }"; ic3_ms="${ic3_ms%ms}"; fi
                    if $in_bmc && [[ "$line" == *"Runtime:"* ]]; then bmc_ms="${line##*: }"; bmc_ms="${bmc_ms%ms}"; fi
                fi
            done < "$RESULTS_FILE"
        fi

        echo "$name,$ic3_proven,$ic3_failed,$ic3_ms,$ic3_mem,$bmc_proven,$bmc_failed,$bmc_ms,$bmc_mem" >> "$summary"
    done

    # Totals row
    awk -F, 'NR>1 {ip+=$2; if_+=$3; bp+=$6; bf+=$7}
        END {printf "TOTAL,%d,%d,,,%d,%d,,\n",ip,if_,bp,bf}' "$summary" >> "$summary"

    echo ""
    echo "Evaluation summary: $summary"
}

# ============================================================================
# Automation Metrics (CSV)
# ============================================================================
generate_automation_metrics() {
    mkdir -p "$METRICS_DIR"
    local metrics="$METRICS_DIR/automation_metrics.csv"

    echo "system,sysml_loc,smv_loc,reward_loc,smv_to_sysml_ratio" > "$metrics"

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name sysml_rel smv_subdir smv_file <<< "$entry"

        if [[ -n "$SINGLE_MODEL" && "$name" != "$SINGLE_MODEL" ]]; then
            continue
        fi

        local sysml_path="$SYSML_DIR/$sysml_rel"
        local smv_path="$SMV_DIR/$smv_subdir/$smv_file"
        local reward_file="$RL_DIR/env.py"

        local sysml_loc=0 smv_loc=0 reward_loc=0

        if [[ -f "$sysml_path" ]]; then
            sysml_loc=$(wc -l < "$sysml_path")
        fi
        if [[ -f "$smv_path" ]]; then
            smv_loc=$(wc -l < "$smv_path")
        fi
        if [[ -f "$reward_file" ]]; then
            reward_loc=$(wc -l < "$reward_file")
        fi

        local ratio=""
        if (( sysml_loc > 0 )); then
            ratio=$(awk "BEGIN {printf \"%.2f\", $smv_loc / $sysml_loc}")
        fi

        echo "$name,$sysml_loc,$smv_loc,$reward_loc,$ratio" >> "$metrics"
    done

    echo "Automation metrics: $metrics"
}

# ============================================================================
# CPU Training Summary (CSV)
# ============================================================================
generate_cpu_training_summary() {
    mkdir -p "$METRICS_DIR"
    local summary="$METRICS_DIR/cpu_training_summary.csv"

    echo "system,seed,cpu_mode,cpu_affinity_core,selected_checkpoint_source,selected_episode,selected_success_rate,selected_override_rate,selected_safety_violation_rate,final_checkpoint_safe,train_seconds,training_peak_rss_mb,eval_success_rate,eval_override_rate,eval_safety_violation_rate,test_success_rate,test_override_rate,test_safety_violation_rate,test_mean_episode_steps,test_n_steps,policy_us_mean,policy_us_p99,shield_us_mean,shield_us_p99,total_us_mean,total_us_p99" > "$summary"

    local wrote=false
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name _ _ _ <<< "$entry"

        if [[ -n "$SINGLE_MODEL" && "$name" != "$SINGLE_MODEL" ]]; then
            continue
        fi

        local i seed seed_summary
        for ((i = 0; i < NUM_SEEDS; i++)); do
            seed="$(seed_for_index "$i")"
            seed_summary="$CPU_RESULTS_DIR/$name/seed_$seed/summary.json"
            if [[ ! -f "$seed_summary" ]]; then
                continue
            fi
            "$PYTHON_BIN" - "$seed_summary" "$summary" "$name" <<'PY'
import csv
import json
import sys

summary_json, out_csv, system = sys.argv[1:4]
with open(summary_json) as f:
    data = json.load(f)
eval_ = data.get("eval", {})
test = data.get("test", {})
best = data.get("best_during_training", {})
execution = data.get("execution", {})
row = {
    "system": system,
    "seed": data.get("seed", ""),
    "cpu_mode": execution.get("cpu_mode", ""),
    "cpu_affinity_core": execution.get("cpu_affinity_core", ""),
    "selected_checkpoint_source": best.get("selected_checkpoint_source", ""),
    "selected_episode": best.get("selected_episode", ""),
    "selected_success_rate": best.get("success_rate", ""),
    "selected_override_rate": best.get("override_rate", ""),
    "selected_safety_violation_rate": best.get("safety_violation_rate", ""),
    "final_checkpoint_safe": best.get("final_checkpoint_safe", ""),
    "train_seconds": data.get("train_seconds", ""),
    "training_peak_rss_mb": data.get("peak_rss_mb", ""),
    "eval_success_rate": eval_.get("success_rate", ""),
    "eval_override_rate": eval_.get("pooled_override_rate", ""),
    "eval_safety_violation_rate": eval_.get("safety_violation_rate", ""),
    "test_success_rate": test.get("success_rate", ""),
    "test_override_rate": test.get("pooled_override_rate", ""),
    "test_safety_violation_rate": test.get("safety_violation_rate", ""),
    "test_mean_episode_steps": test.get("mean_episode_steps", ""),
    "test_n_steps": test.get("n_steps", ""),
    "policy_us_mean": test.get("policy_us_mean", ""),
    "policy_us_p99": test.get("policy_us_p99", ""),
    "shield_us_mean": test.get("shield_us_mean", ""),
    "shield_us_p99": test.get("shield_us_p99", ""),
    "total_us_mean": test.get("total_us_mean", ""),
    "total_us_p99": test.get("total_us_p99", ""),
}
with open(out_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
PY
            wrote=true
        done
    done

    if $wrote; then
        echo "CPU training summary: $summary"
    else
        echo "CPU training summary: $summary (no completed seed outputs found)"
    fi
}

# ============================================================================
# Runtime Monitor Evaluation
# ============================================================================
run_runtime_monitor_eval() {
    require_cpu_training_program || return 1

    local out_dir="$METRICS_DIR/runtime_results"
    local summary="$out_dir/runtime_monitor_summary.csv"
    mkdir -p "$out_dir"

    echo "system,seed,cpu_mode,cpu_affinity_core,episodes,total_steps,success_rate,safety_violation_rate,override_rate,final_checkpoint_safe,policy_us_mean,policy_us_p95,policy_us_p99,shield_us_mean,shield_us_p95,shield_us_p99,total_us_mean,total_us_p95,total_us_p99,inference_peak_rss_mb,eval_seconds,test_seconds" > "$summary"

    local any_ckpt=false
    local seed_pids=()
    declare -A seed_info=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name sysml_rel _ _ <<< "$entry"

        if [[ -n "$SINGLE_MODEL" && "$name" != "$SINGLE_MODEL" ]]; then
            continue
        fi

        local sysml_path="$SYSML_DIR/$sysml_rel"
        local i seed seed_dir ckpt out_json
        for ((i = 0; i < NUM_SEEDS; i++)); do
            seed="$(seed_for_index "$i")"
            seed_dir="$CPU_RESULTS_DIR/$name/seed_$seed"
            ckpt="$seed_dir/best.npz"
            out_json="$seed_dir/inference_summary.json"

            if [[ ! -f "$ckpt" ]]; then
                continue
            fi
            any_ckpt=true

            if [[ "$FORCE_FULL" == true ]] || ! runtime_seed_complete "$out_json"; then
                log "[$name] Runtime monitor evaluation seed $seed"
                local log_file="$out_dir/${name}_seed_${seed}_runtime_eval.log"
                wait_for_available_job_slot
                (
                    run_cpu_bound "$PYTHON_BIN" "$CPU_EVAL_ENTRY" \
                        --program-root "$CPU_TRAINING_REPO" \
                        --model-path "$sysml_path" \
                        --ckpt "$ckpt" \
                        --seed "$seed" \
                        --cpu-mode "$CPU_EXECUTION_MODE" \
                        --cpu-affinity-core "$CPU_AFFINITY_CORE" \
                        --dt 0.1 \
                        --max-steps 5000 \
                        --eval-episodes 100 \
                        --test-episodes 200 \
                        --out "$out_json"
                ) > "$log_file" 2>&1 &
                local pid=$!
                seed_pids+=("$pid")
                seed_info[$pid]="$name|$seed|$log_file|$out_json"
            fi
        done
    done

    wait_for_eval_jobs || return 1

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name sysml_rel _ _ <<< "$entry"

        if [[ -n "$SINGLE_MODEL" && "$name" != "$SINGLE_MODEL" ]]; then
            continue
        fi

        local i seed seed_dir ckpt out_json
        for ((i = 0; i < NUM_SEEDS; i++)); do
            seed="$(seed_for_index "$i")"
            seed_dir="$CPU_RESULTS_DIR/$name/seed_$seed"
            out_json="$seed_dir/inference_summary.json"
            if [[ -f "$out_json" ]]; then
                "$PYTHON_BIN" - "$out_json" "$summary" "$name" <<'PY'
import csv
import json
import sys

in_json, out_csv, system = sys.argv[1:4]
with open(in_json) as f:
    data = json.load(f)
test = data.get("test", {})
execution = data.get("execution", {})
row = {
    "system": system,
    "seed": data.get("seed", ""),
    "cpu_mode": execution.get("cpu_mode", ""),
    "cpu_affinity_core": execution.get("cpu_affinity_core", ""),
    "episodes": test.get("n_episodes", ""),
    "total_steps": test.get("n_steps", ""),
    "success_rate": test.get("success_rate", ""),
    "safety_violation_rate": test.get("safety_violation_rate", ""),
    "override_rate": test.get("pooled_override_rate", ""),
    "final_checkpoint_safe": data.get("final_checkpoint_safe", ""),
    "policy_us_mean": test.get("policy_us_mean", ""),
    "policy_us_p95": test.get("policy_us_p95", ""),
    "policy_us_p99": test.get("policy_us_p99", ""),
    "shield_us_mean": test.get("shield_us_mean", ""),
    "shield_us_p95": test.get("shield_us_p95", ""),
    "shield_us_p99": test.get("shield_us_p99", ""),
    "total_us_mean": test.get("total_us_mean", ""),
    "total_us_p95": test.get("total_us_p95", ""),
    "total_us_p99": test.get("total_us_p99", ""),
    "inference_peak_rss_mb": data.get("inference_peak_rss_mb", ""),
    "eval_seconds": data.get("eval_seconds", ""),
    "test_seconds": data.get("test_seconds", ""),
}
with open(out_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
PY
            fi
        done
    done

    if $any_ckpt; then
        echo "Runtime monitor summary: $summary"
    else
        echo "  (no CPU training checkpoints found, skipping runtime monitor eval)"
    fi
}

# ============================================================================
# Human-Readable Report
# ============================================================================
generate_human_report() {
    if [[ ! -f "$REPORT_ENTRY" ]]; then
        fail "Report generator not found: $REPORT_ENTRY"
        return 1
    fi

    "$PYTHON_BIN" "$REPORT_ENTRY" \
        --metrics-dir "$METRICS_DIR" \
        --out "$METRICS_DIR/pipeline_report.md" \
        --seeds "$(seed_list_display)" \
        --models "$(model_list_display)" \
        --cpu-mode "$CPU_EXECUTION_MODE" \
        --cpu-affinity-core "$CPU_AFFINITY_CORE"
}

main "$@"
