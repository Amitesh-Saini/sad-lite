#!/usr/bin/env bash
# =============================================================================
# SAD-Lite — Run the Whole Pipeline
# =============================================================================
#
# PURPOSE
# Run every stage in dependency order and stop at the first failure:
#   1. Preflight        Python, packages, source and configuration files, Git,
#                       the output location, and a run lock
#   2. Generate         synthetic sessions, separate truth files, manifests
#   3. Features         verified from the generate stage
#   4. Rule scoring     with config/rules.yaml
#   5. Model scoring    Isolation Forest with an explicit seed
#   6. Evaluation       the only stage that reads truth; writes summary.md
#   7. Investigation    one explicitly selected account
#   8. Enforcement      a dry-run preview, only if the case supports a reason
#   9. Game monitoring  every game and window; alert report
#  10. Agent            restricted tools: preview, injection demo, and
#                       (with --proposal-demo) a real local branch
#  11. Summary          where every output is
#
# WHAT THIS SCRIPT DOES NOT DO
# - Provenance: each Python stage records and verifies its own manifests;
#   nothing here re-implements that. A stale or mixed stage stops the run
#   in the stage that reads it.
# - Bans: the enforcement stage is a preview. It never passes --apply and
#   never overrides the evidence policy. If the case report supports no
#   permitted reason at Moderate or above, no ban is previewed, and the run
#   continues; that is a correct outcome.
# - Your configuration: the agent works in a disposable Git repository under
#   reports/agent/workspace holding a copy of config/games.yaml. Your own
#   games.yaml, branches, and history are never touched, and nothing is
#   merged or pushed.
# - Installing anything: missing packages stop the run with a message.
#
# OUTPUT LOCATION AND REPLACEMENT
# Outputs go to the fixed data/ and reports/ folders. If generated outputs
# already exist, the run refuses unless --replace is given. --replace removes
# only this list, and nothing else:
#   data/{generate,truth,features,rules,rules_shared_ip_raised,model}
#   data/.*.tmp-* (interrupted publishes)
#   reports/{comparison_*,feature_distributions.csv,cases,
#            enforcement_preview,disarm,agent}
# It never removes enforcement/ (decisions, archived datasets, evidence),
# reports/calibration (referenced by config/games.yaml), config/, source
# code, or Git history. Everything is removed before the first stage runs,
# so an interrupted run cannot leave older downstream results looking
# current.
#
# CONCURRENCY
# One run at a time: a lock directory (.run_all.lock) is created atomically
# at start and removed on exit. Running the individual scripts by hand while
# this runs is unsupported. If a run was killed, remove .run_all.lock after
# checking that no run is active.
#
# Usage:
#     ./run_all.sh [--seed N] [--model-seed N] [--demo-user user_NNNNN]
#                  [--proposal-demo] [--replace] [--test]
#
#   --seed N           generator seed (default 42)
#   --model-seed N     Isolation Forest seed (default 42)
#   --demo-user ID     account to investigate (default: the model's rank 1,
#                      chosen from the model stage without reading truth)
#   --proposal-demo    let the agent create a real disarm branch in the
#                      disposable workspace (default: preview only)
#   --replace          remove the previous generated results first
#   --test             run the test suite before the pipeline
# =============================================================================

set -Eeuo pipefail

# Make execution independent of the caller's current directory.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CURRENT_STAGE="initialisation"
LOCK_DIR="$PROJECT_ROOT/.run_all.lock"
HAVE_LOCK=0

# Report once: a failure inside $(...) also fails the assignment in the parent shell.
trap 'rc=$?; [ "$BASH_SUBSHELL" -eq 0 ] && printf >&2 "\nFAILED: %s (exit %s)\n" "$CURRENT_STAGE" "$rc"; exit "$rc"' ERR
trap 'if [ "$HAVE_LOCK" = 1 ]; then rm -rf -- "$LOCK_DIR"; fi' EXIT

run_stage() {
    CURRENT_STAGE="$1"
    shift
    printf '\n== %s ==\n' "$CURRENT_STAGE"
    "$@"
}

fail() {
    printf >&2 '\nFAILED: %s: %s\n' "$CURRENT_STAGE" "$1"
    exit 1
}

usage() {
    cat <<'USAGE'
Usage: ./run_all.sh [--seed N] [--model-seed N] [--demo-user user_NNNNN]
                    [--proposal-demo] [--replace] [--test]

  --seed N          generator seed (default 42)
  --model-seed N    Isolation Forest seed (default 42)
  --demo-user ID    account to investigate (default: the model's rank 1)
  --proposal-demo   create a real disarm branch in the disposable workspace
  --replace         remove the previous generated results first
  --test            run the test suite before the pipeline
USAGE
}

# ------------------------------------------------------------
# 1. Arguments.
# ------------------------------------------------------------

SEED=42
MODEL_SEED=42
DEMO_USER=""
PROPOSAL_DEMO=0
REPLACE=0
RUN_TESTS=0

need_value() {
    [ "$#" -ge 2 ] && [ -n "$2" ] || fail "$1 needs a value"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --seed)          need_value "$@"; SEED="$2"; shift 2 ;;
        --model-seed)    need_value "$@"; MODEL_SEED="$2"; shift 2 ;;
        --demo-user)     need_value "$@"; DEMO_USER="$2"; shift 2 ;;
        --proposal-demo) PROPOSAL_DEMO=1; shift ;;
        --replace)       REPLACE=1; shift ;;
        --test)          RUN_TESTS=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               fail "unknown argument '$1' (see --help)" ;;
    esac
done

[[ "$SEED" =~ ^[0-9]{1,9}$ ]] || fail "--seed must be a non-negative integer, not '$SEED'"
[[ "$MODEL_SEED" =~ ^[0-9]{1,9}$ ]] || fail "--model-seed must be a non-negative integer, not '$MODEL_SEED'"
[ -z "$DEMO_USER" ] || [[ "$DEMO_USER" =~ ^user_[0-9]{5}$ ]] || fail "--demo-user must look like user_00042, not '$DEMO_USER'"

# ------------------------------------------------------------
# 2. Lock, then preflight.
# ------------------------------------------------------------

CURRENT_STAGE="Preflight"
printf '\n== %s ==\n' "$CURRENT_STAGE"

if mkdir -- "$LOCK_DIR" 2>/dev/null; then
    HAVE_LOCK=1
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
else
    fail "another run_all.sh holds $LOCK_DIR (pid $(cat "$LOCK_DIR/pid" 2>/dev/null || echo unknown)). If no run is active, remove it."
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "$PYTHON_BIN not found (set PYTHON_BIN to choose another interpreter)"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail "Python 3.10 or newer is required; found $("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')"

MODULES="numpy pandas sklearn yaml"
[ "$RUN_TESTS" = 0 ] || MODULES="$MODULES pytest"

for module in $MODULES; do
    "$PYTHON_BIN" -c "import $module" 2>/dev/null \
        || fail "Python package '$module' is missing; install the pinned versions with: $PYTHON_BIN -m pip install -r requirements.txt"
done

for file in generate_users.py features.py score_rules.py score_model.py evaluate.py investigate.py enforce.py \
            provenance.py game_config.py game_monitor.py disarm_watch.py agent_sim.py pipeline_demo.py \
            config/rules.yaml config/rules_shared_ip_raised.yaml config/games.yaml; do
    [ -f "$file" ] && [ ! -L "$file" ] || fail "required file $file is missing (or is a symlink)"
done

"$PYTHON_BIN" -c 'import game_config; game_config.load("config/games.yaml")' \
    || fail "config/games.yaml is invalid"

command -v git >/dev/null 2>&1 || fail "git is required for the agent stage"

# Generated outputs that a run produces, and only those.
shopt -s nullglob
GENERATED=(
    data/generate data/truth data/features data/rules data/rules_shared_ip_raised data/model data/.*.tmp-*
    reports/comparison_* reports/feature_distributions.csv reports/cases reports/enforcement_preview
    reports/disarm reports/agent
)
EXISTING=()
for path in "${GENERATED[@]}"; do
    [ -e "$path" ] && EXISTING+=("$path")
done
shopt -u nullglob

if [ "${#EXISTING[@]}" -gt 0 ] && [ "$REPLACE" = 0 ]; then
    printf >&2 'Existing generated results:\n'
    printf >&2 '  %s\n' "${EXISTING[@]}"
    fail "generated results already exist; re-run with --replace to remove them and regenerate"
fi

printf 'Python %s, git %s; seeds: generator %s, model %s.\n' \
    "$("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')" "$(git --version | awk '{print $3}')" "$SEED" "$MODEL_SEED"

if [ "$RUN_TESTS" = 1 ]; then
    run_stage "Tests" "$PYTHON_BIN" -m pytest tests/ -q
fi

if [ "${#EXISTING[@]}" -gt 0 ]; then
    CURRENT_STAGE="Replace previous results"
    printf '\n== %s ==\n' "$CURRENT_STAGE"
    printf 'Replacing removes the previous generated results:\n'
    for path in "${EXISTING[@]}"; do
        printf '  %s\n' "$path"
        rm -rf -- "$PROJECT_ROOT/$path"
    done
    printf 'Kept: enforcement/, reports/calibration/, config/, source code, Git history.\n'
fi

# ------------------------------------------------------------
# 3. Detection pipeline.
# ------------------------------------------------------------

run_stage "Generate sessions (seed $SEED)" "$PYTHON_BIN" generate_users.py --seed "$SEED"
run_stage "Build features" "$PYTHON_BIN" features.py
run_stage "Score rules" "$PYTHON_BIN" score_rules.py --config config/rules.yaml --stage rules
run_stage "Score model (seed $MODEL_SEED)" "$PYTHON_BIN" score_model.py --seed "$MODEL_SEED" --stage model
run_stage "Evaluate results" "$PYTHON_BIN" evaluate.py --config config/rules.yaml \
    --experiment-config config/rules_shared_ip_raised.yaml

shopt -s nullglob
SUMMARIES=(reports/comparison_*/summary.md)
shopt -u nullglob
[ "${#SUMMARIES[@]}" -eq 1 ] || fail "expected one evaluation summary, found ${#SUMMARIES[@]}"

# ------------------------------------------------------------
# 4. Account demonstration (no truth; preview only).
# ------------------------------------------------------------

CURRENT_STAGE="Select demonstration account"
if [ -z "$DEMO_USER" ]; then
    DEMO_USER="$("$PYTHON_BIN" pipeline_demo.py select-user --data-dir data)"
    printf '\n== %s ==\nModel rank 1: %s (a review priority, not evidence).\n' "$CURRENT_STAGE" "$DEMO_USER"
fi

CURRENT_STAGE="Investigate $DEMO_USER"
printf '\n== %s ==\n' "$CURRENT_STAGE"
if ! INVESTIGATION="$("$PYTHON_BIN" investigate.py --user-id "$DEMO_USER" --data-dir data --output-dir reports/cases)"; then
    printf '%s\n' "$INVESTIGATION"
    fail "investigate.py did not produce an account case for $DEMO_USER"
fi
printf '%s\n' "$INVESTIGATION"
CASE_JSON="$(printf '%s\n' "$INVESTIGATION" | sed -n 's/^Case files: .* and \(.*\.json\)$/\1/p')"
[ -n "$CASE_JSON" ] && [ -f "$CASE_JSON" ] || fail "investigate.py did not report a case file"

run_stage "Preview enforcement decision (dry run)" "$PYTHON_BIN" pipeline_demo.py enforcement-preview \
    --data-dir data --report "$CASE_JSON" --output-dir reports/enforcement_preview

# ------------------------------------------------------------
# 5. Game-level monitoring and agent demonstration.
# ------------------------------------------------------------

run_stage "Monitor game termination rates" "$PYTHON_BIN" disarm_watch.py scan --data-dir data --config config/games.yaml

CURRENT_STAGE="Prepare disposable agent workspace"
printf '\n== %s ==\n' "$CURRENT_STAGE"
WORKSPACE="reports/agent/workspace"
mkdir -p "$WORKSPACE/config"
cp -- config/games.yaml "$WORKSPACE/config/games.yaml"
git -C "$WORKSPACE" -c init.defaultBranch=main init -q
git -C "$WORKSPACE" add config/games.yaml
git -C "$WORKSPACE" -c user.name="SAD-Lite run_all" -c user.email="run-all@sadlite.invalid" \
    commit -q -m "Copy of config/games.yaml for the agent demonstration"
printf 'Disposable repository %s with a committed copy of config/games.yaml.\n' "$WORKSPACE"

run_stage "Simulate restricted agent workflow (preview)" "$PYTHON_BIN" agent_sim.py run --dry-run \
    --data-dir data --config "$WORKSPACE/config/games.yaml"

run_stage "Injection demonstration" "$PYTHON_BIN" agent_sim.py injection-demo \
    --data-dir data --config "$WORKSPACE/config/games.yaml"

if [ "$PROPOSAL_DEMO" = 1 ]; then
    run_stage "Proposal demonstration (local branch, not merged)" "$PYTHON_BIN" agent_sim.py run \
        --data-dir data --config "$WORKSPACE/config/games.yaml"
fi

# ------------------------------------------------------------
# 6. Summary.
# ------------------------------------------------------------

CURRENT_STAGE="complete"
printf '\nPipeline completed successfully.\n\n'

shopt -s nullglob
printf 'Seeds:                 generator %s, model %s\n' "$SEED" "$MODEL_SEED"
printf 'Evaluation summary:    %s\n' "${SUMMARIES[0]}"
printf 'Investigation report:  %s\n' "${CASE_JSON%.json}.md"
for path in reports/enforcement_preview/preview_*.json; do
    printf 'Enforcement preview:   %s\n' "$path"
done
for path in reports/disarm/scan_*/alerts.md; do
    printf 'Game alerts:           %s\n' "$path"
done
for path in reports/agent/run_*/trace.md; do
    printf 'Agent trace:           %s\n' "$path"
done
if [ "$PROPOSAL_DEMO" = 1 ]; then
    for branch in $(git -C "$WORKSPACE" for-each-ref --format='%(refname:short)' 'refs/heads/disarm/*'); do
        printf 'Proposal branch:       %s in %s (not merged)\n' "$branch" "$WORKSPACE"
    done
    for path in reports/disarm/proposals/merge_request_*.md; do
        printf 'Review description:    %s\n' "$path"
    done
fi
shopt -u nullglob