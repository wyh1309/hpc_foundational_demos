#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FORWARD_DEMO="${FORWARD_DEMO:-${PROJECT_ROOT}/build/src/forward_demo}"

if [[ ! -x "${FORWARD_DEMO}" ]]; then
    echo "forward_demo executable not found: ${FORWARD_DEMO}" >&2
    echo "Set FORWARD_DEMO=/path/to/forward_demo or build with BUILD_CUDA_FORWARD=ON and BUILD_DEMOS=ON." >&2
    exit 1
fi

passed=0
failed=0

run_valid_case() {
    local rays="$1"
    local samples="$2"

    echo
    echo "[VALID] rays=${rays}, samples=${samples}"
    if "${FORWARD_DEMO}" "${rays}" "${samples}"; then
        passed=$((passed + 1))
    else
        echo "case failed: rays=${rays}, samples=${samples}" >&2
        failed=$((failed + 1))
    fi
}

run_rejected_case() {
    local name="$1"
    local rays="$2"
    local samples="$3"

    echo
    echo "[REJECTED INPUT] ${name}: rays=${rays}, samples=${samples}"
    if "${FORWARD_DEMO}" "${rays}" "${samples}" >/dev/null 2>&1; then
        echo "case unexpectedly succeeded: ${name}" >&2
        failed=$((failed + 1))
    else
        echo "rejected as expected"
        passed=$((passed + 1))
    fi
}

# Ray-count and thread-boundary cases.
run_valid_case 1 64
run_valid_case 1000 64
run_valid_case 16384 64

# Sample-count cases.
run_valid_case 4096 1
run_valid_case 4096 64
run_valid_case 4096 256

# The demo accepts only positive dimensions. These cases verify that invalid
# API inputs are rejected before any CUDA work is started.
run_rejected_case "zero rays" 0 64
run_rejected_case "zero samples" 4096 0

echo
echo "Summary: ${passed} passed, ${failed} failed"
(( failed == 0 ))
