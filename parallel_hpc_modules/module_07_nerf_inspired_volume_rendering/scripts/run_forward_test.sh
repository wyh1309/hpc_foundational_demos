#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FORWARD_DEMO="${FORWARD_DEMO:-${PROJECT_ROOT}/build/src/forward_demo}"
USE_SANITIZER="${USE_SANITIZER:-}"

#----------------------------------------------------------------------
# Usage examples:
#
# 1. 原生完整套件（功能+非法输入校验）
#    bash run_forward_test.sh
#
# 2. memcheck：仅边界子集，不跑rejected，速度可控
#    USE_SANITIZER=memcheck bash run_forward_test.sh
#
# 3. racecheck 共享内存竞争 / 同步错误检测
#    USE_SANITIZER=racecheck bash run_forward_test.sh
#
# 4. 指定自定义二进制路径 + memcheck
#    FORWARD_DEMO=./build-cuda/src/forward_demo USE_SANITIZER=memcheck bash run_forward_test.sh

if [[ ! -x "${FORWARD_DEMO}" ]]; then
    echo "forward_demo executable not found: ${FORWARD_DEMO}" >&2
    echo "Set FORWARD_DEMO=/path/to/forward_demo or build with BUILD_CUDA_FORWARD=ON and BUILD_DEMOS=ON." >&2
    exit 1
fi

# 执行命令前缀
EXEC_PREFIX=()
if [[ -n "${USE_SANITIZER}" ]]; then
    EXEC_PREFIX=(compute-sanitizer --tool "${USE_SANITIZER}")
    echo "=== Compute Sanitizer enabled, tool=${USE_SANITIZER} ==="
    echo "=== Sanitizer mode: only run boundary subset, skip rejected-input cases ==="
fi

passed=0
failed=0

# 通用有效用例运行
run_valid_case() {
    local rays="$1"
    local samples="$2"
    echo
    echo "[VALID] rays=${rays}, samples=${samples}"
    if "${EXEC_PREFIX[@]}" "${FORWARD_DEMO}" "${rays}" "${samples}"; then
        passed=$((passed + 1))
    else
        echo "case failed: rays=${rays}, samples=${samples}" >&2
        failed=$((failed + 1))
    fi
}

# 仅原生模式使用：校验非法输入拒绝逻辑
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

if [[ -z "${USE_SANITIZER}" ]]; then
    # ========== 原生模式：全部用例 ==========
    echo "==== Running full test suite (native) ===="
    # ray count cases
    run_valid_case 1 64
    run_valid_case 1000 64
    run_valid_case 16384 64
    # sample count cases
    run_valid_case 4096 1
    run_valid_case 4096 64
    run_valid_case 4096 256
    # invalid input rejection
    run_rejected_case "zero rays" 0 64
    run_rejected_case "zero samples" 4096 0
else
    # ========== Sanitizer模式：只跑边界子集，不跑rejected ==========
    echo "==== Running sanitizer boundary subset ===="
    # 极小、非32对齐边界、中等规模；避开超大尺寸防止sanitizer卡死
    run_valid_case 1 1
    run_valid_case 1 64
    run_valid_case 997 64
    run_valid_case 4096 1
    run_valid_case 4096 64
    run_valid_case 4096 256
fi

echo
echo "Summary: ${passed} passed, ${failed} failed"
(( failed == 0 ))
