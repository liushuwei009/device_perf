#!/usr/bin/env bash
# B300 benchmark launcher. Platform NCCL_* settings are inherited unchanged.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_FILE="$SCRIPT_DIR/bench_b300.py"
die() { echo "错误: $*" >&2; exit 2; }

usage() {
    cat <<'HELP'
B300 GPU 基准测试入口

用法:
  bash run.sh [启动选项] [bench_b300.py 参数]

常用示例:
  bash run.sh                                  # 全部 suites，自动生成结果目录
  bash run.sh --outdir results/b300_001
  bash run.sh --suites fp8,moe --quick
  bash run.sh --plan-only
  bash run.sh --bench-help
  bash run.sh --dry-run --suites gemm

启动选项（也可用对应环境变量）:
  --nnodes N             节点数；NNODES，默认 1
  --node-rank N          节点序号；NODE_RANK，默认 0
  --master-addr HOST     主节点地址；MASTER_ADDR，多机必填
  --master-port PORT     主节点端口；MASTER_PORT，默认 29500
  --nproc-per-node N     每节点进程数；NPROC_PER_NODE，默认本机可见 GPU 数
  --stress-seconds N     稳定性时长；STRESS_SECONDS，默认 300
  --dry-run              只打印 torchrun 命令，不运行、不创建日志目录
  --bench-help           显示 bench_b300.py 参数说明
  -h, --help             显示本说明

bench_b300.py 参数会原样透传，例如 --suites、--quick、--outdir、--spec-file。
多机运行时请在所有节点使用同一个 --outdir（或 B300_OUTDIR）。
HELP
}

NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
STRESS_SECONDS="${STRESS_SECONDS:-300}"
OUTDIR="${B300_OUTDIR:-}"
TORCHRUN_EXTRA="${TORCHRUN_EXTRA:-}"
DRY_RUN=0
BENCH_HELP=0
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --bench-help)
            BENCH_HELP=1
            shift
            ;;
        --target|--target=*)
            die "本仓库仅支持 B300；无需 --target"
            ;;
        --nnodes|--node-rank|--master-addr|--master-port|--nproc-per-node|--stress-seconds)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 缺少参数"
            case "$1" in
                --nnodes) NNODES="$2" ;;
                --node-rank) NODE_RANK="$2" ;;
                --master-addr) MASTER_ADDR="$2" ;;
                --master-port) MASTER_PORT="$2" ;;
                --nproc-per-node) NPROC_PER_NODE="$2" ;;
                --stress-seconds) STRESS_SECONDS="$2" ;;
            esac
            shift 2
            ;;
        --nnodes=*) NNODES="${1#*=}"; shift ;;
        --node-rank=*) NODE_RANK="${1#*=}"; shift ;;
        --master-addr=*) MASTER_ADDR="${1#*=}"; shift ;;
        --master-port=*) MASTER_PORT="${1#*=}"; shift ;;
        --nproc-per-node=*) NPROC_PER_NODE="${1#*=}"; shift ;;
        --stress-seconds=*) STRESS_SECONDS="${1#*=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

PYTHON_BIN="${PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "找不到 Python: $PYTHON_BIN"
[[ -f "$BENCH_FILE" ]] || die "找不到 $BENCH_FILE"

if (( BENCH_HELP )); then
    exec "$PYTHON_BIN" "$BENCH_FILE" --help
fi

# Read quoted environment arguments without evaluating shell substitutions.
parse_env_args() {
    local key="$1" temporary_file
    temporary_file="$(mktemp)" || die "无法创建临时文件"
    if ! "$PYTHON_BIN" -c 'import os, shlex, sys; sys.stdout.buffer.write(b"".join(s.encode()+b"\0" for s in shlex.split(os.environ[sys.argv[1]])))' "$key" > "$temporary_file"; then
        rm -f "$temporary_file"
        die "$key 引号不匹配"
    fi
    PARSED_ARGS=()
    while IFS= read -r -d '' item; do PARSED_ARGS+=("$item"); done < "$temporary_file"
    rm -f "$temporary_file"
}

has_arg() {
    local name="$1" item
    for item in ${ARGS[@]+"${ARGS[@]}"}; do
        [[ "$item" == "$name" || "$item" == "$name="* ]] && return 0
    done
    return 1
}

if has_arg --plan-only || has_arg --report; then
    exec "$PYTHON_BIN" "$BENCH_FILE" ${ARGS[@]+"${ARGS[@]}"}
fi

[[ -z "${LOCAL_RANK:-}" ]] || die "请在普通 shell 启动，不要在 torchrun worker 内嵌套启动"
[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || die "节点数必须为正整数: $NNODES"
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || die "节点序号必须为非负整数: $NODE_RANK"
(( NODE_RANK < NNODES )) || die "节点序号必须小于节点数"
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || die "端口必须为正整数"
(( MASTER_PORT <= 65535 )) || die "端口必须在 1~65535"
[[ "$STRESS_SECONDS" =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]] || die "稳定性秒数必须为正数"

if (( NNODES > 1 )); then
    [[ -n "$MASTER_ADDR" ]] || die "多机运行必须设置 MASTER_ADDR 或 --master-addr"
    if [[ -z "$OUTDIR" ]] && ! has_arg --outdir; then
        die "多机自动目录会因节点时间不同而分裂；请给所有节点传同一个 --outdir 或设置 B300_OUTDIR"
    fi
fi
if [[ -n "$OUTDIR" ]] && has_arg --outdir; then
    die "B300_OUTDIR 与 --outdir 只能选择一个"
fi

if [[ -z "$NPROC_PER_NODE" ]]; then
    NPROC_PER_NODE="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')" ||
        die "无法探测 GPU；请安装/检查 PyTorch，或设置 NPROC_PER_NODE"
fi
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "每节点进程数必须为正整数: $NPROC_PER_NODE"

# A shared explicit output directory is required on multiple nodes. Generate a
# local one only after validation and GPU probing have succeeded.
if [[ -z "$OUTDIR" ]] && ! has_arg --outdir; then
    OUTDIR="$PWD/results/b300_$(date +%Y%m%d-%H%M%S)_$$"
fi

CMD=("$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$NPROC_PER_NODE")
if (( NNODES == 1 )); then
    CMD+=(--standalone --nnodes 1)
else
    CMD+=(--nnodes "$NNODES" --node_rank "$NODE_RANK"
          --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
fi
if [[ -n "$TORCHRUN_EXTRA" ]]; then
    parse_env_args TORCHRUN_EXTRA
    CMD+=(${PARSED_ARGS[@]+"${PARSED_ARGS[@]}"})
fi
CMD+=("$BENCH_FILE")
[[ -z "$OUTDIR" ]] || CMD+=(--outdir "$OUTDIR")
if ! has_arg --stress-seconds; then
    CMD+=(--stress-seconds "$STRESS_SECONDS")
fi
CMD+=(${ARGS[@]+"${ARGS[@]}"})

LOG_ROOT="${LOG_DIR:-$PWD/logs}"
[[ "$DRY_RUN" == 1 ]] || mkdir -p "$LOG_ROOT" || die "无法创建日志目录: $LOG_ROOT"
export PYTHONUNBUFFERED=1

if [[ "$DRY_RUN" == 1 ]]; then
    printf '%s\n' "$(printf '%q ' ${CMD[@]})"
    exit 0
fi

LOG="$LOG_ROOT/b300_node${NODE_RANK}_$(date +%Y%m%d-%H%M%S)_$$.log"
{
    echo "任务=b300 节点数=$NNODES 节点序号=$NODE_RANK 每节点进程=$NPROC_PER_NODE"
    printf '命令: %s\n' "$(printf '%q ' ${CMD[@]})"
    "${CMD[@]}"
} 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
echo "退出码=$rc 日志=$LOG"
exit "$rc"
