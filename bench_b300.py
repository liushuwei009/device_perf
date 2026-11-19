#!/usr/bin/env python3
"""B300 acceptance workloads. No packages are installed or device settings changed.

python bench_b300.py --plan-only
python -m torch.distributed.run --standalone --nproc_per_node=8 bench_b300.py \
    --outdir results/b300_001

Exit 0: selected required tests passed; 1: failure; 2: incomplete/unsupported.
PASS concerns only the selected scope, never a lifetime/whole-product guarantee.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

VERSION = "0.1.0"
SUITES = ("gemm", "fp8", "mxfp8", "nvfp4", "grouped_gemm", "sparse", "memory", "pcie", "nvlink",
          "collectives", "attention", "paged_attention", "mla", "moe", "stress", "features")
SPEC_SOURCE = "https://www.nvidia.com/en-us/data-center/hgx/"
# HGX B300 system values / 8 GPUs. Dense/sparse are separate, particularly FP4.
# HBM capacity, HBM bandwidth and PCIe are intentionally supplied by the OEM profile.
B300_SPEC = dict(bf16_dense_tflops=2250.0, fp8_dense_tflops=4500.0,
                 mxfp8_dense_tflops=4500.0, nvfp4_dense_tflops=13500.0,
                 bf16_sparse_effective_tflops=4500.0, nvfp4_sparse_effective_tflops=18000.0,
                 nvlink_bidirectional_gbps=1800.0, nvlink_one_way_gbps=900.0)
DESCRIPTIONS = {
    "gemm": "BF16 dense GEMM, sampled FP32 references and CUDA kernel names",
    "fp8": "E4M3/E5M2 tensor scaling, quantization error and native scaled GEMM",
    "mxfp8": "Transformer Engine MXFP8 block scaling and native GEMM",
    "nvfp4": "Transformer Engine NVFP4 E2M1 + block/global scales and native GEMM",
    "grouped_gemm": "Transformer Engine native grouped BF16/FP8/MXFP8/NVFP4 GEMM vs per-expert loops",
    "sparse": "2:4 BF16 semi-structured sparse GEMM vs identical pruned dense weights",
    "memory": "large HBM allocation, complete multi-pattern checks, address/random access, copy",
    "pcie": "NUMA-bound pinned H2D/D2H, serial vs all local GPUs concurrently",
    "nvlink": "all local pairs, round-robin concurrent matchings, contention ratios",
    "collectives": "all-reduce/gather/reduce-scatter/all-to-all; optional externally forced NVLS",
    "attention": "flash SDPA MHA/GQA/MQA prefill and long-context decode",
    "paged_attention": "FlashInfer BF16, FP8 KV-cache and native FP8 QKV paged decode",
    "mla": "FlashInfer paged multi-head latent attention vs independent FP32 reference",
    "moe": "balanced multi-expert dispatch -> FP8/NVFP4/BF16 GEMM -> combine",
    "stress": "compute, communication, memory independently then mixed; timed soak and checks",
    "features": "read-only MIG/virtualization/confidential-computing/RAS inventory; no isolation claim",
}


class Unsupported(RuntimeError):
    """A prerequisite is absent, not a numerical or hardware failure."""


class Unverified(RuntimeError):
    """Work completed but evidence is insufficient for an acceptance verdict."""


def now():
    return datetime.now().astimezone().isoformat()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def select_suites(text):
    names = list(SUITES) if text == "all" else list(dict.fromkeys(x.strip() for x in text.split(",") if x.strip()))
    unknown = set(names) - set(SUITES)
    if not names or unknown:
        raise ValueError(f"unknown/empty suites: {sorted(unknown)}; choose {','.join(SUITES)}")
    return names


def round_robin(ranks):
    """Disjoint pairings covering each unordered pair once, including odd sizes."""
    ranks = list(ranks)
    if len(ranks) < 2:
        return []
    if len(ranks) % 2:
        ranks.append(None)
    rounds = []
    for _ in range(len(ranks) - 1):
        rounds.append([(a, b) for a, b in zip(ranks[:len(ranks)//2], reversed(ranks[len(ranks)//2:]))
                       if a is not None and b is not None])
        ranks = [ranks[0], ranks[-1], *ranks[1:-1]]
    return rounds


def memory_budget(free, reserve, fraction, quick=False):
    available = max(0, free - reserve)
    value = int(available * fraction)
    return min(value, 256 * 1024**2) if quick else value


def summarize_rows(rows, expected_ranks, quick=False, non_b300=False):
    required = [r for r in rows if r.get("required", True)]
    failed = [r for r in required if r["status"] == "fail"]
    incomplete = [r for r in required if r["status"] not in ("pass", "fail")]
    actual = {r["rank"] for r in rows}
    missing_ranks = sorted(set(range(expected_ranks)) - actual)
    # A case must be accounted for on every rank, even when only a pair is active.
    coverage = defaultdict(set)
    for r in required:
        coverage[r["case"]].add(r["rank"])
    missing_cases = {name: sorted(set(range(expected_ranks))-ranks)
                     for name, ranks in coverage.items() if len(ranks) != expected_ranks}
    workload_count = sum(r.get("required", True) and r["status"] == "pass" and
                         r["case"].split("/")[0] in (*SUITES, "bf16") for r in rows)
    state = "FAIL" if failed else "INCOMPLETE" if incomplete or missing_ranks or missing_cases or quick or non_b300 or not workload_count else "PASS"
    slow = []
    by_case = defaultdict(list)
    for r in rows:
        if r["status"] == "pass" and r.get("comparable", False):
            metric = r.get("tflops") or r.get("gbps")
            if metric:
                by_case[r["case"]].append((r, metric))
    for case, items in by_case.items():
        reference = statistics.median(x[1] for x in items)
        for r, value in items:
            if value < .9 * reference:
                slow.append(dict(case=case, rank=r["rank"], gpu_uuid=r.get("gpu_uuid"),
                                 value=value, reference=reference, ratio=value/reference))
    if slow and state == "PASS":
        state = "REVIEW"
    return dict(verdict=state, scope="selected tests only", counts=dict(Counter(r["status"] for r in rows)),
                failed=failed, incomplete=incomplete, missing_ranks=missing_ranks,
                missing_cases=missing_cases, slow_items=slow, quick=quick,
                non_b300=non_b300, passed_workload_records=workload_count,
                optional=[r for r in rows if not r.get("required", True)])


def write_report(root, rows, manifest, expected_ranks):
    root = Path(root)
    result = summarize_rows(rows, expected_ranks, manifest["config"]["quick"],
                            any(not x["is_b300"] for x in manifest.get("devices", [])))
    selected = set(manifest["config"]["suites"]) - {"features"}
    covered = {r["case"].split("/")[0] for r in rows}
    if "bf16" in covered:
        covered.add("gemm")
    result["missing_suites"] = sorted(selected-covered)
    if result["missing_suites"] and result["verdict"] != "FAIL":
        result["verdict"] = "INCOMPLETE"
    result.update(manifest=manifest, rows=rows, finished=now())
    dump(root/"summary.json", result)
    # Keep a merged raw stream so CPU-only re-analysis also works with node-local disks.
    with (root/"merged.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    fields = ["case", "rank", "host", "gpu_uuid", "status", "required", "median_ms", "worst_ms",
              "max_rank_median_ms", "tflops", "gbps", "spec_key", "spec_peak", "pct_spec", "reason", "details"]
    with (root/"results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key) for key in fields[:-1]}
            record["details"] = json.dumps({k: v for k, v in row.items() if k not in fields[:-1]}, ensure_ascii=False)
            writer.writerow(record)
    lines = ["# B300 测试报告", "", f"结论：**{result['verdict']}**（仅针对所选项目）。", "",
             f"测试范围：{', '.join(manifest['config']['suites'])}；quick={manifest['config']['quick']}。", "",
             "退出码：0=所选必测项目通过；1=失败；2=缺失/不支持/待复核。quick 或非 B300 开发运行不出正式 PASS。", "",
             "FP4 稠密与稀疏峰值分别配置。PCIe、HBM 带宽及容量采用客户/OEM spec 文件；未配置时只报实测，不借用其他 GPU 规格。", "",
             "MIG/虚拟化/机密计算只采集状态；CUDA kernel 名称不能证明 TMA/TMEM/双 SM 指令覆盖。", "",
             "只有配置并满足客户约定的性能/容量/数量/时长要求，才支持相应交付验收；默认 PASS 只表示所选已实现测试通过。", "",
             "| 测试 | rank | 状态 | ms | TFLOPS | GB/s | spec % | 原因 |", "|---|---:|---|---:|---:|---:|---:|---|"]
    def fmt(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) else "—"
    for r in rows:
        if r["case"].startswith("stress_sample/"):
            continue
        reason = str(r.get("reason", "")).replace("|", "/").replace("\n", " ")[:220]
        lines.append(f"| {r['case']} | {r['rank']} | {r['status']} | {fmt(r.get('median_ms'))} | {fmt(r.get('tflops'))} | {fmt(r.get('gbps'))} | {fmt(r.get('pct_spec'))} | {reason} |")
    lines += ["", f"缺失 ranks：{result['missing_ranks']}；缺失测量点：{result['missing_cases']}；缺失套件：{result['missing_suites']}。", "",
              f"低于同类中位数 90%：{len(result['slow_items'])} 项，需在相同配置下复测。", "",
              f"规格来源：{SPEC_SOURCE}；具体客户配置见 manifest.json。", "",
              "原始 JSONL、CUDA kernel、数值误差、计时轮次、健康快照和监控见运行目录。"]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n")
    return 1 if result["verdict"] == "FAIL" else 0 if result["verdict"] == "PASS" else 2


def command_output(command, timeout=30):
    try:
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return dict(returncode=p.returncode, output=p.stdout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(returncode=-1, output=str(error))


def package_versions():
    versions = {}
    for name in ("torch", "transformer-engine", "transformer-engine-torch", "flashinfer-python", "nvidia-cublas-cu13"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def normalized_uuid(value):
    return str(value).lower().removeprefix("gpu-")


class Context:
    def __init__(self, args):
        import torch
        import torch.distributed as dist
        self.t, self.dist, self.args = torch, dist, args
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(args.gpu)))
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed + self.rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.root = Path(args.outdir).resolve()
        # NCCL environment is established before its first communicator is created.
        if args.nvls_only:
            if os.environ.get("NCCL_ALGO", "NVLS") != "NVLS":
                raise ValueError("--nvls-only requires NCCL_ALGO=NVLS (existing conflicting setting is not overridden)")
            os.environ["NCCL_ALGO"] = "NVLS"
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,TUNING")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("NCCL_DEBUG_FILE", str(self.root.parent/(self.root.name+".nccl.%h.%p.log")))
        if self.world > 1:
            dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout), device_id=self.device)
        creation = None
        if self.rank == 0:
            try:
                self.root.mkdir(parents=True, exist_ok=False)
            except OSError as e:
                creation = str(e)
        creation = self.gather(creation)[0]
        if creation:
            raise RuntimeError(f"output directory must be new: {creation}")
        self.root.mkdir(parents=True, exist_ok=True)  # Each node may have local storage.
        self.rows = []
        props = torch.cuda.get_device_properties(self.device)
        self.info = dict(rank=self.rank, local_rank=self.local_rank, host=socket.gethostname(),
                         gpu_name=props.name, gpu_uuid=str(props.uuid),
                         capability=list(torch.cuda.get_device_capability(self.device)),
                         total_memory_bytes=props.total_memory, sm_count=props.multi_processor_count,
                         is_b300="B300" in props.name.upper())
        self.infos = self.gather(self.info)
        self.nodes = defaultdict(list)
        for info in self.infos:
            self.nodes[info["host"]].append(info["rank"])
        self.node_ranks = self.nodes[self.info["host"]]
        self.node_leader = self.rank == min(self.node_ranks)
        self.spec = dict(B300_SPEC)
        if args.spec_file:
            profile = json.loads(Path(args.spec_file).read_text())
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0
                   for v in profile.values()):
                raise ValueError("spec values must be finite positive numbers")
            self.spec.update(profile)
        self.manifest = dict(started=now(), version=VERSION, config=vars(args), devices=self.infos,
                             packages=package_versions(), torch=torch.__version__, cuda=torch.version.cuda,
                             nccl=list(torch.cuda.nccl.version()), spec=self.spec, spec_source=SPEC_SOURCE,
                             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             environment={k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "CUDA_VISIBLE"))})
        dump(self.root/f"manifest_rank{self.rank}.json", self.manifest)
        if self.rank == 0:
            dump(self.root/"manifest.json", self.manifest)
            shutil.copy2(__file__, self.root/"bench_b300.py")
        self.snapshot("before")
        self.health_before = self.read_ecc()
        self.monitor = self.monitor_log = None
        if self.node_leader:
            self.monitor_log = (self.root/f"telemetry_{self.info['host']}.csv").open("w")
            query = "timestamp,uuid,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used,clocks_event_reasons.active"
            self.monitor = subprocess.Popen(["nvidia-smi", f"--query-gpu={query}", "--format=csv", "-l", "1"],
                                            stdout=self.monitor_log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL)

    def gather(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world
        self.dist.all_gather_object(values, value)
        return values

    def barrier(self):
        if self.world > 1:
            self.dist.barrier()
        self.t.cuda.synchronize(self.device)

    def maximum(self, value):
        if self.world == 1:
            return value
        x = self.t.tensor(value, dtype=self.t.float64, device=self.device)
        self.dist.all_reduce(x, op=self.dist.ReduceOp.MAX)
        return x.item()

    def emit(self, case, status="pass", required=True, **values):
        row = dict(timestamp=now(), case=case, status=status, required=required,
                   rank=self.rank, host=self.info["host"], gpu_uuid=self.info["gpu_uuid"], **values)
        self.rows.append(row)
        with (self.root/f"rank{self.rank:06d}.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        if self.rank == 0:
            print(f"[{status}] {case} {values.get('reason', '')}", flush=True)
        return row

    def case(self, name, function, prerequisite=None, required=True):
        # Negotiate prerequisites before any participating rank enters a collective.
        try:
            reason = prerequisite() if callable(prerequisite) else prerequisite
        except Exception as error:
            reason = f"prerequisite inspection failed: {type(error).__name__}: {error}"
        reasons = self.gather(reason)
        if any(reasons):
            return self.emit(name, "unsupported", required, reason="; ".join(sorted({r for r in reasons if r})))
        try:
            values = function() or {}
            row = self.emit(name, required=required, **values)
        except Unsupported as error:
            row = self.emit(name, "unsupported", required, reason=str(error))
        except Unverified as error:
            row = self.emit(name, "unverified", required, reason=str(error))
        except Exception as error:
            row = self.emit(name, "fail", required, reason=f"{type(error).__name__}: {error}", error=traceback.format_exc())
        self.t.cuda.empty_cache()
        # Failing collectives are fatal: the process-group timeout bounds any peers.
        statuses = self.gather(row["status"])
        if "fail" in statuses and name.startswith(("nvlink/", "collectives/", "moe/", "stress/")):
            raise RuntimeError(f"distributed workload failed: {name}")
        return row

    def measure(self, function, distributed=False, iterations=None):
        t = self.t
        n = iterations or self.args.iters
        for _ in range(self.args.warmup):
            function()
        t.cuda.synchronize(self.device)
        local, worst_rank = [], []
        for _ in range(self.args.repeats):
            if distributed:
                self.barrier()
            start, end = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n):
                function()
            end.record()
            end.synchronize()
            value = start.elapsed_time(end)/n
            local.append(value)
            if distributed:
                worst_rank.append(self.maximum(value))
        result = dict(trials_ms=local, median_ms=statistics.median(local), worst_ms=max(local))
        if distributed:
            result.update(max_rank_trials_ms=worst_rank, max_rank_median_ms=statistics.median(worst_rank))
        return result

    def kernels(self, function):
        t = self.t
        function()
        t.cuda.synchronize(self.device)
        try:
            with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,
                                                t.profiler.ProfilerActivity.CUDA]) as profile:
                function()
                t.cuda.synchronize(self.device)
            names = sorted({e.name for e in profile.events() if "CUDA" in str(e.device_type)})
            return dict(cuda_kernels=names, kernel_evidence="observed" if names else "unavailable",
                        instruction_coverage="TMA/TMEM/two-SM instructions not independently verified")
        except (RuntimeError, AssertionError) as error:
            return dict(cuda_kernels=[], kernel_evidence="unavailable", profile_error=str(error))

    def perf(self, function, flops=None, moved=None, spec_key=None, distributed=False, **details):
        stats = self.measure(function, distributed)
        elapsed = stats.get("max_rank_median_ms", stats["median_ms"])/1000
        if flops:
            details["tflops"] = flops/elapsed/1e12
        if moved:
            details["gbps"] = moved/elapsed/1e9
        if spec_key and spec_key in self.spec:
            value = details.get("tflops", details.get("gbps"))
            details.update(spec_key=spec_key, spec_peak=self.spec[spec_key], pct_spec=value/self.spec[spec_key]*100)
            if self.args.min_efficiency and value/self.spec[spec_key] < self.args.min_efficiency:
                details.update(status="fail", reason=f"below configured efficiency {self.args.min_efficiency:.1%}")
        return dict(**stats, **details)

    def snapshot(self, label):
        if not self.node_leader:
            return
        for name, command in {"gpu": ["nvidia-smi", "-q", "-x"],
                              "topology": ["nvidia-smi", "topo", "-m"],
                              "nvlink": ["nvidia-smi", "nvlink", "--errorcounters"],
                              "kernel": ["dmesg", "--ctime"]}.items():
            dump(self.root/f"{label}_{self.info['host']}_{name}.json", command_output(command))

    def read_ecc(self):
        data = command_output(["nvidia-smi", "--query-gpu=uuid,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total", "--format=csv,noheader,nounits"])
        for row in csv.reader(data["output"].splitlines()):
            if len(row) == 3 and normalized_uuid(row[0].strip()) == normalized_uuid(self.info["gpu_uuid"]):
                try:
                    return dict(uncorrected=int(row[1]), corrected=int(row[2]))
                except ValueError:
                    pass
        return None

    def close_monitor(self):
        if self.monitor:
            self.monitor.terminate()
            try:
                self.monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.monitor.kill()
                self.monitor.wait()
            self.monitor_log.close()



def te_workspace_kwargs(ctx, function, group_count=None):
    """TE 2.x explicit workspaces and newer internally managed workspaces."""
    params = inspect.signature(function).parameters
    kwargs = {}
    if "workspace" in params:
        kwargs["workspace"] = ctx.t.empty(32*1024**2+1024, dtype=ctx.t.uint8, device=ctx.device)
    if "workspaces" in params:
        import transformer_engine_torch as tex
        kwargs["workspaces"] = [ctx.t.empty(32*1024**2+1024, dtype=ctx.t.uint8, device=ctx.device)
                                for _ in range(tex.get_num_cublas_streams())]
    if group_count is not None and "quantization_params" in params:
        kwargs["quantization_params"] = [None]*group_count
    return kwargs


class Matrix:
    """Native low-precision adapters. No dequantize-to-BF16 fallback is timed."""
    def __init__(self, ctx, kind, a, weight, fp8_format="e4m3", use_te=False):
        t = ctx.t
        self.kind, self.a, self.weight, self.ctx = kind, a, weight, ctx
        self.use_te = use_te
        self.metadata = dict(precision=kind, input_shape=list(a.shape), weight_shape=list(weight.shape))
        if kind == "bf16":
            self.run = lambda: a @ weight.t()
            self.dequantize = lambda: (a, weight)
            self.quantize = lambda: (a, weight)
        elif kind == "fp8" and not use_te:
            if not hasattr(t, "_scaled_mm"):
                raise Unsupported("torch._scaled_mm is unavailable")
            dtype = t.float8_e4m3fn if fp8_format == "e4m3" else t.float8_e5m2
            def quantize(x, target_dtype):
                scale = (x.float().abs().amax()/t.finfo(target_dtype).max).clamp_min(1e-12)
                return (x.float()/scale).clamp(-t.finfo(target_dtype).max, t.finfo(target_dtype).max).to(target_dtype), scale
            # cuBLASLt supports E4M3xE4M3 and mixed E5M2/E4M3, not E5M2xE5M2.
            self.quantize = lambda: (quantize(a, dtype), quantize(weight, t.float8_e4m3fn))
            (self.qa, self.sa), (self.qw, self.sw) = self.quantize()
            def native():
                out = t._scaled_mm(self.qa, self.qw.t(), scale_a=self.sa.reshape(1),
                                   scale_b=self.sw.reshape(1), out_dtype=t.bfloat16, use_fast_accum=False)
                return out[0] if isinstance(out, tuple) else out
            self.run = native
            self.dequantize = lambda: (self.qa.float()*self.sa, self.qw.float()*self.sw)
            self.metadata.update(backend="torch._scaled_mm/cuBLASLt", format=fp8_format+" activation x e4m3 weight", scaling="per tensor")
        else:
            try:
                import transformer_engine_torch as tex
                from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
                if kind == "fp8":
                    from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
                    factory = lambda: Float8CurrentScalingQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, device=ctx.device)
                elif kind == "mxfp8":
                    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
                    factory = lambda: MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
                elif kind == "nvfp4":
                    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
                    factory = lambda: NVFP4Quantizer(rowwise=True, columnwise=True, with_rht=False)
                else:
                    raise ValueError(kind)
            except (ImportError, OSError) as error:
                raise Unsupported(f"Transformer Engine {kind} dependency unavailable: {error}") from error
            self.q_a, self.q_w = factory(), factory()
            self.quantize = lambda: (self.q_a(a), self.q_w(weight))
            self.qa, self.qw = self.quantize()
            expected = {"fp8": "Float8", "mxfp8": "MXFP8", "nvfp4": "NVFP4"}[kind]
            if expected not in type(self.qa).__name__ or expected not in type(self.qw).__name__:
                raise RuntimeError(f"expected {expected} quantized tensors, got {type(self.qa)}")
            workspace = te_workspace_kwargs(ctx, general_gemm)
            self.native_gemm = lambda weight, activation: general_gemm(weight, activation, out_dtype=t.bfloat16, layout="TN", **workspace)[0]
            self.run = lambda: self.native_gemm(self.qw, self.qa)
            self.dequantize = lambda: (self.qa.dequantize(dtype=t.float32), self.qw.dequantize(dtype=t.float32))
            self.metadata.update(backend="TransformerEngine.general_gemm/native quantized inputs",
                                 quantized_type=type(self.qa).__name__,
                                 scaling={"fp8": "per tensor current scaling", "mxfp8": "32-element E8M0 blocks",
                                          "nvfp4": "16-element E4M3 blocks + FP32 global scale"}[kind])

    def dynamic(self, activation):
        """MoE path: quantization of new activations belongs inside the timed step."""
        if self.kind == "bf16":
            return activation @ self.weight.t()
        if self.kind == "fp8" and not self.use_te:
            t = self.ctx.t
            scale = (activation.float().abs().amax()/t.finfo(self.qa.dtype).max).clamp_min(1e-12)
            qa = (activation.float()/scale).to(self.qa.dtype)
            out = t._scaled_mm(qa, self.qw.t(), scale_a=scale.reshape(1), scale_b=self.sw.reshape(1),
                               out_dtype=t.bfloat16, use_fast_accum=False)
            return out[0] if isinstance(out, tuple) else out
        return self.native_gemm(self.qw, self.q_a(activation))


def precision_reason(ctx, kind):
    cc = tuple(ctx.info["capability"])
    if kind == "fp8" and cc < (8, 9):
        return "native FP8 GEMM requires supported SM89/SM90/Blackwell hardware"
    if kind in ("mxfp8", "nvfp4"):
        if cc[0] < 10:
            return f"native {kind} block-scaled GEMM requires Blackwell"
        try:
            if importlib.util.find_spec("transformer_engine") is None:
                return "install a B300-compatible Transformer Engine build before this test"
        except (ImportError, ValueError):
            return "Transformer Engine is unavailable"
    return None


def error_metrics(ctx, actual, expected, limit):
    t = ctx.t
    actual, expected = actual.float(), expected.float()
    if not bool(t.isfinite(actual).all()) or not bool(t.isfinite(expected).all()):
        raise AssertionError("NaN/Inf in numerical check")
    diff = actual - expected
    nrmse = float((diff.square().mean()/(expected.square().mean().clamp_min(1e-20))).sqrt())
    maximum = float(diff.abs().amax())
    if nrmse > limit:
        raise AssertionError(f"NRMSE {nrmse:.6f} exceeds {limit:.6f}; max_abs={maximum:.6f}")
    return dict(nrmse=nrmse, max_abs_error=maximum, limit=limit)


def matrix_check(ctx, matrix):
    # Sample rows/columns but include the full reduction dimension.
    t = ctx.t
    actual = matrix.run()
    indices_m = t.linspace(0, matrix.a.shape[0]-1, min(16, matrix.a.shape[0]), device=ctx.device).long()
    indices_n = t.linspace(0, matrix.weight.shape[0]-1, min(64, matrix.weight.shape[0]), device=ctx.device).long()
    target = matrix.a[indices_m].float() @ matrix.weight[indices_n].float().t()
    got = actual[indices_m][:, indices_n]
    limit = dict(bf16=.015, fp8=ctx.args.fp8_error, mxfp8=ctx.args.mxfp8_error, nvfp4=ctx.args.nvfp4_error)[matrix.kind]
    accuracy = error_metrics(ctx, got, target, limit)
    da, dw = matrix.dequantize()
    quantized_reference = da[indices_m].float() @ dw[indices_n].float().t()
    kernel_accuracy = error_metrics(ctx, got, quantized_reference, .02)
    return dict(accuracy_vs_original=accuracy, accuracy_vs_dequantized=kernel_accuracy,
                verification="sampled output rows/columns, full K FP32 reference")


def test_gemm(ctx, kind):
    t, args = ctx.t, ctx.args
    shapes = [(256, 512, 512)] if args.quick else [(4096, 4096, 4096), (8192, 8192, 8192),
                                                               (128, 8192, 4096), (4096, 14336, 4096)]
    formats = ("e4m3", "e5m2") if kind == "fp8" else ("default",)
    distributions = ("normal",) if args.quick else ("normal", "channel_scaled")
    for m, n, k in shapes:
        for fmt in formats:
            for distribution in distributions:
                def execute(m=m, n=n, k=k, fmt=fmt, distribution=distribution):
                    a = t.randn(m, k, device=ctx.device, dtype=t.bfloat16)
                    w = t.randn(n, k, device=ctx.device, dtype=t.bfloat16)
                    if distribution == "channel_scaled":
                        scale = t.logspace(-1, 1, k, device=ctx.device).to(t.bfloat16)
                        a.mul_(scale)
                        w.div_(scale)
                    matrix = Matrix(ctx, kind, a, w, fmt)
                    check = matrix_check(ctx, matrix)
                    quantization = ctx.measure(matrix.quantize) if kind != "bf16" else None
                    dequantization = ctx.measure(matrix.dequantize) if kind != "bf16" else None
                    kernels = ctx.kernels(matrix.run)
                    peak = f"{kind}_dense_tflops"
                    return ctx.perf(matrix.run, flops=2*m*n*k,
                                    spec_key=peak if m >= 2048 else None, comparable=True,
                                    quantization=quantization, dequantization=dequantization,
                                    **matrix.metadata, **check, **kernels)
                ctx.case(f"{kind}/{fmt}/{distribution}/{m}x{n}x{k}", execute,
                         prerequisite=lambda: precision_reason(ctx, kind))



def test_grouped_gemm(ctx):
    t = ctx.t
    for kind in ("bf16", "fp8", "mxfp8", "nvfp4"):
        def prerequisite(kind=kind):
            if kind == "bf16":
                return None if hasattr(t, "_grouped_mm") else "torch._grouped_mm is unavailable in this PyTorch build"
            if kind == "fp8":
                if not hasattr(t, "_scaled_grouped_mm"):
                    return "torch._scaled_grouped_mm is unavailable in this PyTorch build; grouped GEMM runs natively for bf16 only"
                return precision_reason(ctx, kind)
            if tuple(ctx.info["capability"])[0] < 10:
                return f"native {kind} block-scaled GEMM requires Blackwell"
            return (f"native {kind} grouped GEMM has no PyTorch kernel in this build; "
                    "grouped GEMM runs natively without Transformer Engine for bf16/fp8")
        def execute(kind=kind):
            lengths = (128, 256) if ctx.args.quick else (256, 512, 1024, 2048)
            hidden = output = 256 if ctx.args.quick else 4096
            experts = len(lengths)
            a = t.randn(sum(lengths), hidden, dtype=t.bfloat16, device=ctx.device)
            weights = t.randn(experts, hidden, output, dtype=t.bfloat16, device=ctx.device)
            offs = t.tensor(lengths, dtype=t.int32, device=ctx.device).cumsum(0).to(t.int32)
            reference, start = [], 0
            for m, weight in zip(lengths, weights):
                reference.append(a[start:start+m].float() @ weight.float())
                start += m
            if kind == "bf16":
                fn = lambda: t._grouped_mm(a, weights, offs)
            else:
                def quant(x):
                    scale = (x.detach().abs().amax().float().clamp_min(1e-12) / 448.0).reshape(1)
                    return (x.float() / scale).clamp(-448, 448).to(t.float8_e4m3fn), scale
                qa, scale_a = quant(a)
                qw, scale_b = quant(weights.transpose(-2, -1).contiguous())  # [G,N,K] row-major
                grouped_b = qw.transpose(-2, -1)  # [G,K,N] column-major, as grouped kernels expect
                def fn():
                    return t._scaled_grouped_mm(qa, grouped_b, scale_a=scale_a, scale_b=scale_b,
                                                offs=offs, out_dtype=t.bfloat16)
            try:
                out = fn()
            except RuntimeError as error:  # kernel not implemented for this arch/format combination
                raise Unsupported(f"torch grouped {kind} kernel rejected this configuration: {error}") from error
            if isinstance(out, tuple):
                out = out[0]
            limit = dict(bf16=.02, fp8=ctx.args.fp8_error, mxfp8=ctx.args.mxfp8_error, nvfp4=ctx.args.nvfp4_error)[kind]
            checks = []
            for m, ref, chunk in zip(lengths, reference, out.split(list(lengths))):
                rows = min(16, m)
                checks.append(error_metrics(ctx, chunk[:rows, :64], ref[:rows, :64], limit))
            def loop():
                results, start = [], 0
                for m, weight in zip(lengths, weights):
                    results.append(a[start:start+m] @ weight)
                    start += m
                return results
            baseline = ctx.measure(loop)
            result = ctx.perf(fn, flops=2*sum(lengths)*hidden*output, comparable=True,
                              backend="torch._grouped_mm" if kind == "bf16" else "torch._scaled_grouped_mm",
                              precision=kind, expert_token_counts=list(lengths), accuracy=checks,
                              loop_baseline=baseline, **ctx.kernels(fn))
            result["speedup_vs_native_loop"] = baseline["median_ms"]/result["median_ms"]
            return result
        ctx.case(f"grouped_gemm/{kind}/unbalanced_experts", execute, prerequisite)


def test_sparse(ctx):
    t = ctx.t
    size = 512 if ctx.args.quick else 8192
    def execute():
        from torch.sparse import to_sparse_semi_structured
        a = t.randn(size, size, device=ctx.device, dtype=t.bfloat16)
        w = t.randn(size, size//4, 4, device=ctx.device, dtype=t.bfloat16)
        selected = w.abs().topk(2, dim=-1).indices
        mask = t.zeros_like(w, dtype=t.bool).scatter_(-1, selected, True)
        w = (w*mask).reshape(size, size)
        if not bool((w.reshape(size, size//4, 4).ne(0).sum(-1) <= 2).all()):
            raise AssertionError("invalid 2:4 mask")
        sparse = to_sparse_semi_structured(w)
        fn = lambda: t.mm(sparse, a.t()).t()
        dense = lambda: a @ w.t()
        check = error_metrics(ctx, fn()[:32, :64], (a[:32].float() @ w[:64].float().t()), .02)
        base = ctx.measure(dense)
        result = ctx.perf(fn, flops=2*size**3, spec_key="bf16_sparse_effective_tflops",
                          backend=type(sparse).__name__, accuracy=check, dense_baseline=base,
                          flops_convention="dense-equivalent effective FLOPs; executed nonzeros about half",
                          comparable=True, **ctx.kernels(fn))
        result["speedup_vs_same_pruned_dense"] = base["median_ms"]/result["median_ms"]
        return result
    ctx.case("sparse/bf16_2of4", execute)


def test_memory(ctx):
    t, args = ctx.t, ctx.args
    def patterns():
        free, total = t.cuda.mem_get_info(ctx.device)
        n = memory_budget(free, int(args.reserve_gib*1024**3), args.memory_fraction, args.quick)//4*4
        if n < 32*1024**2:
            raise Unsupported("insufficient free memory after reserve")
        buffer = t.empty(n, dtype=t.uint8, device=ctx.device)
        chunk = min(n, 16*1024**2)
        complete = []
        for pattern in (0x00, 0xFF, 0x55, 0xAA):
            buffer.fill_(pattern)
            for offset in range(0, n, chunk):
                if not bool((buffer[offset:offset+chunk] == pattern).all()):
                    raise AssertionError(f"HBM mismatch pattern={pattern} offset={offset}")
            complete.append(pattern)
        words = buffer.view(t.int32)
        step = chunk//4
        for offset in range(0, words.numel(), step):
            ids = t.arange(offset, min(offset+step, words.numel()), device=ctx.device, dtype=t.int64)
            words[offset:offset+ids.numel()].copy_((ids ^ 0x5A5A5A5A).int())
        for offset in range(0, words.numel(), step):
            ids = t.arange(offset, min(offset+step, words.numel()), device=ctx.device, dtype=t.int64)
            if not bool((words[offset:offset+ids.numel()] == (ids ^ 0x5A5A5A5A).int()).all()):
                raise AssertionError(f"HBM address pattern mismatch offset={offset}")
        ids = t.randint(words.numel(), (min(1024**2, words.numel()),), device=ctx.device)
        if not bool((words[ids] == (ids ^ 0x5A5A5A5A).int()).all()):
            raise AssertionError("random HBM gather mismatch")
        stats = ctx.perf(lambda: words[ids], moved=ids.numel()*4,
                         metric_note="logical 4-byte random read only; excludes index traffic and is not physical HBM utilization")
        return dict(tested_bytes=n, total_memory_bytes=total, fraction_of_physical=n/total,
                    fraction_of_free=n/free, byte_patterns=complete, address_pattern="word index XOR 0x5a5a5a5a",
                    full_buffer_verified=True, random_read=stats)
    ctx.case("memory/full_patterns_and_random", patterns)
    def copy():
        n = 64*1024**2 if args.quick else 1024**3
        a = t.full((n,), 37, device=ctx.device, dtype=t.uint8)
        b = t.empty_like(a)
        b.copy_(a)
        if not bool((b == 37).all()):
            raise AssertionError("HBM copy corrupted")
        return ctx.perf(lambda: b.copy_(a), moved=2*n, spec_key="hbm_gbps", comparable=True,
                        buffer_bytes=n, convention="source read + destination write")
    ctx.case("memory/hbm_copy", copy)


def bind_numa(ctx):
    # Resolve logical CUDA indices through UUID, not physical nvidia-smi indices.
    data = command_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"])
    for uuid, bus in csv.reader(data["output"].splitlines()):
        if normalized_uuid(uuid.strip()) == normalized_uuid(ctx.info["gpu_uuid"]):
            path = Path("/sys/bus/pci/devices")/bus.strip().lower()[-12:]
            cpus = set()
            for value in (path/"local_cpulist").read_text().strip().split(","):
                ends = list(map(int, value.split("-")))
                cpus.update(range(ends[0], ends[-1]+1))
            cpus &= os.sched_getaffinity(0)
            if not cpus:
                raise Unsupported("GPU-local NUMA CPUs are unavailable")
            os.sched_setaffinity(0, cpus)
            return dict(cpu_affinity=sorted(cpus), pci_bus=bus.strip(),
                        link_speed=(path/"current_link_speed").read_text().strip(),
                        link_width=(path/"current_link_width").read_text().strip())
    raise Unsupported("cannot map CUDA UUID to physical PCIe device")


def test_pcie(ctx):
    t = ctx.t
    # Every rank enters every round; serial ranks run alone per whole job.
    affinity = {}
    def prepare():
        nonlocal affinity
        try:
            affinity = bind_numa(ctx)
        except (OSError, Unsupported, ValueError) as error:
            return str(error)
        return None
    reason = prepare()
    n = 32*1024**2 if ctx.args.quick else 512*1024**2
    def transfer(active, direction):
        host = t.full((n,), 19 if direction == "H2D" else 0, dtype=t.uint8, pin_memory=True)
        gpu = t.full((n,), 0 if direction == "H2D" else 19, dtype=t.uint8, device=ctx.device)
        fn = (lambda: gpu.copy_(host, non_blocking=True)) if direction == "H2D" else (lambda: host.copy_(gpu, non_blocking=True))
        operation = fn if active else lambda: None
        operation()
        t.cuda.synchronize()
        if active and not bool(((gpu if direction == "H2D" else host) == 19).all()):
            raise AssertionError("PCIe transfer data mismatch")
        result = ctx.perf(operation, moved=n if active else None, distributed=True,
                          spec_key="pcie_one_way_gbps" if active else None,
                          active=active, **affinity)
        return result
    for direction in ("H2D", "D2H"):
        for rank in range(ctx.world):
            ctx.case(f"pcie/serial/{direction}/rank{rank}", lambda rank=rank: transfer(ctx.rank == rank, direction), reason)
        ctx.case(f"pcie/concurrent/{direction}", lambda: transfer(True, direction), reason)


def require_distributed(ctx):
    return "launch with torchrun and at least two GPUs" if ctx.world < 2 else None



def nvlink_table_path(text, source, destination):
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    lines = [line.split() for line in text.splitlines() if line.split()]
    header = next((line for line in lines if source in line and destination in line
                   and "X" not in line and line[0].startswith("GPU")), None)
    row = next((line for line in lines if line[0] == source and "X" in line), None)
    position = 1+header.index(destination) if header else None
    return bool(row and position is not None and position < len(row) and row[position].startswith("NV"))


def topology_is_nvlink(ctx, first, second):
    info = {r["rank"]: r for r in ctx.infos}
    if info[first]["host"] != info[second]["host"]:
        return False
    # Each host reads its own mapping; negotiate failures before pair operations.
    if ctx.rank not in (first, second):
        return True
    query = command_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
    physical = {normalized_uuid(uuid.strip()): index.strip() for index, uuid in csv.reader(query["output"].splitlines())}
    a = "GPU" + physical[normalized_uuid(info[first]["gpu_uuid"])]
    b = "GPU" + physical[normalized_uuid(info[second]["gpu_uuid"])]
    text = command_output(["nvidia-smi", "topo", "-m"])["output"]
    return nvlink_table_path(text, a, b)


def test_nvlink(ctx):
    t = ctx.t
    schedule = []
    all_pairs = []
    rounds_by_node = [round_robin(ranks) for ranks in ctx.nodes.values()]
    for rounds in rounds_by_node:
        all_pairs.extend(pair for wave in rounds for pair in wave)
    for i in range(max((len(x) for x in rounds_by_node), default=0)):
        schedule.append([pair for rounds in rounds_by_node if i < len(rounds) for pair in rounds[i]])
    if not all_pairs:
        ctx.case("nvlink/local_pairs", lambda: None, "requires at least two visible GPUs on the same node")
        return
    if ctx.args.quick:
        all_pairs, schedule = all_pairs[:2], schedule[:1]
    n = 4*1024**2 if ctx.args.quick else 256*1024**2
    baselines = {}
    def reason(pairs):
        if ctx.world < 2:
            return require_distributed(ctx)
        return None if all(topology_is_nvlink(ctx, a, b) for a, b in pairs) else "GPU pair has no confirmed NVLink path in nvidia-smi topology"
    def transfer(pairs, concurrent):
        partner = next((b if ctx.rank == a else a for a, b in pairs if ctx.rank in (a, b)), None)
        send = t.full((n,), ctx.rank % 251+1, dtype=t.uint8, device=ctx.device)
        recv = t.zeros_like(send)
        def fn():
            if partner is not None:
                req = ctx.dist.batch_isend_irecv([ctx.dist.P2POp(ctx.dist.isend, send, partner),
                                                 ctx.dist.P2POp(ctx.dist.irecv, recv, partner)])
                for item in req:
                    item.wait()
        ctx.barrier()
        fn()
        if partner is not None and not bool((recv == partner % 251+1).all()):
            raise AssertionError("peer transfer corrupted")
        result = ctx.perf(fn, distributed=True, moved=2*n if partner is not None else None,
                          spec_key="nvlink_bidirectional_gbps" if partner is not None else None,
                          active=partner is not None, peer=partner, pairings=pairs,
                          convention="per participating GPU TX + RX; NCCL sendrecv, not isolated lane utilization")
        if partner is not None:
            pair = tuple(sorted((ctx.rank, partner)))
            if not concurrent:
                baselines[pair] = result["gbps"]
            elif pair in baselines:
                ratio = result["gbps"]/baselines[pair]
                result.update(concurrent_to_pair_baseline=ratio)
                if ratio < ctx.args.min_concurrent_ratio:
                    result.update(status="unverified", reason=f"concurrent/pair ratio {ratio:.3f} requires retest")
        return result
    for a, b in all_pairs:
        ctx.case(f"nvlink/pair/{a}-{b}", lambda a=a, b=b: transfer([(a, b)], False), lambda a=a, b=b: reason([(a, b)]))
    for i, pairs in enumerate(schedule):
        ctx.case(f"nvlink/concurrent/wave{i}", lambda pairs=pairs: transfer(pairs, True), lambda pairs=pairs: reason(pairs))


def test_collectives(ctx):
    t, d = ctx.t, ctx.dist
    sizes = (8192, 1024**2) if ctx.args.quick else (8192, 128*1024, 1024**2, 16*1024**2, 128*1024**2)
    ops = ("all_reduce",) if ctx.args.nvls_only else ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
    for n in sizes:
        for op in ops:
            def execute(n=n, op=op):
                per = max(1, n//4//ctx.world)
                shard = t.full((per,), ctx.rank+1, device=ctx.device, dtype=t.float32)
                full = t.arange(1, ctx.world+1, device=ctx.device).repeat_interleave(per).float()
                result = t.empty_like(full if op in ("all_gather", "all_to_all") else shard)
                if op == "all_reduce":
                    # MAX is idempotent and avoids repeated SUM overflow.
                    fn = lambda: d.all_reduce(shard, op=d.ReduceOp.MAX)
                    check = lambda: bool((shard == ctx.world).all())
                    moved = shard.numel()*4*2*(ctx.world-1)/ctx.world
                elif op == "all_gather":
                    fn = lambda: d.all_gather_into_tensor(result, shard)
                    check = lambda: bool((result == full).all())
                    moved = result.numel()*4*(ctx.world-1)/ctx.world
                elif op == "reduce_scatter":
                    fn = lambda: d.reduce_scatter_tensor(result, full)
                    check = lambda: bool((result == ctx.world*(ctx.rank+1)).all())
                    moved = full.numel()*4*(ctx.world-1)/ctx.world
                else:
                    send = (ctx.rank*ctx.world+t.arange(ctx.world, device=ctx.device)).repeat_interleave(per).float()
                    fn = lambda: d.all_to_all_single(result, send)
                    expected = (t.arange(ctx.world, device=ctx.device)*ctx.world+ctx.rank).repeat_interleave(per)
                    check = lambda: bool((result == expected).all())
                    moved = send.numel()*4
                ctx.barrier()
                fn()
                if not check():
                    raise AssertionError(f"{op} reference mismatch")
                result_stats = ctx.perf(fn, moved=moved, distributed=True,
                                        algorithm_requested=os.environ.get("NCCL_ALGO", "auto"),
                                        protocol_requested=os.environ.get("NCCL_PROTO", "auto"),
                                        algorithm_evidence="NCCL_ALGO restricted to NVLS; successful native all-reduce" if ctx.args.nvls_only else "auto selection; inspect NCCL logs",
                                        nccl_logs=os.environ.get("NCCL_DEBUG_FILE", "stderr"),
                                        convention="algorithm-adjusted busbw; not physical link utilization")
                if not check():
                    raise AssertionError(f"{op} post-timing mismatch")
                return result_stats
            ctx.case(f"collectives/{op}/{n}", execute, lambda: require_distributed(ctx))


def attention_reference(ctx, q, k, v, causal=False):
    """Independent FP32 attention for a few heads/query positions, full KV length."""
    t = ctx.t
    # q/k/v are B,H,S,D; verify the first query positions in every batch.
    heads = min(2, q.shape[1])
    queries = min(8, q.shape[2])
    qsmall = q[:, :heads, :queries].float()
    mapping = t.arange(heads, device=ctx.device)//(q.shape[1]//k.shape[1])
    kk, vv = k[:, mapping].float(), v[:, mapping].float()
    scores = qsmall @ kk.transpose(-1, -2)/math.sqrt(q.shape[-1])
    if causal:
        mask = t.arange(k.shape[2], device=ctx.device)[None, :] > t.arange(queries, device=ctx.device)[:, None]
        scores.masked_fill_(mask, -float("inf"))
    return scores.softmax(-1) @ vv


def test_attention(ctx):
    t = ctx.t
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    contexts = (128,) if ctx.args.quick else (2048, 8192, 32768)
    configs = [("prefill", 1, seq, seq, heads) for seq in contexts for heads in (32, 8, 1)]
    configs += [("decode", batch, 1, seq, heads)
                for batch in ((1,) if ctx.args.quick else (1, 4))
                for seq in ((512,) if ctx.args.quick else (8192, 32768, 131072)) for heads in (8, 1)]
    for mode, batch, qlen, kvlen, heads in configs:
        def execute(mode=mode, batch=batch, qlen=qlen, kvlen=kvlen, heads=heads):
            q = t.randn(batch, 32, qlen, 128, device=ctx.device, dtype=t.bfloat16)
            k = t.randn(batch, heads, kvlen, 128, device=ctx.device, dtype=t.bfloat16)
            v = t.randn_like(k)
            def fn():
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=mode == "prefill", enable_gqa=heads != 32)
            out = fn()
            ref = attention_reference(ctx, q, k, v, causal=mode == "prefill")
            check = error_metrics(ctx, out[:, :ref.shape[1], :ref.shape[2]], ref, .025)
            return ctx.perf(fn, comparable=True, accuracy=check, backend="torch SDPA forced FLASH_ATTENTION",
                            q_heads=32, kv_heads=heads, batch=batch, q_length=qlen, kv_length=kvlen,
                            tokens_per_call=batch*qlen, **ctx.kernels(fn))
        ctx.case(f"attention/{mode}/B{batch}_Q{qlen}_KV{kvlen}_Hkv{heads}", execute)


def test_paged_attention(ctx):
    t = ctx.t
    def dependency():
        if importlib.util.find_spec("flashinfer") is None:
            return "FlashInfer is not installed (no package will be installed automatically)"
        return None
    for precision in ("bf16", "fp8", "fp8_qkv"):
        for batch in ((1,) if ctx.args.quick else (1, 8)):
            def execute(precision=precision, batch=batch):
                from flashinfer import BatchDecodeWithPagedKVCacheWrapper
                length, page, hq, hkv, dim = (251 if ctx.args.quick else 16379), 16, 32, 8, 128
                pages = math.ceil(length/page)
                # Reverse the physical page order so this exercises indirection.
                indices = t.arange(batch*pages-1, -1, -1, device=ctx.device, dtype=t.int32)
                indptr = t.arange(0, (batch+1)*pages, pages, device=ctx.device, dtype=t.int32)
                last = t.full((batch,), length % page or page, device=ctx.device, dtype=t.int32)
                cache = t.randn(batch*pages, 2, page, hkv, dim, device=ctx.device, dtype=t.bfloat16)
                query = t.randn(batch, hq, dim, device=ctx.device, dtype=t.bfloat16)
                dtype = t.bfloat16 if precision == "bf16" else t.float8_e4m3fn
                scale = 1.0 if precision == "bf16" else float(cache.float().abs().amax()/t.finfo(dtype).max)
                packed = cache if precision == "bf16" else (cache.float()/scale).to(dtype)
                wrapper = BatchDecodeWithPagedKVCacheWrapper(t.empty(128*1024**2, device=ctx.device, dtype=t.uint8), "NHD", use_tensor_cores=True)
                qscale = float(query.float().abs().amax()/t.finfo(dtype).max) if precision == "fp8_qkv" else 1.0
                quantized_query = (query.float()/qscale).to(dtype) if precision == "fp8_qkv" else query
                extra_plan = dict(o_data_type=t.bfloat16) if precision == "fp8_qkv" else {}
                wrapper.plan(indptr, indices, last, hq, hkv, dim, page, q_data_type=quantized_query.dtype,
                             kv_data_type=dtype, pos_encoding_mode="NONE", **extra_plan)
                fn = lambda: wrapper.run(quantized_query, packed, q_scale=qscale, k_scale=scale, v_scale=scale)
                out = fn()
                # Reconstruct logical pages for the reference, including a partial last page.
                refs = []
                for b in range(batch):
                    logical = cache[indices[b*pages:(b+1)*pages].long()].reshape(pages, 2, page, hkv, dim)
                    k = logical[:, 0].reshape(-1, hkv, dim)[:length].transpose(0, 1)[None]
                    v = logical[:, 1].reshape(-1, hkv, dim)[:length].transpose(0, 1)[None]
                    refs.append(attention_reference(ctx, query[b:b+1, :, None], k, v)[:, :, 0])
                ref = t.cat(refs)
                check = error_metrics(ctx, out[:, :ref.shape[1]], ref, .03 if precision == "bf16" else .09)
                result = ctx.perf(fn, comparable=True, accuracy=check, backend="FlashInfer paged decode",
                                  cache_precision=precision, q_precision="fp8" if precision == "fp8_qkv" else "bf16",
                                  batch=batch, context_length=length,
                                  page_size=page, physical_pages_reordered=True, **ctx.kernels(fn))
                result["tokens_per_second"] = batch/result["median_ms"]*1000
                return result
            def prerequisite(precision=precision):
                if precision == "fp8_qkv" and tuple(ctx.info["capability"]) < (9, 0):
                    return "native FP8 QKV attention requires Hopper/Blackwell and a compatible FlashInfer backend"
                return dependency()
            ctx.case(f"paged_attention/{precision}/B{batch}", execute, prerequisite)



def test_mla(ctx):
    t = ctx.t
    def prerequisite():
        return "FlashInfer MLA requires the flashinfer package" if importlib.util.find_spec("flashinfer") is None else None
    for batch in ((1,) if ctx.args.quick else (1, 8)):
        def execute(batch=batch):
            try:
                from flashinfer.mla import BatchMLAPagedAttentionWrapper
            except ImportError as error:
                raise Unsupported(f"FlashInfer build has no MLA wrapper: {error}") from error
            length, page, heads, latent, rope = (251 if ctx.args.quick else 16379), 16, 32, 512, 64
            pages = math.ceil(length/page)
            ckv = t.randn(batch*pages, page, latent, device=ctx.device, dtype=t.bfloat16)
            kpe = t.randn(batch*pages, page, rope, device=ctx.device, dtype=t.bfloat16)
            qn = t.randn(batch, heads, latent, device=ctx.device, dtype=t.bfloat16)
            qp = t.randn(batch, heads, rope, device=ctx.device, dtype=t.bfloat16)
            qo = t.arange(batch+1, device=ctx.device, dtype=t.int32)
            kv = t.arange(0, (batch+1)*pages, pages, device=ctx.device, dtype=t.int32)
            indices = t.arange(batch*pages-1, -1, -1, device=ctx.device, dtype=t.int32)
            lengths = t.full((batch,), length, device=ctx.device, dtype=t.int32)
            scale = 1/math.sqrt(latent+rope)
            wrapper = BatchMLAPagedAttentionWrapper(t.empty(128*1024**2, device=ctx.device, dtype=t.uint8), backend="auto")
            wrapper.plan(qo, kv, indices, lengths, heads, latent, rope, page, False, scale, t.bfloat16, t.bfloat16)
            fn = lambda: wrapper.run(qn, qp, ckv, kpe)
            actual = fn()
            references = []
            for b in range(batch):
                order = indices[b*pages:(b+1)*pages].long()
                c = ckv[order].reshape(-1, latent)[:length].float()
                k = kpe[order].reshape(-1, rope)[:length].float()
                score = (qn[b, :2].float() @ c.t() + qp[b, :2].float() @ k.t())*scale
                references.append(score.softmax(-1) @ c)
            accuracy = error_metrics(ctx, actual[:, :2], t.stack(references), .03)
            result = ctx.perf(fn, accuracy=accuracy, comparable=True, backend="FlashInfer BatchMLAPagedAttentionWrapper",
                              batch=batch, context_length=length, latent_dim=latent, rope_dim=rope,
                              physical_pages_reordered=True, **ctx.kernels(fn))
            result["tokens_per_second"] = batch/result["median_ms"]*1000
            return result
        ctx.case(f"mla/paged_decode/B{batch}", execute, prerequisite)


def expert_weight(ctx, destination, expert, hidden, output):
    t = ctx.t
    generator = t.Generator(device=ctx.device).manual_seed(ctx.args.seed+10000+destination*97+expert)
    return t.randn(output, hidden, device=ctx.device, dtype=t.bfloat16, generator=generator)/math.sqrt(hidden)


def test_moe(ctx):
    t, d = ctx.t, ctx.dist
    experts, tokens, hidden, output = (2, 64, 256, 256) if ctx.args.quick else (4, 512, 4096, 4096)
    for kind in ctx.args.moe_precisions:
        def prerequisite(kind=kind):
            return require_distributed(ctx) or precision_reason(ctx, kind)
        def execute(kind=kind):
            send = t.randn(ctx.world, experts, tokens, hidden, device=ctx.device, dtype=t.bfloat16)
            recv = t.empty_like(send)
            expert_out = t.empty(ctx.world, experts, tokens, output, device=ctx.device, dtype=t.bfloat16)
            returned = t.empty_like(expert_out)
            matrices = []
            for e in range(experts):
                a = t.zeros(ctx.world*tokens, hidden, device=ctx.device, dtype=t.bfloat16)
                matrices.append(Matrix(ctx, kind, a, expert_weight(ctx, ctx.rank, e, hidden, output)))
            def fn():
                d.all_to_all_single(recv, send)
                for e, matrix in enumerate(matrices):
                    activation = recv[:, e].reshape(-1, hidden).contiguous()
                    expert_out[:, e].copy_(matrix.dynamic(activation).reshape(ctx.world, tokens, output))
                d.all_to_all_single(returned, expert_out)
                return returned
            ctx.barrier()
            fn()
            errors = []
            limit = dict(bf16=.02, fp8=ctx.args.fp8_error, mxfp8=ctx.args.mxfp8_error, nvfp4=ctx.args.nvfp4_error)[kind]
            for destination in range(ctx.world):
                for e in range(experts):
                    weight = expert_weight(ctx, destination, e, hidden, output)
                    ref = send[destination, e, :8].float() @ weight[:64].float().t()
                    errors.append(error_metrics(ctx, returned[destination, e, :8, :64], ref, limit))
            result = ctx.perf(fn, distributed=True, backend=matrices[0].metadata.get("backend", "torch.mm"),
                              precision=kind, experts_per_rank=experts, tokens_per_expert_source=tokens,
                              routing="balanced synthetic top-1; distinct rank/expert weights; not a learned router",
                              includes="dispatch + activation quantization + per-expert native GEMMs + combine",
                              expert_execution="native GEMMs in a loop, not a fused grouped low-precision kernel",
                              max_reference_nrmse=max(x["nrmse"] for x in errors))
            result["tokens_per_second_per_rank"] = ctx.world*experts*tokens/result["max_rank_median_ms"]*1000
            return result
        ctx.case(f"moe/{kind}/dispatch_compute_combine", execute, prerequisite)


def test_stress(ctx):
    t, d, args = ctx.t, ctx.dist, ctx.args
    reasons = {kind: precision_reason(ctx, kind) or require_distributed(ctx) for kind in args.stress_precisions}
    all_reasons = ctx.gather(reasons)
    runnable = [kind for kind in args.stress_precisions if not any(r[kind] for r in all_reasons)]
    for kind in args.stress_precisions:
        def execute(kind=kind):
            size = 512 if args.quick else 4096
            a = t.randn(size, size, device=ctx.device, dtype=t.bfloat16)
            w = t.randn_like(a)
            matrix = Matrix(ctx, kind, a, w)
            check = matrix_check(ctx, matrix)
            expected = matrix.run().clone()
            free, _ = t.cuda.mem_get_info(ctx.device)
            amount = memory_budget(free, int(args.reserve_gib*1024**3), args.stress_memory_fraction, args.quick)
            n = amount//2
            if n < 16*1024**2:
                raise Unsupported("insufficient free memory for mixed stress ballast")
            source = t.full((n,), 71, device=ctx.device, dtype=t.uint8)
            target = t.zeros_like(source)
            message = t.empty((2*1024**2 if args.quick else 32*1024**2,), device=ctx.device, dtype=t.float32)
            cs, ns, ms = t.cuda.Stream(), t.cuda.Stream(), t.cuda.Stream()
            iteration = 0
            def block(mode, count=10):
                nonlocal iteration
                ctx.barrier()
                iteration += 1
                begin = time.perf_counter()
                measured = {}
                tensors = {}
                streams = dict(compute=cs, comm=ns, memory=ms)
                modes = ("compute", "comm", "memory") if mode == "mixed" else (mode,)
                for item in modes:
                    start, end = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
                    with t.cuda.stream(streams[item]):
                        if item == "comm":
                            message.fill_(ctx.rank+iteration)
                        start.record()
                        for _ in range(count):
                            if item == "compute":
                                tensors["compute"] = matrix.run()
                            elif item == "comm":
                                d.all_reduce(message, op=d.ReduceOp.MAX)
                            else:
                                target.copy_(source)
                        end.record()
                    measured[item] = (start, end)
                t.cuda.synchronize()
                stats = {item+"_ms": pair[0].elapsed_time(pair[1])/count for item, pair in measured.items()}
                stats["wall_seconds"] = ctx.maximum(time.perf_counter()-begin)
                if "compute" in modes:
                    # Exact repeatability is not assumed; use a small tolerance against a validated result.
                    error_metrics(ctx, tensors["compute"], expected, .005)
                if "comm" in modes and not bool((message == ctx.world-1+iteration).all()):
                    raise AssertionError("stress collective result corruption")
                if "memory" in modes:
                    # Check rotating chunks during soak; full-memory sweeps are a separate suite.
                    offset = (iteration*1024**2) % max(1, n-1024**2)
                    if not bool((target[offset:offset+1024**2] == 71).all()):
                        raise AssertionError("stress ballast corruption")
                return stats
            baselines = {}
            for mode in ("compute", "comm", "memory", "mixed"):
                block(mode, 2)
                trials = [block(mode) for _ in range(args.repeats)]
                baselines[mode] = {key: statistics.median(x[key] for x in trials) for key in trials[0]}
            duration = args.stress_seconds/max(1, len(runnable))
            deadline = time.monotonic()+duration
            history = []
            next_log = time.monotonic()
            stop = t.zeros((), dtype=t.int32, device=ctx.device)
            started = time.monotonic()
            while True:
                if ctx.rank == 0:
                    stop.fill_(time.monotonic() >= deadline)
                d.broadcast(stop, 0)
                if stop.item():
                    break
                stats = block("mixed")
                history.append(stats)
                if time.monotonic() >= next_log:
                    ctx.emit(f"stress_sample/{kind}", required=False, elapsed_seconds=time.monotonic()-started, **stats)
                    next_log = time.monotonic()+1
            if not history:
                raise AssertionError("stress run completed no mixed blocks")
            degradations = {mode: (statistics.median(r[mode+"_ms"] for r in history)/baselines[mode][mode+"_ms"]-1)*100
                            for mode in ("compute", "comm", "memory")}
            first = statistics.median(x["compute_ms"] for x in history[:max(1, len(history)//10)])
            last = statistics.median(x["compute_ms"] for x in history[-max(1, len(history)//10):])
            result = dict(precision=kind, target_seconds=duration, actual_seconds=time.monotonic()-started,
                          blocks=len(history), ballast_bytes=2*n, baselines=baselines,
                          concurrent_degradation_pct=degradations, compute_drift_pct=(last/first-1)*100,
                          accuracy=check)
            if any(value > args.max_degradation_pct for value in degradations.values()):
                result.update(status="unverified", reason="mixed/isolated interference exceeds diagnostic threshold; investigate shared resources")
            return result
        ctx.case(f"stress/{kind}/mixed_soak", execute, reasons[kind])


def test_features(ctx):
    def inventory():
        raw = command_output(["nvidia-smi", "-q", "-x"])
        if raw["returncode"]:
            raise Unsupported(raw["output"])
        tree = ET.fromstring(raw["output"])
        info = {}
        for gpu in tree.findall("gpu"):
            if normalized_uuid(gpu.findtext("uuid", "")) == normalized_uuid(ctx.info["gpu_uuid"]):
                for section in ("mig_mode", "gpu_virtualization_mode", "confidential_compute", "gpu_recovery_action",
                                "retired_pages", "row_remapper", "ecc_errors"):
                    node = gpu.find(section)
                    info[section] = ET.tostring(node, encoding="unicode") if node is not None else "not exposed"
        if not info:
            raise Unverified("selected GPU absent from physical inventory (MIG UUID may identify an instance)")
        return dict(status="observed", features=info,
                    functional_coverage="inventory only; tenant isolation, reset containment and confidential attestation not exercised")
    ctx.case("features/mig_virtualization_confidential_ras", inventory, required=ctx.args.require_isolation_tests)
    ctx.emit("features/instruction_audit", "unverified", required=False,
             reason="CUDA kernel names are recorded; instruction-level TMA/TMEM/two-SM proof requires Nsight/SASS inspection")


def health_checks(ctx):
    def ecc():
        after = ctx.read_ecc()
        if ctx.health_before is None or after is None:
            raise Unverified("selected GPU ECC counters unavailable")
        delta = {key: after[key]-ctx.health_before[key] for key in after}
        if any(value < 0 for value in delta.values()):
            raise Unverified("ECC counters reset during test; cannot compare")
        if delta["uncorrected"]:
            raise AssertionError(f"new uncorrected ECC: {delta}")
        result = dict(before=ctx.health_before, after=after, delta=delta)
        if delta["corrected"] or ctx.health_before["uncorrected"]:
            result.update(status="unverified", reason="corrected ECC increase or pre-existing uncorrected ECC requires disposition")
        return result
    ctx.case("health/ecc", ecc)
    def ras():
        raw = command_output(["nvidia-smi", "-q", "-x"])
        if raw["returncode"]:
            raise Unverified("GPU RAS inventory unavailable")
        gpu = next((x for x in ET.fromstring(raw["output"]).findall("gpu")
                    if normalized_uuid(x.findtext("uuid", "")) == normalized_uuid(ctx.info["gpu_uuid"])), None)
        if gpu is None:
            raise Unverified("selected GPU UUID not found in RAS XML")
        values = {}
        for section in ("retired_pages", "row_remapper"):
            node = gpu.find(section)
            if node is not None:
                for item in node.iter():
                    if len(item) == 0 and item.text:
                        values[f"{section}/{item.tag}"] = item.text.strip()
        action = gpu.findtext("gpu_recovery_action", "Not exposed").strip()
        if action.lower() not in ("none", "n/a", "not exposed"):
            raise AssertionError(f"GPU recovery action required: {action}")
        failures = {key: value for key, value in values.items()
                    if any(x in key.lower() for x in ("pending", "failure"))
                    and value.lower() not in ("no", "false", "0", "n/a", "not supported")}
        if failures:
            raise Unverified(f"pending page retirement/remapping or remap failure: {failures}")
        return dict(ras_values=values, recovery_action=action,
                    note="historical retired/remapped page counts preserved for supplier disposition")
    ctx.case("health/ras", ras)
    def kernel():
        before = json.loads((ctx.root/f"before_{ctx.info['host']}_kernel.json").read_text())
        after = json.loads((ctx.root/f"after_{ctx.info['host']}_kernel.json").read_text())
        if before["returncode"] or after["returncode"]:
            raise Unverified("kernel log inaccessible; Xid/SXid cannot be verified")
        baseline = set(before["output"].splitlines())
        extra = [line for line in after["output"].splitlines()
                 if line not in baseline and re.search(r"NVRM: Xid|SXid", line)]
        if extra:
            raise AssertionError(f"new host-scope Xid/SXid requires investigation: {extra[-5:]}")
        return dict(scope="entire host kernel log", new_errors=[])
    ctx.case("health/xid", kernel)
    def telemetry():
        path = ctx.root/f"telemetry_{ctx.info['host']}.csv"
        if not path.exists():
            raise Unverified("monitor output unavailable")
        records = []
        for row in csv.DictReader(path.open(), skipinitialspace=True):
            try:
                row = {str(k).strip(): str(v).strip() for k, v in row.items()}
                if normalized_uuid(row["uuid"]) != normalized_uuid(ctx.info["gpu_uuid"]):
                    continue
                temperature = float(next(v for k, v in row.items() if k.startswith("temperature.gpu")))
                watts = float(next(v.split()[0] for k, v in row.items() if k.startswith("power.draw")))
                throttle = int(next(v for k, v in row.items() if k.startswith("clocks_event_reasons.active")), 0)
                records.append((temperature, watts, throttle))
            except (ValueError, KeyError, StopIteration):
                continue
        if not records:
            raise Unverified("no usable monitoring samples for selected GPU UUID")
        max_temp = max(x[0] for x in records)
        thermal = sum(bool(x[2] & 0x60) for x in records)
        result = dict(samples=len(records), max_temperature_c=max_temp,
                      max_power_w=max(x[1] for x in records), thermal_throttle_samples=thermal)
        if ctx.args.max_temperature and max_temp > ctx.args.max_temperature:
            result.update(status="fail", reason="temperature exceeds configured OEM threshold")
        elif thermal:
            result.update(status="unverified", reason="thermal throttling observed; investigate cooling and configured limits")
        return result
    ctx.case("health/telemetry", telemetry)
    def link_errors():
        before = json.loads((ctx.root/f"before_{ctx.info['host']}_nvlink.json").read_text())
        after = json.loads((ctx.root/f"after_{ctx.info['host']}_nvlink.json").read_text())
        if before["returncode"] or after["returncode"]:
            raise Unverified("NVLink error counters unavailable")
        # Preserve line identities: do not compare unrelated counters by position alone.
        def counters(text):
            result, gpu, link = {}, "", ""
            for line in text.splitlines():
                if line.lstrip().startswith("GPU "):
                    gpu = line.strip()
                if line.lstrip().startswith("Link "):
                    link = line.strip()
                match = re.match(r"\s*(.*(?:Errors|Error|Replay|Recovery)[^:]*):\s*(\d+)\s*$", line, re.I)
                if match:
                    result[(gpu, link, match[1].strip())] = int(match[2])
            return result
        a, b = counters(before["output"]), counters(after["output"])
        if not a or set(a) != set(b):
            raise Unverified("NVLink counters absent or identities changed")
        changes = {str(key): b[key]-a[key] for key in a if b[key] != a[key]}
        if changes:
            raise Unverified(f"host-scope NVLink error/replay counters changed: {changes}")
        return dict(scope="entire host NVLink error counters", checked=len(a), changes=changes)
    if "nvlink" in ctx.args.suites or "collectives" in ctx.args.suites or "stress" in ctx.args.suites:
        ctx.case("health/nvlink_counters", link_errors)


def run(args):
    ctx = Context(args)
    fatal = False
    try:
        bad = [x for x in ctx.infos if not x["is_b300"]]
        if args.expected_gpus_per_node:
            counts = {host: len(ranks) for host, ranks in ctx.nodes.items()}
            ctx.emit("inventory/gpu_count", "pass" if all(n == args.expected_gpus_per_node for n in counts.values()) else "fail",
                     expected_per_node=args.expected_gpus_per_node, actual_per_node=counts)
        if "memory_min_bytes" in ctx.spec:
            ctx.emit("inventory/memory_capacity", "pass" if ctx.info["total_memory_bytes"] >= ctx.spec["memory_min_bytes"] else "fail",
                     actual_bytes=ctx.info["total_memory_bytes"], minimum_bytes=ctx.spec["memory_min_bytes"])
        ctx.emit("inventory/model", "unverified" if bad else "pass",
                 reason="non-B300 hardware: capability-gated cases run, unsupported ones are marked; no B300 acceptance claim" if bad else "B300 device name verified",
                 device=ctx.info)
        for suite in args.suites:
            if suite in ("gemm", "fp8", "mxfp8", "nvfp4"):
                test_gemm(ctx, "bf16" if suite == "gemm" else suite)
            else:
                {"grouped_gemm": test_grouped_gemm, "sparse": test_sparse, "memory": test_memory, "pcie": test_pcie,
                 "nvlink": test_nvlink, "collectives": test_collectives,
                 "attention": test_attention, "paged_attention": test_paged_attention, "mla": test_mla,
                 "moe": test_moe, "stress": test_stress, "features": test_features}[suite](ctx)
    except Exception as error:
        fatal = True
        ctx.emit("execution/fatal", "fail", reason=str(error), error=traceback.format_exc())
    finally:
        ctx.close_monitor()
        try:
            ctx.barrier()
            ctx.snapshot("after")
            ctx.barrier()
            health_checks(ctx)
        except Exception as error:
            ctx.emit("health/collection", "unverified", reason=str(error))
    if ctx.node_leader:
        for path in ctx.root.parent.glob(ctx.root.name+".nccl.*.log"):
            shutil.copy2(path, ctx.root/path.name)
    try:
        gathered = ctx.gather(ctx.rows)
    except Exception:
        gathered = [ctx.rows]
        fatal = True
    rows = [row for group in gathered for row in group]
    code = 1 if fatal else 0
    if ctx.rank == 0:
        code = max(code, write_report(ctx.root, rows, ctx.manifest, ctx.world))
        print(f"Report: {ctx.root/'REPORT.md'}; exit={code}", flush=True)
    if ctx.world > 1 and not fatal:
        status = ctx.t.tensor(code if ctx.rank == 0 else 0, dtype=ctx.t.int32, device=ctx.device)
        ctx.dist.broadcast(status, src=0)
        code = int(status.item())
        ctx.dist.destroy_process_group()
    return code


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="", help="new run directory; must not already exist")
    p.add_argument("--suites", default="all", help="all or comma-separated: "+",".join(SUITES))
    p.add_argument("--plan-only", action="store_true", help="print plan without importing torch or touching GPUs")
    p.add_argument("--report", metavar="DIRECTORY", help="regenerate report from existing per-rank JSONL; CPU only")
    p.add_argument("--spec-file", help="JSON numeric per-GPU OEM spec overrides, including hbm_gbps/pcie_one_way_gbps")
    p.add_argument("--nvls-only", action="store_true", help="restrict collective test to all-reduce with NCCL_ALGO=NVLS; cannot combine with other suites")
    p.add_argument("--expected-gpus-per-node", type=int, default=0, help="OEM GPU count to verify on each host; 0 means unspecified")
    p.add_argument("--require-isolation-tests", action="store_true", help="mark read-only MIG/virtualization/confidential inventory insufficient for acceptance")
    p.add_argument("--gpu", type=int, default=0, help="GPU for direct Python run; torchrun uses LOCAL_RANK")
    p.add_argument("--quick", action="store_true", help="small smoke workloads; cannot return formal PASS")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--timeout", type=int, default=180, help="distributed operation timeout in seconds")
    p.add_argument("--memory-fraction", type=float, default=.9, help="fraction of free memory after reserve for full pattern test")
    p.add_argument("--reserve-gib", type=float, default=2.0)
    p.add_argument("--stress-memory-fraction", type=float, default=.5)
    p.add_argument("--stress-seconds", type=float, default=300, help="total mixed soak seconds split across runnable precisions; excludes baselines")
    p.add_argument("--stress-precisions", default="bf16,fp8,nvfp4")
    p.add_argument("--moe-precisions", default="bf16,fp8,nvfp4")
    p.add_argument("--min-efficiency", type=float, default=0, help="optional agreed measured/spec floor; 0 reports percentages only")
    p.add_argument("--min-concurrent-ratio", type=float, default=.9)
    p.add_argument("--max-degradation-pct", type=float, default=10)
    p.add_argument("--max-temperature", type=float, default=0, help="OEM temperature ceiling in C; 0 uses throttle evidence only")
    p.add_argument("--fp8-error", type=float, default=.06, help="maximum sampled output NRMSE vs original FP32 GEMM")
    p.add_argument("--mxfp8-error", type=float, default=.08)
    p.add_argument("--nvfp4-error", type=float, default=.25)
    p.add_argument("--version", action="version", version=f"bench_b300 {VERSION}")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.report:
        root = Path(args.report)
        manifest = json.loads((root/"manifest.json").read_text())
        sources = [root/"merged.jsonl"] if (root/"merged.jsonl").exists() else sorted(root.glob("rank*.jsonl"))
        rows = [json.loads(line) for path in sources for line in path.read_text().splitlines()]
        return write_report(root, rows, manifest, len(manifest["devices"]))
    try:
        args.suites = select_suites(args.suites)
        if args.require_isolation_tests and "features" not in args.suites:
            raise ValueError("--require-isolation-tests requires the features suite")
        if args.nvls_only and args.suites != ["collectives"]:
            raise ValueError("--nvls-only must be used with --suites collectives in a separate run")
        if args.expected_gpus_per_node < 0:
            raise ValueError("--expected-gpus-per-node must be nonnegative")
        for name in ("stress_precisions", "moe_precisions"):
            values = list(dict.fromkeys(x.strip() for x in getattr(args, name).split(",") if x.strip()))
            if not values or set(values)-{"bf16", "fp8", "mxfp8", "nvfp4"}:
                raise ValueError(f"--{name.replace('_', '-')} must contain bf16/fp8/mxfp8/nvfp4")
            setattr(args, name, values)
        for name in ("iters", "repeats", "cpu_threads", "timeout", "stress_seconds", "reserve_gib",
                     "max_degradation_pct", "fp8_error", "mxfp8_error", "nvfp4_error"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
        if args.warmup < 0 or args.gpu < 0:
            raise ValueError("--warmup and --gpu must be nonnegative")
        for name in ("memory_fraction", "stress_memory_fraction", "min_concurrent_ratio"):
            if not 0 < getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must lie in (0,1)")
        if not 0 <= args.min_efficiency <= 1 or not math.isfinite(args.max_temperature) or args.max_temperature < 0:
            raise ValueError("invalid efficiency/temperature threshold")
    except ValueError as error:
        p.error(str(error))
    if args.plan_only:
        print(json.dumps(dict(version=VERSION, config=vars(args), plan={x: DESCRIPTIONS[x] for x in args.suites},
                              dependencies=package_versions(), spec_defaults=B300_SPEC,
                              optional_features="read-only inventory; not functional/security qualification"), indent=2))
        return 0
    if not args.outdir:
        p.error("--outdir must specify a new directory")
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA GPU unavailable; use --plan-only for a CPU-only plan", file=sys.stderr)
            return 2
        with torch.no_grad():
            return run(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
