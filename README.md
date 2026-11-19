# device_perf

面向 NVIDIA B300 GPU 的测试仓库。入口脚本会通过 torchrun 启动 [bench_b300.py](bench_b300.py)，按可见 GPU 数创建进程；基准脚本自身会校验设备名称，非 B300 运行不会给出正式 PASS。

## 快速开始

    bash run.sh                                  # 全部测试项，自动生成 results/b300_时间戳
    bash run.sh --outdir results/b300_001       # 指定新的结果目录
    bash run.sh --suites fp8,moe --quick        # 冒烟测试
    bash run.sh --plan-only                     # 只打印计划，不导入 torch、不触碰 GPU
    bash run.sh --bench-help                    # 查看基准脚本全部参数

结果目录包含 REPORT.md、summary.json、results.csv、每 rank JSONL、监控快照和运行清单。退出码 0 表示所选必测项通过，1 表示失败，2 表示缺失、不支持或待复核；--quick 只用于开发验证，不能作为正式验收结论。

## 多节点

所有节点必须传同一个结果目录，并保持 GPU 数和测试参数一致：

    # 节点 0
    bash run.sh --nnodes 2 --node-rank 0 --master-addr <host0> \
      --nproc-per-node 8 --outdir /shared/results/b300_001

    # 节点 1
    bash run.sh --nnodes 2 --node-rank 1 --master-addr <host0> \
      --nproc-per-node 8 --outdir /shared/results/b300_001

NNODES、NODE_RANK、MASTER_ADDR、MASTER_PORT、NPROC_PER_NODE、STRESS_SECONDS、B300_OUTDIR、LOG_DIR、TORCHRUN_EXTRA 环境变量分别对应同名命令行选项和扩展参数。平台注入的 NCCL_* 配置会原样继承。

## 常用测试范围

--suites 支持 gemm、fp8、mxfp8、nvfp4、grouped_gemm、sparse、memory、pcie、nvlink、collectives、attention、paged_attention、mla、moe、stress、features，也可写 all。例如只测互连与通信：

    bash run.sh --suites nvlink,collectives --outdir results/b300_comm
