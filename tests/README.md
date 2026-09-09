# SparseLightningIndexerGradKlLoss 测试

本目录按题面组织输入覆盖、数值正确性和固定 Mock 测试，运行平台为 Atlas A2/A3。

## 目录与分类

```text
tests/
  st/cases/
    coverage.json       # 47 项输入类型、范围和边界案例
    correctness.json    # 15 项精度、掩码、Scatter 和确定性案例
    mock.json           # 固定 6 项 Mock
  common/               # CaseSpec、加载器、参考公式和 dtype 编解码
  st/scripts/           # 构建、数据生成、执行、Profiler 和结果汇总
  st/runner/            # ACLNN/NPU runner
  ut/                   # Host 契约、参考数学、分类和工具自检
```

三份 JSON 分别定义各类案例。`common/cases.py` 提供共用结构并加载 Coverage/Correctness；
`common/mock_cases.py` 提供 Mock 六例的加载、固定输入与参考计算。
`full` 为 Coverage 与 Correctness 的并集，共 62 项；Mock 使用独立 Profiler 入口。
Coverage 审计检查 `coverage.json` 中的输入规格和范围。

`ut/` 的对应关系：

| 文件 | 用途 |
|---|---|
| `test_host_contract.py`、`host_contract.cpp` | Coverage：输入约束、输出 Shape/DType |
| `test_reference.py` | Correctness：参考数学与有限差分 |
| `test_profile_mapping.py` | Mock：Profiler 记录映射 |
| `test_case_matrix.py`、`test_data_generation.py` | 分类、数据生成和缓存检查 |

## 案例选择

在工程根目录执行：

```bash
python3 -m tests.st.scripts.gen_data --list --suite coverage
python3 -m tests.st.scripts.gen_data --list --suite correctness
python3 -m tests.st.scripts.run_tests --layer correctness --suite coverage --allow-large --rtol 0.002 --atol 0.000001
python3 -m tests.st.scripts.run_tests --layer correctness --suite correctness --rtol 0.002 --atol 0.000001
```

`--layer correctness` 表示执行数值比较，`--suite` 选择案例集合。
单例或少量案例使用 `--suite custom --case <case_id>`，可重复指定 `--case`。
大尺寸案例通过 `--allow-large` 显式启用。

## 参考计算与比较

正式案例使用 BSND 输入、四维 BNSG 统计量和 int64 实际长度。
当前 Coverage 的 Query 存储长度取值为 0、1、5、7；Query 空 Tensor 执行启动检查。
数值参考适用于零 RoPE、有效行至少有一个有效选中位置的输入。

参考实现使用 Float64 计算题面的 KL 和三个梯度公式，scaleValue 按 double 读取。
重复 sparseIndices 保留重复概率项，dKeyIndex 通过 `np.add.at` 累加；
分布计算使用经过索引有效性及右下对齐因果掩码筛选的位置。KL 采用 log-softmax 稳定计算。

四输出按 `abs(actual - reference) <= atol + rtol * abs(reference)` 比较，
golden 按输出 dtype 存储。运行时显式指定有限、非负的 `--rtol` 和 `--atol`；
本文命令给出的是本地比较阈值示例。

Runner 将输出预填为 NaN，以检查完整写回；输出前后的保护区填入固定标记，用于检测越界写入。
每次重复运行都比较数值误差，整网确定性开启时还检查输出逐字节一致。

## 构建与验证

构建默认 28 路并发，支持 24～28 路。默认产物根目录为
`/tmp/sparse_lightning_indexer_grad_kl_loss_new`。任务启动前检查 CPU 负载及 NPU 占用。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m tests.st.scripts.run_tests --layer static
python3 -m tests.st.scripts.build --target bootstrap
python3 -m tests.st.scripts.build --target host
python3 -m tests.st.scripts.build --target kernel --representative-case baseline_f16
python3 -m tests.st.scripts.run_tests --layer workspace --suite custom --case query_empty_special_case
python3 -m tests.st.scripts.run_tests --layer smoke --suite custom --case query_empty_special_case --case baseline_f16 --rtol 0.01 --atol 0.000001
python3 -m tests.st.scripts.run_tests --layer correctness --rtol 0.01 --atol 0.000001
python3 -m tests.st.scripts.build --target final
python3 -m tests.st.scripts.run_tests --layer full-test --allow-large --rtol 0.01 --atol 0.000001
```

按顺序执行，任一阶段失败即停止。静态阶段通过真实 Host Context 检查输入约束，
代表模板准备好后执行 ACLNN Workspace 查询和 NPU 验证。
`host` 构建 Host、Schema 和 Tiling；`kernel` 按代表案例选择 dtype 模板。
`final` 执行干净全量构建，目标目录应为空或尚未创建。

`run_tests --layer pipeline` 串行执行开发阶段检查；
追加 `--stage final --allow-large` 可执行干净全量构建和完整矩阵。
`full-test` 会校验源码、编译环境及产物哈希，确保运行包对应当前构建。

## 结果与产物

每次真机运行在 `results/run-*/` 保存：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 命令、案例配置、源码和构建产物哈希 |
| `runner.log` | 四输出误差、未选 Key 清零检查、设备耗时和运行状态 |
| `summary.json` | 汇总后的逐案例结果，由下面的命令生成 |

```bash
python3 -m tests.st.scripts.summarize_results /绝对路径/results/run-...
```

汇总器校验产物及数据哈希，应在重建同路径产物之前执行。
数据缓存逐文件校验 SHA256；Kernel 缓存以源码、共享 Tiling、编译器、生成配置和 Host 库标识版本。
Runner 核对实际加载的 ACLNN 库。运行时致命错误会结束进程，并收集该进程的最新日志。

需要复用固定 final 目录时，可通过 `archive_build --commit <已验证提交>` 归档原有构建。
归档工具核验源码和产物哈希，移动构建目录并记录路径映射。

## 固定六例 Mock

`st/cases/mock.json` 按用户提供的 Case Shape 结果配置六例，案例 ID 为 `mock_case_1`～`mock_case_6`。

| Case | B / S1 / S2 | N1 / Nidx1 | Dq / Didx / Drope | K | scale | 核心类型 | 实际长度 |
|---|---|---|---|---|---|---|---|
| 1 | 1 / 2 / 6 | 4 / 4 | 32 / 16 / 8 | 2 | 0.1768 | FP16 | 省略 |
| 2 | 2 / 3 / 8 | 4 / 4 | 32 / 16 / 8 | 3 | 0.1768 | FP16 | int32：q=(2,3)，k=(5,8) |
| 3 | 1 / 4 / 10 | 8 / 8 | 32 / 16 / 8 | 3 | 0.125 | FP32 | 省略 |
| 4 | 1 / 8 / 16 | 8 / 8 | 64 / 32 / 16 | 3 | 0.125 | FP16 | 省略 |
| 5 | 2 / 4 / 16 | 8 / 8 | 64 / 32 / 16 | 3 | 0.125 | FP16 | 省略 |
| 6 | 1 / 8 / 32 | 16 / 16 | 64 / 32 / 16 | 4 | 0.125 | FP32 | 省略 |

全部采用 BSND 输入和 BNSG 统计量；weights 及 RoPE dtype 与核心类型一致。
`common/mock_cases.py` 只生成这六例的固定输入与四输出参考，数据版本为 `mock-six-v1`。
四个 FP16 Case 使用固定 seed=20260908，两个 FP32 Case 使用固定数值构造；RoPE 均为零。
FP32 Mock 参考内部重算完整可见区间的 softmax，实际输入的统计量仍为固定 FP32 数据。
比较阈值固定为 rtol=0.002、atol=1e-6。跨版本性能比较使用相同配置、数据和比较方式。

完成当前源码的干净全量构建后运行：

```bash
python3 -m tests.st.scripts.profile_mock_workloads \
  --build-dir /path/to/final-build --rounds 3 --metrics PipeUtilization
```

一个 runner 执行六例并比较四输出，随后解析 Profiler 数据。
Case 2 的实际长度明确通过 int32 Tensor 传入。数值失败继续记录其他例，运行时致命错误立即退出。
默认三轮，结果保存在 `results/mock-profile-*/`：
`summary.json` 记录各次 Kernel 耗时，`profiler/` 保存原始 CSV，
`manifest.json` 记录配置、输入和构建产物。性能分析使用数值比较及运行检查均通过的结果。

## 依赖与来源

运行依赖为 CANN SDK 与 NumPy（BSD-3-Clause）。
构建脚手架、内容寻址缓存、ACL 资源管理和 dtype 编解码复用自本地相邻工程
`SparseLightningIndexerGradKlLoss`，起点提交为 `2948a5da2e83971233799f38e46151f139c670d2`。
