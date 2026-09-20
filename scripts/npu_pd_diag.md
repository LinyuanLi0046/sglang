# NPU PD 长期诊断：使用说明

这是 opt-in 诊断，不是故障修复。默认关闭；不修改 MF、绑核、传输额度、业务 timeout、KV 分配、模型算子或既有 stream wait。

## 1. 当前建议

你现在难以复现，先在 P/D 两端使用 `cpu` 长期留痕。它不创建/record/query 诊断 NPU Event，可捕获 bootstrap、预分配、metadata、KV job、MF 调用、完成通知、结果取回 CPU 的进展。

若要进一步区分“CPU 已提交到哪层”和“设备实际完成到哪层”，改成 `coarse` 后重启服务。`stage` 在少数层记录更细的 attention/AG/RS/MoE/副流边界，扰动更高，通常第二步再用。

三个环境变量必须在启动 Python **前** 设置。`cpu` 是默认 level。代码启用范围是 `server_args.device == npu` 且 disaggregation mode 为 prefill/decode；其他实例不创建 recorder。

## 2. 启动 P 和 D

下面假设远端仓库为 `/data2/hw_lly/sglang`；替换成实际路径。保留你现有启动脚本的其余参数、网卡、绑核、timeout 不变。

P 的 shell：

```bash
export SGLANG_NPU_PD_DIAG=1
export SGLANG_NPU_PD_DIAG_LEVEL=cpu
export SGLANG_NPU_PD_DIAG_DIR=/dev/shm/sglang_pd_diag/run_0920_P
# 在这个 shell 中执行原来的 P 启动命令/脚本
```

D 的 shell：

```bash
export SGLANG_NPU_PD_DIAG=1
export SGLANG_NPU_PD_DIAG_LEVEL=cpu
export SGLANG_NPU_PD_DIAG_DIR=/dev/shm/sglang_pd_diag/run_0920_D
# 在这个 shell 中执行原来的 D 启动命令/脚本
```

每次重启服务仍建议使用一个新的 run 目录；不要在运行时删除/截断 `.mmap`。collector 会跳过同一 source、role/rank/device 下已被新一轮替代且确认退出的旧记录。采集期间退出的 worker 先保存一次最终现场，再停止重复扫描；源文件不会删除。新 manifest 记录 PID namespace；跨 namespace、权限不足或旧 manifest 无法确认身份时保留为未知并继续读取，不把它误判为退出。

启动前检查共享内存：

```bash
df -h /dev/shm
```

每个 worker 固定约 **22.02 MiB** mmap，8 个 worker 约 177 MiB，另留余量。容器默认 64 MiB `/dev/shm` 不够。初始化会尝试 `posix_fallocate`，空间不足则告警并关闭该 worker 的诊断，不应拿缺失 rank 的结果当作完整现场。也可选用一个容量足够的现有 tmpfs；普通磁盘映射不推荐用于低扰动场景。

## 3. 独立运行 collector

collector 只依赖 Python 标准库，不需要 torch/NPU。必须由**另一个 shell、docker exec 会话或独立服务**启动，不能让 SGLang 的 Python 进程创建它：仅设置 `setsid` 不能防止父进程按进程树杀掉它。

这不等于能抵抗整个容器退出/删除：如果 SGLang 就是容器 PID 1，整个容器终止仍会结束同容器 collector。持久化 output 应放在宿主机挂载的数据目录；若需要容器级故障后仍继续采集，需在容器外独立部署并处理文件可见性/PID namespace，当前脚本不会自动修改容器配置。

P、D 在同一个容器/同一 PID namespace，且两个目录均可见时，**只启动一个 collector**，通过两个 `--source` 同时采集。任何一侧触发时都会保存两侧当前已发现的全部 rank：

```bash
cd /data2/hw_lly/sglang
mkdir -p /data2/pd_diag_artifacts/run_0920
nohup python scripts/npu_pd_diag_collect.py \
  --source /dev/shm/sglang_pd_diag/run_0920_P \
  --source /dev/shm/sglang_pd_diag/run_0920_D \
  --output /data2/pd_diag_artifacts/run_0920 \
  > /data2/pd_diag_artifacts/run_0920/collector.log 2>&1 &
echo "collector PID=$!"
```

如果 P/D 在不同容器或不同机器，在各自容器中各开一个独立 shell，分别启动 collector：

```bash
# P 容器；D 侧将路径中的 _P 改为 _D
cd /data2/hw_lly/sglang
mkdir -p /data2/pd_diag_artifacts/run_0920_P
nohup python scripts/npu_pd_diag_collect.py \
  --source /dev/shm/sglang_pd_diag/run_0920_P \
  --output /data2/pd_diag_artifacts/run_0920_P \
  > /data2/pd_diag_artifacts/run_0920_P/collector.log 2>&1 &
```

不要让两个 collector 使用同一个 output 目录。`collector.lock` 防止重复运行；若 collector 被 SIGKILL 遗留该文件，先确认文件中 PID 的进程确实已退出，再手动移走锁文件，或用新的 output 目录。

为了保存服务日志尾部，可追加 `--log-file /绝对路径/P.log`，最多 8 个，每个截取末尾 1 MiB。不自动猜测/扫描整个日志盘。服务日志可能含业务内容，请注意现场目录权限。

## 4. 触发规则

| 条件 | 默认动作 |
| --- | --- |
| 具体在途 CPU/MF/job 或设备观察 5 秒无进展 | 保存所有已发现 rank 的现场 |
| 同一执行停滞达到 10 秒 | 先留证，再尝试一次 Python/native 栈 |
| 单个 PD 请求 30 秒无实际进展 | `PD_WAIT_LONG`，仅保存现场，不自动抓栈 |
| worker 异常、watchdog 标记、进程退出/zombie | 下一次扫描即留证，不等 5/10 秒 |

collector 每秒扫描；所以实际触发通常在阈值后的约一个扫描周期内。抓栈每个子命令最多约 5 秒，总预算约 20 秒，超时只结束 collector 自己的 py-spy 子进程，不杀 SGLang。抓栈期间可能出现一段采集间隔，ring 覆盖会明确标记缺口；抓栈前的证据已经落盘。

首次 PD 长等待另外保存到 `incident-first-wait`。这组请求在明确记录 PD 终态/清理或确认 worker 退出前，后续轮询不会覆盖首次现场；短暂取得进展、暂时不触发超时或读取撕裂都不解除保护。这组等待全部结束后，下一组长等待可以替换它。这不是为每个并发 room 保留无限份完整 mmap；其他请求仍有滚动现场和独立 lifecycle 记录。滚动等待快照仍按 `--pd-wait-after` 节流，不因不断出现新的慢请求而每秒备份全部 raw。正常等待也可能达到阈值，触发本身不表示故障。

`py-spy` 必须已经在 collector 的 PATH 中，且当前用户/容器具有合法 attach 权限。工具不会修改 ptrace、容器 capability、cpuset 或安全策略。未安装或权限不足会记录原因，其他采集继续。完全不希望自动 attach 时追加 `--no-stack`。

正常单次 prefill/MF 若确实可能超过 10 秒，可只改采集阈值，例如：

```bash
python scripts/npu_pd_diag_collect.py \
  --source /dev/shm/sglang_pd_diag/run_0920_P \
  --output /data2/pd_diag_artifacts/run_slow_prefill \
  --snapshot-after 10 --stack-after 20 --pd-wait-after 30
```

这些数值不取消/重试请求，也不改变 `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` 或 scheduler watchdog。

## 5. 怎样确认生效

```bash
ls -lh /dev/shm/sglang_pd_diag/run_0920_P
ls -lh /dev/shm/sglang_pd_diag/run_0920_D
ls -lh /data2/pd_diag_artifacts/run_0920
```

每个 worker 应有一对 `.json` manifest 和 `.mmap`；TP4 的 P 应有 4 对，D 同理。collector 约每 5 秒 flush：

- `history-*.jsonl`：4 个 64 MiB 段，保存执行事件及发生变化的精简 observation；不再每秒重复写完整设备事件/请求片段。
- `lifecycle-*.jsonl`：独立的 4 个 16 MiB 段，保存 PD 阶段、metadata、终态、取消及覆盖缺口。高频设备事件不会挤掉这个文件中的请求记录；它仍是按容量轮转，不保证任意 QPS 下固定的保留时长。
- startup `EVENT_WARM` 不作为活动设备工作反复写入 observation；原始 mmap 中的数据仍然保留。

可在另一个 output 目录做一次只读快照，不抓栈：

```bash
python scripts/npu_pd_diag_collect.py \
  --source /dev/shm/sglang_pd_diag/run_0920_P \
  --source /dev/shm/sglang_pd_diag/run_0920_D \
  --output /data2/pd_diag_artifacts/manual_check_01 --once
```

滚动快照保留在 `incident-0` / `incident-1`，优先保留最近一次带栈的现场；首次等待现场见 `incident-first-wait`。每份包含 `report.json`、近期 history、独立 lifecycle、**全部已发现 worker 的原始 mmap 及配套 manifest**、`/proc` 线程状态、可选日志尾部和栈。不再使用先复制 P、额度耗尽后漏掉 D 的 160 MiB 总预算。

P4+D4 每份 raw 合计约 **176.2 MiB**；三份现场的 raw 约 **528.5 MiB**，另加 history/lifecycle、日志和栈，建议 output 所在磁盘至少预留 **2 GiB**。最多 64 个当前 recorder，更多 rank 的空间需求按实际数量增加。只有发生采集触发才复制 raw，不在正常每秒扫描时重复备份。复制是运行中 best-effort 快照，不会暂停模型/设备，记录自身的 CRC/sequence 用于检测撕裂。

检查每份 `report.json`：`raw_complete=true`，`raw_artifacts` 对 P4+D4 应有 8 项且 `saved=true`，每项都带对应源路径；保存失败会记录错误并打印告警，不再静默漏掉 D。这只表示已发现文件的复制完整，不保证未接入 collector 的 peer/rank 已被采集，也不消除原 recorder 的 coverage gap。

离线查看概要，无需连接 NPU：

```bash
python scripts/npu_pd_diag_collect.py \
  --report /data2/pd_diag_artifacts/run_0920/incident-0/report.json
```

复现后保留 **P/D 两端**整个 output 目录及相应 run 的 manifest/mmap、服务日志、底层 plog。不要只截取最后一条 ERROR。跨机器的 monotonic 时间不能直接比较，manifest 同时记录墙钟和 monotonic 锚点；配对依赖 room/rid，而不是 PID 或本地 attempt 数字相同。

## 6. 读报告时的关键语义

- `requests[].fragments`：不同 producer 线程的请求视角；合并时使用同一请求实际进展的最大时间，不被其他请求/健康检查刷新。未分配请求槽为 `slot=-1`，未知应收数量为 `-1`，不是 0。
- P `seen/need` 表示本地已收到/应收到 metadata；D `sent/need_sent` 表示发送调用返回数/目标数。发送返回不等于 P 已收到。
- D `done/need_done/source_mask` 表示完成通知的去重计数和前 63 个源 rank 摘要；不是底层 CQ/WQE 状态。
- `calls` 中 `MF_NATIVE_ENTER`、`RESULT_TO_HOST_ENTER` 可区分 MF 调用未返回与 CPU 在取回结果。调用内发生的异常在原有异常转换为 `-1` 前留痕。
- `JOB_CANCELLED` 只表示尚未执行的 Future 已取消；运行中的 native 调用直到真实 return/exception 才退出 active 表。`ABORT_ACK`、`clear`、逻辑 KV release 均不证明 native drain。
- `KV_RELEASE_ENTER` 的 generation 是能读到的 **CPU request-slot generation**；不是物理 page generation，也不是已证明物理页释放。radix 持有的页可能仍然有效。
- `KV_PAGE_SAMPLE` 是已经转到 CPU 的 page-index 数组的前后各 4 项摘要，不额外读取设备。乐观 prefill 重排和 D rebootstrap 会切换本地 attempt；wire 只携带 room 的通知无法单凭记录排除旧 attempt 的迟到包。
- `EVENT_QUERY_ENTER` 长期未 return：只能证明诊断 query 本身未返回/观察受阻；不能据此证明某个 attention 算子坏了。`EVENT_PENDING` 也需结合 status/最近 query 判读。
- `Work.wait()` / `wait_stream()` 的 CPU 返回不等于设备完成。stage Event 在原有依赖之后记录，不新增依赖。
- `PD_WAIT_LONG` 是长等待，不是“RDMA 死锁”判决。预分配预算不足、metadata 不齐、完成通知不足，要看各自字段及另一端现场。
- `UNKNOWN`、`coverage` 的 table overflow、ring lost、torn read、Event 池不足/观察过期不能解读成“正常空闲”或“对端没发送”。

## 7. 覆盖边界与验证

当前设备标记针对 eager、非 MTP；Graph/MTP 开启时只做 CPU 留痕，并记录 `EVENT_UNSUPPORTED`，不插入 capture/replay。当前事故的 P4+EP4、无 MTP、无 Graph、无 scheduler overlap 是主要定位范围。普通 decode 及既有内部副流不增加新的 wait。

不记录 MF 内部具体 QP/CQ/WQE 释放，不映射 runtime task ID 到具体算子，不为物理 KV 页新增 generation。只凭这些记录不能保证每次都得到唯一根因；它的作用是区分阶段、保留在途状态，并缩小下一步证据范围。跨容器 PID namespace 不匹配不会 attach。collector 不自动 SSH 拉取对端。

本地 CPU 检查：

```bash
python scripts/npu_pd_diag_selftest.py
```

已覆盖：5000 正常请求后的等待记录保留、跨 producer 完成回收、健康检查排除、重复轮询不推进时钟、room 重用、取消与 native 生命周期分离、线程池上下文、假 Event 未完成不复用/query 进入可见、撕裂检测、关闭模式 decorator 保留原函数、现场轮转，以及无动态属性的 IPC 请求/重试、一个 collector 保存 P4+D4 全部 raw、旧轮次过滤、首次等待保护和 lifecycle 独立轮转。该脚本不替代真实 Ascend 验证。

远端建议先 `cpu` 小流量确认所有 rank 和文件增长，再做与关闭诊断的 TTFT/吞吐对照。之后选择一个短时 `coarse` 验证 `event_flags` ready、Event 能完成、没有持续 `EVENT_POOL_FULL`/coverage gap。真实 NPU Event ABI、record/query 开销、GIL 行为和吞吐损耗尚需远端实测；不能承诺零扰动。

关闭：取消 `SGLANG_NPU_PD_DIAG` 或设为 `0` 后重启服务；独立 collector 按它自己的 PID 正常停止。程序不自动删除 worker mmap，以便崩溃后继续取证。
