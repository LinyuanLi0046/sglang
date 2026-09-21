# NPU PD 长期诊断：使用说明

这是 opt-in 诊断，不是故障修复。默认关闭；不修改 MF、绑核、传输额度、业务 timeout、KV 分配、模型算子或既有 stream wait。

**当前落盘策略：无损压缩、增量记录，不设置保留容量上限，不自动删除或覆盖历史。** history、lifecycle 只按文件大小分段，所有旧分段保留。首次异常/升级保存完整压缩现场，持续 PD 等待每 30 秒只保存状态报告，不再重复复制全部 mmap。目标是当前 P+D 工作负载下新增诊断文件低于 **1 GB/10 分钟**；这是测量和优化目标，不是达到后丢弃证据的硬配额。内存 ring 仍有界，覆盖、撕裂和写盘失败必须查看 coverage。请使用持久化数据盘并监控空间，不要把无限增长的落盘目录放进 `/dev/shm`。本次不改 router、无需重建原生模块。

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

v2 每个 recorder 固定约 **38.03 MiB** mmap，含独立低频生命周期环。P4+D4 加两个单进程 HTTP/Tokenizer recorder，共 10 份约 **380.3 MiB**，另留余量；多 HTTP worker 会增加份数。容器默认 64 MiB `/dev/shm` 不够。初始化会尝试 `posix_fallocate`，空间不足则告警并关闭该进程的诊断，不应拿缺失进程的结果当作完整现场。也可选用一个容量足够的现有 tmpfs；普通磁盘映射不推荐用于低扰动场景。

新增 `prefill_api` / `decode_api` recorder 始终为 CPU 模式，记录 HTTP、请求转换/分词、IPC 提交和取消，使用 task-local context 关联并发请求，不读取/缓存请求正文，不创建 NPU Event。原 scheduler 的 `ARRIVED` 接上 IPC 后半段。`FRONT_IPC_RETURN` 只说明本地提交返回，不代表 scheduler 已收到。

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

为了同时保存服务日志，可追加 `--log-file /绝对路径/P.log`。仅在完整现场时无损压缩复制显式指定的文件，不限制个数或只留末尾；周期性等待状态报告不重复复制。很大的历史服务日志会增加采集 CPU/I/O，也可能超过 1 GB/10 分钟目标，建议单独保留原服务日志、不传此参数。不自动扫描日志盘。服务日志可能含业务内容，请注意现场目录权限。

## 4. 触发规则

| 条件 | 默认动作 |
| --- | --- |
| 具体在途 CPU/MF/job 或设备观察 5 秒无进展 | 保存所有已发现 rank 的现场 |
| 同一执行停滞达到 10 秒 | 先留证，再尝试一次 Python/native 栈 |
| 单个 PD 请求 30 秒无实际进展 | `PD_WAIT_LONG`，仅保存现场，不自动抓栈 |
| worker 异常、watchdog 标记、进程退出/zombie | 下一次扫描即留证，不等 5/10 秒 |

collector 每秒扫描；所以实际触发通常在阈值后的约一个扫描周期内。抓栈每个子命令最多约 5 秒，总预算约 20 秒，超时只结束 collector 自己的 py-spy 子进程，不杀 SGLang。抓栈期间可能出现一段采集间隔，ring 覆盖会明确标记缺口；抓栈前的证据已经落盘。

首次 PD 长等待保存完整压缩现场 `incident-<时间戳>-first-wait`；持续等待每 30 秒保存 `incident-<时间戳>-status`，包含所有已发现进程的当前请求/调用/设备事件/job 状态和日志引用，不复制 raw、`/proc` 或服务日志。增量事件始终持续落盘。新的执行停滞、升级抓栈、异常、进程退出、手动 `--once` 仍保存完整压缩 `incident-<时间戳>-snapshot`。旧目录永久保留。重叠等待属于同一组，原请求取消不会使其最初现场被覆盖；新一轮等待仍保留首份完整现场。触发本身不表示故障。

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

每个 recorder 应有一对 `.json` manifest 和 `.mmap`；TP4 单 HTTP worker 的 P 应有 5 对（4 scheduler + 1 API），D 同理。collector 约每 5 秒 flush：

- `history-*.jsonl.gz`：按解压前约 64 MiB 分段，保存执行事件及变化的精简 observation。
- `lifecycle-*.jsonl.gz`：按解压前约 16 MiB 分段，保存请求入口、PD 阶段、metadata、终态、取消及覆盖缺口；`PD_MODEL` 只进 history，不进 lifecycle，低频事件也不再重复写入 history。v2 producer mmap 使用独立低频环，避免逐 token 事件覆盖请求边界。
- 两类均为一级 gzip **无损压缩**，所有段永久保留；压缩只在 collector 进程，不在 SGLang 执行线程。约每 64 KiB/flush 写入一个完整 gzip member，运行中的已 flush 部分可直接读取；进程强杀时尚未写入/未写完的尾部仍可能丢失。
- startup `EVENT_WARM` 不作为活动设备工作反复写入 observation；原始 mmap 中的数据仍然保留。

可在另一个 output 目录做一次只读快照，不抓栈：

```bash
python scripts/npu_pd_diag_collect.py \
  --source /dev/shm/sglang_pd_diag/run_0920_P \
  --source /dev/shm/sglang_pd_diag/run_0920_D \
  --output /data2/pd_diag_artifacts/manual_check_01 --once
```

完整现场包含 `report.json`、**全部已发现 recorder 的 `.mmap.gz` 及配套 manifest**、`.json.gz` 格式的完整选定 `/proc` 文本、线程列表、可选压缩服务日志及栈。mmap 解压后逐字节一致；不是采样或截断。`/proc` 从头读取，不再 SEEK_END。堆栈输出和 report 不按大小截断，但抓栈时间上限保留，以避免长时间 attach。周期性 `status` 只保留当前状态及增量日志引用，其 `raw_included=false`、`raw_complete=null`，明确区别于复制失败。

完整 history/lifecycle 保留在 output 根目录。report 的 `journals` 字段记录相对路径和快照时文件长度，避免每次现场重复复制全部增长中的历史。**交付现场要带整个 output 目录，不能只拷贝一个 incident 子目录。** 新 collector 可以读取 v1 旧 mmap，但旧 recorder 本身没有独立低频环；要启用隔离和前端补点，必须重启升级后的服务。

P4+D4 加两个 API 的**常驻 mmap**仍约 380.3 MiB，不会每 30 秒累加；它也不等于整个 collector 的 RSS。现在不再每 30 秒落盘 380.3 MiB：完整 raw 一级 gzip 压缩，持续等待仅追加状态。对提供的 0921-02 旧数据离线测试，8 个当前 worker raw 从 176.16 MiB 压到 2.26 MiB；旧 history/lifecycle 合计 250.44 MiB，使用新分块 writer 压到约 15.41 MiB。这是已保留样本、本机测试，不是远端任意负载的体积或时延保证。

每分钟输出 `DIAG_OUTPUT_VOLUME` 到 collector.log 和 lifecycle：`bytes_written` 是本 collector 最近 600 秒新增文件字节数，目标 `target_bytes=1000000000`；超过时 `over_target=true`，不会删旧记录或停止写入。使用一个 collector 同时采集 P/D 才是合计值；分开部署则需相加。计数包含压缩 journals、现场文件及显式附加的服务日志副本，不包含常驻 source mmap、外部服务日志原件和重定向的 collector.log；重启后窗口重新累计。大量全新故障或巨大的附加服务日志仍可能超过目标，需远端验证。

没有自动容量清理：空间不足会明确报错。压缩消耗 collector CPU；完整现场复制及抓栈仍可能延迟下一次 ring 读取，coverage 会记录缺口，因此不承诺零扰动。

检查**完整现场**的 `report.json`：`raw_complete=true`，`raw_artifacts` 对 P4+D4 加两个 API 应有 10 项且 `saved=true`，每项带源路径、解压后 `bytes` 和压缩后 `stored_bytes`；失败会记录错误并告警。这只表示已发现文件复制完整，不保证未接入的 peer/进程已采集，也不消除 ring coverage gap。注意 `lifecycle_ring_lost`、`lifecycle_torn_reads` 和 `lifecycle_isolated`。周期性状态报告的 `previous_raw_snapshot` 指向前一份完整现场，不能把旧 raw 当成该状态报告时刻的 raw。

离线需要按旧格式读取 raw 时，可在完整现场目录执行 `gzip -dk -- *.mmap.gz`，保留压缩原件并生成 `.mmap`；解压需要额外磁盘空间。JSONL 可用 `gzip -cd lifecycle-0.jsonl.gz` 或 Python 标准库 `gzip.open(path, 'rt')` 阅读，不需要安装额外包。

离线查看概要，无需连接 NPU：

```bash
python scripts/npu_pd_diag_collect.py \
  --report /data2/pd_diag_artifacts/run_0920/incident-实际时间戳-snapshot/report.json
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

## 6.1 不改 router 的请求链路定位

保留现有 router 启动方式和原生模块，不需要给 router 设置新的环境变量或重新安装。新增记录全部位于 P/D Python 侧：

- `FRONT_HTTP_ENTER/PEER`：进入服务端 ASGI，以及本地/对端地址。
- `FRONT_CLIENT_TRACE/TRACEPARENT`：只记录已收到的 `x-request-id` / `traceparent`，不读取其他 header。
- `FRONT_GENERATE`：请求已规范化，记录实际 `rid` / `bootstrap_room`。
- `FRONT_TOKENIZE_ENTER/RETURN/ERROR`：分词前处理开始、返回、异常。
- `FRONT_IPC_ENTER/RETURN/ERROR`：提交 Scheduler 的调用边界。
- `FRONT_HTTP_HEADERS/DISCONNECT/ERROR/END`、`FRONT_ABORT/ABORT_CAUSE`：响应头、断开观察、异常、结束和取消来源。

同一 API 进程内用 `call` 关联 HTTP、分词和 IPC，跨 P/D 用 `bootstrap_room`；`rid` 不要求相同。若客户端能够为每次请求附加唯一 `x-request-id`，并且现有 router 确实透传，两端会记录它；应先用一条测试请求确认透传，不能仅凭客户端设置就假设服务端已收到。不修改或解析原始 body，不改原 rid/room。

边界：未经 router 补点，不能直接知道两路 future 是否被执行、router 是否在等连接池或提前收到另一侧错误；这部分保留现有 router 日志辅助判断。HTTP 校验失败发生在 `FRONT_GENERATE` 之前时可能没有 room，但仍有 HTTP call、状态码及已收到的关联 header。`HTTP_HEADERS=200` 不代表流式响应全部完成；`HTTP_END.done=1` 只表示 ASGI 应用正常返回，不证明业务成功。

## 7. 覆盖边界与验证

当前设备标记针对 eager、非 MTP；Graph/MTP 开启时只做 CPU 留痕，并记录 `EVENT_UNSUPPORTED`，不插入 capture/replay。当前事故的 P4+EP4、无 MTP、无 Graph、无 scheduler overlap 是主要定位范围。普通 decode 及既有内部副流不增加新的 wait。

不记录 MF 内部具体 QP/CQ/WQE 释放，不映射 runtime task ID 到具体算子，不为物理 KV 页新增 generation。只凭这些记录不能保证每次都得到唯一根因；它的作用是区分阶段、保留在途状态，并缩小下一步证据范围。跨容器 PID namespace 不匹配不会 attach。collector 不自动 SSH 拉取对端。

本地 CPU 检查：

```bash
python scripts/npu_pd_diag_selftest.py
```

已覆盖：5000 正常请求后的等待记录保留、跨 producer 完成回收、健康检查排除、重复轮询不推进时钟、room 重用、取消与 native 生命周期分离、线程池上下文、假 Event 未完成不复用/query 进入可见、撕裂检测、关闭模式 decorator 保留原函数、无动态属性的 IPC 请求/重试、P4+D4 全部 raw、旧轮次过滤、重叠等待保护、独立低频 ring、日志超过四段及重启后不丢旧段、现场不覆盖、不可 seek 的 procfs、并发 ASGI context 隔离和取消原样传播。该脚本不替代真实 Ascend 验证。

远端建议先 `cpu` 小流量确认所有 rank 和文件增长，再做与关闭诊断的 TTFT/吞吐对照。之后选择一个短时 `coarse` 验证 `event_flags` ready、Event 能完成、没有持续 `EVENT_POOL_FULL`/coverage gap。真实 NPU Event ABI、record/query 开销、GIL 行为和吞吐损耗尚需远端实测；不能承诺零扰动。

关闭：取消 `SGLANG_NPU_PD_DIAG` 或设为 `0` 后重启服务；独立 collector 按它自己的 PID 正常停止。程序不自动删除 worker mmap，以便崩溃后继续取证。
