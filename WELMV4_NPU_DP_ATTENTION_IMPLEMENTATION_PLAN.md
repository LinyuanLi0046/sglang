# WeLMv4 NPU DP-Attention 正式实施方案

> 基线：`009eb25525ada59efaf6aa1d1c70badbe7518f62`  
> 文档状态：正式实施方案，尚未实施  
> 适用范围：WeLMv4 / WeLMv4 NextN、NPU、DP Attention、KV Mirror、Spec V2 MTP、NPU Graph、scheduler overlap

## 1. 结论

本方案改为 **能力与通信契约驱动**，不再枚举固定拓扑组合。`Attention TP` 仍由框架按 `TP / (Attention DP × Attention CP)` 解析；WeLMv4 只检查所选 MoE transport 是否满足本阶段的数据布局契约。

以下四种当前重点配置都在方案覆盖范围内，但它们只是验收样例，不是启动白名单：

| 外层 TP | Attention DP | Attention TP | MoE EP | MoE TP | MoE transport |
|---:|---:|---:|---:|---:|---|
| 4 | 2 | 2 | 1 | 4 | 纯 TP-MoE |
| 4 | 4 | 1 | 1 | 4 | 纯 TP-MoE |
| 4 | 2 | 2 | 4 | 1 | prefill normal AllGather；decode/mirror AG+本地排序 |
| 4 | 4 | 1 | 4 | 1 | prefill normal AllGather；decode/mirror AG+本地排序 |

核心规则只有两类：

- 没有 EP：DP-local attention rows 先按 row-layout gather 成纯 TP-MoE 所需 FULL rows，执行 TP-sharded MoE，再按 row-layout combine 回原 DP shard。该规则同时覆盖 DP2 和 DP4。
- 同时开启 DP Attention 和 EP：普通非 mirror prefill 必须使用 DeepEP NORMAL AllGather；decode、target mirror、target verify 必须先显式 AllGather 成 `MOE_FULL_REPLICATED`，再走本地 sort/initrouting/local experts/EP AllReduce。physical MTP layer 的 resolved `moe_ep_size>1` 时，draft decode/extend 也必须使用同一 explicit-AG local-sort transport，不能按 shape 绕过。
- 每层有效的 `self_attn.use_o_norm=True` 时，attention 后处理恰好调用一次 `mmq_style_norm_after_attn`；删除 `use_dp_o_norm_after_attn` 重叠分支。
- 不开启 DP Attention 时，不执行新的 DP+EP transport 校验、不创建新执行器，不改变现有纯 TP、TP+EP、Graph、MTP、overlap 和多流路径。

当前 blanket reject 仍不能直接删除：必须先建立显式 layout/row-layout 转换，再把门禁收窄为上述能力校验，而不是换成另一组固定数字组合。

## 2. 范围和非目标

### 2.1 首版必须支持

- `TP4+DP2/DP4+纯TP4 MoE`。
- `TP4+DP2/DP4+EP4`，并满足本方案的分阶段 EP transport 契约。
- 普通非 mirror prefill。
- 首个 mirror consumer 和后续 mirror 层。
- eager decode 与 NPU Graph decode。
- Spec V2 target verify。
- Spec V2 draft decode、draft extend 和 prefill 后的 draft seed。
- decode/target verify/draft/draft extend 的 Graph padding 开启或关闭；普通 prefill 首版保持 eager，不扩展 breakable prefill Graph。
- scheduler overlap 开启或关闭。
- 一个 Attention DP shard 暂时没有请求的 idle-rank 情况。
- 每个 DP shard 请求数不相等。
- WeLMv4 full attention 和 SWA attention。

### 2.2 DP+EP 的能力校验

- MTP 沿用已经确定的 Spec V2 范围：EAGLE/NEXTN、`topk=1`、greedy、固定 `S>=1`，verify 宽度 `D=S+1`；随机采样、rejection sampling 和 `topk>1` 不在首版范围。
- WeLMv4 DP Attention + Spec V2 首版保持 `speculative_skip_dp_mlp_sync=False`。这是后文依赖的 scheduler phase-alignment 契约；若显式绕过该同步，必须先另行实现并验证跨 DP 的 prefill/verify 混合轮次，不能静默进入本方案路径。
- mirror 配置沿用 config 的两类既有 pair，不能混为一个 first consumer：target 内部是 `source1->consumer47, ... source15->consumer33`，因此 target first consumer 是 33、suffix 是 34..47；唯一跨模型 pair 严格为 `source0->physical NextN consumer48`。`num_target_hidden_layers=48`、`num_nextn_predict_layers=1`，只有一层 physical MTP layer，不新增 `welmv4_target_num_hidden_layers` 或 `welmv4_mtp_layer_id`。
- 仅当 `enable_dp_attention=True` 且 `moe_ep_size>1` 时增加 WeLMv4 model-specific 校验：
  - resolved backend 是 DeepEP；
  - ordinary prefill 的 resolved transport 是 NORMAL-AllGather，而不是 AllToAll；
  - runner 构造后确认 normal 通信组覆盖实际 MoE-EP group。
- decode/mirror/verify 的 resolved executor 必须具备 `DP_LOCAL -> EP_FULL -> local sort/initrouting -> EP AllReduce -> DP_LOCAL` 能力；当前 NPU local-EP 实现要求 `moe_tp_size=1`、EP group 覆盖实际专家分片，并且 shared expert 在 EP rank 上是完整副本（否则必须另有显式 shared-TP combine）。若这些能力不满足，按缺失能力报错，不按 TP/DP 数字组合报错。
- 当 `moe_ep_size=1` 时不检查 DeepEP 环境变量，直接选择 pure-TP MoE transport；因此 `TP4+DP4+纯TP4 MoE` 不会被错误拒绝。
- PP、CP、MoE-DP、TBO、elastic EP/EPLB 等仍服从框架已有通用校验；本方案不再为它们编写 model-specific 数字组合白名单。
- 这些检查只约束“WeLMv4 + DP Attention + EP”新路径；不开 DP Attention 时原有配置不受影响。

### 2.3 非目标

- 不重写 NPU Graph backend 的 replay/update 顺序。
- 不新增模型专用 WAR barrier。
- 不修改非 DP 的 DeepEP normal、LL、local-EP、OProj MatMul+RS、stream wait 语义。
- 不把现有纯 TP 非 DP 路径一起迁入新状态机。
- 不在本次顺便优化 DP2/DP4 新 head shape 的 attention 性能；先保证正确性，性能作为后续单独工作。
- 不新增新的模型配置字段；沿用当前 config 和已解析的并行状态。
- 不在首版扩展 online weight reload/remote reload 协议；不能只写一句“reload 后重绑”而漏掉 updater、Graph invalidation 和跨 replica 同步。该组合后续单独接入，首版不在 forward 增加 reload 相关检查。

## 3. 为什么当前代码不能直接解除门禁

当前 `welmv4.py` 中存在三个根本问题。

### 3.1 `o_norm` 前的数据仍可能是 OProj partial

DP Attention 下，RowParallel OProj 的输出在 attention-TP 组内是 partial。`o_norm` 是非线性操作，必须先得到每个 token 的完整 OProj 和：

```text
正确：sum(partial) -> o_norm -> residual add -> post-attention norm
错误：o_norm(partial) -> sum
```

当前 `use_mmq_norm_after_attn` 和 `use_dp_o_norm_after_attn` 条件会重叠，只是依赖 `if/elif` 顺序没有同时执行；这既难读，也容易在重构时破坏数学顺序。

### 3.2 forward mode 不能代表 tensor layout

当前 `has_full_replicated_moe_input` 根据 decode/verify/NextN 等 forward mode 推断输入是 FULL。这个推断在纯 TP 下可能碰巧成立，但 DP Attention 下：

```text
TP_ATTN_FULL = [ab, ab, cd, cd]
EP_FULL      = [abcd, abcd, abcd, abcd]
```

两者完全不同。decode/verify 只能说明“处于哪个阶段”，不能说明 tensor 已经在所有 EP rank 上完整复制。

### 3.3 mirror 的 DP 分支只修改了元数据，没有完成数据搬运

当前 mirror DP 代码存在以下问题：

- 只写当前 rank 的 `global_num_tokens_gpu[dp_rank]`，其他 DP shard 的真实请求数没有同步。
- `scale_seq_factor` 通过 prompt token 数推导 request 数，不成立。
- CPU counts、全局 buffer 长度和其他 rank 的 GPU counts 可能仍是旧的 prompt `T`。
- 修改通用 `global_num_tokens_*` 会同时影响 attention、KV、logits 和 DP gathered buffer。
- 最关键的是，它没有把 DP-local 数据真正转换为 local-EP 需要的 `EP_FULL`。

因此不能继续在现有大 forward 中增加零散条件。

## 4. 统一术语和数据布局

### 4.1 行数记号

- `T_i`：DP shard `i` 的普通 prefill token 行数。
- `B_i`：DP shard `i` 的请求数，也是 decode/mirror query 的真实行数。
- `S`：speculative proposal step 数。
- `D=S+1`：固定 target verify 宽度。
- `V_i`：target verify 的实际行数；固定宽度时为 `B_i × D`。
- `R_i`：draft 阶段实际送入模型的物理行数。当前 `DRAFT_EXTEND_V2` 固定为 `B_i × W`，其中 `W=num_draft_tokens+front_offset`；不能按 accept length 把它压缩成 ragged rows。
- `schedule_mode_i`：scheduler MLP-sync 时 DP shard `i` 的 mode。它描述同步时刻的 `EXTEND/MIXED`、`DECODE` 或 `IDLE`，不冒充 Spec worker 后续生成的 `TARGET_VERIFY`。

当前代码的可达 phase 组合必须分两条链路理解：

```text
非 Spec DP Attention：
  scheduler/target 可出现 {ordinary prefill, ordinary decode, idle}

Spec V2 + DP Attention（speculative_skip_dp_mlp_sync=False）：
  prefill round       = {ordinary prefill, idle}
  scheduler decode    = {DECODE, idle}
  target verify round = {TARGET_VERIFY, idle}
```

Spec V2 中 `TARGET_VERIFY` 是 scheduler 同步完成后由 Eagle worker 把所有非 idle `DECODE` 确定性改写得到的；因此同一次 target forward 不会出现 ordinary prefill、ordinary decode 和 `TARGET_VERIFY` 任意混合。全局 scheduler mode 只用于非 Spec mixed round 的逐 slot 行数解释，以及识别 idle slot；Spec target verify 的物理行数直接使用已经按宽度 `D` 缩放的 `global_num_tokens`。

### 4.2 显式布局

新 DP 执行器只使用显式布局枚举，不再从 forward mode、shape 或旧布尔字段猜测：

```python
class WelmDpLayout(Enum):
    DP_LOCAL_TP_ATTN_FULL = auto()
    EP_SCATTERED = auto()
    MOE_FULL_REPLICATED = auto()
```

含义如下：

- `DP_LOCAL_TP_ATTN_FULL`：每个 rank 只持有本 DP shard 的完整 token 行；同一个 attention-TP 组内存在相同副本。
- `EP_SCATTERED`：外层四个 rank 分别持有互不重复的 token 分片。
- `MOE_FULL_REPLICATED`：进入 MoE 前，collective domain 内各 rank 都持有所有 DP shard 的完整行；该布局也适用于无 EP 的 TP-MoE，不把实现方式写进名字。

Attention 的 OProj partial 是一个短暂内部状态，不允许作为 MoE 输入布局暴露。

### 4.3 运行时状态

```python
@dataclass
class WelmDpLayerState:
    hidden_states: torch.Tensor
    residual: Optional[torch.Tensor]
    hidden_layout: WelmDpLayout
    residual_layout: Optional[WelmDpLayout]
    row_layout: "WelmDpMoeRowLayout"
```

hidden 与 residual 的 layout 必须分别记录；DP gather 后 hidden 可能已经是 `MOE_FULL_REPLICATED`，而不进入 MoE 的 residual 仍是 `DP_LOCAL_TP_ATTN_FULL`。`row_layout` 描述哪些行是真实行、哪些是 padding，不能和任一 tensor layout 合并为一个 `num_token_non_padded` 标量。

当前 WeLMv4 MoE 不消费独立的 `hidden_states_fp32`：router 直接读取 hidden，`torch.mm` 自己产生 FP32 logits。因此首版不创建、不搬运额外 router FP32 副本，也不在状态机中为它增加 layout。以后只有真实 consumer 出现时才扩展该 contract。residual 不进入 MoE 时可以保持 DP-local，但必须在 layer 输出重组时与本地 hidden 精确对齐。

普通 DP+EP non-mirror prefill 是一个明确例外：它的层间 hidden、residual 都是 `EP_SCATTERED`；下一层只把本地 norm 后的 attention hidden AllGather 回 DP-local full rows，residual 在 attention 期间仍保持 scattered，等 attention 输出再次 ReduceScatter 后才重新对齐。新状态机用两个独立 layout 字段表达这一点。

## 5. 能力驱动的校验和执行计划

### 5.1 校验时机

分三级解析/校验，但每一项事实只保留一个责任点，禁止同一条件在参数解析、runner 构造和每层 forward 重复判断：

1. `server_args.py` 的早期 model-specific adjustment 只声明构造前必须生效的 `enable_dp_lm_head=True`；DP/EP/speculative/A2A 全部完成归一化后，再从 `resolved_view()` 做一次 transport 静态校验并给出用户可读错误。禁止在未解析状态提前判断 backend/NORMAL mode。
2. target 通信组建立后、构造 draft `TpModelWorker` **之前**，`EagleDraftWorker` 根据 target 的已知 group handles、scheduler 已经生成的 resolved `draft_server_args` 和 physical MTP config 生成 provisional immutable draft plan，并派生与 draft 模型拓扑一致的 `draft_ps`。这是构造/加载权重所需的先验，不能等 draft `ModelRunner` 建好后再生成，也不能在 Eagle 内第二次调用 `draft_server_args_copy()`；它只在收到的 resolved draft copy 上追加 `welmv4_draft.build` override。
3. target/draft `ModelRunner` 在各自初始化完成后做一次 resolved 校验：实际 group size/rank、权重分片尺寸、OProj reduce contract、local-EP/shared-expert capability 与 plan 一致；不允许静默改写 provisional plan。

校验分工固定如下：

- 启动期晚校验只检查已经解析且无需模型实例即可判断的条件，例如 DP+EP 的 backend、NORMAL AllGather/AllToAll 选择。
- runner/model 构造后只检查必须看实际 group 或模块分片才能确认的契约；成功后把结果冻结进 plan。
- release forward 热路径不重复检查 backend、group identity、模块权重 coverage 或 context 恢复；只保留会随 batch 改变且关系到越界/死锁的必要检查，例如 row count、slot capacity、phase/layout。更细的 tensor-layout 和 collective-owner 断言只在 debug/test 模式启用。
- 已由框架通用校验或 WeLM 现有 mirror/config 校验保证的事实直接复用，不再增加第二套同义门禁。

不能只做第一层，因为 speculative draft 会临时切换 TP group，而全局 `is_dp_attention_enabled()` 不代表 draft 的实际执行拓扑；也不能只 patch 全局 group，因为 `ModelRunner.ps`、loader 和兼容性检查同样读取并行事实。

plan 进入模型构造器的通路必须显式闭环：`TpModelWorker -> ModelRunner.runner_parallel_plan` 使用普通构造参数；`ModelRunner.load_model()` 再在调用通用 loader 的整个期间进入 WeLM 专用、**仅构造期** 的 `welm_runner_build_plan_context(plan)`。WeLM model/DecoderLayer `__init__` 只读取一次该 carrier、存为实例 plan 并据此创建 executor；forward/Graph/compile 不再读取 carrier。这样无需修改通用 model loader 的 kwargs 协议，也不能等 model 已构造/权重已 postprocess 后才注入 plan。

### 5.2 transport 判定

不比较固定 TP/DP/EP 数字组合，直接按能力选择：

```python
class WelmMoeTransport(Enum):
    DP_GATHER_TP_MOE = auto()
    DEEPEP_NORMAL_AG_SCATTERED = auto()
    EXPLICIT_AG_LOCAL_SORT_EP_AR = auto()
    LOCAL_REPLICA_MOE = auto()  # physical MTP layer without EP


def resolve_group_transport(runner_plan, has_ordinary_prefill, layer_id):
    # 只在 _welm_dp_executor 内调用；NON_DP 根本不会进入 transport enum
    if runner_plan.role is TARGET_DP:
        if runner_plan.moe_ep_group is None:
            return DP_GATHER_TP_MOE
        if has_ordinary_prefill and runner_plan.prefill_is_scattered_at(layer_id):
            return DEEPEP_NORMAL_AG_SCATTERED
        return EXPLICIT_AG_LOCAL_SORT_EP_AR

    if runner_plan.role is DRAFT_REPLICA_LOCAL:
        if runner_plan.moe_ep_group is None:
            return LOCAL_REPLICA_MOE
        return EXPLICIT_AG_LOCAL_SORT_EP_AR

    raise AssertionError("DP executor received unsupported runner role")
```

这里使用的 capability 已在 runner plan finalize 时一次性验证；batch planner 只按现有 group-wide `is_extend_in_batch` 和静态 mirror layer plan 选择已验证 handler，不在每层重复 `require(backend/group/weight)`。`is_extend_in_batch` 已由 scheduler MLP-sync 取全组 OR，因此同一 collective domain 的所有 rank 得到相同 transport 和 collective 顺序：

```text
TARGET_DP + no EP                   -> DP_GATHER_TP_MOE
TARGET_DP + EP + 本轮含 ordinary prefill，且当前层仍在 non-mirror prefix
                                         -> DEEPEP_NORMAL_AG_SCATTERED
TARGET_DP + EP + mirror suffix 或普通 decode/target verify round
                                         -> EXPLICIT_AG_LOCAL_SORT_EP_AR
DRAFT_REPLICA_LOCAL + no EP         -> LOCAL_REPLICA_MOE
DRAFT_REPLICA_LOCAL + EP            -> EXPLICIT_AG_LOCAL_SORT_EP_AR
```

静态 finalize 的能力要求是：

```python
if runner_role is TARGET_DP:
    if target_moe_ep_size == 1:
        require(pure_tp_routed_and_shared_contract)
    else:
        require(deepep_backend)
        require(normal_use_allgather and not normal_use_alltoall)
        require(normal_group_covers_moe_ep_group)
        require(local_ep_kernel_available)
        require(shared_expert_is_ep_replicated)
elif runner_role is DRAFT_REPLICA_LOCAL:
    if draft_moe_ep_size == 1:
        require(routed_and_shared_weights_cover_draft_tp_group)
    else:
        require(deepep_backend)
        require(local_ep_kernel_available)
        require(shared_expert_is_ep_replicated)
```

这里 `enable_dp_attention` 只用于启动时决定 target 是否创建 `TARGET_DP` role；进入执行计划后不再参与 transport 分派。尤其 `DRAFT_REPLICA_LOCAL` 的内部 DP flag 固定为 false，但它仍必须选择 `LOCAL_REPLICA_MOE` 或 draft 的 explicit-AG local-EP transport。

这个判定不需要完整 phase vector 来选择 transport。完整 scheduler modes 仅在 **非 Spec** 的 `{prefill, decode, idle}` mixed round 中解释每个 slot 的物理行数；Spec V2 的两类 round 已由 scheduler 契约对齐，不能为理论上不可达的 `prefill + TARGET_VERIFY + ordinary decode` 增加 handler、状态转换或运行时校验。

能力不满足时：

- WeLMv4 + DP Attention + EP：明确打印缺失能力，例如 normal strategy 不是 AllGather、local-EP group 不覆盖专家分片或误开 AllToAll；不打印固定允许组合。
- WeLMv4 + DP Attention、但没有 EP：直接使用 pure-TP transport，不检查 DeepEP。
- 非 WeLMv4：不受这套 model-specific transport 校验影响。
- WeLMv4 但未启用 DP Attention：完全跳过，不影响现有任意 EP backend。

### 5.3 runner role 不能用全局 bool 代替

增加只读执行角色：

```python
class WelmRunnerRole(Enum):
    TARGET_DP = auto()
    DRAFT_REPLICA_LOCAL = auto()
```

- `TARGET_DP`：target attention 本身采用 attention DP，OProj 可能是 attn-TP partial。
- `DRAFT_REPLICA_LOCAL`：draft attention 在自己的 draft-TP group 中运行，但 batch 仍属于某个 target DP shard；它的 MoE 传输不能错误地把 replica-local 当作全局 FULL。

不开 DP Attention 时不创建 runner plan，也不创建 executor，不需要再用一个 `NON_DP` enum 值表达“对象不存在”。`DRAFT_REPLICA_LOCAL` 的 draft 内部 `is_dp_attention_enabled()` 虽为 false，但只要 outer target 使用本方案的 WeLM DP Spec V2，physical MTP layer48 仍按 injected draft plan 构造 `_welm_dp_executor`，以处理 local-replica/EP transport、row-layout 和 target-DP bridge。executor 的创建条件是存在 role 为 `TARGET_DP` 或 `DRAFT_REPLICA_LOCAL` 的 plan，不能退化为读取全局 DP bool。

batch 执行计划只记录不可从其他字段无歧义推出的事实：

```python
attention_finish: AttentionFinishKind
moe_transport: WelmMoeTransport
```

这样 target 和 draft 可以共享布局代码，但不会错误地共享 attention reduction 逻辑。

### 5.4 draft 只有一个并行事实源

provisional draft plan 必须同时包含：

```python
draft_parallel_state: ParallelState
draft_attn_tp_group: ProcessGroup
draft_moe_tp_group: ProcessGroup
draft_moe_ep_group: Optional[ProcessGroup]
outer_dp_bridge_group: ProcessGroup
outer_target_dp_rank: int
outer_target_dp_size: int
process_global_rank: int
world_group: ProcessGroup
```

`draft_parallel_state` 表示 draft 模型自身的构造/加载拓扑：其 `tp_*`、`attn_tp_*`、`attn_dp_*`、`moe_ep_*`、`moe_dp_*` 等所有会被 `ModelRunner` 消费的字段都必须与 resolved draft groups 一致。batch 属于哪个 target-DP shard 则保存在独立的 `outer_target_dp_*` 字段中，不能借用一份仍带 target TP4 值的 `ps` 同时表达两套拓扑。

TP4+DP4+无 EP 的 physical MTP 示例中，draft model 的 TP/attnTP/MoE-TP 都是 1；outer target DP identity 仍是四个 shard 之一。`EagleDraftWorker` 传给 draft `TpModelWorker` 的必须是该 `draft_ps`，不能继续只做 `replace(ps, pp_rank=0)`。但 `draft_ps.tp_rank=0` 只是模型分片 rank，绝不能冒充物理 WORLD rank：四个进程的 seed/object broadcast root、remote/IPC resource identity 和日志 rank 仍分别是各自 `world_group.rank_in_group`。

仅传入 `draft_ps` 仍然不够。当前模块构造和通用层会同时读取 `ModelRunner.ps`、`get_parallel()` live topology、`is_dp_attention_enabled()`、`get_attention_dp_size/rank()`，以及构造期缓存到模块上的 TP/group 字段。首版因此生成不可变 `DraftParallelRuntimeBundle`，并由一个统一、可嵌套、异常安全的 `draft_parallel_runtime_context(bundle)` 在以下整个生命周期内生效：

- draft model/module 构造；
- draft 权重加载、post-load 和共享 embedding/head 绑定；
- attention/MoE backend 初始化；
- Graph capture；
- draft prefill seed、draft decode、draft extend 的 eager 与 replay 前向。

该 bundle/context 必须把所有 draft-facing live 并行事实一次性切到 provisional plan：

```text
TP size/rank/group
Attention-TP size/rank/group
Attention-DP size/rank = 1/0，draft 内部 DP-attention enabled = false
MoE-TP size/rank/group
MoE-EP、MoE-DP size/rank/group
PP/CP/DCP 中 draft_ps 已解析的字段
draft logits/head 采用的 group policy，以及 speculative MoE/A2A backend
物理 WORLD/process identity（只读，不被 draft model rank 覆盖）
```

outer target DP rank/size/group **不覆盖成 draft DP**，而是只通过 `outer_target_dp_*` / `outer_dp_bridge_group` 显式传给 transport；物理 process/WORLD identity 也独立保存且永不由 `draft_ps` 推导。这样 generic draft layer 看见的是一套自洽的本模型拓扑，进程级资源与跨 target shard 桥接又不会丢失。

构造期 raw `server_args`/config-bag 读取不能靠 topology context 修复。scheduler 现有 `draft_server_args_copy()` 已经得到 resolved copy；`EagleDraftWorker` 只通过正式 `ServerArgs.override("welmv4_draft.build", ...)` 在这份 copy 上写入 draft TP/DP/EP、`enable_dp_attention=False`、resolved `enable_dp_lm_head=True` 和 speculative backend 等 role-sensitive 字段。在 `get_context().preserve_config()` 内临时 publish 后，与 runtime context 一起构造/加载 draft worker，退出时恢复 target config。运行期禁止 generic draft forward 再读取 raw target `server_args` 决定 role/layout，统一读取 immutable plan 或构造时缓存的 resolved runner field。

现有 speculative MoE context 不读取临时 publish 的 draft args，而读取初始化阶段已经写入 `MoeFlags.speculative_runner_backend/speculative_a2a_backend`。因此 bundle 创建时必须读取这两个 resolved flag，逐项断言与 draft args/physical layer48 capability 一致；禁止重新调用 `initialize_moe_config(draft_args)` 覆盖 target 状态。之后只由统一 `draft_parallel_runtime_context()` 进入一次两个 speculative MoE context，所有原来在调用点并列进入它们的写法都删除，避免双套嵌套。

这里故意不把 `ContextVar.get()` 加进全局 canonical getter：这些 getter 可能位于 `torch.compile`/Graph-visible 路径，通用查询会造成 graph break 并影响非 DP 模型。`draft_parallel_runtime_context()` 沿用当前 Spec worker 已有的 scoped-global 架构，但补齐全部事实：在首次写入前完成 snapshot，通过一个 `ExitStack` 同时切换 TP、Attention-TP、MoE-TP、必要的 MoE-EP/MoE-DP、Attention-DP rank/size+enabled flag、draft 自己的 offloader，以及现有 speculative MoE runner/A2A context；`finally` 逆序恢复。`_WORLD`、物理 process rank 和 outer bridge 永不 patch。恢复 identity/value 的详细断言只放在 context 单测/debug，不在每次 release forward 执行。

这种 scoped global 只允许当前 scheduler 的单 Python launch thread 使用；scheduler overlap 是 NPU device stream 重叠，不会在另一个 Python thread 同时进入 target forward。context 覆盖每次 draft Python 构造/capture/eager/replay launch，device kernel 提交完即可退出，因为已提交 work 不再读取 Python global。若以后引入真正 host-side target/draft 并发，必须把所有 dynamic getter 改为显式 group/plan 参数后才能开放，不能继续复用本首版 context。

构造/加载完成后由 `ModelRunner.finalize_welm_runner_plan()` 做一次必要的 runner 闭环校验：`ModelRunner.ps`、live group size/rank、QKV/OProj 的实际 TP/reduce policy、MoE TP/EP 分片必须与 draft plan 一致。共享 embedding/head 与 `LogitsProcessor` 不归这里重复验证，唯一 owner 是第5.5/12.4节的 `DraftSharedModulePlan.build()`。不要逐层重复校验等价字段，也不要在每次 draft forward 重做这组检查；context 的异常恢复由 context 单测覆盖。

`welm_runner_build_plan_context` 可使用 model-local `ContextVar`，但只允许在上述 Python 构造/加载边界读取；它与“禁止给 canonical getter 增加 ContextVar”的规则不冲突。进入/退出用 token 恢复，非 WeLM loader 不进入该 context。

### 5.5 NextN 共享参数也必须服从 draft 分片契约

当前 `eagle_worker_v2.py::init_lm_head()` 会把 target 的 embedding/head 对象直接交给 NextN，而 `welmv4_nextn.py::set_embed_and_head()` 实际绑定四类对象：`embed_tokens`、`oe_embed`、replicated `oe_gate_up_proj` 和 `lm_head`。是否可以共享不能由“同一个模型”推断，必须逐模块比较其真实 shard group、vocab coverage、dtype/quant layout 与 draft plan：

- target 的 `embed_tokens` / `oe_embed` 在 DP Attention 下已经按 target attnTP 构造；只有该 group 与 `draft_attn_tp_group` 成员/rank 映射完全一致、每个 draft replica 的 vocab coverage 正确时才直接共享。
- `oe_gate_up_proj` 当前是 replicated；确认每 rank 都是完整权重后可直接共享。
- `lm_head` 只有在 `enable_dp_lm_head=True` 时才按 attnTP 构造；false 时仍是 outer-TP vocab shard，不能交给较小的 draft TP，否则 TP4+DP4 draft TP1 只看到四分之一 vocab。

首版采用现有能力的最小闭环：对 **WeLMv4 + DP Attention + EAGLE/NEXTN Spec V2**，在任何 target/draft module 构造之前把 resolved `enable_dp_lm_head` 自动设为 `True` 并记录一次 info 日志。这不是拒绝门禁，也不比较 TP/DP 数字；它使 target base/OE embedding、LM head、draft TP 和 draft `LogitsProcessor` 全部使用同一个 target-attnTP vocab group，避免新增启动期权重重组和每-token outer-TP 通信：

```text
target_attn_tp_group == draft_attn_tp_group == vocab_group
embed/OE lookup reduce: vocab_group
draft logits gather: vocab_group
outer TP/DP: 只服务 target MoE transport，不参与 draft vocab/logits
```

TP4+DP4 时该 group 退化为 TP1，每 rank 持有完整 embedding/head；TP4+DP2 时每个 DP replica 内由 TP2 保存/收集完整 vocab。数值只是验收例子，解析规则只看 model/runner role 和实际 group identity。

`DraftSharedModulePlan.build()` 是该契约唯一校验 owner：它在绑定前一次性验证 base embedding、每个 OE embedding、LM head 的 `use_attn_tp_group`、真实 shard interval/TP size/rank 与 `vocab_group`，确认 OE gate 是完整 replica，并确认 draft `LogitsProcessor` 的 gather policy；随后生成只读 plan。`set_embed_and_head()` 只消费该 plan，不复验同一组事实。任一失败按 `vocab group/layout capability mismatch` 报错；Graph capture 前必须完成。未来若要允许 resolved `enable_dp_lm_head=False`，再单独实现启动期 head materialize/reshard；首版不在运行时增加 outer-TP head bridge。

## 6. `o_norm` 和 post-attention norm 的唯一正确路径

### 6.1 统一规则

DP 执行器中：

```python
use_mmq_norm_after_attn = self.self_attn.use_o_norm
```

不再定义 `use_dp_o_norm_after_attn`。只要该物理层有效的 `self.self_attn.use_o_norm=True`，就必须且只能调用一次：

```python
mmq_style_norm_after_attn(
    complete_attention_rows,
    matching_residual_rows,
    self.self_attn.o_norm.weight,
    self.post_attention_layernorm.weight,
    self.post_attention_layernorm.eps,
    ...,
)
```

这里的 `complete_attention_rows` 指的是每一行都已经完成 attention-TP 求和，而不是要求所有 token 都复制到所有 rank。

### 6.2 DP+EP ordinary prefill：ReduceScatter 后 MMQ

所有 DP Attention + EP 的普通非 mirror prefill 都需要向 DeepEP NORMAL 提供 `EP_SCATTERED`：

```text
OProj attn-TP partial / DP_LOCAL_TP_ATTN_FULL
    -> attn-TP ReduceScatter
    -> 每个 rank 得到互不重复、数值完整的 token rows
    -> MMQ o_norm + residual add + postnorm
    -> EP_SCATTERED
```

TP4/DP2 时 ReduceScatter 使用 size=2 的 attention-TP group，不是外层 TP4。residual 只做对应的行切片，不参与求和。

TP4/DP4 时 `attn_tp_size=1`，该转换退化为本地 identity。

不能直接复用当前 pure-TP 的 OProj MatMul+ReduceScatter 开关，因为它绑定的是外层 TP group。首版 DP 路径先关闭该 fused 选择，使用明确的 attnTP primitive；以后只有在 fused OProj 能显式接收 attnTP group 且经过数值验证后才允许接回。

这里 hidden 和 residual 在层间的布局必须写全，不能把 residual 留成隐式例外：

```text
本层 attention 输出 partial/full rows
  -> attnTP ReduceScatter（或 attnTP=1 identity）
  -> 对 residual 取完全相同的 row slice
  -> MMQ(hidden_scattered, residual_scattered)
  -> hidden/residual 都是 EP_SCATTERED
  -> DeepEP NORMAL MoE 后仍为 EP_SCATTERED

下一层 prepare_attention：
  -> 在 scattered hidden/residual 上先做本地 input norm
  -> 只把 normalized attention hidden 做 attnTP AllGather
  -> attention hidden = DP_LOCAL_TP_ATTN_FULL
  -> residual 仍为 EP_SCATTERED
  -> attention 结束再次 RS 后与 residual 对齐，再进入 MMQ
```

因此本阶段 `WelmDpLayerState.hidden_layout` 与 `residual_layout` 在 prepare-attention 期间可以不同；所有 shape/layout assertion 必须按 tensor 分别执行。

### 6.3 pure-TP MoE / decode-like local-EP：AllReduce 后 MMQ

无 EP 的 pure-TP MoE，以及有 EP 时 mirror/decode/verify 的 explicit-AG local-EP，都必须先保留每个 DP shard 的完整本地行：

```text
OProj attn-TP partial
    -> attn-TP AllReduce
    -> 完整 DP-local rows
    -> MMQ o_norm + residual add + postnorm
    -> DP_LOCAL_TP_ATTN_FULL
```

随后才执行 DP gather。绝不能先 DP gather partial 再对错误布局做 norm，也不能对已经 AllReduce 的副本再 ReduceScatter 求和。

### 6.4 `o_norm=False`

如果以后某个同架构 checkpoint 的 `o_norm=False`：

- 保留显式的 layout transition；
- post-attention norm 可以复用现有 communicator 数学；
- 仍不允许从 forward mode 猜布局。

首版的核心验收对象是当前 `o_norm=True` 配置。

`residual_after_layernorm` 仍决定 input norm/residual 的生成方式，但不再决定是否使用 MMQ。每层有效的 `self_attn.use_o_norm=True` 时恰好调用一次 MMQ；全局 config 的 `o_norm=true` 不代表 prenorm layer1 的有效值也为 true。Attention.forward 在该分支必须 `skip_o_norm=True`，不能再有 explicit o_norm fallback。

MMQ 调用当下只要求 complete attention rows 与 matching residual rows 具有相同 shape/layout。MMQ 后的布局取决于 transport：`DEEPEP_NORMAL_AG_SCATTERED` 中 hidden/residual 都是 `EP_SCATTERED`；`DP_GATHER_TP_MOE` 和 `EXPLICIT_AG_LOCAL_SORT_EP_AR` 中 hidden/residual 是 DP-local，随后只 gather MoE hidden，residual 保持 DP-local。`return_fp32_out` 只服务于现有 residual/norm 数学契约，不为 router 额外制造状态。

### 6.5 为什么现有非 DP 路径不一起强制改成 MMQ

用户要求的“`o_norm=True` 必须走 MMQ”落实在新的 DP 执行器中。非 DP 当前已经有经过验证的 PPLN、OProj MatMul+RS 和 fused norm 数值顺序；本次把它也重写会扩大回归面。因此：

- DP：effective `use_o_norm=True` 时唯一 MMQ 分支且恰好执行一次；effective false 的 prenorm 层不伪造 o_norm。
- 非 DP：保持当前精确行为。
- 如果以后要统一非 DP，也应作为独立提交和独立精度/性能验收进行。

## 7. 无 EP：`DP_GATHER_TP_MOE`

### 7.1 通用数据流

当 `moe_ep_size=1` 时，不检查 DeepEP backend 或环境变量，也不按 TP/DP 数字组合分支。MoE 权重由实际 pure-TP MoE group 分片，统一流程为：

```text
DP_LOCAL_TP_ATTN_FULL(real_rows_i)
  -> input norm / QKV / attention，KV cache 始终 DP-local
  -> OProj partial（若 attnTP=1 则已完整）
  -> attnTP AllReduce；attnTP=1 时 identity
  -> MMQ o_norm + residual add + postnorm
  -> row-layout-aware DP gather；每个 DP shard 只贡献一个 attnTP 副本
  -> OUTER_TP_FULL=[slot0+slot1+...]x moeTP
  -> router + TP-sharded routed/shared experts
  -> Qwen2MoeSparseMoeBlock 内部把 routed/shared partial 相加
  -> 由 MLP block 在 resolved moeTP group 上恰好一次 MoE-combine AllReduce
  -> 按 row-layout 取回本 DP shard
  -> DP_LOCAL_TP_ATTN_FULL(real_rows_i)
```

关键约束：

- 不能走 DeepEP dispatch、initrouting 或 local-EP。
- residual 始终保持本 DP shard 的 local layout；只 gather MoE 实际消费的 hidden。
- shared expert 和 routed experts 都是 pure-TP partial 时，应先相加，再做一次 outer-TP combine。
- 首版明确把这次 combine 的唯一责任方设为现有 `Qwen2MoeSparseMoeBlock.forward()`：DP executor 传入 FULL rows、保持 `use_reduce_scatter=False`，MLP block 先相加 routed/shared partial，再在 resolved MoE-TP group 上执行一次 MoE-combine AllReduce；executor 随后只按 `row_layout.slot_offsets` slice，绝不能再做第二次 MoE combine。DP replicate-gather 内部可能使用的 AllReduce 不属于这个计数。
- 构造期断言 routed `experts.reduce_results=False`、shared expert 的 row-parallel down-proj 不自行 reduce，且 `Qwen2MoeSparseMoeBlock.tp_size/group` 与 plan 的 `moe_tp_group` 一致；否则“block 内唯一 combine”不成立。
- eager/SUM_LEN 使用上述 AllReduce + local slice；不同 DP shard 行数不等或某个 shard idle 时仍然正确。Graph/MAX_LEN 首版也沿用同一数学路径。
- `reduce_scatterv` 或 Graph 等长 ReduceScatter 仅作为后续性能优化；接入时必须把 combine owner 从 MLP block 显式切换给 executor，并验证 trace 中仍只有一次 collective，不能在首版同时保留两条 finalize。

planner 忽略 DeepEP 环境还不够，模型内部也不能继续用全局 `moe_a2a_backend.is_deepep()` 决定 shared-expert 分片或是否跳过 block 尾部 combine。DP executor 将冻结 runner plan 的 `moe_ep_group` 与 batch 的 `moe_transport` 作为只读构造/执行契约传给 MoE block：`moe_ep_group is None` 时 shared expert 按 resolved MoE-TP 分片并执行上述 combine，即使环境里残留 DeepEP backend 也不改变；不开 DP Attention 的 `_forward_non_dp()` 仍保持原 backend 条件。

### 7.2 `attnTP>1` 示例：TP4+DP2+纯 TP4 MoE

```text
rank0/1: DP0，attention-TP 内持有同一批 rows
rank2/3: DP1，attention-TP 内持有同一批 rows
模型层输入/输出：[ab, ab, cd, cd]
MoE FULL 输入：  [abcd, abcd, abcd, abcd]
```

当前 config 的全局 `24` 个 Q heads、`2` 个 KV heads 在这个示例中得到本地 `q_heads=12, kv_heads=1`。DP gather 必须去除 attnTP2 的副本；不能把四个 rank 都当成独立 DP shard，否则 token 和 MoE 输出都会重复计数。

若以后使用 `reduce_scatterv` 优化 outer-TP combine，该示例的 sizes 是 `[slot0/2, slot0/2, slot1/2, slot1/2]`，随后仍需在各 attnTP2 group 内恢复同一 DP shard 的副本。

### 7.3 `attnTP=1` 示例：TP4+DP4+纯 TP4 MoE

```text
rank0: DP0
rank1: DP1
rank2: DP2
rank3: DP3

attention 输入/输出：[a, b, c, d]
MoE FULL 输入：     [abcd, abcd, abcd, abcd]
```

这个配置是本方案明确支持的验收项。attention 没有 TP collective：OProj 已是完整行，attention finish 为 identity，随后直接执行 MMQ。DP4 gather 中四个 rank 各贡献一个唯一 slot；TP4 routed/shared experts 的 partial output 在 MLP block 内只做一次 TP4 AllReduce，再由 executor 按自己的 `slot_offsets` 取回本 DP shard。Graph 仍使用四段固定 slot 和 segmented valid mask，不能把 padding 行送入 router。

### 7.4 各阶段行数

- ordinary prefill：`T_i`。
- target first mirror consumer layer33：从 `T_i` 精确切换为 `B_i`，不能使用 `T_i / scale`。
- target mirror suffix layer34..47/decode：`B_i`。
- target verify：`V_i`，固定宽度时 `B_i × D`。
- draft/draft extend：使用真实 `R_i`，但 transport 按第12.3节从 draft 实际权重分片和 group 独立解析，不能盲用 target 的 pure-TP transport。

target 的 ordinary prefill、mirror、decode、verify 使用同一 `DP_GATHER_TP_MOE`，只替换 row layout；draft 是独立解析的 runner。

## 8. 有 EP 的 ordinary non-mirror prefill：`DEEPEP_NORMAL_AG_SCATTERED`

### 8.1 通用数据流

同时开启 DP Attention 和 EP 时，普通非 mirror prefill 固定使用 DeepEP NORMAL AllGather，而不是 AllToAll 或 decode-like local sort：

```text
DP_LOCAL_TP_ATTN_FULL(T_i)
  -> attention / OProj partial
  -> attnTP ReduceScatter；attnTP=1 时 identity
  -> MMQ norm on complete, non-duplicated rows
  -> EP_SCATTERED
  -> DeepEP NORMAL AllGather + initrouting + experts + combine
  -> EP_SCATTERED
  -> 下一层 attention 前 attnTP AllGather；attnTP=1 时 identity
  -> DP_LOCAL_TP_ATTN_FULL(T_i)
```

这一路保持既定的 normal AllGather 语义。shared expert 多流只能与 normal dispatch 中的通信窗口并行，wait 位置保持现状；本方案不新增 gate 多流。

### 8.2 `attnTP>1` 示例：TP4+DP2+EP4

rank0/1 属于 DP0，rank2/3 属于 DP1；每个 rank 只拥有 EP4 中自己的 local experts，`moeTP=1`。当前全局 Q/KV heads 对应本地 `q_heads=12, kv_heads=1`。attnTP2 ReduceScatter 同时完成 OProj partial 求和和副本去除；下一层前的 attnTP2 AllGather 恢复本 DP shard 完整行。

### 8.3 `attnTP=1` 示例：TP4+DP4+EP4

四个 rank 分别是 DP0..DP3，当前全局 Q/KV heads 对应本地 `q_heads=24, kv_heads=2`。因为每个 rank 原本就持有唯一、数值完整的本地行，attention finish 和下一层 restore 都退化为 identity；这些行天然满足 `EP_SCATTERED`，直接进入 DeepEP NORMAL AllGather。

## 9. 有 EP 的 mirror/decode/verify：`EXPLICIT_AG_LOCAL_SORT_EP_AR`

### 9.1 首个 target mirror consumer

target 内部首个 consumer 是 layer33（对应 source15），不是跨模型 physical consumer48。它发生唯一一次 target 语义行数转换：

```text
prompt rows T_i -> request rows B_i
```

上一层 ordinary prefill 输出的 hidden/residual 仍是 `EP_SCATTERED`。layer33 必须先在本地 scattered rows 上完成 input norm，保留同布局的 FP32 residual；只 AllGather normalized attention hidden，不能先 gather hidden 后再拿 scattered residual 做 norm：

```text
EP_SCATTERED prompt input from layer32
  -> 对本地 scattered hidden/residual 做 input norm
  -> 只对 normalized hidden 做 attnTP AllGather，恢复本 DP shard 的 prompt rows
  -> custom_last_index 选择 B_i 个 query rows
  -> 从本 shard 的 mirror K/V 对 T_i 长度上下文做 attention
  -> OProj partial
  -> attnTP AllReduce；attnTP=1 时 identity
  -> 从仍为 scattered 的 FP32 residual 按 owner 精确重建 B_i 行
  -> MMQ norm
  -> DP gather，并去掉 attnTP 内的重复副本
  -> EP_FULL=[B_0+B_1+...]x EP
  -> local sort + initrouting + local experts
  -> EP AllReduce routed output
  -> add replicated shared-expert output once
  -> 按 row layout 取回本 DP shard
  -> DP_LOCAL_TP_ATTN_FULL(B_i)
```

这里的 AllGather 是显式 DP-row gather：先形成 `MOE_FULL_REPLICATED`，再执行 local sort/initrouting；它不是 ordinary prefill 的 DeepEP NORMAL dispatcher。

### 9.2 mirror suffix、decode、target verify

target layer34..47 已经接收 layer33 产生的 B 行；decode 与固定宽度 target verify 分别使用 `B_i` 和 `V_i=B_i×D`。它们统一执行：

```text
DP_LOCAL_TP_ATTN_FULL(real_rows_i)
  -> attention
  -> attnTP AllReduce；attnTP=1 时 identity
  -> MMQ norm
  -> deduplicating DP gather -> MOE_FULL_REPLICATED
  -> local sort + initrouting + local experts
  -> EP AllReduce -> shared output add once
  -> DP local slice
  -> DP_LOCAL_TP_ATTN_FULL(real_rows_i)
```

local-EP 形成闭环后，不应再把输出伪装成 `EP_SCATTERED` 交给 generic communicator。

该闭环的“EP AllReduce 后只加一次 shared output”有一个硬前提：shared expert 必须是每个 EP rank 上的完整副本。当前 DeepEP 构造契约满足这一点；若 resolved module 显示 shared expert 仍按 draft/outer TP 分片，则必须先实现显式 shared-TP combine，否则该 transport capability 不成立并应报错，不能直接相加 partial shared output。

### 9.3 DP2/DP4 差异只是 collective 退化，不是分派白名单

- TP4+DP2+EP4：attnTP2 必须先 AllReduce，DP gather 时去掉同 shard 的第二份副本。
- TP4+DP4+EP4：attnTP1，attention collective 退化为 identity，四个 rank 各贡献自己的唯一 DP slot。
- 两者都由实际 `attn_tp_group`、EP group 和 row-layout 驱动；代码中不得出现 `if dp_size == 2/4` 来决定是否支持。

### 9.4 attention/RoPE 的 P0 检查

当前 `welmv4_sink_prefill_attention.py` 的通用 LPT 路径按 `num_q_heads/num_kv_heads` 参数化；若干小 M 优化分支只允许 `q_heads=6, kv_heads=1`。因此：

- DP2 的 `12/1` 和 DP4 的 `24/2` 应回退通用路径，不能误入 6/1 专用 kernel。
- 首版不扩大专用 kernel 的 constexpr 范围。
- full attention、SWA、prefill、verify、decode、Graph capture/replay 都必须做正确性测试。
- RoPE 当前也是参数化实现，但必须验证 `12/1` 和 `24/2`；不能因为 shape 可构造就视为已验收。
- 性能若明显下降，后续单独增加 NPU kernel 优化，不与本次布局修复混合。

## 10. mirror 的精确 metadata

### 10.1 同步真实请求数

在现有 `MLPSyncBatchInfo` 同一轮 all-gather payload 中增加 `num_requests`：

```text
[num_tokens, num_tokens_for_logprob, num_requests]
```

得到：

```python
global_num_requests_cpu = [B_0, B_1, ...]
```

不新增第二次 collective，不使用 `num_tokens_for_logprob` 代替请求数，也不在模型 forward 中调用 `.tolist()` 做 D2H。首版不再增加一份通用 `ForwardBatch.global_num_requests_gpu`：

- eager ordinary prefill 需要的 request-row device tensor 由 WeLM row metadata 在 ForwardBatch 边界一次性创建并持有；
- ordinary decode/target verify 的物理 row counts 已分别存在于现有 `global_num_tokens_gpu`（`B_i` / `V_i`）中；
- target Graph 的请求 bucket 已由现有 `original_global_num_tokens_cpu` 选择，不需要第二份 Graph request-count buffer。

### 10.2 独立 row-layout

新增 ForwardBatch-owned metadata：

```python
@dataclass
class WelmDpMoeRowLayout:
    real_rows_cpu: list[int]
    real_rows_gpu: torch.Tensor
    slot_rows_cpu: list[int]
    slot_offsets_cpu: list[int]
    local_real_rows: int
    local_slot_rows: int
    global_slot_rows: int
    slot_stride: Optional[int]
    padding_mode: DpPaddingMode
```

它不能覆盖：

- `global_num_tokens_cpu/gpu`；
- `global_num_tokens_for_logprob_*`；
- `num_token_non_padded`；
- attention/KV 的 seq-len metadata；
- logits gather metadata。

普通 prompt、mirror request、ordinary decode、target verify 和 draft 各自选择自己的物理 row view；同一个 ordinary-prefill ForwardBatch 在 first mirror consumer 前选择 `T_i`，从 first mirror consumer 起选择 `B_i`。这个选择只切换预先构造的 counts/offsets view，不在每层分配 tensor。

`MLPSyncBatchInfo` 已同步每个 DP shard 的 `local_forward_mode`；新实现把它作为 **scheduler-time** `global_schedule_modes` 发布。它和 `global_num_tokens/global_num_requests` 的使用规则固定为：

- 非 Spec mixed round：逐 slot 的 `EXTEND/MIXED` 使用 `T_i`，`DECODE` 使用 `B_i`，`IDLE` 使用 0；这是保留完整 modes vector 的唯一原因。
- Spec prefill round：只可能是 ordinary prefill/idle，使用 `T_i/0`。
- Spec target verify round：scheduler-time `DECODE` 的所有非零 slot 都确定性解释为 `TARGET_VERIFY`，直接使用已按 `D` 缩放的 `global_num_tokens=V_i`；不得把同步值声称为已经是 `TARGET_VERIFY`。
- transport 不读取逐 shard modes vector；它只读取 group-wide `is_extend_in_batch` 与静态 mirror layer plan。这样既支持非 Spec mixed prefill/decode，又不为 Spec 中不可达的四相组合建立状态机。

对 DP+EP，layer33 的 scattered-to-local transition 只需要覆盖真实可达输入：

- Spec prefill round：ordinary prefill shard在 scattered hidden/residual 上本地 input norm，gather normalized hidden，再按 `custom_last_index` 选 `B_i`，residual 用 owner 规则重建 `B_i`；idle shard构造 typed empty并参加统一 collective。
- 非 Spec mixed round：ordinary prefill shard执行同一 `T_i->B_i` 转换；ordinary decode shard保持 `B_i` 语义并从自己的 scattered slot 恢复 matching hidden/residual；idle shard为 0 行。
- Spec target verify round从 layer0 起就是 explicit-AG local-sort transport，输入为 `V_i`，不会从 ordinary-prefill scattered prefix 切换过来。

若 mirror 关闭且非 Spec mixed group 一直使用 NORMAL 到 layer47，target 发布边界仍要分别恢复 ordinary prefill 的 `T_i` 和 ordinary decode 的 `B_i`。Spec V2 不增加同时恢复 prefill/verify 的分支，因为该组合在支持的 scheduler 契约下不可达。

### 10.3 first mirror residual

这里的 first mirror 指 target layer33。它的 query row 来自每个请求最后一个 prompt token，residual 可能仍是 attnTP-scattered FP32：

- 根据 `custom_last_index` 判断 owner attnTP rank；
- owner rank取出对应 residual，非 owner写零；
- 在 attnTP group 中求和，重建精确 `[B_i, hidden]` FP32 residual；
- 再进入 MMQ fused norm。

不能用全局 prompt index 直接索引某个 rank 的 local residual。

## 11. Graph padding

### 11.1 固定 slot，而不是“前 N 行有效”

Graph 中全局 staging buffer 布局为：

```text
[dp0_real, dp0_pad, dp1_real, dp1_pad, ...]
```

- `slot_rows` 在 capture 时固定。
- replay 只更新物理 real rows。target decode/verify 直接复用 Graph 中现有的 `global_num_tokens_gpu` 固定地址，不再注册第二份等价 buffer。
- TP4/DP2 时，decode/verify slot 按当前 DP Graph buffer 约束满足 attnTP2 对齐，避免 gather/scatter primitive 出现不一致 shape。ordinary prefill 首版保持 eager，其 SUM_LEN/MAX_LEN 对齐沿用现有 DP padding 结果，不纳入 prefill Graph capture。
- TP4/DP4 时 `attn_tp_size=1`，无需额外对齐。

### 11.2 scalar `num_token_non_padded` 不够

单个标量只能表达：

```text
[real, real, real, pad, pad]
```

不能表达 DP slot 中间有洞的情况。因此 MoE routing 使用 segmented valid mask：

```python
dp_id = row // slot_stride
row_in_slot = row % slot_stride
valid = row_in_slot < real_rows_gpu[dp_id]
```

所有 transport 的无效行都必须满足：

```text
topk_weight = 0
```

expert id 按 transport 生成：当前 Ascend local-EP/NORMAL dispatcher 已验证负 ID，可用 `id=-1, weight=0`；pure-TP Ascend dispatcher 没有同等负 ID 契约，`DP_GATHER_TP_MOE` 使用合法 sink expert ID（例如 0）并令 weight=0，或在接入前为 pure-TP dispatcher补齐并验证负 ID 能力。普通 attention 仍保留自己现有的 scalar metadata；不要为了 MoE mask 改写 attention 的 `num_token_non_padded`。

### 11.3 collective 序列必须 rank-uniform

所有 rank，包括本地 `real_rows=0` 的 idle rank，都必须按照全局 execution plan 执行同样的 collective 序列和固定 shape。禁止：

```python
if hidden_states.shape[0] > 0:
    collective()
```

也禁止根据本 rank 的 forward mode 或真实请求数选择不同通信分支。

idle rank 在进入 collective 前跳过纯本地 attention、OProj 和 MMQ（这些算子不保证零 grid 合法），直接构造 shape/dtype/layout 正确的空本地结果；随后仍与其他 rank执行完全相同的 gather/dispatch/combine collective。

### 11.4 Graph 独立性

以下 Graph 继续拥有各自独立的 runner、静态输入和命中判断：

- target verify Graph；
- draft decode Graph；
- draft extend Graph。

允许 target 命中而 draft eager fallback，也允许相反组合。eager 和 Graph 必须共用相同 row-layout 语义。这里不要求为每个 runner再新增一个命名为 `real_rows_gpu` 的物理 tensor：target 直接把现有 `global_num_tokens_gpu` 作为 row-layout 的 Graph-resident real rows；draft runner 只有在 resolved transport 确实包含 outer/EP collective 时，才把其已有 global-row buffer 扩展为相同用途。

这里的“独立”只指 target 与各 draft runner 之间可以独立命中；**同一个 target collective domain 内不能各 rank 独立选 bucket**。现有 target Graph key 的 batch-size 单位是“请求数 B”，不能与模型内部物理 token rows 混用：

```text
decode:        request_rows_i=B_i，K=1，physical_rows_i=B_i
target verify: request_rows_i=B_i，K=D，physical_rows_i=V_i=B_i×D

B_graph = common request-count bucket selected from synchronized B_i
slot_rows/slot_stride = B_graph×K
global_slot_rows = outer_dp_size×B_graph×K
```

当前 `decode_cuda_graph_runner.py::can_run_graph()` 已在 `require_mlp_tp_gather` 下使用 `max(original_global_num_tokens_cpu)` 选择请求 bucket；对 target verify，`original_global_num_tokens_cpu` 正是同步前的 `B_i`，而 `global_num_tokens_cpu/gpu` 才是缩放后的 `V_i`。因此这里是一个必须保留并补测试的既有契约，不重新实现 Graph key 选择，也不用新增 `global_num_requests_gpu`：

```text
existing original_global_num_tokens_cpu -> B_graph
existing global_num_tokens_gpu          -> physical B_i or V_i rows
captured_req_width                       -> K=1 or D
```

禁止先用 `V_i` 选 bucket 后又乘 `D`。target decode/verify Graph 内包含 outer-TP/EP collective，所有参与 rank 继续依赖现有同步 metadata 与 group-wide `can_run_dp_cuda_graph` 得到同一个 key 和 replay/fallback 结论。

不新增一套 Graph candidate/success manifest。当前 runner 的 `capture_bs` 来自同一 resolved ServerArgs，各 rank捕获顺序本来一致；初始化失败继续沿用框架现有语义。运行期复用 scheduler MLP-sync 已经 all-gather 的 `original_global_num_tokens_cpu` 和 group-wide `can_run_dp_cuda_graph=min(local_can_run)`，只为 WeLM row-layout补充 request counts/slot metadata，不再增加第二次同步 collective。

```text
synchronized row counts
  -> deterministic target graph key/bucket
  -> all ranks 用现有同步 metadata 和 group-wide can-run 得到同一决策
  -> all ranks replay the same target variant，或全部 eager fallback
```

- idle rank 也选择同一 target variant、以同一静态 collective shape 参与 replay。
- Graph padding 开启时由 `max(B_i)` 选择共同请求 bucket `B_graph`，再按本 phase 的固定 `K` 计算 token slot；padding 关闭时要求 `max(B_i)` 本身存在 exact captured key。其他 rank 的较小 `B_i` 仍放入同一个固定 slot并由 valid mask 屏蔽，不要求所有 `B_i` 相等。
- 任一 rank 按既有 group-wide 条件 miss 时，整个 target group 本轮回退 eager；禁止部分 rank replay、部分 rank eager。WeLM 的 backend/group/weight capability 已在 runner finalize 时静态冻结，不再发明一组每轮本地 Graph eligibility，也不扩展 scheduler payload。
- fallback 决策复用现有 captured keys、同步 row metadata 和 `can_run_dp_cuda_graph`，不新增运行时 collective。
- `LOCAL_REPLICA_MOE` 且不含 outer collective 的 draft Graph 仍按自己的 `draft_tp_group` 和本地物理 rows 选 bucket。只有 resolved draft transport 含 outer/EP collective 时，draft runner 才复用 outer scheduler 已同步的 request counts和 group-wide can-run，统一该 group 的 bucket/fallback；draft decode、draft extend、target verify 三者仍互不绑定。

不修改 `npu_cudagraph_backend.py` 的 replay/update 线程顺序。

## 12. MTP / Spec V2

### 12.1 不新增独立 worker

继续接入现有 `eagle_worker_v2.py` / Spec V2 体系，不新增 WeLM 专用 worker。WeLM 的特殊性放在模型 DP executor、row-layout 和 ngram/KV mirror metadata 中。

### 12.2 target verify

target verify 使用 `TARGET_DP` role：

- 固定宽度时 `V_i=B_i×D`，`D=S+1`。
- Graph slot 是 `B_graph×D`。
- ragged 模式若以后开放，必须直接携带 `V_i`，不能继续按固定公式推导。
- target 每层按 runner plan 已解析的 `moe_transport` 执行；无 EP 为 `DP_GATHER_TP_MOE`，有 EP 为 `EXPLICIT_AG_LOCAL_SORT_EP_AR`。

### 12.3 draft 与 draft extend

`draft_tp_context()` 当前只替换 TP group，“全局 enable_dp_attention=true”、Attention-DP rank/size、MoE-TP 等 target runtime 事实仍然存在，而且传入 draft `TpModelWorker` 的 `ps` 仍可能描述 target TP。因而不能在模型中用全局 bool 判断 draft 是否要走 target 的 attention reduction，也不能把 context patch 当成 `draft_ps` 的替代品。按第5.4节升级后，draft context 必须接收完整 `DraftParallelRuntimeBundle` 并一次性 scope 所有相关事实；draft 内 `is_dp_attention_enabled()` 为 false、Attention-DP 固定 1/0，跨 target-DP 的通信只允许使用 plan 中显式保存的 bridge group。

draft 使用 `DRAFT_REPLICA_LOCAL` role；这个名字表示“batch 属于一个 target-DP shard”，不表示 draft attention 自己再次启用 DP：

- attention reduction 依据 draft runner 实际 OProj 契约决定；若 draft attention 已在 draft-TP group 内完成 TP reduce，就不能重复 AllReduce。
- batch 仍然属于 target 的某个 DP shard，但 draft MoE transport 必须依据 **实际 speculative backend、实际 group 和 physical MTP layer 的权重分片** 单独解析，不能复制 target transport。
- draft decode 使用实际 draft rows。
- 当前 `prepare_for_draft_extend()` 构造固定 `num_window_tokens=W`，物理输入和 MoE rows 均为 `B_i×W`，`extend_lens`/`extend_seq_lens` 每项都是 `W`。不能按 accept length 或“有效接受 token 数”缩成 ragged rows；Graph padding 的无效请求另由 segmented valid mask 屏蔽。
- prefill 后的 draft seed 使用每请求一行的真实 `B_i`。

draft runner 不使用 target 数字组合派生表，而是根据 physical layer48 的实际 EP 状态选择首版的两条能力路径：

| resolved draft capability | 条件 | draft MoE transport |
|---|---|---|
| `LOCAL_REPLICA_MOE` | physical MTP layer 无 EP；routed/shared expert 都按 draft-TP group 构造和加载 | 所有 `R_i` 在本 target-DP replica 内完成，不跨 target-DP gather |
| `EXPLICIT_AG_LOCAL_SORT_EP_AR` | physical MTP layer 实际启用 EP，且满足 DP+EP local-sort 能力 | 显式 gather 到 outer EP group，local sort/initrouting/local experts，EP AllReduce 后切回原 shard |

这里必须修复当前 `draft_tp_context()` 只 patch 普通 `_TP`、不 patch `_MOE_TP`，且没有同步 DP runtime/config context 的隐患。首版在 **无 EP 的 physical MTP layer** 上明确令 `draft_moe_tp_group == draft_attn_tp_group`；在构造 draft worker 前先派生匹配的 `draft_ps`，并在模型构造、权重加载、attention/backend 初始化、Graph capture 和每次 eager/replay launch 的完整 draft runtime context 中保持 group、live getter 与 `draft_ps` 一致：

- target 为 TP4+DP2 时，每个 draft TP2 group 各自拥有一份完整 physical layer48 replica，MoE 在组内按 TP2 分片。
- target 为 TP4+DP4 时，draft TP1，physical layer48 的 routed/shared experts 都在每个 rank 完整复制，因此 MoE 完全本地。

不能只因为 draft attention TP 为 1 就认定 routed expert 已经完整；必须由构造和 loader 使用的 group 证明。否则会出现 shared expert 按 TP1 完整、FusedMoE routed expert 却仍按 outer TP4 partial 的混合契约，产生静默精度错误。

如果未来要让无 EP 的 physical MTP layer 继续按 outer TP group 分片，必须另行实现 `DP_GATHER_TP_MOE` draft bridge、outer-TP combine 和 shared-output 去重；首版不静默进入这条未实现路径。实际 group/权重 coverage 不一致时按内部 capability 缺失报错，这不是 TP/DP 数字白名单。

`draft_tp_context()` 必须同时发布 draft attention-TP 和上述一致的 draft MoE plan。因而 draft 的 DP-local -> EP_FULL gather 和回取绝不能隐式调用当前 `get_tp_group()`：

- attnTP reduction/副本去重使用 plan 中的 `draft_attn_tp_group`；
- local-replica MTP 使用 plan 中的实际 `draft_moe_tp_group`；跨 target-DP shard bridge 或 EP local-EP 使用显式保存的 `outer_dp_bridge_group` / `draft_moe_ep_group`；
- 所有 transport primitive 都显式接收 group handle，不能依赖 context 中的全局 getter。

在 draft worker 初始化完成后，必须校验：

- draft TP group 的 size/rank；
- speculative MoE A2A backend；physical MTP layer 的 `moe_ep_size>1` 时它必须解析为 DeepEP，不能用 `none` 构造一组不完整 local experts。
- draft MoE 的 EP/TP 权重布局；
- routed expert 与 shared expert 是否属于同一个完整/分片契约；
- 选择 `EXPLICIT_AG_LOCAL_SORT_EP_AR` 时 shared expert 是否为 EP-replicated；若不是，是否存在已验证的显式 shared-TP combine。首版没有后者时必须拒绝。
- OProj 是否返回 partial；
- 哪个外层 DP shard拥有该 draft batch。

不允许从 target 的全局参数猜这些值。共享模块不在本清单复验，统一交给下一节的 `DraftSharedModulePlan.build()`。

### 12.4 NextN 共享 embedding/head 的绑定时序

在 target/draft module 构造前已经按第5.5节把 `enable_dp_lm_head` 解析为 true。`init_lm_head()` 仍不能无条件调用 `set_embed_and_head(target_objects)`：它先执行逐模块 shard-contract 校验，然后形成不可变 `DraftSharedModulePlan`，记录统一的 attnTP vocab group、vocab start/end、replica id 和 weight version。

该 plan 在 draft model/loader 初始化完成后、attention backend/Graph capture 前校验并绑定 direct-shared 对象；后续 eager/replay 只读，不在 forward 中动态换对象。首版不声称支持该组合的 online weight reload，也不添加半套 updater/Graph invalidation hook。这样 pure TP、EP、Graph 开关和 overlap 开关看到的是同一组启动期绑定参数，不会因执行模式改变 logits shard。

### 12.5 target 到 draft 的 hidden 边界

target layer47 必须先完成本层 MoE combine、恢复本 DP shard 的 local rows，再执行最终 residual/norm 和 hidden 发布；但不能把所有 phase 的行数都强制写成 `B_i`。layer47/LogitsOutput 的正确边界是：

| target phase | layer47 / target model 输出行数 | 何时变成每请求一行 |
|---|---:|---|
| mirror prefill seed（layer33 已收缩） | `B_i` | 已在 target 内完成 |
| mirror 关闭的 ordinary prefill | `T_i` | 保持 token rows；由 target 顶层现有 last-token/logits 逻辑处理，DecoderLayer 不提前压缩 |
| decode | `B_i` | 本来就是每请求一行 |
| `TARGET_VERIFY` | `V_i=B_i×D` | target 返回后，由 Spec V2 worker 按 accept/select index 选择为 `B_i`，再构造下一轮 draft input |

因此边界 invariant 是“`DP_LOCAL_TP_ATTN_FULL(current_phase_rows_i)`”，不是恒定 B 行：

- 不能发布 MoE gather 后的 global FULL rows；
- 不能发布 outer-TP partial components；
- target verify 不能在 layer47 提前丢弃 `D` 个候选的行，否则 verify logits、`accept_index` 与 hidden/KV 会错位；
- worker 完成接受选择后，交给 draft 的 hidden 才必须与每请求一行的 token/ngram seed 一一对应。

这条边界断言对 TP4+DP4+纯 TP4 MoE 尤其重要：TP4 MoE 的 AllReduce+local slice 必须发生在 target 内部，不能把 combine 延后到 physical MTP layer48；但 local slice 仍保留当前 phase 的 `B_i` 或 `V_i`。

### 12.6 ngram embedding 与 KV mirror

- target in-model mirror 由第9节处理：layer33 完成 target 的 `T_i->B_i`，layer34..47 保持 B 行。
- 唯一 cross-model pair 是 source0->physical consumer48。source0 写入 `model_specific_states` 的 mirror K/V 行数跟随 target phase：prefill seed 是本 shard `T_i`，target verify 是 `V_i`，普通 decode 是 `B_i`；这些 K/V 始终保持 target-DP local，不跨 DP gather。
- prefill seed 交给 NextN 的 main hidden 已经过 layer33 收缩，是 B 行；target verify 则必须先保留 V 行，等 Spec V2 worker 用现有 `mirrored_kv_indices` / `accept_index` 同步选择 hidden 与对应 mirror K/V 后，才形成下一轮每请求 B 行的 seed。不能用 reshape、除以 D 或 Graph padded shape 代替索引选择。
- consumer48 不得再次对已经由 worker 选成 B 行的 main hidden 做 target first-mirror transition。
- NextN 的 token/OE embedding 在 prefill seed 中先按 T 行构造，以保留正确 ngram 历史，再由 `welmv4_nextn.py` 的 `custom_last_index` 收缩到 B 行；它与 B 行 target hidden 拼接后，使用本 shard source0 的 T 行 external mirror K/V 做 attention。
- consumer48 attention 输入是本 draft replica 的 DP-local B 行，不从 `EP_SCATTERED` 再做 attnTP AllGather。只有 resolved draft transport 需要 outer TP/EP bridge 时才临时 gather MoE hidden；K/V cache 始终不 gather。
- ngram embedding 的行数跟随 draft 阶段物理 rows，不从 Graph padded tensor shape 反推。
- mirror source/consumer 配置仍沿用 config，不新增 layer-id 字段。
- target/draft Graph hit 或 eager fallback 都必须复用同一套 `mirrored_kv_indices` / accept-selection 语义，不改变 ngram/KV mirror 的行顺序。

## 13. stream、overlap 和同步

### 13.1 scheduler overlap

支持 scheduler overlap 开/关，要求：

- row-layout 是 ForwardBatch-owned，不使用跨 batch 的可变全局状态。
- 新 DP executor 的 gather/combine API 禁止调用 `set_dp_buffer_len()`、`get_dp_global_num_tokens()` 或通过 `_DpGatheredBufferWrapper` 读取可能被后一批覆盖的 counts/buffer length；eager 直接接收本 ForwardBatch 的 row-layout，Graph 使用该 Graph runner 独占的静态 buffer。
- collective plan 由同步后的全局 metadata 决定。
- idle rank 也参与 collective。
- 不新增 WeLM 专用 WAR barrier。

### 13.2 shared expert 多流

- `DP_GATHER_TP_MOE` / `LOCAL_REPLICA_MOE`：保留当前 no-A2A decode/mirror 的既有 stream policy。
- `DEEPEP_NORMAL_AG_SCATTERED`：保留当前“shared expert 只与 DeepEP normal AllGather/dispatch 通信窗口重叠”的 ordinary-prefill 方案，wait 位置不变。
- `EXPLICIT_AG_LOCAL_SORT_EP_AR`：mirror/decode/verify/draft 复用当前 decode-like shared-expert/local-EP stream policy，EP AllReduce 完成后只加一次 shared output。
- 不新增 gate 多流，不改变现有 gate/attention stream 行为。

### 13.3 非 DP

不开 DP Attention 时，stream 创建、launch、event、wait 顺序必须与基线完全相同。

## 14. 最终 DecoderLayer 结构

### 14.1 顶层只做静态分派

```python
def forward(self, positions, hidden_states, forward_batch, residual):
    if self._welm_dp_executor is None:
        return self._forward_non_dp(
            positions, hidden_states, forward_batch, residual
        )
    return self._welm_dp_executor.forward(
        layer=self,
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        residual=residual,
    )
```

`_forward_non_dp()` 首版保持当前代码操作顺序，不顺便重写。

### 14.2 DP executor 的可读结构

```python
def forward(...):
    # ForwardBatch 边界已消费 scheduler modes；这里只做无分配的 layer view 选择。
    row_layout = forward_batch.welm_dp_moe_row_layout.for_layer(self.layer_id)
    plan = self.planner.resolve(
        local_phase=forward_batch.forward_mode,
        has_ordinary_prefill=forward_batch.is_extend_in_batch,
        row_layout=row_layout,
    )
    state = self.prepare_attention_input(plan, hidden_states, residual)

    if plan.local_attention_rows > 0:
        if plan.is_first_mirror_consumer:
            state = self.select_mirror_query_and_residual(plan, state)
        attn_output = layer.self_attn(
            positions=positions,
            hidden_states=state.hidden_states,
            forward_batch=forward_batch,
            skip_o_norm=True,
            skip_o_proj_all_reduce=plan.attention_finish.needs_external_reduce,
        )
        state = self.finish_attention_and_norm(plan, state, attn_output)
    else:
        # idle rank：不 launch 0-grid attention/MMQ，但保留 dtype/device/layout
        state = self.make_typed_empty_after_attention(plan, state)

    # idle rank 仍必须进入下面与全组一致的 gather/dispatch/combine 顺序
    state = self.prepare_moe_layout(plan, state)
    state = self.run_moe(plan, state)
    state = self.restore_layer_output(plan, state)
    return state.hidden_states, state.residual
```

### 14.3 plan 只负责语义，不直接改 tensor

```python
@dataclass(frozen=True)
class WelmBatchExecutionPlan:
    local_phase: WelmForwardPhase
    has_ordinary_prefill: bool
    local_attention_rows: int
    attention_finish: AttentionFinishKind
    moe_transport: WelmMoeTransport
    input_hidden_layout: WelmDpLayout
    input_residual_layout: Optional[WelmDpLayout]
    moe_hidden_layout: WelmDpLayout
    output_hidden_layout: WelmDpLayout
    output_residual_layout: Optional[WelmDpLayout]
```

本 rank `forward_mode` 只决定 `local_phase`；rank-uniform `moe_transport` 和 collective 顺序由冻结的 `WelmRunnerParallelPlan + has_ordinary_prefill + layer_id` 派生。`global_schedule_modes` 只在进入 executor 时构造逐 slot row-layout，构造完成后不再参与每层 transport 分派，也不复制进 `WelmBatchExecutionPlan`。group handles、runner role、分片 provenance 只存在于 `WelmRunnerParallelPlan`。`local_attention_rows` 必须取当前层选择后的 `row_layout.local_real_rows`：ordinary prefill 是 `T_i`、mirror/decode 是 `B_i`、verify 是 `V_i`、draft extend 是 `R_i`；`global_num_requests` 只服务于 ordinary-prefill 的 `T_i->B_i` 与 eager row metadata，target Graph bucket继续使用现有 `original_global_num_tokens_cpu`。各 tensor layout 由实际转换函数返回，release 仅携带状态，细粒度断言放在 debug/test。尤其 `DRAFT_REPLICA_LOCAL` 必须进入 `LOCAL_REPLICA_MOE` 或 draft explicit-AG local-EP handler，不能因内部 DP flag 为 false 当成“没有 executor”。命名固定为：启动期静态对象 `WelmRunnerParallelPlan`，每 batch 动态对象 `WelmBatchExecutionPlan`，tensor 状态 `WelmDpLayerState`，构造/forward scoped globals 才叫 `DraftParallelRuntimeBundle`。

### 14.4 删除/替换的混乱条件

DP forward 中删除：

- `use_dp_o_norm_after_attn`；
- 根据 `if/elif` 优先级解决两个 norm bool 重叠的写法；
- `has_full_replicated_moe_input`；
- 根据 decode/verify 猜 FULL layout；
- mirror 中直接修改 `global_num_tokens_*`；
- 根据 `scale_seq_factor` 推导 mirror request rows；
- local-EP capability 与 runtime layout 混为一个 bool 的写法。

替换为：

```python
local_ep_kernel_available = ...  # 静态能力
handler = self.transport_handlers[batch_plan.moe_transport]
# handler 的输入 layout 是 contract；debug 可 assert，release 不因 layout 静默改选另一条 transport
```

## 15. 具体文件修改点

### 15.1 `python/sglang/srt/server_args.py`

- 早期 `_handle_model_specific_adjustments()` 只处理构造前必须生效的声明：对 `WeLMv4 && enable_dp_attention && EAGLE/NEXTN Spec V2` 自动解析 `enable_dp_lm_head=True` 并记录一次 info；兼容算法 alias，但不在这里判断尚未归一化的 EP/A2A transport。
- DP/MoE/speculative/A2A 全部归一化后，从一次 `resolved_view()` 调用 model-specific late validator；在所有新增检查之前先判断 `WeLMv4 && enable_dp_attention`，不满足立即返回。
- 仅当 resolved `moe_ep_size>1` 时检查 `backend=deepep` 且 ordinary prefill 的 resolved transport 为 NORMAL-AllGather。resolved strategy 是唯一真值；raw env 只用于错误信息说明来源，不对等价 alias/default 重复门禁。通信组覆盖与 kernel 能力留给 runner 构造后一次性校验。
- 不比较 `TP/DP/EP` 固定数字 tuple。PP/CP/MoE-DP 等继续使用框架已有通用校验；当前 local-sort 实现对 `moeTP=1`、EP group 覆盖所有实际专家分片的要求作为 resolved kernel/group capability 校验。
- 错误信息打印缺失能力和实际 backend/group，不打印固定允许组合。
- 不对非 DP WeLM 或其他模型增加新门禁。

### 15.2 runner 构造、进程身份与量化校验

涉及：

- `python/sglang/srt/managers/tp_worker.py`；
- `python/sglang/srt/model_executor/model_runner.py`；
- `python/sglang/srt/model_executor/model_runner_components/moe_ep_setup.py`；
- `python/sglang/srt/model_executor/model_runner_components/load_model_utils.py`；
- `python/sglang/srt/model_executor/model_runner_components/remote_instance_weight_transporter.py`；
- `python/sglang/srt/utils/offloader.py`。

- `TpModelWorker`/`_init_model_runner()` 增加可选只读 `runner_parallel_plan` 参数，并从所有实际创建 `ModelRunner` 的分支（包括 speculative/multi-layer runner wrapper）原样透传；默认 `None`，所有现有 caller 行为不变。不要依赖隐藏 ContextVar snapshot 传计划，也不新增“一层 MTP”重复门禁——现有 `num_nextn_predict_layers`/mirror config 校验继续是唯一 owner。
- target runner 在 `init_torch_distributed()` 建好通信组后、`initialize()/load_model()` 之前构建并发布 target `WelmRunnerParallelPlan`，保证 DecoderLayer 构造时就能看到已校验计划。
- draft runner 不负责首次生成自己的 plan；它接收 `EagleDraftWorker` 在构造前生成的 provisional plan 与 `draft_ps`，并在 model/loader 初始化前绑定，随后只做 actual-module/group 二次校验。
- `ModelRunner.load_model()` 对 WeLM runner 在调用 `load_model_with_memory_saver` 的整个 constructor/load/postprocess 生命周期外包一层 `welm_runner_build_plan_context(self.runner_parallel_plan)`；helper 在 WeLM 分支内延迟 import，避免通用 loader/model import cycle，通用 loader 不改 kwargs 协议。WeLM model/DecoderLayer constructor 必须在该 context 内读取并缓存 plan，退出后 forward 不再读取 build carrier。
- target 和 draft 分别记录实际 role、TP group、Attention DP group、MoE group、OProj output contract。
- `ModelRunner` 缓存 `effective_dp_attention`/runner role：target 从 target plan 得到 true，`DRAFT_REPLICA_LOCAL` 得到 false；draft-sensitive downstream 不再读取 outer target 的 raw args。
- draft plan 同时保存 patched draft-attention-TP group、实际 draft-MoE TP/EP group 和未被 context 替换的 outer DP bridge group；不能在进入 `draft_tp_context()` 后再通过全局 getter 猜任一 group。
- `draft_ps` 只参与模型分片。`TpModelWorker` 的进程级 broadcast root 改用 `world_group.rank_in_group`，不能再由 `ps.tp_size * pp_rank + tp_rank` 推导；loader 的 shard 选择使用 draft model rank，而 IPC/remote session、日志和进程资源使用 plan 中的物理 WORLD rank。二者分别命名，禁止复用一个 `rank` 变量。
- draft 构造/加载完成后，`ModelRunner.finalize_welm_runner_plan()` 只做一次 runner/QKV/OProj/MoE 的可观测契约校验：group size/rank、shard interval/coverage 和 collective 使用的显式 plan handle。module 没有 group handle 时不声称能 post-hoc 证明 identity；构造时记录 shard provenance/group uid，或只验证可观测 coverage。运行期不重做模块遍历。
- 全局 offloader 不能被 draft `ModelRunner` 永久覆盖。target/draft 各自持有 runner-local offloader，统一 draft runtime context 只在 draft 构造/forward 期间 scope 它并在 `finally` 恢复 target；不开 offload 时为 no-op。
- 把只读 plan 注入模型，不依赖进程级全局 bool 在 forward 中反复判断。
- draft `LogitsProcessor` 构造时必须看到 resolved attnTP vocab-group policy；不能读取 target outer-TP head policy 后缓存错误的 all-gather 策略。
- `check_quantized_moe_compatibility()` 增加可选 `resolved_moe_tp_size`；默认 `None` 时逐行保持现有公式。它只做能由 size 表达的 intermediate-size/量化 block 数值对齐，不读取隐式 global group，也不重复检查 rank/coverage；后者由上一条一次性 post-load validator 负责。

### 15.3 `python/sglang/srt/models/welmv4.py`

- 将当前非 DP forward 原样隔离为 `_forward_non_dp()`。
- 顶层 `forward()` 只做 `_welm_dp_executor is None` 静态分派。
- `_welm_dp_executor` 按 injected runner plan 创建：role 为 `TARGET_DP` 或 `DRAFT_REPLICA_LOCAL` 时创建；没有 plan 时为 `None`。不能因 draft context 内 `is_dp_attention_enabled()=false` 把 physical layer48 错分到 `_forward_non_dp()`。
- 用“DP plan 已存在且当前 phase 所需 transport capability 已验证”的构造期断言替换当前 `dp && o_norm && mlp_mode != FULL` blanket reject；在对应 transport 真正接通前按实施阶段逐项开放。
- 当前 `_update_kv_mirror_full_metadata()` 保留给现有 pure-TP 非 DP 路径，建议明确重命名为 `_update_pure_tp_kv_mirror_full_metadata()`；DP 不再调用。
- `supports_welm_local_ep_moe` 拆成 kernel capability 和 runtime selection。
- DP 路径删除重叠 norm bool、`has_full_replicated_moe_input` 和 mirror metadata mutation。
- `_mask_npu_padded_topk()` 扩展为可接收 segmented `valid_row_mask`；旧 scalar 调用保持原行为。
- `DP_GATHER_TP_MOE` 调用现有 MoE block 时固定 `use_reduce_scatter=False`；构造完成后一次性确认 routed/shared 子模块不会自行 reduce，由 block 在 resolved MoE-TP group 内完成 partial 相加和唯一一次 MoE-combine AllReduce，DP executor 只做 local row slice。双重 combine 检查只放在 debug collective trace/测试中，不进入每层 release 热路径。
- DP executor 向 MoE block 显式传入只读 `resolved_transport/stream_policy`；shared-expert 的分片、combine 和普通 prefill stream overlap 不再从全局 DeepEP bool 或旧 `welmv4_npu_deepep_*` 布局字段猜测。`moe_ep_size=1` 始终按 pure-TP MoE 契约，`DEEPEP_NORMAL_AG_SCATTERED` 保留已有 normal-AG overlap wait 点。
- target layer47 的输出边界由 executor 的 phase/layout contract 保证；release forward 不再追加同义断言。debug/test 下检查 MoE 已完成 combine 且 hidden/residual 已恢复为本 DP shard 的 current-phase rows，禁止 FULL/partial tensor 越界，也禁止在 worker accept-selection 前把 V 提前压成 B。
- 不改变非 DP 的 OProj MatMul+RS、DeepEP、mirror、stream 和 final-layer component 行为。

### 15.4 `python/sglang/srt/models/welmv4_dp_attention.py`（新增）

集中放置：

- `WelmRunnerRole`、`WelmDpLayout`、`WelmMoeTransport`；
- `WelmBatchExecutionPlan`、`WelmDpLayerState`、`DraftParallelRuntimeBundle`、`DraftSharedModulePlan` 和 planner；shared-module plan 在这里定义，避免 worker 与 NextN model 反向 import。
- model-local `welm_runner_build_plan_context()` / `get_welm_runner_build_plan_for_init()`；只供 WeLM constructor/load Python 边界使用，禁止在 forward/torch.compile/Graph-visible 代码中读取。
- ordinary prefill 的 attnTP ReduceScatter + MMQ；
- FULL/local-EP 的 attnTP AllReduce + MMQ；
- first mirror `T_i -> B_i` 和 FP32 residual owner reconstruction；
- 无 EP 的 `DP_GATHER_TP_MOE` transport；
- physical MTP 无 EP 的 `LOCAL_REPLICA_MOE` transport；
- 有 EP 的 `DEEPEP_NORMAL_AG_SCATTERED` 和 `EXPLICIT_AG_LOCAL_SORT_EP_AR` transport；
- debug/test 专用的阶段边界 layout assertion；release hot path 不逐层重复检查静态 group、权重或 collective owner。

该文件是模型内部执行器，不是新的 Spec worker。

### 15.5 `python/sglang/srt/layers/communicator.py`

- 新增 WeLM DP 专用的纯布局 primitive，例如：

```python
reduce_attn_partial_to_scattered(hidden, residual, row_layout)
```

- 只做 attnTP ReduceScatter 和 residual 对应行切片，不做 norm。
- 新接口显式接收 plan 的 `attn_tp_group`，绝不能在实现内调用可能指向外层 TP4 的 `get_tp_group()`；现有 communicator 旧接口保持原 getter 语义。
- 现有 communicator 默认分派和已有调用者不改。

### 15.6 `python/sglang/srt/layers/dp_attention.py`

- 增加显式 row-layout 版本的 replicate gather、scatter 和 AllReduce 后 local-slot slice；首版不在这里再次实现 TP-MoE AllReduce。
- 新接口显式接收 `attn_tp_group` 和 outer/EP group，禁止在实现内部假定 `get_tp_group()` 总是外层 TP4；这是 draft context 下的正确性要求。
- SUM_LEN 和 MAX_LEN 都从 `WelmDpMoeRowLayout` 取得 sizes/offsets。
- replicate gather 只贡献每个 Attention DP shard 的一个副本，避免 DP2 数值乘 2。
- Graph 模式支持固定 slot 和 per-DP real rows。
- 新接口只读取显式传入的 `WelmDpMoeRowLayout`/Graph static buffer，不读取进程级 `set_dp_buffer_len/get_dp_global_num_tokens` 状态，保证 scheduler overlap 下两批 metadata 不串扰。
- 新增单一 scoped `draft_dp_attention_context(enabled=False, rank=0, size=1)`：进入前 snapshot `_ATTN_DP_*` 与相关 DP flags，统一写入并在 `finally` 逆序恢复；禁止在多个调用点分别手写这些 global。恢复值检查只在 debug/test 启用。
- canonical getter 本身不增加 `ContextVar`/role 分支，避免进入 compile/Graph-visible 路径；非 draft 调用和非 DP 模型完全不变。
- 旧 `dp_gather_replicate()`、`dp_scatter()` 和其他模型调用保持不变。

### 15.7 `python/sglang/srt/managers/scheduler_components/dp_attn.py`

- `MLPSyncBatchInfo` 增加 `num_requests` 和 `global_num_requests`。
- 复用现有同步 payload 中已经存在的 `local_forward_mode` 槽位，发布语义明确的 `global_schedule_modes`。它只表示 scheduler 同步时的 mode，不声称包含 worker 后生成的 `TARGET_VERIFY`。
- request counts 与 scheduler modes 合并进现有 all-gather payload，不新增 collective；不新增 WeLM 专用 Graph eligibility 列。
- idle batch 同样发布 0 request rows。
- 现有 `is_extend_in_batch=max(...)` 继续作为 rank-uniform transport 的唯一 round-level 判据。非 Spec mixed round 才消费完整 modes vector解释各 slot 行数；Spec V2 依赖默认的额外 MLP-sync 将 prefill 与 speculative decode 分轮。

### 15.8 `python/sglang/srt/managers/schedule_batch.py`

- 增加 `global_num_requests` 与 `global_schedule_modes` 字段并传给 ForwardBatch；字段注释明确 scheduler-time 语义。
- target verify/draft 的实际 row count 继续由对应 spec info/物理输入构造提供；当前 draft extend 明确使用固定 `B×W`，不从 accept length 推导。

### 15.9 `python/sglang/srt/model_executor/forward_batch_info.py`

- 增加 `global_num_requests_cpu` 与只读 `global_schedule_modes`；不增加通用 `global_num_requests_gpu`。
- 增加 `welm_dp_moe_row_layout`，在 ForwardBatch 边界一次性构造 prompt/request/verify/draft 所需 counts/offset views；`for_layer(layer_id)` 只选择已有 view，不分配 tensor。
- 非 Spec 时由 `global_schedule_modes` 区分各 slot 的 `T_i/B_i/0`；Spec verify 时把 scheduler `DECODE` slots 按 worker stage确定性解释为 `V_i/0`。transport 只读取 `is_extend_in_batch` 和静态 mirror plan。
- 保持通用 `global_num_tokens_*` 和 `num_token_non_padded` 语义不变。
- target Graph row-layout 直接引用现有 Graph-resident `global_num_tokens_gpu`；eager prefill 的 request rows tensor只在 ForwardBatch 边界创建一次。

### 15.10 `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`

- **首版预期不修改。** target decode/verify 已有 `global_num_tokens_gpu` slot 正好承载本轮物理 `B_i/V_i`，row-layout直接引用该固定地址。
- 只有实际实现证明现有 slot 无法被 captured ForwardBatch引用时，才增加一个 WeLM alias/view；不得分配内容等价的第二份 tensor，也不得把 Python dataclass 注册成 Graph slot。

### 15.11 `python/sglang/srt/model_executor/runner_utils/buffers.py`

- **首版预期不修改。** target runner继续复用 `DecodeInputBuffers.global_num_tokens_gpu` 及 registry 的既有 replay copy。
- draft decode/extend 的条件化 metadata 由各自 runner-owned buffer处理，不把 draft-only 字段塞入通用 target buffer dataclass。

### 15.12 `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`

- 当前实现已经用 `max(original_global_num_tokens_cpu)` 选择 DP 请求 bucket，并用 `captured_req_width` 区分 decode `K=1` 与 verify `K=D`；保留该逻辑并增加回归测试，不重新实现 bucket planner。
- 当前 `can_run_graph()` 已复用 `can_run_dp_cuda_graph` 的 group-wide AND，继续沿用现有 eager fallback，不增加 candidate/success manifest或运行时 collective。
- captured ForwardBatch 必须让 `welm_dp_moe_row_layout` 引用 runner现有固定 `global_num_tokens_gpu`。优先在通用 ForwardBatch row-layout builder 完成；只有 dummy/capture 构造确实绕过该 builder 时，才在本文件补一次 helper调用。
- 不修改其他模型 Graph key、capture 顺序、padding 策略和 replay backend。

### 15.13 Spec V2 Graph runners

涉及：

- `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`；
- `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`；
- NPU 对应 graph runner adapter（仅在其拥有独立 load/replay metadata 时修改）。

收敛后的改动：

- target verify、draft decode、draft extend 的独立 `can_run_graph()`/eager fallback 当前已经存在，保持原调用顺序，不重写成新的统一判定器。
- `LOCAL_REPLICA_MOE` 且不含 outer collective 时不增加 row-sync metadata，draft Graph继续按实际 draft group和本地物理 rows 独立选 bucket。
- 只有 resolved draft transport 含 outer/EP collective 时，才让对应 draft runner-owned buffer保存同步 real rows，并使用 outer scheduler 已有 counts/`can_run_dp_cuda_graph` 统一该 collective group 的 key/fallback；draft decode 与 draft extend 仍使用各自既有 buffer，不能共享可变 metadata。
- NPU adapter 仅在它实际拥有独立 load/replay metadata 时补同一字段；若只是转调通用 runner则不修改。
- 不修改 Graph backend replay 本体，也不把 target/draft/draft-extend 的 hit/miss 绑定在一起。

### 15.14 `python/sglang/srt/distributed/parallel_state.py`

- 保持现有 `patch_tensor_parallel_group()` 的签名和所有调用者不变。
- 新增 `patch_draft_parallel_groups(bundle)`，在单个 context 内 snapshot 并切换 `_TP`、`_ATTN_TP`、`_MOE_TP`；physical MTP 的 resolved plan 不同于 target 时同时切换 `_MOE_EP`、`_MOE_DP` 和 NORMAL communicator。PP/Attention-CP/DCP 首版必须为已有支持值并校验；若不相同则同样从 bundle 显式切换，不能猜。
- 所有 group 在 context 外预创建并写入 immutable bundle；context 内禁止 `new_group/destroy_group`。`_WORLD`、outer target-DP/TP bridge 和 PDMux group 永不 patch，跨 shard transport 只用 plan 的显式 handle。
- 全部 snapshot 完成后才首次写入，`finally` 逆序恢复；release 只保留同一 bundle 的轻量计数式嵌套 guard，不同 bundle 嵌套立即拒绝。恢复 identity/value 的逐项检查放在 debug/test。
- canonical getter 本身不增加 ContextVar/runner-role 分支，避免影响 compile 和所有现有非 DP/其他模型调用者。

### 15.15 `python/sglang/srt/speculative/spec_utils.py`

- 将 `draft_tp_context()` 扩展为可接收只读 draft runner role/parallel plan，退出 context 后恢复。
- 普通模型或无 plan 时仍使用原有 TP-only patch 和 speculative MoE context；WeLM DP 用一个 `ExitStack` 同时进入第15.14节的完整 group context、第15.6节的 draft DP context，以及现有 `speculative_moe_backend_context()` / `speculative_moe_a2a_backend_context()`。
- bundle 创建时已经一次性校验 speculative MoE runner/A2A flags；每次 ExitStack 不再重复读取/断言，只消费冻结值，也不从临时 draft args 重新初始化。
- 现有 MoE context 仍按框架原语义 scoped 写入并在 `finally` 恢复；WeLM 不修改通用 MoE getter，也不引入 compile-visible ContextVar。恢复值检查只在 debug/test，不进入每次 release draft launch。
- context 进入/退出集中在这里；model、loader、backend 和 runner 禁止分别手写 flag/group patch。当前 scheduler 只允许单 Python launch thread；真正 host 并发不在首版开放。
- 禁止模型只读取全局 `is_dp_attention_enabled()` 判断 draft 数据布局。

### 15.16 `python/sglang/srt/speculative/eagle_worker_v2.py`

- target verify、draft、draft extend 在建立 ForwardBatch 时填入各自 phase 和实际 row metadata。
- 在 `TpModelWorker(is_draft_worker=True)` 之前，根据 outer target groups/config 创建 provisional draft plan，并用它派生 `draft_ps` 和 draft-private `draft_server_args`；把二者传给 worker，同时把 outer target-DP identity/group 只保存在 plan 中。不能继续把 target `ps` 仅改 `pp_rank` 后直接传入。
- scheduler 已经通过 `draft_worker_common.py::draft_server_args_copy()` 生成 resolved draft copy；Eagle 直接接收它并只追加正式 `ServerArgs.override("welmv4_draft.build", ...)`。禁止再次 copy、从裸 target args 重建、原地改 target args 或使用 test-only runtime override；首版不修改 `draft_worker_common.py`。
- bundle 创建时读取并校验已初始化的 speculative MoE runner/A2A flags；所有 WeLM draft 调用点只进入统一 `draft_tp_context(bundle)`，删除外层并列的两个 speculative MoE context，禁止重新初始化 process-global MoE config。
- draft worker 构造期是否进入完整 runtime context 由 `draft_parallel_plan.requires_runtime_context` 决定，不能继续只用 `speculative_algorithm.is_eagle3()`；WeLM NEXTN 会解析为 EAGLE，同样必须在模型构造和权重加载前进入一致 context。
- model construction/load、attention backend init、Graph capture、draft prefill seed、draft decode 和 draft extend 的每一个 `draft_tp_context()` 调用都传同一份只读 draft parallel plan，不能只修 eager forward。
- `alloc_memory_pool()` 中的 `draft_worker.alloc_memory_pool`、ngram table share、`init_lm_head()/set_embed_and_head()` 全部进入同一 runtime context；不能让 head 绑定/校验在 target live group 下裸跑。
- `init_lm_head()` 调用唯一 owner `DraftSharedModulePlan.build()`，一次性校验 embedding/OE/head 的可观测 shard interval、vocab-group rank/size、replica coverage 和 draft LogitsProcessor policy；Eagle 本身不再复验模块事实。
- target/draft Graph 命中和 eager fallback 独立。
- 不创建 WeLM 专用 worker，不改变 Spec V2 调度主流程。

### 15.17 `python/sglang/srt/models/welmv4_nextn.py`

- 只传递 draft runner role、row-layout 和现有 hidden/ngram 信息。
- 不复制 DP executor，不重新实现一套 decoder layer。
- 保持 physical MTP layer48 的逻辑权重名称/映射方式不变；其实际 TP/MoE-TP shard group 必须来自 draft plan，确保 routed/shared expert 使用一致分片。
- `set_embed_and_head()` 只接收并绑定已验证的 shared-module plan，不重复断言其构造事实；调用方无法提供 plan 时才拒绝，禁止盲绑 target object。
- consumer48 入口只保留无 D2H 的轻量动态边界检查：`hidden.shape[0] == row_layout.local_real_rows == B_i` 且 layout tag 为 DP-local；不遍历 module/group，也不重复验证静态权重或 collective owner。

### 15.18 NPU attention/RoPE 文件

首版预期不修改：

- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`；
- `python/sglang/srt/hardware_backend/npu/attention/welmv4_sink_prefill_attention.py`；
- `python/sglang/srt/layers/welmv4_op.py`。

原因是 DP 适配应先修正模型层的数据布局，不能为布局错误再造 attention 特例。只有 `12/1` 或 `24/2` 正确性测试证明通用 kernel 存在真实缺口时，才增加单独、可验证的 NPU kernel 修复。

### 15.19 `python/sglang/srt/speculative/eagle_info.py`（首版预期不修改）

- topk1 固定宽度 verify 已能由 request rows 和 `D` 得到物理 `B×D`。
- 当前 draft extend 的 `B×W` 由 `prepare_for_draft_extend()` 写入 `extend_num_tokens/extend_lens`，不需要在 EagleInfo 中再造另一套 row count。
- `eagle_worker_v2.py` 构造 `WelmDpMoeRowLayout` 时直接消费这些既有字段，并增加一致性断言；如果实现时发现 Graph runner 无法取得 per-DP 物理 rows，才在本文件增加明确字段，不能从 padded tensor shape 猜。

### 15.20 `python/sglang/srt/model_executor/model_runner_components/ngram_embedding_manager.py`（首版预期不修改）

- 沿用当前 `NgramEmbeddingInfo.create()` 的 `input_ids`、`extend_start_loc`、`extend_seq_lens` 物理行语义。
- 新 MoE row-layout 不覆盖 ngram metadata；consumer48 仍先对 T 行生成 token/OE history，再用 `custom_last_index` 选 B 行。
- Graph padding 只通过现有静态 ngram buffers 和请求有效性处理，不用 MoE 的 segmented mask 改写 ngram offsets。
- 若验证发现现有 create/replay 仍读取 Graph padded shape 而非物理 rows，再把修复限制在该 manager；不在 DecoderLayer 中补偿。

## 16. 对非 DP 行为的影响

目标是 **零行为影响**。

当 target/server 未开启 DP Attention、runner plan 不存在时：

- 不运行新的 WeLM DP+EP transport 能力校验。
- `_welm_dp_executor=None`，直接进入 `_forward_non_dp()`。
- 不使用新 row-layout。
- 不增加 collective。
- 不改变 collective group、shape、顺序。
- 不改变 Graph capture bucket、Graph 数量或 replay metadata。
- 不改变 pure TP prefill scattered fast path。
- 不改变 TP+EP 的 DeepEP normal/LL/local-EP。
- 不改变 mirror FULL 路径。
- 不改变 MTP step1/stepN、target/draft Graph 独立回退。
- 不改变 scheduler overlap、shared expert stream、gate/attention stream 和 wait 位置。
- 不新增 Triton constexpr 变体或反复编译点。
- OProj `reduce_results`、local-EP dispatcher construction、embedding/head group、final component rebuild 等构造期决策都必须读取注入的 runner plan；不能残留一个全局 `is_dp_attention_enabled()` 让 draft 或现有非 DP 路径意外改变行为。
- `local_ep_kernel_available` 拆分后，非 DP 的最终布尔表达式必须与基线逐项等价，不能因为“capability 更通用”而让非 DP 选择到新 runtime path。

为了保证这一点，首版不会把 `welmv4_npu_deepep_scattered`、`welmv4_npu_deepep_full_mirror` 等现有字段强行重命名并迁移所有调用；它们先被限制在 `_forward_non_dp()`。稳定后可做纯机械清理。

## 17. 实施顺序

### 阶段 0：冻结基线

- 记录当前非 DP 三类路径的 greedy output IDs、collective trace、Graph bucket、显存和性能。
- 保存 TP4、TP4+EP4、MTP step1/step3、Graph/overlap/multi-stream 组合。

### 阶段 1：建立非 DP 防火墙

- 先把现有 forward 原样移入 `_forward_non_dp()`。
- 新顶层 forward 暂时仍只调用该非 DP 路径。
- 验证代码移动前后完全一致。

### 阶段 2：capability plan 和 row metadata

- 增加按实际 group/backend/权重覆盖解析的 runner capability plan；不增加数字 topology 白名单。
- 在 draft worker 构造前派生 `draft_ps`、draft-private `draft_server_args` 和 `DraftParallelRuntimeBundle`；先只做四方一致性与 context 恢复校验，不运行新数据流。
- 打通 `TpModelWorker -> ModelRunner -> WeLM build-plan carrier -> model/DecoderLayer constructor`，在 executor 构造后断言实例 plan identity；carrier 退出后 forward 读取应被静态检查禁止。
- 把量化 MoE 兼容校验改为可接收 resolved MoE-TP size，覆盖 `draftTP < EP` 但实际 local expert layout 合法的场景，同时保留所有 block/intermediate 对齐检查。
- 解析并校验统一的 attnTP vocab group，确保 target head、shared embedding/OE 和 draft LogitsProcessor 的 shard contract 一致。
- 同步 exact `num_requests`。
- 增加独立 `WelmDpMoeRowLayout`。
- 此阶段不开放当前 reject。

### 阶段 3：无 EP 的 `DP_GATHER_TP_MOE` eager

- 实现 attnTP AllReduce + MMQ。
- 实现 replicate gather + FULL TP-MoE；复用 MLP block 内唯一一次 TP AllReduce，executor 只做 row-layout slice。
- 同时覆盖 `attnTP>1` 和 `attnTP=1`，并覆盖普通 prefill、mirror、decode、verify。

### 阶段 4：有 EP 的 ordinary-prefill NORMAL AllGather

- 实现 attnTP ReduceScatter + MMQ -> `EP_SCATTERED`。
- 接入 DeepEP normal AllGather + initrouting。
- 验证 attnTP collective/identity 两种情况、非对齐 `T_i` 和 idle DP shard。

### 阶段 5：有 EP 的 decode-like explicit-AG local-EP 闭环

- 实现 first mirror `T_i -> B_i`。
- 实现 DP gather -> EP_FULL -> local experts -> EP AR -> local slice。
- 删除 DP mirror 的临时 metadata mutation。

### 阶段 6：MTP eager

- target verify role。
- draft DP-local role，以及构造/加载/backend/forward 全程一致的 `draft_ps + draft_server_args + DraftParallelRuntimeBundle`；验证退出 context 后 target topology/config 完整恢复。
- 无 EP physical layer48 先实现 `LOCAL_REPLICA_MOE`；有 EP layer48 接入 `EXPLICIT_AG_LOCAL_SORT_EP_AR`。
- draft decode/draft extend 的真实 row metadata。
- target layer47 在发布 aux hidden 前完成 MoE combine 和 DP-local slice。
- ngram embedding、KV mirror、draft/target-verify 精度验证。

### 阶段 7：Graph 与 scheduler overlap

- target/draft/draft-extend 独立静态 buffers。
- 复用现有统一 capture 顺序和 scheduler MLP-sync；把 WeLM Graph eligibility 合入既有 group-wide `can_run_dp_cuda_graph`。
- target collective domain 用同步 request counts 统一 Graph key/bucket，任一 rank miss 时全体 eager；draft local graph 仍独立。
- segmented padding mask。
- Graph hit/fallback 交叉组合。
- overlap 开/关和 idle rank collective 顺序。

### 阶段 8：可读性清理

- 只删除 DP path 已经不可达的旧分支。
- 现有字段重命名另起纯机械提交。
- 不在功能提交中混入非 DP 重构。

## 18. 验证矩阵

### 18.1 四个重点配置与通用 transport

至少验证以下四个当前重点配置；它们是验收样例，不是代码白名单：

| 外层 TP | Attention DP | Attention TP | MoE EP | MoE TP |
|---:|---:|---:|---:|---:|
| 4 | 2 | 2 | 1 | 4 |
| 4 | 4 | 1 | 1 | 4 |
| 4 | 2 | 2 | 4 | 1 |
| 4 | 4 | 1 | 4 | 1 |

每个 resolved transport 都验证：

- ordinary prefill：短 prompt、长 prompt、非对齐 token 数、所有 DP shard 行数可不等。
- mirror first consumer：至少两个 shard 的 `B_i` 不等，以及任意一个 shard 为 0。
- decode：BS=1、非对齐 BS、最大 Graph bucket。
- target verify：step1 (`D=2`) 和 step3 (`D=4`)。
- draft decode / draft extend：Graph hit 和 eager fallback。
- TP4+DP4+纯 TP4 MoE 下验证 physical layer48 的 shared/routed expert 均按 draft TP1 完整复制；构造、loader、eager、Graph 所见 group 完全一致。
- TP4+DP2+EP4 和 TP4+DP4+EP4 的 physical layer48 在量化配置下必须使用 resolved MoE-TP group 完成 compatibility check，不得命中旧 `draft_tp_size % ep_size` 公式；错误 block size 仍必须被拒绝。
- 在 model constructor 与 weight postprocess 中断言 build-plan carrier 可见且 identity一致，退出 loader 后 carrier 已恢复；executor 在构造期已经按 role 建好，禁止 post-load 临时补注入。
- draft 构造完成后一次性验证 `draft_ps/live getter/module cache/group` 一致；context 的进入/退出、异常注入和嵌套恢复由专门单测/debug instrumentation 验证。release forward 只执行必要的 snapshot/restore，不重复模块遍历和同义断言。
- shared base/OE embedding 与 LM head 的 attnTP shard interval、vocab coverage、draft LogitsProcessor gather group 一致；TP4+DP4 draft TP1 必须看到完整 vocab，TP4+DP2 每个 replica 的 TP2 合并必须覆盖完整 vocab。
- target layer47：mirror-enabled prefill/decode 是本 shard `B_i` 行，mirror-disabled ordinary prefill 是 `T_i` 行，target verify 是 `V_i` 行；都不能是 FULL 或 TP partial，verify 的 `V_i->B_i` 只允许在 worker accept-selection 后发生。DP+EP mirror-off prefill 在发布前还必须从 `EP_SCATTERED` 恢复本 shard `T_i` 行。
- 无 EP target 每层的 MoE collective trace 按 purpose/group 标记，恰好一次 `moe_combine` AllReduce；DP replicate-gather 的内部 collective 不计入，禁止 MLP block 与 DP executor 双重 combine。
- Graph padding 开/关。
- scheduler overlap 开/关。
- NPU multi-stream 开/关。
- full attention 与 SWA。
- 198 条以上长稳压、请求逐步退出和 idle-rank 阶段。

能力负例也必须覆盖：

- DP Attention + EP，却没有使用 DeepEP NORMAL AllGather，启动时报清楚缺失能力。
- DP Attention + EP 且误开 NORMAL AllToAll，启动时拒绝。
- resolved local-sort/local-EP kernel 或 group coverage 不满足，构造 runner plan 时拒绝。
- EP local-sort transport 下 shared expert 不是 EP-replicated 且没有显式 shared-TP combine 时拒绝。
- 无 EP 时，无论 DeepEP 环境变量为何，都不能因此拒绝 `DP_GATHER_TP_MOE`。
- WeLM DP Spec 输入 `enable_dp_lm_head=False` 时应在构造任何 vocab module 前自动解析为 true 并记录 info，而不是启动拒绝或运行时拼接 outer-TP head。
- 不开 DP Attention 时，上述 model-specific 校验全部跳过。

### 18.2 Graph 交叉命中

至少覆盖：

- target Graph hit，draft eager。
- target eager，draft Graph hit。
- draft decode Graph hit，draft extend eager。
- draft decode eager，draft extend Graph hit。
- 所有 Graph 均 hit。
- 所有 Graph 均 miss。
- target 各 rank 本地可用 bucket 集不完全相同：只选共同 key；没有共同 key 时全体 eager。
- 用同步 metadata 制造 group-wide `can_run_graph` miss：所有 rank 都选择 eager，collective shape/顺序保持一致；WeLM eligibility 必须经既有 MLP-sync AND 后再进入 runner。
- Graph padding 关闭且各 DP shard shape 不同：只要同步后的 `max(B_i)` 有 exact captured key，仍共同 replay；没有该 exact key时全体 eager。开启时按同步 rows 选择同一个 common bucket。
- target verify 对 step1/step3 断言 Graph key 始终是请求 bucket `B_graph`，静态 token slot 分别是 `B_graph×2` / `B_graph×4`，不能出现 `D` 二次放大。

### 18.3 非 DP 回归

以基线为金标准：

- TP1/TP4。
- TP4+EP4 各个当前可用 EP backend。
- Graph 开/关、padding 开/关。
- mirror 开/关。
- MTP 关、step1、step3。
- overlap 和 multi-stream 开/关。
- BF16/MXFP8 当前已验证组合。

## 19. 验收标准

### 19.1 正确性

- DP greedy 输出与单卡/非 DP reference 对齐。
- Graph 开关不改变 output IDs。
- overlap 开关不改变 output IDs。
- 四个重点配置和所有 resolved transports 在 full/SWA、step1/step3 下均无复读或接受链偏移。
- mirror source/consumer 的 KV 和 residual row 对齐。
- padding 行永远不参与有效专家路由。

### 19.2 布局断言

debug 模式下每个阶段断言：

```text
进入 attention：DP_LOCAL_TP_ATTN_FULL
进入 DeepEP normal：EP_SCATTERED
进入 local-EP：MOE_FULL_REPLICATED
离开 layer：DP_LOCAL_TP_ATTN_FULL 或明确的下一层 SCATTERED 状态
```

ordinary DP+EP prefill 还要在 debug 模式逐 tensor 断言：层间 `hidden/residual=EP_SCATTERED`；下一层 attention 前 normalized hidden 为 `DP_LOCAL_TP_ATTN_FULL`、residual 仍为 `EP_SCATTERED`；attention RS 后二者才重新成为 matching scattered rows。

### 19.3 非 DP 隔离

- 无新增 collective。
- collective group/shape/顺序不变。
- Graph capture 数量和 bucket 不变。
- greedy output IDs 完全一致。
- stream/event/wait trace 不变。
- canonical parallel/MoE/DP getter 无 ContextVar/runner-role 新分支；torch.compile graph-break 数和既有 compiled graph cache key 不变。
- 不新增 model-specific WAR barrier。
- 不新增 attention Triton 编译变体。

### 19.4 性能

- 本次功能提交先以“无明显非 DP 回退、DP 路径可稳定运行”为门槛。
- DP2 `q12/kv1`、DP4 `q24/kv2` 落入通用 attention kernel 导致的性能差异单独记录，不通过错误放宽 6/1 专用 kernel 解决。
- local-EP 和 FULL TP-MoE 的 gather/combine 不允许出现按层 D2H 或动态 tensor 分配。

## 20. 最终预期

实施后，WeLMv4 DecoderLayer 将由两个清晰、互不污染的体系组成：

```text
非 DP：当前已验证路径，行为不变

DP：runner role + resolved groups/capabilities + tensor layout + row layout
    -> 正确完成 attention reduction
    -> 唯一 MMQ o_norm/postnorm
    -> 按 has-EP、phase 和实际 group/权重覆盖选择 MoE transport
    -> 正确处理 mirror、Graph padding、MTP 和 idle rank
```

这能够解决当前代码中 DP 条件散落、norm 分支重叠、forward mode 冒充布局、mirror 只修 metadata 等问题，同时把不开 DP Attention 的回归风险控制在最小范围。
