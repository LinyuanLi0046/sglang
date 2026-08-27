# WeLMV4 NPU MTP（Spec V2）正式改动方案

> 适用代码库：`D:\sglang-tx\NEWSGLANG\sglang`  
> 方案基线：`6b61e3a1e13958a5b6307871838417c645eb31ae`  
> 对照实现：`D:\sglang-tx\sglang-perf-welm-v4-optimization\sglang-perf-welm-v4-optimization` 的 CUDA WeLMV4 MTP 路径  
> 方案状态：已按本方案完成代码实施与静态复审；NPU 真机四象限和 ModelSlim MXFP8 专项回归待执行

## 1. 目标和边界

本方案在 NEWSGLANG 现有 NPU 路径中，为 WeLMV4 接入已有的 EAGLE/NEXTN Spec V2 体系，支持：

- `topk=1`、greedy；固定 draft 步数 `S >= 1`，target verify 宽度 `D=S+1`。
- NPU Graph 开启或关闭。
- scheduler overlap 开启或关闭。
- 纯 TP，或 `TP=EP`。
- prefill 始终 eager；decode 的 `draft`、`target_verify`、`draft_extend` 分别独立判断是否命中 Graph，任一阶段均可独立 eager fallback。
- WeLMV4 特有的 over-encoding/ngram embedding 和 KV mirror。
- checkpoint 中逻辑 `layer_id=48` 的唯一 MTP layer；runtime physical/cache slot为0。

首版的验证范围不包含随机采样、rejection sampling、`topk>1`、speculative adaptive、attention-DP、TBO、PDMux、混合 decode、disaggregation、EPLB、LoRA。这里是实施与测试范围，不新增对应的启动门禁；用户侧继续保证不启用这些组合。

精度/量化类型也不新增门禁。BF16 和已经专项验证的 ModelSlim MXFP8 都沿用现有 WeLMV4 权重加载与算子路径。

## 2. 核心结论

### 2.1 不新增 worker

不创建 `welmv4_mtp_worker.py`，也不创建 `WelMMTPTopK1Worker`。WeLMV4 继续使用现有：

- `EAGLEWorkerV2`：编排 target 与 draft。
- `EagleDraftWorker`：执行 `draft` 和 `draft_extend`。
- `run_eagle_verify`：执行 `target_verify`、采样和 accept bookkeeping。
- 现有 target、draft、draft-extend Graph runner。

WeLMV4 的差异只通过模型类型分支、两个小的数据契约和 NPU 算子接入，不复制 Spec V2 的调度、采样、KV cache 或 overlap 逻辑。

### 2.2 阶段命名统一

本文及代码注释只使用以下名称：

1. `draft`
2. `target_verify`
3. `draft_extend`

不再使用 `proposal` 指代 `draft`，也不新增独立的 “proposal graph”。现有 `EAGLEDraftNpuGraphRunner` 已经负责 Graph 模式下的完整 draft 循环。

### 2.3 只补 WeLMV4 的模型语义和 Graph 数据契约

通用 Spec V2 在 NPU 上已经工作。WeLMV4 需要补的是以下模型语义：

1. ngram/OE 历史在 speculative token 链上的计算与 accepted token 提交。
2. target source layer 0 产生的 mirror K/V，传递给 draft physical/cache slot 0（checkpoint/config逻辑 layer 48）的 `draft_extend`。
3. topk1 draft 的 Q-only frozen-KV 语义：attention/RoPE position 和 KV prefix 在多步 draft 中保持不变。
4. draft physical/cache layer 0 与 checkpoint/config logical layer 48 的彻底分离。
5. target verify、draft、draft-extend 三类 Graph 各自完整携带上述语义所需的静态输入、输出和 metadata。

不修改 NPUGraph 的 update/replay 并发实现，不改 scheduler 的通用 overlap 协议，也不引入 CUDA 分支里的专用 FutureMap或整段专用 draft runner。权重加载阶段只保留一个短生命周期的 layer48 QKV staging module，用于在 ModelSlim runtime relayout 前完成 source0/consumer48 的原始权重配对；它在 target load 完成后立即释放，不进入 forward/Graph。WeLMV4 的 `forward_mtp()` 必须走 Triton，这是本方案的 P0 正确性要求，不允许回退 FIA。

## 3. layer 0、layer 48 和 local layer 0 的准确含义

配置必须解释为一个逻辑 layer namespace：

- target transformer layers：逻辑 `[0, 48)`。
- 唯一 MTP layer：checkpoint/config 逻辑 `48`，draft runtime physical/cache slot `0`。
- `kv_mirror_imitated_layers[0]=0`：MTP mirror source是target逻辑layer0。
- `kv_mirror_layers[0]=48`：MTP mirror consumer是逻辑layer48。

因此严格语义是 `source0 -> consumer48`。`48` 同时是：

- checkpoint 中 MTP 权重所在的 layer id；
- 唯一 MTP layer 的 checkpoint/config logical id；
- `kv_mirror_layers` 的第一个、且唯一越过target边界的MTP consumer。

`config.json`还包含`1→47, 2→46, ..., 15→33`，这些是target内部mirror pair，
不是额外MTP层，不能被MTP校验拒绝。

draft runtime 为节省 KV cache，只实例化一层，并把它存储为 draft 本地 physical slot 0。这个 local slot 0 只是存储压缩：

```text
checkpoint/logical layer 48  ->  draft physical layer slot 0
```

它绝不表示 “layer 48 等价于 target layer 0”，也不改变 mirror 的 `0 -> 48` 语义。

模型构造/加载时只做语义完整性校验：

```python
kv_mirror_imitated_layers[0] == 0
kv_mirror_layers[0] == 48
仅有一个 kv_mirror consumer >= num_target_hidden_layers
num_nextn_predict_layers == 1
num_target_hidden_layers == 48
num_target_hidden_layers + physical_layer_id == 48
```

这不是部署配置白名单，而是防止把错误权重或错误 mirror 对应关系静默加载。

### 3.1 logical48 的 layerwise 属性

所有layerwise数组都按logical/config layer id索引，不能按draft physical slot0索引。
对当前`config.json`：

```text
sliding_window_size_layerwise[48] = 512
enable_attn_sink_layerwise[48] = true
```

NEWSGLANG把checkpoint窗口解释为“包含当前token的总跨度”，所以raw 512规范化为
`sliding_window_size=511`（左侧511个历史token+当前token，总跨度512）。raw
262144达到有效full-attention上限并规范化为`-1`。因此logical48/physical0是SWA
层，不是full-attention层。

`use_sliding_window=false`和`max_window_layers`不得覆盖WeLM显式的49项layerwise
数组。sink同样取index48并加载checkpoint layer48自己的per-head`attn_sink`权重；
同一个physical MTP层在每个speculative step复用同一组window/sink/norm/MLP属性。

## 4. 最终数据流

### 4.1 target prefill（eager）

1. target 正常执行 prompt prefill。
2. target layer 0 除本层 Q/K/V 外，同时产生为 consumer48 训练的 `mirror_k/mirror_v`。
3. mirror K/V 以 `LogitsProcessorOutput.model_specific_states` 返回；它们仍是 consumer48 的原始 K/V projection 输出，尚未经过 consumer48 自己的 K norm/RoPE。
4. target 正常采样首个 bonus token；现有 `NgramEmbeddingManager.update_after_decode()` 把该 token 写入 canonical token table。
5. `_draft_extend_for_prefill()` 构造 shifted draft input，将 target hidden states、mirror states 一并交给 NextN 模型。
6. draft physical slot 0（逻辑 48）计算自己的 Q，读取 source0 mirror K/V，再按 layer48 的 K norm/RoPE 写入 draft KV cache。

prefill 不捕获 Graph。

### 4.2 steady decode

每轮按以下顺序执行：

```text
draft
  -> target_verify
  -> eagle_sample / accept_index / accept_lens
  -> commit accepted tokens to canonical ngram table
  -> on_publish
  -> draft_extend
```

具体语义：

1. `draft` 的 active layer48始终是KVMirror consumer：draft step只计算Q，读取
   已由前一轮draft-extend用source0 mirror K/V填好的frozen draft KV cache；不在
   speculative step上用layer48 hidden state重新计算K/V。
2. topk1 draft 的 attention/RoPE position 固定为进入本轮时的 `base_position = seq_lens - 1`；KV prefix length 同样固定为进入本轮时的 `seq_lens`。所有 step 都不得执行通用 EAGLE 的 position/metadata 递增。
3. `draft_forward()` 的第一个实际输入是前一轮 `draft_extend` 已经产出的第一个 draft token，不是当前 bonus/root。因此它的 logical token column 是 `seq_lens+1`，OE entry history从 canonical table 的 `seq_lens`（当前bonus/root）、`seq_lens-1` 读取。每一步 hash 后把当前 draft input token推入 draft-local history；后续 draft token只更新local history，不提前写canonical token table。
4. `target_verify` 对宽度 `D` 的线性 topk1 candidate 链计算 OE hash；第0行 logical column 是 verify 前的 `old_seq_lens`，后续行依次加1。target 同时输出每个 verify row 对应的 source0 mirror K/V。
5. `eagle_sample()` 得到 `predict`、`accept_lens`、`accept_index`。
6. 固定形状 NPU kernel 只把 accepted emitted token 写入 canonical table。verify 前 `token_table[old_seq_lens]` 已经保存 root/bonus，因此第 `j` 个 accepted `predict` 必须写入 `old_seq_lens + 1 + j`。`accept_lens` 已包含本轮 trailing bonus，每个请求写入恰好 `A=accept_lens[b]` 个 token。
7. `accept_index` 同时作为 mirror row 的选择映射，随 target 输出 states 交给 `draft_extend`。
8. `draft_extend`按`accept_index`读取target source0 mirror K/V，执行layer48自己的
  K norm/RoPE后更新frozen draft KV cache，同时产生下一轮draft的第一个query。
   它的第0个 input token logical column 是 `old_seq_lens + 1`，即相对对应的 target query/mirror row 向后偏移1；不能从 attention `positions` 反推 OE logical column。

当`S>1`时不会再次运行target source0。后续step复用同一physical layer48、同一
mirror-origin KV cache，并通过token/OE embedding与recurrent hidden state更新Q。
perf的merged extend-draft路径会让`model_specific_states`在整个step loop内保持可见，
但那是同一轮mirror KV-fill数据，不是每个step产生一份新mirror。

topk1 是线性树，verify row、hidden-state row、mirror-state row 和 `accept_index` 使用同一索引空间。bonus 的 mirror 也遵循现有 Spec V2/CUDA 参考实现的 verify-row 对齐，不额外执行一次 target forward。

### 4.3 两套坐标必须分离

`attention_positions` 只服务 RoPE/attention；`oe_logical_columns` 只服务 canonical token table/ngram。对 steady decode，设 verify 前 `old_seq_lens=L`、接受数 `A`：

| 阶段 | attention/query position | OE logical token column |
|---|---|---|
| 本轮 draft 第0个实际forward输入 | `L-1`，后续 step 仍固定 | `L+1`，后续 step 逐次 `+1`；entry history最新token在`L` |
| target_verify row `j` | 沿用 Spec V2 verify position | `L+j` |
| accepted `predict[j]` commit | 不适用 | 写入 `L+1+j` |
| draft_extend row `j` | target row `accept_index[j]` 对应位置 | `L+1+j` |
| 下一轮 draft 第0个实际forward输入 | `L+A-1` | `L+A+1`；entry history最新token在`L+A` |

禁止使用一个 `positions` 加模糊的 `history_shift` 同时推导这两套坐标。每个 hash helper 必须接收明确的 logical column start/entry history；Graph 模式也使用同一公式。

## 5. 两个新增数据契约

### 5.1 mirror output contract

在 `LogitsProcessorOutput` 增加通用可选字段：

```python
model_specific_states: Optional[Dict[str, Any]] = None
```

WeLMV4 使用固定 key：

```python
WELMV4_MTP_MIRROR_STATES_KEY = "welmv4_mtp_mirror_states"
```

value 结构：

```python
{
    "welmv4_mtp_mirror_states": {
        48: (mirror_k, mirror_v),
    }
}
```

其中 `mirror_k/mirror_v` shape 为 `[num_target_rows, local_kv_width]`，第一维严格与 target forward 输出行对齐。

### 5.2 draft-extend input contract

在 `EagleDraftInput` 和 `EagleDraftExtendInput` 增加 mirror 字段，并在 `EagleDraftInput` 增加 frozen-KV 标记：

```python
model_specific_states: Optional[Dict[str, Any]] = None
mirrored_kv_indices: Optional[torch.Tensor] = None
welmv4_mtp_frozen_kv: bool = False
```

- prefill draft-extend：target rows 与 draft-extend rows 等宽，默认 identity；`mirrored_kv_indices` 可为 `None`。
- decode draft-extend：`mirrored_kv_indices = accept_index.reshape(-1)`；负数 padding 在 Graph input copy 时安全替换为 0，实际有效性由 `num_accept_tokens` 控制。
- `welmv4_mtp_frozen_kv=True` 只用于 WeLMV4 topk1 steady draft。它同时约束 worker position、Ascend eager metadata 和 NPU Graph replay metadata；不能只解释为 `save_kv_cache=False`。
- 两个字段只在 `target_verify -> draft_extend` 的同一轮中使用。draft-extend 完成后立即从 `next_draft_input` 清空，不进入下一轮 `filter_batch/merge_batch`，也不进入 overlap FutureMap。

OE hash 不作为跨阶段大 tensor 传递。target_verify、draft 和 draft-extend 都从同一 canonical token table 加各自当前 token 链，在设备侧计算本阶段 hash。

## 6. ngram/OE 设计

### 6.1 canonical token table 只有一个 owner

target `ModelRunner.ngram_embedding_manager.table` 是唯一 canonical table，按 `req_pool_idx` 存储已经正式进入输出序列的 token。

draft 初始化后：

- draft manager 通过 `NgramEmbeddingManager.share_table_from(target_manager)` 复用 target table，而不是再维护一份独立历史。
- NextN 模型不再额外保存第二个 table 引用；target/draft/Graph 都从 `forward_batch.ngram_embedding_info.token_table` 读取同一地址，避免出现两个所有权入口。
- Graph capture 前必须完成 table 共享，保证 capture/replay 看到同一稳定地址。

写入规则：

- 普通 target prefill/decode：继续使用现有 `update_after_decode()`。
- speculative draft：不写 canonical table。
- target verify：在 accept 结果产生后，由新增 `commit_speculative_accepts()` 写入 accepted token。
- rejected token 永不写入。

### 6.2 OE hash 数学保持现有 WeLMV4 语义

仅支持当前 checkpoint 的四路 grams `(2, 2, 3, 3)`。对当前 token `x0`、前一 token `x1`、前二 token `x2`：

```text
packed2 = x0 + x1 * vocab_size
packed3 = packed2 + x2 * vocab_size^2
hash(x) = (x * 2654435761) & 0xffffffff
id[0] = hash(packed2) % oe_vocab_sizes[0]
id[1] = hash(packed2) % oe_vocab_sizes[1]
id[2] = hash(packed3) % oe_vocab_sizes[2]
id[3] = hash(packed3) % oe_vocab_sizes[3]
```

边界位置的历史 token 填充值必须与当前 `welmv4.py` eager fallback 完全一致，以 CPU reference test 锁定，不在新路径中重新定义。

### 6.3 四类 NPU helper

在 `welmv4_npu_op.py` 增加以下固定形状/固定上界 helper；wrapper 名称可按项目风格微调，但职责不得混合：

1. `welmv4_mtp_init_oe_history_npu`
   - 输入：canonical table、`req_pool_indices`、显式 `current_token_columns`。
   - 输出：每个请求当前输入 token之前的两 token history，即读取 `column-1`、`column-2`，越界填充值与现有 WeLM eager 语义一致。
   - steady draft step0 的 `current_token_columns=seq_lens+1`，因此读取的entry history是`seq_lens/seq_lens-1`；禁止传 attention position `seq_lens-1`。

2. `welmv4_mtp_oe_hash_step_4way_npu`
   - 输入：当前 draft token `[B]`、history `[B,2]`。
   - 输出：hashed ids `[4,B]` 和 next history `[B,2]`。
   - draft 循环使用 ping-pong history buffer；无 `.item()`、`nonzero()` 或动态 shape。

3. `welmv4_mtp_oe_hash_linear_4way_npu`
   - 输入：flat token rows、request row start/len 或固定宽度 `W`、canonical table、`req_pool_indices`、显式 `logical_column_starts`。
   - helper 对每个请求把当前 flat token rows 视为连续 token 链；链内前序 token直接取本次 input，只有 row0 之前的历史从 canonical table读取。
   - `target_verify`：`W=D`、所有 D rows 有效、`logical_column_starts=old_seq_lens`。
   - decode `draft_extend`：`W=D`、`valid_lens=accept_lens`、`logical_column_starts=old_seq_lens+1`。也可以由 Graph 内已有 metadata 等价计算为 `extend_prefix_lens+1`，但必须通过单测证明等价。
   - shifted prefill draft-extend：ragged request layout，首行 logical column 为 target 对应行的 column `+1`。首个 shifted token `x[p+1]` 的前一 token是 canonical table 中的 `x[p]`，不能错误读取成 `x[p-1]`。
   - 输出固定布局 `[4, num_rows]`。

4. `welmv4_mtp_commit_accepted_tokens_npu`
   - 输入：canonical table、`predict`、`accept_index[B,D]`、`accept_lens[B]`、old `seq_lens[B]`、`req_pool_indices[B]`。
   - grid 固定覆盖 `[B,D]`，只在 `j < accept_lens[b]` 时写入：

```python
valid = j < accept_lens[b] and accept_index[b, j] >= 0
safe_src = max(accept_index[b, j], 0)
dst = token_table[req_pool_indices[b], old_seq_lens[b] + 1 + j]
src = predict[safe_src]
# only perform dst = src when valid
```

   - invalid/padding index 先安全 clamp，再由 `valid` mask 阻止读写。kernel 必须先形成 safe source index，不能在 mask 生效前发生负索引读取。

### 6.4 各阶段如何调用

- ordinary target EXTEND/DECODE：原实现不变。
- target TARGET_VERIFY：`welmv4.py::_compute_oe_embedding()` 检测该 mode，调用 fixed-width linear helper。
- draft DECODE：`EagleDraftWorker.draft_forward()` 在每个实际 MTP forward 前传入显式prev1/prev2 token history；step0 logical column=`seq_lens+1`，后续step逐次`+1`。NextN模型在forward内计算step hash，不从attention position重建历史。
- draft DRAFT_EXTEND_V2：NextN 模型用 `num_accept_tokens` 调 linear helper。
- draft prefill EXTEND：NextN 模型用 shifted ragged helper。

`S=1` 时 draft loop 不一定执行新的 MTP forward，但 target verify 和 draft-extend 仍各自计算正确的 OE hash。

## 7. KV mirror 设计

### 7.1 target source0 输出

保留当前 `LayerManager.post_init()` 的权重重写方式：加载 layer48 后，把 layer48 的 K/V projection 权重拼入 target layer0 的 source projection。为保证 ModelSlim MXFP8 的 `weight/weight_scale` 仍处于 checkpoint `[N,K]` 原始布局时完成拼接，target load 临时实例化只含 layer48 QKV 的 staging layer；配对结束后立即从 model 和 `LayerManager` 删除。随后 draft 正常加载自己的 layer48，并只提取 Q projection，不再次改写已经 relayout 的 target source0。target layer0 一次 GEMM 产出：

```text
q0, k0, v0, mirror_k_for_48, mirror_v_for_48
```

当前 `KVMirrorManager` 仍是类级全局字典；`ForwardBatch` 虽然已经有通用的
`model_specific_states` 字段，但 WeLMV4 尚未使用。source0→consumer48 是
target→draft 的跨模型、跨阶段数据，不能继续依赖该全局字典。

target layer0 把 `(mirror_k_for_48, mirror_v_for_48)` 写入本次
`ForwardBatch.model_specific_states`。顶层 CausalLM forward 再把该字典挂到
`LogitsProcessorOutput`。旧 `KVMirrorManager` 仅保留给与本功能无关的、同一次
model forward 内的 legacy mirror pair；严格的 source0→consumer48 路径不得
写入或读取它。

### 7.2 consumer48 的两种 attention 模式

必须把当前基于 `is_extend_without_speculative()` 的判断改成明确的 mode 判断：

```python
use_external_mirror_fill = self.is_nextn and (
    forward_batch.forward_mode.is_extend_without_speculative()
    or forward_batch.forward_mode.is_draft_extend(include_v2=True)
    or getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False)
)
use_frozen_kv_query = self.is_nextn and forward_batch.forward_mode.is_decode()
```

- prefill `EXTEND`和decode `DRAFT_EXTEND_V2`：Q-only，并从
  `spec_info.model_specific_states[KEY][48]`读取source0 K/V完成cache fill。
- `draft/DECODE`：Q-only、`save_kv_cache=False`、`out_cache_loc` 不参与本层写入，只读取已经由mirror填好的 layer48 frozen KV cache；不得执行full QKV。所有 step 的 `positions=seq_lens-1`，attention metadata 的 KV length固定为进入本轮时的 `seq_lens`。
- 若沿用perf merged extend-draft，则KV-fill rows与draft query在一个forward中
  合并；语义仍是external mirror fill + Q-only query。

perf的`NextnMirrorQProjection`会显式拒绝普通active full-QKV draft decode，这说明
full QKV不是该模型的正确fallback。NEWSGLANG必须在model/attention backend中提供
Q-only frozen-KV decode hook，或在现有`EAGLEWorkerV2`内整合最小merged路径；不新建
WeLM专用worker。

consumer48 取到原始 mirror K/V 后，继续执行 layer48 自己的 K norm、RoPE、attention 和 cache store。禁止在 target 侧提前应用 layer48 RoPE。

### 7.3 Graph 模式下的 mirror 输入输出

target verify Graph：

- Graph capture 得到静态 mirror output tensors。
- `NPUGraphRunner.execute()` 返回结果时，像处理 hidden states 一样，把 `model_specific_states` 中各 mirror tensor按 `raw_num_token` 切片后保留。

draft-extend Graph：

- 为每个 mirror pair 分配静态 K/V input buffer `[max_num_token, local_kv_width]`。
- 分配静态 index buffer `[max_num_token]`。
- capture 时把这些 buffer 绑定到 dummy `EagleDraftExtendInput`。
- replay 前把本轮 target states 和 `mirrored_kv_indices` copy 到对应静态 buffer；
  padding rows清零，padding index置 0。
- OE hash不新增Graph input buffer。target/draft共享的canonical token table地址固定，
  replay使用已有静态 `input_ids/req_pool_indices/seq_lens` 在Graph内执行MTP hash，
  与普通 WeLM decode Graph 使用同一数据所有权规则。
- eager fallback 直接使用本轮 tensor，不经过静态 buffer。

draft Graph：

- 不添加 external mirror input buffer；它只读已经 mirror-fill 的 KV cache并计算Q。
- capture dummy `EagleDraftInput` 必须设置 `welmv4_mtp_frozen_kv=True`。
- replay 前仍复用现有 `seq_lens/positions/req_pool_indices/topk/hidden_states` 静态 buffer；其中真实 `positions` 在进入 runner 前已经写成 `seq_lens-1`。
- captured `draft_forward()` 对 WeLM 不执行 `positions.add_(1)`，所以 capture 后也不能执行通用的 `positions.sub_(S-1)` 恢复逻辑。
- NPU replay 给每个 draft step 更新 attention metadata 时，所有 step 都使用同一个真实 `seq_lens`，不能继续使用 `seq_lens + step + 1`。
- shared canonical table 必须已经出现在现有 `ngram_embedding_info` Graph buffer 中；capture 前断言其地址与 target manager table相同。OE step hash在Graph内从该固定地址和 draft-local ping-pong history计算。

本方案不采用 perf 的独立 merged draft Graph；若未来改用 merged 路径，仍必须保持三阶段独立命中/回退和相同 mirror 静态输入契约。

## 8. Graph 与 overlap 四象限

### 8.1 三个 Graph 决策必须独立

沿用现有独立判断：

```python
draft_can_graph = (
    draft_graph_runner is not None
    and draft_graph_runner.can_run_graph(draft_forward_batch)
)

target_can_graph = (
    target_graph_runner is not None
    and target_graph_runner.can_run(target_verify_forward_batch)
)

draft_extend_can_graph = (
    draft_extend_graph_runner is not None
    and draft_extend_graph_runner.can_run_graph(draft_extend_forward_batch)
)
```

变量名按当前类真实 API 使用（当前代码中部分 runner 方法名是 `can_run_graph`），原则是三者绝不互相推导。以下组合都必须正确：

- draft Graph，target verify eager，draft-extend Graph。
- draft eager，target verify Graph，draft-extend eager。
- target verify Graph 命中，但 draft 或 draft-extend 因 bucket 不匹配回退 eager。
- 三阶段都 eager 或都 Graph。

不新增 Graph hit/miss/fallback 指标。现有 eager fallback 足够保证正确性；首版通过测试矩阵覆盖独立 fallback。

独立判断只决定执行器，不得改变本轮数据所有权：target Graph/eager 都返回同形状 mirror states；draft Graph/eager 都使用 frozen position/KV metadata；draft-extend Graph/eager 都使用相同 `accept_index` gather 和 `logical_column_starts=old_seq_lens+1`。

### 8.2 overlap 关闭

- 数据按同一调用栈同步传递。
- canonical token commit 在 `run_eagle_verify()` 内完成。
- eager/Graph 只影响每个阶段的执行器，不改变数据契约。

### 8.3 overlap 开启

- 不修改 scheduler overlap 协议。
- `on_publish` 的位置仍保持在 target prefill/verify 之后、draft-extend 之前。
- accepted token commit 必须在 `run_eagle_verify()` 返回前完成，因此必然先于 `on_publish`。
- mirror states 与现有 target hidden states具有相同生命周期：
  `batch_result.logits_output` 和 `extra_keep_alive_refs` 持有引用。
- mirror K/V 是 target forward 的本轮私有输出；draft-extend Graph 的 mirror/index
  静态 buffer 只归 draft-extend runner 所有。scheduler、FutureMap 和 schedule stream
  都不读写这些 tensor，因此不需要 WAR barrier。
- target verify、mirror静态input copy和draft-extend replay都提交到当前forward
  stream，天然保持 target→copy→replay 顺序；禁止另开 staging thread/stream。
- canonical token table沿用现有 `NgramEmbeddingManager` 的所有权规则：accepted-token
  commit在forward stream上先于draft-extend hash；steady decode的schedule prep不会
  改写这些请求行。不得为了WeLM增加scheduler barrier或新同步协议。
- `scheduler.py` 不改；不要求 `SGLANG_ENABLE_WAR_BARRIER`。
- 不扩展 FutureMap，不修改 `overlap_utils.py`，不增加新的 scheduler record 类型。

### 8.4 四象限预期

| Graph | overlap | 行为 |
|---|---|---|
| off | off | 三阶段 eager；直接传 mirror；设备侧 fixed-shape ngram commit |
| on | off | 三阶段独立 Graph/eager；target 输出和 draft-extend 输入用静态 buffer |
| off | on | eager tensor 生命周期随 batch result；commit 先于 publish |
| on | on | target→mirror/index静态copy→draft-extend replay同属forward stream；scheduler不接触私有buffer，不增加WAR barrier |

## 9. 具体文件修改范围

下面按实施顺序列出生产代码的最小修改范围。行号以基线 commit 为准，实施时以类/函数名为锚点。

### 9.1 `python/sglang/srt/server_args.py`

位置：WeLMV4 分支，当前约 5069–5130 行。

改动：

- 删除 5111–5116 行对所有 speculative algorithm 的 blanket reject。
- 不替换为新的大白名单或部署参数门禁。
- 保留现有与本任务无关的 WeLMV4 限制及 backend 选择逻辑。
- 更新 “speculative 被拒绝” 的过时注释。

结果：配置可以进入现有 NEXTN/EAGLE Spec V2 初始化流程。

### 9.2 `python/sglang/srt/configs/model_config.py`

位置：`WELMV4_MODEL_ARCHS`、`_config_draft_model()`、hybrid layer 分类。

改动：

- 将 `WeLMV4MoeForCausalLMNextN` 加入 WeLMV4 architecture 集合。
- 在 `_config_draft_model()` 增加转换：

```python
if is_draft_model and arch == "WeLMV4MoeForCausalLM":
    if hf_config.num_target_hidden_layers is None:
        hf_config.num_target_hidden_layers = hf_config.num_hidden_layers  # 48
    hf_config.architectures[0] = "WeLMV4MoeForCausalLMNextN"
    hf_config.num_nextn_predict_layers = 1
    hf_config.num_hidden_layers = hf_config.num_nextn_predict_layers  # physical draft layers
```

- 不新增 `welmv4_target_num_hidden_layers` 或 `welmv4_mtp_layer_id`。统一沿用
  perf 已有的 `num_target_hidden_layers`；logical layer id按
  `num_target_hidden_layers + physical_layer_id` 计算。
- NextN hybrid/SWA分类不能按local layer0误取target layer0的配置；physical slot0
  使用logical id `num_target_hidden_layers + 0 = 48`选择checkpoint中layer48的
  窗口属性。
- `get_hybrid_layer_ids()` 对 NextN 调用 WeLM layerwise helper 时显式使用
  `layer_offset=num_target_hidden_layers`，返回本地 hybrid id `[0]`；不得返回 logical48 作为本地 pool id。
- `get_attention_sliding_window_size()`/KV pool sizing 对 NextN physical0读取 logical48 的 raw window 512，并规范化为 left-history 511。
- `_detect_attention_sinks()`/`has_attention_sinks` 把 NextN architecture 视为 WeLMV4，并读取 logical48 的 `enable_attn_sink_layerwise[48]=true`。
- `ModelConfig.num_attention_layers` 对 draft 为 1，避免分配 48 层 draft KV cache。

### 9.3 `python/sglang/srt/models/welmv4_nextn.py`（新增）

新增 `WeLMV4ModelNextN` 和 `WeLMV4MoeForCausalLMNextN`，复用 `welmv4.py` 中的 block、OE embedding、logits processor 和 loader 基础设施。

必须实现：

- 只实例化一个 `Qwen2MoeDecoderLayer(..., layer_id=0, config_layer_id=48, is_nextn=True)`；参数名可以按项目风格调整，但两个 id 不得继续共用一个变量。
- 该层 logical/config id取`config.num_target_hidden_layers + physical_layer_id`，本例为
  `48 + 0 = 48`；physical cache layer id保持0。
- NextN 覆盖/实现 `get_attention_sliding_window_size()` 时传 `num_layers=1, layer_offset=num_target_hidden_layers`，确保KV pool为physical0分配但读取logical48的SWA配置。
- MTP 输入融合：token/OE embedding 经 `enorm`，target hidden state 经 `hnorm`，concat 后过 `eh_proj`，再送入唯一 decoder layer。
- active draft step使用Q-only frozen-KV attention；`EXTEND`/`DRAFT_EXTEND_V2`
  使用external source0 mirror K/V完成cache fill。不得把full QKV作为active decode
  fallback。
- 通过 `forward_batch.ngram_embedding_info.token_table` 读取共享 canonical table，不在模型上注册或保存第二份 table 引用；分别执行 shifted prefill、draft step 和 fixed-width draft-extend hash。
- `lm_head`/base embedding/OE embedding 继续通过 Spec worker 现有 `get_embed_and_head()` / `set_embed_and_head()` 与 target 共享。
- `load_weights()` 调用/复用 `WeLMV4MoeForCausalLM.load_weights(..., is_nextn=True)`。
- `EntryClass = WeLMV4MoeForCausalLMNextN`。

当前 `ModelRegistry` 会用 `pkgutil.iter_modules()` 扫描 `sglang.srt.models` 下所有带 `EntryClass` 的模块，因此新增该文件后不需要再修改显式registry映射；单测必须直接调用`ModelRegistry.resolve_model_cls(["WeLMV4MoeForCausalLMNextN"])`确认未被import error静默跳过。

构造时不得调用 target CausalLM 的完整 `__init__()`，否则会清空已经注册好的 target `LayerManager.decoder_layer`。NextN 构造只初始化自己的 module，并让当前 target 0–47 与 draft logical48 同时存在于 load-time `LayerManager` 中，供 mirror projection fixup 使用。

### 9.4 `python/sglang/srt/models/welmv4.py`

改动点：

1. 定义 mirror state key 和 get/set helper。
2. 在 `Qwen2MoeDecoderLayer`、attention 以及所有 layerwise 属性使用点引入明确的 `physical_layer_id/config_layer_id` 分工：
   - RadixAttention、KV pool、local decoder module index使用 physical0；
   - window/sink/prenorm/o_norm/MLP/expert layerwise配置、LayerManager key、checkpoint命名使用 logical48。
   - 删除/替换当前NextN通过`layer_idx + len(LayerManager.decoder_layer)`推导consumer id的构造顺序依赖，直接使用`config_layer_id=48`。
3. target source layer0 产生 mirror K/V 时：
   - source0→consumer48 只写入本次 `forward_batch.model_specific_states`；
   - 不把该跨阶段 pair 写入类级 `KVMirrorManager`。
4. consumer48 只读取 `spec_info.model_specific_states`，并用
   `mirrored_kv_indices` 选择行；普通、与本功能无关的同模型 mirror pair 仍可保留
   现有 manager。
5. 把NextN active路径统一为Q-only：draft-extend读取external mirror做KV fill，
   draft decode设置`save_kv_cache=False`并读取frozen mirror cache；同步修正
   `forward()`和`get_qkv_prefetch_weight()`，不得出现full-QKV active fallback。
6. 把 OE embedding 中 “hashed ids -> 四路 embedding -> concat -> projection -> add base embedding” 抽成共享 helper，供 target 与 NextN 调用。
7. TARGET_VERIFY 模式调用 `welmv4_mtp_oe_hash_linear_4way_npu()`，显式传 `logical_column_starts=old_seq_lens`，不使用普通 decode 的单 token 假设。
8. 顶层 `forward()` 先得到 `logits_output`，再挂载 `forward_batch.model_specific_states` 后返回；split-prefill 不在首版范围，不为其增加 speculative 分支。
9. `load_weights(is_nextn=True)` 使用perf已有的
   `config.num_target_hidden_layers` 做layer48→local decoder0映射，避免依赖构造
   顺序隐式推断target层数；不引入WeLM私有别名字段。
10. `post_init_after_load_weights(is_nextn=True)`验证唯一跨target边界的MTP pair
   `(consumer=48, source=0)`，允许配置中的target内部mirror pairs；完成source0
   extra K/V与consumer48 Q-only projection fixup。
11. target loader为跨模型consumer48创建临时QKV staging：先在原始checkpoint布局
    上把layer48 K/V并入source0，再释放staging；draft loader随后只提取layer48 Q。
    staging缺少任一Q/K/V权重时立即失败，避免用未初始化权重静默配对。
12. 更新顶层 forward 中 “speculative rejected” 的过时注释。

### 9.5 `python/sglang/srt/layers/logits_processor.py`

在 `LogitsProcessorOutput` 增加：

```python
model_specific_states: Optional[Dict[str, Any]] = None
```

它是通用可选扩展，默认 `None`，其他模型行为不变。

### 9.6 `python/sglang/srt/model_executor/model_runner_components/ngram_embedding_manager.py`

改动：

- 新增 `share_table_from(owner)`，返回共享同一table地址的新manager；保留 frozen dataclass，不做可变字段赋值。
- 共享前校验 `enabled/n/k/table shape/table dtype` 一致。
- 新增 `commit_speculative_accepts(...)`，NPU 路径调用 fixed-shape accepted-token commit kernel；函数契约明确规定 `old_seq_lens` 是 verify 前长度，目标列从 `old_seq_lens+1` 开始，不能复用普通 sampling 从 `seq_lens` 开始的写法。
- Graph 初始化前断言 draft manager 的 `table.data_ptr()` 与 target manager相同，且现有 `ngram_embedding_info.token_table` 指向该地址；不在 model 中另存 table。
- eager reference 路径可用二维固定循环/张量操作实现，主要用于单测；运行时 NPU 不允许 `nonzero()` 动态压缩。
- 更新文件头说明：该 manager 同时服务 LongCat 与 WeLMV4，而不再只描述 LongCat。

### 9.7 `python/sglang/srt/layers/welmv4_npu_op.py`

复用现有 ragged prefill/decode OE hash helper，并新增 active-draft显式history hash和accepted-token commit两个NPU helper；职责合起来覆盖第6.3节的四类语义，不复制已有kernel。

实现要求：

- 所有 decode/verify/draft-extend 路径 shape 固定或有固定上界。
- 不在 Graph 内调用 `.item()`、CPU round-trip、`nonzero()`、boolean compaction。
- hash 乘法严格复现 uint32 overflow。
- 所有 hash API 使用显式 `current_token_columns` 或 `logical_column_starts`；禁止从 attention `positions` 隐式推导 OE token column。
- accepted commit 的 destination 固定为 `old_seq_lens+1+j`，并在形成 safe source index 后才读取 `predict`。
- `TP>1` 下输入 token/hash 在各 rank 一致；OE embedding 的 TP 汇聚继续沿用当前模型实现。
- padding rows 的读取有安全地址，写入受有效 mask 控制。

### 9.8 `python/sglang/srt/speculative/eagle_info.py`

改动：

- 为 `EagleDraftInput` 增加：
  - `model_specific_states`
  - `mirrored_kv_indices`
  - `welmv4_mtp_prev1_tokens` / `welmv4_mtp_prev2_tokens`（仅 draft loop 内临时使用）
  - `welmv4_mtp_frozen_kv: bool = False`
- 为 `EagleDraftExtendInput` 增加：
  - `model_specific_states`
  - `mirrored_kv_indices`
- idle constructor 的可选 tensor/dict字段默认 `None`，`welmv4_mtp_frozen_kv` 默认 `False`。
- `welmv4_mtp_frozen_kv` 在真实 steady draft 和 draft Graph capture dummy input 中固定为 `True`；其他模型/阶段为 `False`。
- mirror 两字段在 draft-extend 后清空，不加入 FutureMap。
- 若防御性支持 `filter_batch/merge_batch`，只允许字段为 `None`；检测到未清空则显式报内部生命周期错误，避免错误地跨轮 merge 大 tensor。

### 9.9 `python/sglang/srt/speculative/eagle_worker_v2.py`

`EAGLEWorkerV2` 不派生新子类，只增加小型 WeLMV4 hook：

初始化：

- 识别 draft architecture `WeLMV4MoeForCausalLMNextN`。
- 在 Graph capture 前，让 draft manager 共享 target canonical table，并校验 Graph `ngram_embedding_info` 使用同一地址；不把 table 绑定成 NextN model 的第二个属性。

`EagleDraftWorker.prepare_for_draft()` 返回后、执行独立 Graph 判断前：

- 设置 `spec_info.welmv4_mtp_frozen_kv=True`。
- 把 `forward_batch.positions` 从通用 EAGLE 的 `seq_lens` 改为 `seq_lens-1`，并保存为本轮固定 base position。
- attention KV prefix length保持进入本轮时的 `seq_lens`；不为后续 step增加长度。
- Q-only layer忽略通用 draft write slots，不让 `out_cache_loc` 导致任何 layer48 K/V store。

`EagleDraftWorker.draft_forward()`：

- 在 WeLMV4 topk1 draft loop 开始时初始化 `[B,2]` OE history。
- history初始化使用 `current_token_columns=seq_lens+1`，读取 canonical table 的 `seq_lens/seq_lens-1`；当前第一个draft `input_ids`尚未提交到table，不能把它误当成`table[seq_lens]`。
- 每个实际 MTP forward 前把显式prev1/prev2历史赋给spec info；NextN forward内的
  NPU helper据此计算`[4,B]` hash。
- draft 选出下一 token 后只更新 local history。
- WeLM 分支跳过当前 NPU topk1路径的 `forward_batch.positions.add_(1)`；每一步 forward 前均断言/恢复 `positions==base_positions`。
- draft 结束清空临时 hash 引用。
- 不改变通用 `build_eagle_verify_input()` 的树构造。

`_draft_extend_for_prefill()`：

- 新增 `target_model_specific_states` 参数，调用处传 `batch_output.logits_output.model_specific_states`。
- 构造 `EagleDraftExtendInput` 时携带 states；mirror index 为 identity/`None`。
- NextN 使用 shifted prefill hash，logical column比对应 target hidden/mirror row明确 `+1`。

`_draft_extend_for_decode()`：

- 从 `batch_result.next_draft_input` 取出 target states 和 flattened `accept_index`。
- 把 target states 和 flattened `accept_index` 放入 `EagleDraftExtendInput`，再执行
  现有独立 Graph 判断。NextN 在 forward 内用本轮 input ids、共享 canonical table
  和 `num_accept_tokens` 调用 fixed-width OE hash helper；`logical_column_starts`
  明确为 verify 前 `old_seq_lens+1`，或由 `extend_prefix_lens+1` 等价得到；不跨阶段传大 hash tensor。
- draft-extend 完成后清空 transient states/index，再填充下一轮正常的 topk、hidden states、bonus token。

外层 `forward_batch_generation()`：

- 保持 `on_publish` 位置不变。
- 不把 target Graph 命中结果传给 draft/draft-extend；三个阶段继续独立 fallback。

### 9.10 `python/sglang/srt/speculative/eagle_worker_common.py`

在 `run_eagle_verify()` 中，紧跟 `eagle_sample()`：

```python
predict, accept_lens, accept_index = eagle_sample(...)

old_seq_lens = batch.seq_lens  # verify 前的 L，语义只读

if is_welmv4_target and ngram_manager.enabled:
    ngram_manager.commit_speculative_accepts(
        predict=predict,
        accept_index=accept_index,
        accept_lens=accept_lens,
        old_seq_lens=batch.seq_lens,
        req_pool_indices=batch.req_pool_indices,
    )
```

随后构造：

```python
next_draft_input = EagleDraftInput(
    bonus_tokens=bonus_tokens,
    model_specific_states=logits_output.model_specific_states,
    mirrored_kv_indices=accept_index.reshape(-1),
)
```

`commit_speculative_accepts()` 对第 `j` 个有效接受项执行：

```python
safe_src = clamp(accept_index[b, j], min=0)
if j < accept_lens[b] and accept_index[b, j] >= 0:
    token_table[req_pool_indices[b], old_seq_lens[b] + 1 + j] = predict[safe_src]
```

不得写 `old_seq_lens+j`，也不得把 candidate/draft token替代 `predict` 提交到 canonical table。commit kernel提交后，同一 forward stream上的 draft-extend Graph/eager hash才能读取 `old_seq_lens+1...old_seq_lens+A`。

内部 dispatch 只按 target model architecture 和 manager 是否 enabled 识别 WeLMV4，不增加用户启动参数门禁。

### 9.11 `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`

改动：

- 将 `WeLMV4MoeForCausalLMNextN` 加入 WeLM Graph architecture 识别（即使通常由专用 draft runner执行，也保持 runner 判定完整）；这不表示 WeLM MTP可以回退FIA。
- 在 `execute()` 重建 `LogitsProcessorOutput` 时保留 `model_specific_states`。
- 新增小 helper：对 `welmv4_mtp_mirror_states` 的每个 K/V tensor 按 `raw_num_token` 切片，字典本身浅复制，不能修改 capture-time 原字典。
- capture 输出中的 mirror tensor必须保持静态地址；`execute()` 返回的是按真实 row数创建的 view/slice，不能原地缩小或覆盖 capture buffer。
- 其他 model-specific state 原样或按其既有规则处理；首版只为 WeLM mirror 定义 row slicing。

不改变 `_update_inputs()`、`replay_with_input_update()` 或 backend 线程顺序。

### 9.12 `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`

该文件只增加三个默认 no-op 的 protected hook，避免在 NPU subclass 复制整个 capture/execute：

```python
_init_model_specific_buffers()
_bind_model_specific_capture_inputs(spec_info, num_tokens)
_copy_model_specific_replay_inputs(forward_batch, raw_bs, bs, num_tokens)
```

调用点分别位于：

- 通用静态 buffer 分配完成、Graph capture 开始之前；
- capture dummy `EagleDraftExtendInput` 构造完成后；
- replay 的通用 input copy 阶段。

model-specific hook 属于现有 replay input-load 阶段，mirror/index copy 和随后的Graph
replay都在当前forward stream提交。不得为其新增thread、stream、event或WAR barrier。
默认 no-op，CUDA 和其他模型行为不变。该文件不增加 WeLMV4 算法逻辑。

### 9.13 `python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_extend_npu_graph_runner.py`

实现上节三个 hook：

- 仅在 draft architecture 为 `WeLMV4MoeForCausalLMNextN` 时分配 mirror K/V/index
  静态 buffer。
- buffer 宽度从 logical48 attention 的本 rank `kv_size` 获取。
- capture 时绑定到 `EagleDraftExtendInput.model_specific_states` 和 `mirrored_kv_indices`。
- replay copy 时：
  - K/V 实际行 copy；padding 行清零；
  - index 先填 0，再 copy/clamp 本轮 index；
  - `num_accept_tokens` 继续作为有效 row 权威 mask。
- fixed-width decode draft-extend 的 Graph `input_ids/positions/extend_seq_lens` 仍保留 `B*D`；OE helper从 `extend_prefix_lens+1` 得到每请求 logical column start，padding row只走安全读地址且不影响下一轮 history。
- WeLM Triton capture时把dummy request的`seq_lens`准备为`D`、`prefix_lens=0`，避免Ascend默认Graph fill value 0形成`q_lens=D, kv_lens=0`的非法capture metadata；真实replay时再由静态buffer覆盖。
- OE hash继续在Graph内读取固定地址的canonical table和已有Graph input buffers，
  不增加model-specific hash buffer。
- `can_run_graph()`、bucket 选择和 eager fallback 不增加 WeLM 特例。

### 9.14 draft Graph runner 的 frozen-KV 修正

涉及：

- `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`
- `python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py`

`eagle_draft_cuda_graph_runner.py`：

- capture dummy `EagleDraftInput` 在 WeLM NextN 时设置 `welmv4_mtp_frozen_kv=True`。
- 继续复用已有静态 `seq_lens/positions/req_pool_indices/topk/hidden_states/ngram_embedding_info`，不增加 mirror buffer。
- capture 执行 `draft_forward()` 后，只有普通 EAGLE 才执行 `positions.sub_(S-1)`；WeLM frozen-KV路径没有递增，必须跳过该恢复操作。
- capture 前断言 static `ngram_embedding_info.token_table` 与 target/draft共享 canonical table地址一致。

`eagle_draft_npu_graph_runner.py`：

- `_replay_graph()` 当前为每个 step构造 `seq_lens_cpu + step + 1`。WeLM frozen-KV分支改为每个 step都传同一个 `seq_lens_cpu`。
- 其他模型仍保留原有递增行为。
- 该修改只决定 replay metadata 输入值，不改变 backend 的 thread/replay 顺序。

### 9.15 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`

位置：当前 `self.is_welm_v4` architecture 判断和 `forward_mtp()`。

改动：

- target 和 NextN architecture 都识别为 WeLMV4；NextN steady draft DECODE/Graph继续命中现有 WeLM sink Triton decode分支，不能因 architecture漏判落到通用FIA。
- 增加统一的 `is_welmv4_mtp_frozen_kv(spec_info)` 判断，供 eager 和 Graph metadata 共用：
  - `init_forward_metadata()` 构造 draft block table、`seq_lens_cpu_int` 时，不增加 `speculative_step_id+1`；
  - `_apply_cuda_graph_metadata()` 计算 `max_len`、block table和 `metadata.seq_lens` 时，同样不增加 step offset；
  - 普通 EAGLE/其他模型维持现有行为。
- 保留 `forward_mtp()` 现有的 K/V cache store、TARGET_VERIFY fixed-width Q 长度、
  DRAFT_EXTEND_V2 的累计 Q 长度、Graph padding恢复和 SWA block-table 选择。当前
  Spec V2 decode draft-extend 实际把每请求 `extend_seq_lens` 填成固定 `D`；代码形式
  仍按 per-request lengths 做 cumulative，不能把 `num_accept_tokens` 误当成 attention
  Q 长度。`num_accept_tokens` 只决定有效 accepted row、mirror选择和 OE history。
- 在 `forward_mtp()` 取到 cache、Q 长度和 block table 后，增加 WeLM sink Triton
  dispatch。原因是 `forward_extend()` 在 TARGET_VERIFY/DRAFT_EXTEND_V2 时会提前
  return 到 `forward_mtp()`，不会经过普通 extend/decode/decode_graph 的 WeLM
  Triton 分支；当前代码会使 MTP sink layer 落到 FIA。
- 不能直接调用 `attention_sinks_triton()` 或 dedicated decode wrapper：它们是一条
  query/请求的 decode kernel，而 MTP 每请求有 `D=S+1` 条 query。
- 首版只有 topk=1，verify tree 是线性 causal chain，因此 Full+Sink 和 SWA+Sink
  都复用/封装现有 prefill-style causal Triton kernel：每请求 Q 长度来自当前 MTP
  metadata，SWA 使用
  `block_tables_swa`。如果后续支持 topk>1/tree mask，必须新增可接收 arbitrary
  MTP mask 的 kernel，不能继续冒充普通 causal prefill。
- 为Triton显式构造设备侧`q_lens/kv_lens/prefix_lens/cu_q_lens`：
  - `TARGET_VERIFY`：`q_lens=D`，`kv_lens=forward_batch.seq_lens+D`，`prefix_lens=forward_batch.seq_lens`；
  - `DRAFT_EXTEND_V2`：`q_lens=extend_seq_lens`（当前固定D），`kv_lens=forward_batch.seq_lens`（已经包含本次D rows），`prefix_lens=kv_lens-q_lens`；
  - `cu_q_lens`由`q_lens`累加得到。不得把`DRAFT_EXTEND_V2`的KV length再次`+D`，也不得为构造这些值新增D2H同步。
- Graph replay按真实batch size更新长度buffer：真实请求`q_lens=D`，padding请求`q_lens=0, kv_lens=0, prefix_lens=0`；padding的`cu_q_lens`保持不增长，query/output尾部保持0。不能沿用Ascend默认`seq_len_fill_value=0`同时又给padding请求固定`q_lens=D`。
- Triton wrapper必须显式校验当前 topk1 `mtp_mask` 与“prefix + 线性 causal D rows”等价；只能在等价时用 causal prefill kernel，不能无条件忽略传入的 MTP mask。
- eager 时只对真实 query rows运行 kernel，再补零恢复输入 shape；Graph 时按 capture
  bucket运行，padding request 使用现有安全 `seq_lens/req_pool_indices/block table`
  填充值。eager从真实batch metadata得到上述长度；Graph capture直接为bucket创建固定上界buffer并在replay input-copy阶段更新真实rows，padding rows使用安全长度。
  Full/SWA helper 的 schedule/cu_q_lens cache 必须按本次 MTP metadata构造，不能
  复用上一次 ordinary extend 的 cache。
- **P0 强制要求**：WeLMV4 的 `TARGET_VERIFY` 和 `DRAFT_EXTEND_V2` 必须命中 Triton dispatch。当前 `config.json` 的49层 `enable_attn_sink_layerwise` 全部为 `true`，包括 logical48；即使 FIA v2能接收 `learnable_sink=sinks`，也不得把 FIA作为 WeLM MTP fallback。
- eager、Graph、overlap on/off使用同一个 Triton wrapper；不允许只在 eager走Triton、Graph回退FIA。
- 若 WeLM MTP metadata不满足首版 topk1线性 causal contract，应显式报内部不支持错误，不能静默落到 FIA。FIA MTP路径只保留给其他模型。
- 不新建 WeLM 专用 attention backend。

### 9.16 `python/sglang/srt/hardware_backend/npu/attention/sink_full_attention.py`

Full+Sink prefill Graph 的静态task schedule在较小真实batch回放较大bucket时会跳过
`q_len=0`的padding request。kernel入口对零宽request直接`continue`，wrapper把输出
buffer改为capture内`zeros_like`初始化；这样被跳过的尾部行不会保留上一次Graph
replay的旧值并进入后续projection/MoE。SWA wrapper本来就是零初始化，只需保持其
动态`cu_q_lens`对padding request不增长。

## 10. 明确不改的架构边界

### 10.1 `npu_cudagraph_backend.py`

当前：

```python
thread.start()
graph.replay()
thread.join()
```

保持原样。不修改 Graph update/replay 时序，不增加锁，不增加 WeLM 特例。其他 NPU MTP 模型已经依赖该实现；WeLM 的问题是输入/输出语义未接入，不是 backend 竞态。

### 10.2 不新增 draft Graph runner

不新增独立 WeLM draft Graph runner。只按第9.14节在现有 runner中修正 frozen position的 capture cleanup和NPU per-step metadata replay；draft-local OE history/hash仍在现有 `draft_forward()` Graph capture内执行，canonical table地址在capture前共享。

### 10.3 overlap/scheduler

以下全部不改：

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/overlap_utils.py`
- scheduler的FutureMap、publish/read-done/WAR barrier实现
- `GenerationBatchResult`结构和`on_publish`位置

不设置也不要求`SGLANG_ENABLE_WAR_BARRIER`。WeLM新增的mirror state通过本轮私有
tensor和draft-extend runner私有Graph buffer传递，schedule stream不接触；hash/table
沿用当前NgramEmbeddingManager的数据所有权。mirror handoff复用`EagleDraftInput`
transient字段和已有logits/hidden-state生命周期。

### 10.4 其他明确不做

- 不新增 `welmv4_mtp_worker.py`。
- 不新增 WeLM 专用 draft NPU Graph runner。
- 不移植 CUDA 分支的整段专用/merged draft Graph、随机采样、topk>1、rejection sampling代码。
- 不新增 `welmv4_mtp_*_graph_hit/miss/fallback_reason` 指标。
- 不把 graph on/off 或 overlap on/off 做成 WeLM 启动门禁。

## 11. 测试方案

### 11.1 配置与权重单测

新增/扩展 config test，验证：

- target architecture 转换为 `WeLMV4MoeForCausalLMNextN`。
- `ModelRegistry` 可解析 `WeLMV4MoeForCausalLMNextN` 到新增NextN类。
- `num_target_hidden_layers=48`、physical `num_hidden_layers=1`，并验证physical0
  映射到logical48。
- physical0 的 RadixAttention/KV pool id为0，但 window/sink/norm/MLP layerwise读取全部命中config index48；验证 raw window512→left-history511、sink=true。
- NextN `get_hybrid_layer_ids()` 返回本地 `[0]`，而不是把logical48当成本地pool id。
- layer48 checkpoint 权重只加载到 `decoder_layers.0`。
- target layer0 source projection包含layer48 K/V；consumer48 active路径只有Q-only
  projection，不允许full-QKV decode fallback。
- 错误 mirror pair、多个 MTP layer、缺 layer48 权重明确失败。

### 11.2 ngram/OE 单测

建议新增：

`test/registered/unit/npu/test_welmv4_mtp_ngram.py`

覆盖：

- step hash 与 CPU reference 完全一致。
- prompt 长度 0/1/2 和正常长 history。
- shifted prefill 首行历史偏移正确。
- target verify `B x D` 线性链 hash 正确。
- 固定示例验证双坐标：draft首个实际forward token column=`L+1`/attention position=`L-1`，entry history最新token在`L`；target verify start=`L`，draft-extend start=`L+1`，下一轮draft forward start=`L+A+1`。
- draft-extend 不同 `accept_lens` 的有效/padding row，decode首行严格使用logical column `L+1`。
- accepted commit：`A=1...D`，多 req slot，不同 old seq len。
- commit 前在 `table[L]` 放置root sentinel，commit后断言sentinel未被覆盖，accepted `predict`只出现在`L+1...L+A`。
- `accept_lens` 包含 bonus；bonus 只写一次。
- rejected/padding 行不修改 table。
- Graph capture/replay 连续两轮后 canonical table 等于正式 emitted token 序列。

### 11.3 mirror 单测

建议新增：

`test/registered/unit/npu/test_welmv4_mtp_mirror.py`

覆盖：

- 配置严格解析为 `source0 -> consumer48`。
- target source state key 使用 consumer48。
- prefill identity mirror rows。
- decode `accept_index` gather。
- invalid/padding index 安全且不污染有效 row。
- consumer48在EXTEND/DRAFT_EXTEND_V2走Q-only+external K/V fill；draft DECODE
  走Q-only+frozen mirror cache且不写新K/V。
- K/V 在 consumer48 应用 norm/RoPE 后写 cache，而不是 target 提前处理。

### 11.4 Graph contract 单测

建议新增：

`test/registered/unit/npu/test_welmv4_mtp_graph_contract.py`

覆盖：

- target NPUGraphRunner 返回 mirror states，并按 raw rows 切片。
- draft-extend capture dummy state绑定到静态 K/V/index buffer。
- replay padding后实际 rows 与 eager 一致。
- draft Graph capture dummy设置frozen标志，capture后不执行通用`positions.sub_(S-1)`；连续多次replay的所有step positions始终为`seq_lens-1`。
- eager Ascend metadata和NPU draft Graph replay metadata的所有step KV length都等于进入本轮时的`seq_lens`，不得出现`+step+1`。
- draft Graph static `ngram_embedding_info.token_table` 与target/draft manager table的`data_ptr()`一致。
- 分别强制 draft、target_verify、draft_extend 的 `can_run*` 返回 true/false，覆盖独立 fallback 组合。
- 验证未调用/修改 `NPUCudaGraphBackend` 的 update/replay 实现。

### 11.5 Triton `forward_mtp` P0 单测

建议新增/扩展 Ascend attention test，覆盖：

- WeLM target `TARGET_VERIFY` 的 Full+Sink、SWA+Sink 均调用 Triton MTP wrapper。
- WeLM NextN `DRAFT_EXTEND_V2` logical48/SWA+Sink 调用 Triton MTP wrapper。
- eager和Graph capture/replay输出与对应 causal CPU/reference mask一致，Graph padding row不影响真实row。
- Graph replay检查真实request的`q_lens=D`、padding request的`q_lens=kv_lens=0`；draft-extend capture dummy使用`q_lens=kv_lens=D`，不存在非法长度组合。
- monkeypatch FIA MTP入口为直接失败，以上WeLM用例仍全部通过，证明不存在静默 FIA fallback。
- 验证K/V cache store发生在Triton读取前，SWA使用`block_tables_swa`，sink使用logical48权重，per-request `cu_q_lens`与`D`/`extend_seq_lens`一致。

### 11.6 overlap 顺序单测

覆盖：

- speculative accepted-token commit 发生在 `on_publish` 之前。
- mirror transient state 在 draft-extend 之后清空。
- 不创建 FutureMap 字段。
- overlap下scheduler不会读写WeLM mirror/index私有Graph buffer；不依赖WAR barrier。

### 11.7 NPU 集成矩阵

至少执行：

| Graph | overlap | S | 并行 |
|---|---|---:|---|
| off | off | 1、2、4 | pure TP |
| on | off | 1、2、4 | pure TP |
| off | on | 1、2、4 | pure TP |
| on | on | 1、2、4 | pure TP |
| off/on | off/on | 1、2、4 | TP=EP |

每组使用 topk1 greedy 和固定 prompts，检查：

- token 输出与 eager target/reference 一致。
- Graph on/off、overlap on/off 四象限输出一致。
- 每轮 canonical table 等于请求正式 token 序列。
- draft physical slot0 实际加载 checkpoint layer48。
- accepted source0 mirror K/V 与 draft consumer48 cache 对齐（允许模型 dtype 对应误差）。
- 强制某一个 Graph bucket miss 时，其余两个阶段仍可命中并保持正确。
- 记录/断言 WeLM `forward_mtp` 实际 dispatch为Triton；任何 FIA调用都判测试失败。

## 12. 验收标准

功能验收：

- WeLMV4 可通过现有 EAGLE/NEXTN Spec V2 启动，不再命中 blanket reject。
- `S>=1`、`D=S+1`、topk1 greedy 可持续生成。
- graph on/off、overlap on/off 均正确。
- pure TP 和 TP=EP 均通过。

语义验收：

- logical layer48/physical slot0 映射清楚且权重加载唯一。
- mirror 严格是 source0→consumer48。
- draft 不污染 canonical ngram table。
- accept 后 canonical table 无漏写、重写或 rejected suffix。
- accepted `predict[j]` 固定写入`old_seq_lens+1+j`，root所在`old_seq_lens`不被覆盖。
- topk1 draft所有step复用`attention_position=seq_lens-1`和KV length=`seq_lens`，同时OE logical history逐token前进。
- `DRAFT_EXTEND_V2` 确实消费 external mirror K/V。
- WeLM `TARGET_VERIFY`/`DRAFT_EXTEND_V2` 的`forward_mtp`在eager和Graph下都强制使用Triton，不允许FIA fallback。

架构验收：

- 没有新增专用 worker。
- 没有新增独立 draft Graph runner。
- 没有修改 NPU Graph backend replay/update。
- 没有新增 FutureMap/scheduler overlap 协议。
- 三阶段 Graph 独立判断并保留 eager fallback。

## 13. 实施顺序

1. ModelConfig architecture转换、physical0/logical48彻底拆分、NextN单层模型和layer48权重加载。
2. eager模式下的mirror output/input contract和Q-only frozen-KV draft；先跑Graph off/overlap off。
3. 实现双坐标ngram、draft-local hash、target verify hash和`L+1+j` accepted commit。
4. 在Ascend `forward_mtp()`接入强制Triton MTP路径并完成eager正确性测试；未通过前不进入Graph阶段。
5. 直接接入现有`EAGLEWorkerV2`，完成Graph off × overlap off/on。
6. 修正draft Graph frozen metadata，target NPUGraphRunner保留mirror output，draft-extend NPU Graph增加静态mirror inputs。
7. 完成三阶段Graph独立fallback、Graph×overlap四象限以及Triton无FIA回退测试。
8. 最后执行pure TP、TP=EP和ModelSlim MXFP8专项回归。

每一阶段都以同一组 deterministic prompts 对比前一阶段；不在语义未闭环时提前做性能优化。

## 14. 修订后复审结论

复审依据：当前 `config.json`、perf CUDA WeLMV4 MTP参考路径，以及NEWSGLANG基线中的Spec V2 worker、ngram manager、Ascend attention和三类Graph runner。

| 原P0/Graph问题 | 修订后的闭环设计 | 方案级状态 |
|---|---|---|
| accepted commit覆盖root | destination改为`old_seq_lens+1+j`，并增加root sentinel测试 | 已解决 |
| attention position与OE token列混用 | 定义两套坐标；draft attention/logical=`L-1/L+1`且entry history到`L`，verify=`L`，draft-extend=`L+1` | 已解决 |
| 只设`save_kv_cache=False`，draft仍递增position/KV length | worker跳过position递增；Ascend eager metadata和NPU Graph replay都固定KV length=`seq_lens` | 已解决 |
| physical0错误读取config index0 | decoder/attention/layerwise/load显式拆分physical0与logical48；SWA/sink按index48 | 已解决 |
| target/draft/draft-extend Graph语义耦合或状态丢失 | 三阶段独立判断；target保留mirror output；draft修正frozen metadata；draft-extend使用静态mirror输入 | 已解决 |
| Graph capture后通用position恢复破坏frozen position | WeLM capture跳过`positions.sub_(S-1)` | 已解决 |
| `forward_mtp`落入FIA | WeLM target verify/draft-extend强制Triton；eager/Graph一致，FIA仅供其他模型 | 已解决 |
| overlap被误认为需要WAR barrier | mirror私有tensor/runner buffer和canonical table操作保持同一forward stream顺序，不改scheduler | 已解决 |

方案级复审没有发现仍未定义的数据坐标、Graph输入输出或fallback路径。特别是：

- `num_target_hidden_layers` 是唯一target层数配置字段；没有新增WeLM私有layer id字段。
- source0→consumer48跨阶段mirror与target内部mirror pair职责分离。
- topk1多步draft的Q-only frozen-KV语义在eager和Graph有同一个控制标记和metadata规则。
- Triton `forward_mtp` 是P0硬要求，不是可选优化，也没有FIA兜底。
- scheduler overlap、WAR barrier和`npu_cudagraph_backend.py`均不需要模型特改。

以上“已解决”现在同时表示对应生产改动已经落入本代码树，并通过 Python
语法编译、`git diff --check` 和代码级链路复审。由于当前 Windows 验证环境没有
安装本仓库运行所需的 `torch`/`pytest`/NPU runtime，第11节中的 NPU 真机测试尚未
执行；尤其是 Triton mask 等价、连续两轮 canonical table、固定 draft KV length、
三阶段混合 Graph 命中四项仍是合入/上线前的硬验收项，不能以 eager/FIA fallback
掩盖失败。

## 15. 本次实施落地状态

- 已新增单层 `WeLMV4MoeForCausalLMNextN`，checkpoint/config logical48 与 runtime
  physical/cache0 分离；权重层数统一使用 `num_target_hidden_layers`。
- 已加入仅存在于target load期间的layer48 QKV staging，在ModelSlim原始布局上完成
  source0/consumer48配对并校验Q/K/V齐全；配对后立即释放，forward/Graph不持有它。
- 已接入 source0→consumer48 的跨阶段 mirror state；prefill/verify 的 mirror-fill、
  多步 draft 的 Q-only frozen KV 在 eager、Graph 下使用同一数据契约。
- 已让 target/draft 共用 canonical ngram table，增加 draft-local 显式 history hash，
  accepted commit 的目标列固定为 `old_seq_lens + 1 + j`。
- 已在 Ascend `forward_mtp()` 中强制 WeLM Full/SWA+Sink Triton 路径；WeLM 分支不会
  进入 FIA fallback。
- 已补齐 target Graph mirror output、draft Graph frozen metadata、draft-extend Graph
  静态 mirror/ngram inputs、padding q_len=0和Full+Sink输出清零处理；三个阶段仍分别调用各自现有
  `can_run_graph()`，没有新增联动门禁。
- 未新增专用 worker，未修改 scheduler/WAR barrier，未修改
  `npu_cudagraph_backend.py` 的 update/replay 顺序。
