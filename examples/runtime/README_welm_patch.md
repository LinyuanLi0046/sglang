# WeLM 推理补丁迁移

本仓库已合入原 `welm_patch` 的三个功能，不需要再执行外部覆盖脚本：

- `SGLANG_FORWARD_UNKNOWN_TOOLS=1` 时，基础工具解析器允许流式返回未在请求 `tools` 中注册的工具名，与非流式行为保持一致。默认仍关闭。
- `--tool-call-parser qwen25` 的非流式 `<tool_call>` JSON 解析失败时，使用 `json-repair` 修复后再次解析。流式仍使用原有增量 JSON 解析器。
- 频率惩罚支持指定 token 免罚和按生成长度增加倍率。张量在当前 batch 的设备上创建，去除原补丁在 import 时枚举所有 CUDA 设备的逻辑；生成长度按请求维护并随 batch 合并、过滤。

迁移以仓库提交 `0504090947f447d1dc609c52d15614bc808aea7a` 为基线，保留了现有 `StructuralTag`、`get_model_structural_tag`、原生结构约束和 `parses_required_natively` 等接口。原补丁整文件覆盖会删除这些接口，导致 GLM47 等解析器在服务启动时导入失败。

## 部署

在实际启动服务的 Python 环境安装新增依赖（标准、NPU、CPU、XPU 和 other 安装清单均已声明）：

```bash
python -m pip install json-repair
```

将合入后的仓库同步到部署机，并确保 `PYTHONPATH` 或 editable install 指向该仓库。例如用户日志中的部署目录是：

```bash
export PYTHONPATH=/data2/hw_sgz/02_code/sglang/python${PYTHONPATH:+:$PYTHONPATH}
python -c 'import sglang; print(sglang.__file__)'
python -c 'from sglang.srt.function_call.function_call_parser import FunctionCallParser; from sglang.srt.function_call.base_format_detector import StructuralTag; print("parser imports OK")'
```

原 WeLM 环境变量已保存在同目录的 `welm_env.sh`。从仓库根目录启动时可使用：

```bash
source examples/runtime/welm_env.sh
# 在现有 launch_server 命令中加上：
# --tool-call-parser qwen25
```

`welm_env.sh` 中的 token ID 列表来自原模型部署，须与使用的 tokenizer 对应。该文件启用未知工具透传；如果希望只允许请求声明的工具，可在 source 后设置 `SGLANG_FORWARD_UNKNOWN_TOOLS=0`。

频率惩罚只对请求中的非零 `frequency_penalty` 生效。相关环境变量：

| 变量 | 含义 |
| --- | --- |
| `FREQUENCY_PENALTY_EXCLUDE_TOKENS` | 非负整数的 JSON 列表；未设置或 `[]` 表示不排除 |
| `FREQUENCY_PENALTY_FACTOR_START` | 非负整数，倍率开始增长前的长度；原配置为 `0` |
| `FREQUENCY_PENALTY_FACTOR_OFFSET` | 正整数，增长间隔；原配置为 `2048` |

START 和 OFFSET 同时设置时，第 `n` 次累计输出 token 的增量倍率为 `1 + max(n - START, 0) // OFFSET`，与原补丁单请求公式一致。只放大本次新增惩罚，不追溯放大已累计惩罚；免罚 token 仍计入长度。缺少任一倍率变量时不缩放。非法配置会在初始化实际使用的频率惩罚器时给出包含变量名的错误。

## 回归测试

在完整 SGLang 开发环境，从仓库根目录运行：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/function_call/test_welm_patch.py \
  test/registered/unit/function_call/test_unknown_tool_name.py \
  test/registered/unit/function_call/test_function_call_parser.py \
  test/registered/unit/sampling/test_welm_frequency_penalty.py
```

新增测试覆盖流式/非流式未知工具、混合多工具调用、异常 JSON 修复、现有结构约束接口、免罚 token、倍率边界及 batch 合并/过滤。CPU 测试验证逻辑；NPU 上的模型加载、算子执行和实际流式请求需要在部署环境验收。

本次本地验证结果：213 项通过。由于宿主机未安装完整推理依赖，测试使用临时包路径绕过 `sglang` 顶层及 penaltylib 的启动导入，实际解析器、协议、XGrammar 和 PyTorch 均使用真实实现；没有模拟被测逻辑。另有 15 项需要 DeepSeek tokenizer 的既有测试因缺少 `sentencepiece` 等环境依赖而排除。此结果不代表完整 `launch_server` 或 NPU 模型启动验证。外部安装包也已在临时目录验证正常安装、重复安装、修复旧补丁、拒绝未知文件版本和导入失败后的完整回滚。
