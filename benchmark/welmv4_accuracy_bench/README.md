# WeLM v4 Accuracy Benchmark

本目录提供一套基于 [OpenCompass](https://github.com/open-compass/opencompass)
+ [SGLang](https://github.com/TopIdiot/sglang) 的精度对齐工具，用于在 WeLM v4 模型
（`sglang` 中通过 `--enable-over-encoding` 启用）上运行 GSM8K / MMLU-Pro / C-Eval / CMMLU
等基准，复现并对齐参考精度。

## 目录结构

```text
benchmark/welmv4_accuracy_bench/
├── README.md                     # 本说明
├── requirements.txt              # 评估侧 Python 依赖
├── eval_template.py              # OpenCompass 数据集与评估配置（已固化 datasets= 列表）
├── generate_script.py            # 用环境变量拼装最终 eval.py（注入 model 配置）
├── run.sh                        # 一键评估入口
├── datasets/data/                # 评估用数据集（gsm8k / mmlu_pro / ceval / cmmlu，共 ~21MB）
└── opencompass_patch/
    └── welmv4_api.py             # 注册到 opencompass.models 的 WeLM API 适配器
```

> 数据集已随交付包一同提供，无需额外下载。

## 1. 准备 SGLang 推理服务

> 该步骤在部署 GPU 的机器上执行。

1) 拉取代码：

```bash
git clone -b main_v056 https://github.com/TopIdiot/sglang.git
cd sglang
pip install --upgrade pip
pip install -e "python"
```

2) 安装 custom_ops（WeLM v4 over-encoding 自定义算子）：

```bash
cd sgl-kernel/3rdparty/custom_ops
python3 setup.py install
# 或：bash install.sh
cd -
```

3) 启动服务（按实际机器调整 `--tp-size` / `--mem-fraction-static` / `--port`）：

```bash
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name welmv4 \
    --host "${HOST_IP}" \
    --port "${PORT}" \
    --tp-size 4 \
    --nnodes 1 \
    --node-rank 0 \
    --mem-fraction-static 0.80 \
    --context-length 16384 \
    --trust-remote-code \
    --enable-over-encoding \
    --enable-kv-mirror \
    --attention-backend fa3 \
    --disable-overlap-schedule
```

服务起来后可以用以下命令快速验证：

```bash
curl "http://${HOST_IP}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"welmv4","prompt":"1+1=","max_tokens":4,"temperature":0}'
```

## 2. 准备评估环境

> 评估端可以与推理服务部署在同一台机器，也可以是另一台能访问推理 HTTP 接口的机器。

```bash
pip install -r benchmark/welmv4_accuracy_bench/requirements.txt
```

把 `welmv4_api.py` 注册到 OpenCompass：

```bash
OC_DIR=$(python -c "import opencompass, os; print(os.path.dirname(opencompass.__file__))")
cp benchmark/welmv4_accuracy_bench/opencompass_patch/welmv4_api.py \
   "${OC_DIR}/models/welmv4_api.py"

# 仅在 __init__.py 没有该 import 时追加一次
grep -q 'from .welmv4_api import WeLM' "${OC_DIR}/models/__init__.py" \
  || echo 'from .welmv4_api import WeLM' >> "${OC_DIR}/models/__init__.py"
```

## 3. 数据集

本交付包已在 `datasets/data/` 中内置以下版本（共约 21MB），开箱即用，无需额外下载：

| 子目录                    | 来源                          | 体积  |
|---------------------------|-------------------------------|-------|
| `datasets/data/gsm8k`     | `opencompass/gsm8k`           | ~11M  |
| `datasets/data/mmlu_pro`  | `opencompass/mmlu_pro`        | ~4.0M |
| `datasets/data/ceval`     | `opencompass/ceval-exam`      | ~3.9M |
| `datasets/data/cmmlu`     | `opencompass/cmmlu`           | ~2.7M |

`run.sh` 默认设置 `COMPASS_DATA_CACHE=./datasets`，OpenCompass 会直接从本目录读取，
不会触发联网下载。如需替换为更新版本，覆盖对应子目录即可。

## 4. 一键评估

修改 `run.sh` 中的：

- `MODEL_PATH`：服务端 `--model-path` 指向的权重目录（评估端只用作 tokenizer 路径）。
- `OPENAI_BASE_URL`：步骤 1 启动的 sglang `v1/completions` 地址。

然后执行：

```bash
cd benchmark/welmv4_accuracy_bench
bash run.sh
```

执行结束后，结果会输出到 `outputs/` 目录（OpenCompass 默认产物），包含每个子任务的
精度数值、详细预测、汇总表格。

## 5. 验收方法

请将以下两项一并交付回我们：

1. `outputs/` 目录的完整压缩包，包含每个数据集的 `summary/` 与 `predictions/`。
2. `run.sh` 的完整 stdout/stderr 日志（建议 `bash run.sh 2>&1 | tee run.log`）。

我们会以下表数据集为基线对齐精度：

| 数据集     | 子任务范围                | 评估指标 |
|------------|---------------------------|----------|
| GSM8K      | 全量                      | accuracy |
| MMLU-Pro   | 全量（按 category 平均）  | accuracy |
| C-Eval     | val（5-shot）             | accuracy |
| CMMLU      | test（5-shot）            | accuracy |

## 6. 常见问题

- **`OpenAI API key is not set.`**：`welmv4_api.py` 强制读取 `OPENAI_API_KEY`，
  对接 sglang 时填任意非空值即可，`run.sh` 默认设为 `EMPTY`。
- **`requests.ConnectionError`**：检查 `OPENAI_BASE_URL` 是否能从评估机直连 sglang
  端口；防火墙、`HOST_IP` 是否对外暴露。
- **复跑得到不一致的精度**：`temperature=0`、`top_p=1.0` 已在配置里固定；如仍有抖动，
  确认 sglang 启动参数与上文完全一致，特别是 `--disable-overlap-schedule`、`--enable-over-encoding`、
  `--enable-kv-mirror`。
