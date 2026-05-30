unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

pkill -9 python | pkill -9 sglang | pkill -9 VLLM
pkill -9 python | pkill -9 sglang | pkill -9 VLLM

source /home/x00452483/cann-0506/cann/set_env.sh

export PYTHONPATH=/home/l00951279/sglang/python:$PYTHONPATH

export HCCL_TOPO_FILE_PATH=/etc/server_8p_noroce.json
export HCCL_BUFFSIZE=400
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68

export ACL_DEVICE_SYNC_TIMEOUT=60

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

MODEL_PATH=/home/weights/LongCat-Flash-Chat_0527

# [DEBUG]
# export ENABLE_PROFILING=1
# export PROFILING_BS=8
# export PROFILING_STAGE="decode"
# export PROFILING_step=50
# export ASCEND_LAUNCH_BLOCKING=1
# export ASCEND_GLOBAL_LOG_LEVEL=4
# export ASCEND_MODULE_LOG_LEVEL=RUNTIME=4

# [FIA]  
export ASCEND_USE_FIA=1
export ASCEND_USE_FIA_V2=1

# [MLAPO]  
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_NPU_LONGCAT_MLAPROLOG=1

# [MTP]
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_NPU_AFFINITY_EARLY_BIND=1
export SGLANG_NPU_AFFINITY_ALL_THREADS=1
export SGLANG_NPU_AFFINITY_DISABLE_SMT=1
export SGLANG_NPU_AFFINITY_PCORES_PER_PROC=8
export SGLANG_NPU_MEMORY_PREFERRED_BIND=1

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp 8 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--host 127.0.0.1 --port 6677 \
--max-running-requests 8 \
--mem-fraction-static 0.5 \
--context-length 6144 \
--chunked-prefill-size 16384 \
--quantization modelslim \
--prefill-round-robin-balance \
--speculative-draft-model-path ${MODEL_PATH} \
--speculative-algorithm NEXTN \
--speculative-num-steps 2 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 3 \
--kv-cache-dtype "fp8_e4m3" \
--enable-longcat-double-stream \
--disable-radix-cache \