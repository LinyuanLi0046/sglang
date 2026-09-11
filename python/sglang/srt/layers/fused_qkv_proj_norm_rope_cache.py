# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FusedQkvProjNormRopeCache -- WeLMv4 QKV 投影全融合算子(P2,Cube+Vector 复合核).

将 sglang WeLMv4 prefill 路径 kv-mirror 层的完整 9 步序列(matmul_fusion.md 步骤 1-8)
融合为 1 个 MIX_AIC_1_2 复合核:

    qkv(M,2560) = hidden(M,2048) @ W(2560,2048)^T      (Cube, MMAD fp32 累加)
    ├── [0:1536]   6 x Q head   : 尾 64 维 RoPE ──────────────► q(M,1536)     ► FA
    ├── [1536:1792] K head      : RMSNorm → 尾 64 RoPE ─┬─────► k(M,256)      ► FA
    │                                                 └─────► k_cache[slot]  (散写)
    ├── [1792:2048] V head      : fp32→bf16 RN ──────────────► v_cache[slot]  (散写, 不物化 v)
    ├── [2048:2304] mirror_k    : 直通 ──────────────────────► mirror_k(M,256) (KVMirror 激活)
    └── [2304:2560] mirror_v    : 直通 ──────────────────────► mirror_v(M,256) (KVMirror 激活)

qkv 中间张量彻底消除:L0C fp32 经 FIXPIPE(dual_dst_ctl=1, split-M) 直达 AIV 的
CrossCore UB 通道,epilogue 就地消费.

维度布局(每 TP rank,与 profile 对应:MatMulV3 16500,2048;2560,2048 → 16500,2560):

    hidden: (M, 2048) bf16;weight: (2560, 2048) bf16(nn.Linear (N,K) 布局)
    每 head 256 维,rope_dim=64(尾部 RoPE,GPT-NeoX rotate-half)
    cos_sin_cache: (max_pos, 64) fp32,每行前 32 cos,后 32 sin

结构(参照 test_cv_mix_tiled_channel_first_npu.py + samples/matmul.py + P1 向量核):

    Cube:  baseM=128 * baseN=256(= 1 个 head-unit)/ tile,baseK=64
           L1(128/256 * 256, depth2) → L0A/L0B(depth2) → L0C(128*256 f32, depth2)
           K=2048 沿 kL1=256 分 8 轮,每轮 4 次 MMAD 累加
    交接:  mem_copy(cv_ub, l0c, fixpipe(f32, dual_dst_ctl=1), partition=split_m)
           cv_ub (64,256) f32 depth2 CrossCore -- 每 AIV 收窄为本核 M/2 行
    Vec:   按 10 个 unit 分派(Q*6 / K / V / mirror_k / mirror_v),行内处理与
           P1 (welmv4_qkv_post_fusion) 同构;M 尾块由 tile_view 裁剪 +
           balanced split-M 自然覆盖
    多核:  grid = min(ceil(M/128), 32) 个 block(1 AIC + 2 AIV),M 向 grid-stride

数值说明:MMAD fp32 累加与 aclnnMatmul 同类,但累加顺序(tiling)不同,bf16 结果
允许 ±1-2 ulp 差异;AIV 侧在进入 RMSNorm/RoPE 前先把 fp32 tile 舍入回 bf16 再升
fp32(复刻原始"qkv 以 bf16 落 GM,后处理 kernel 读回"的舍入语义;直通段/V/mirror
的 fp32→bf16 存储本身就等价);K 的 norm 结果同样先舍回 bf16 再进 rope.

特化条件(host 侧由调用方保证):
    - positions 模式二选一(trace 期分档,epilogue 数学完全相同):
        * prefill(positions_contiguous=True):单请求连续位置,
          kernel 仅读每 AIV 半区首位置 + 一次 bulk cos/sin DMA;
          表尾不足 TILE_VEC_M 行时自动退化为逐行加载(无越界);
        * decode(positions_contiguous=False):任意逐 token 位置
          (batched decode 各请求的末位置),逐行 1x64 gather--与
          生产 generic rope kernel 的寻址一致;
    - slot_mapping 元素为 cache 行索引或 -1(跳过);
    - K = hidden_size = 2048,N = 2560/2048 固化(WeLMv4 每 TP rank
      结构常数;M 为 Dim 动态维,prefill/decode 共用同一产物).
"""

import torch

from cannbotdsl import ChannelKind, Dim, MemLoc, Tensor, TensorSpec, dtypes, select
from cannbotdsl.channel import Channel
from cannbotdsl.lang import const_expr
from cannbotdsl.lang import range as dsl_range
from cannbotdsl.lang import jit, kernel, vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops import matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    full_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup,
    vdups,
    vgts,
    vload,
    vload_unpack,
    vmaxs,
    vmul,
    vmuls,
    vreduce_sum,
    vselect,
    vsqrt,
    vstore_pack,
    vsub,
    update_mask,
)
from cannbotdsl.tensor import (
    local_slice,
    make_partition_tiler,
    partition_view,
    tile_view,
)
from cannbotdsl.types import RoundingMode
from cannbotdsl.types import bfloat16 as BF16, float32 as F32

# ---- model structure constants (per TP rank, WeLMv4 NPU path) ----
HIDDEN = 2048  # K of the projection
HEAD_DIM = 256
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE_DIM = ROPE_DIM // 2
NUM_Q_HEADS = 6
NUM_UNITS = 10  # 6 Q + K + V + mirror_k + mirror_v
QKV_WIDTH = NUM_UNITS * HEAD_DIM  # 2560
UNIT_K = NUM_Q_HEADS
UNIT_V = NUM_Q_HEADS + 1
UNIT_MK = NUM_Q_HEADS + 2

# ---- tiling constants (arch35: L1 512K, L0A/B 64K, L0C 256K, AIC 32) ----
# Cube tiles are PAIR_M=256 rows: two consecutive BASE_M=128 m-tiles form ONE
# contiguous (256, K) GM region of hidden, so a single A load serves both and
# the B (weight) slice is streamed once per pair instead of once per tile —
# L1 traffic drops from (0.5+1)MB x n_units x 129 tiles = 1.95GB to
# (1+1)MB x n_units x 65 pairs = 1.30GB.
PAIR_M = 256
BASE_N = 256  # exactly one head-unit per N tile (epilogue granularity)
BASE_K = 64  # L0B (256,64) bf16 depth2 == 64KB; depth2 pipelining beats halved issue count (measured: depth1@128 is 31% slower)
K_L1 = 256
K_L1_TILES = HIDDEN // K_L1  # 8
K_L0_PER_L1 = K_L1 // BASE_K  # 4
TILE_VEC_M = PAIR_M // 2  # per-AIV rows after split-M
AIC_NUM = 32

VL = 64  # vector register elements (2048b / 32b)
AVG = 1.0 / HEAD_DIM
_F32_MAX = float.fromhex("0x1.fffffep+127")


@kernel
class FusedQkvProjNormRopeCacheKernel:
    def __init__(
        self,
        return_v: bool = False,
        has_mirror: bool = True,
        positions_contiguous: bool = True,
        positions_segmented: bool = False,
    ):
        # Trace-time constants: fold the optional v materialization, the
        # mirror block-store, and the cos/sin load mode away before IR
        # emission. ``has_mirror`` must be a host-side fact (plain layers
        # pass (1, 256) mirror stand-ins whose shape cannot satisfy the
        # block store's static partition check). ``positions_contiguous``
        # selects the prefill (bulk cos/sin DMA on a runtime-contiguous
        # position run) vs decode (per-token gather for arbitrary
        # positions) cos/sin preload; both artifacts share the identical
        # epilogue math.
        self.return_v = bool(return_v)
        self.has_mirror = bool(has_mirror)
        self.positions_contiguous = bool(positions_contiguous)
        self.positions_segmented = bool(positions_segmented)

    @jit
    def _copy_segment_cos_sin(self, cs_ch, gm_cos_sin, dst_row, pos, remaining):
        # Each framework segment is <=64 rows. Use existing static tile-view
        # shapes with runtime tile coordinates: coordinates are TILE indices,
        # not row offsets. Dyadic pieces also handle unaligned request ends
        # without dynamic UB slice offsets or another staging buffer.
        while remaining > 0:
            # DSL branch results must be initialized in the enclosing scope.
            copy_rows = 1
            if remaining >= 64 and dst_row % 64 == 0:
                copy_rows = 64
                mem_copy(
                    tile_view(cs_ch, (64, ROPE_DIM), (dst_row // 64, 0)),
                    tile_view(
                        gm_cos_sin[pos : pos + 64, 0:ROPE_DIM],
                        (64, ROPE_DIM), (0, 0),
                    ),
                )
            elif remaining >= 32 and dst_row % 32 == 0:
                copy_rows = 32
                mem_copy(
                    tile_view(cs_ch, (32, ROPE_DIM), (dst_row // 32, 0)),
                    tile_view(
                        gm_cos_sin[pos : pos + 32, 0:ROPE_DIM],
                        (32, ROPE_DIM), (0, 0),
                    ),
                )
            elif remaining >= 16 and dst_row % 16 == 0:
                copy_rows = 16
                mem_copy(
                    tile_view(cs_ch, (16, ROPE_DIM), (dst_row // 16, 0)),
                    tile_view(
                        gm_cos_sin[pos : pos + 16, 0:ROPE_DIM],
                        (16, ROPE_DIM), (0, 0),
                    ),
                )
            elif remaining >= 8 and dst_row % 8 == 0:
                copy_rows = 8
                mem_copy(
                    tile_view(cs_ch, (8, ROPE_DIM), (dst_row // 8, 0)),
                    tile_view(
                        gm_cos_sin[pos : pos + 8, 0:ROPE_DIM],
                        (8, ROPE_DIM), (0, 0),
                    ),
                )
            elif remaining >= 4 and dst_row % 4 == 0:
                copy_rows = 4
                mem_copy(
                    tile_view(cs_ch, (4, ROPE_DIM), (dst_row // 4, 0)),
                    tile_view(
                        gm_cos_sin[pos : pos + 4, 0:ROPE_DIM],
                        (4, ROPE_DIM), (0, 0),
                    ),
                )
            elif remaining >= 2 and dst_row % 2 == 0:
                copy_rows = 2
                mem_copy(
                    tile_view(cs_ch, (2, ROPE_DIM), (dst_row // 2, 0)),
                    tile_view(
                        gm_cos_sin[pos : pos + 2, 0:ROPE_DIM],
                        (2, ROPE_DIM), (0, 0),
                    ),
                )
            else:
                mem_copy(
                    tile_view(cs_ch, (1, ROPE_DIM), (dst_row, 0)),
                    tile_view(gm_cos_sin, (1, ROPE_DIM), (pos, 0)),
                )
            dst_row += copy_rows
            pos += copy_rows
            remaining -= copy_rows

    @jit
    def _load_segmented_cos_sin(
        self, cs_ch, gm_cos_sin, gm_positions, gm_segment_tile_starts, row0, rows_here
    ):
        # The final entry is the real-token sentinel, excluding EP padding.
        # Reuse the framework's request-local <=64-row tiles. The AIV split
        # can start/end inside a tile, especially in a balanced tail pair.
        num_tiles = gm_segment_tile_starts.shape[0] - 1
        # Framework metadata stays int32; DSL coordinates/selects use int64.
        real_tokens = dtypes.int64(gm_segment_tile_starts[num_tiles])
        row_end = row0 + rows_here
        real_end = select(row_end < real_tokens, row_end, real_tokens)
        cursor = row0
        if cursor < real_end:
            lo = 0
            hi = num_tiles
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if dtypes.int64(gm_segment_tile_starts[mid]) <= cursor:
                    lo = mid
                else:
                    hi = mid
            tile_id = lo
            while cursor < real_end:
                tile_end = dtypes.int64(gm_segment_tile_starts[tile_id + 1])
                end = select(tile_end < real_end, tile_end, real_end)
                pos = gm_positions[cursor]
                self._copy_segment_cos_sin(
                    cs_ch, gm_cos_sin, cursor - row0, pos, end - cursor
                )
                cursor = end
                tile_id += 1

        # No segment describes padding. Read its supplied (dummy) positions,
        # rather than extrapolating the last request beyond the RoPE table.
        padding_start = select(real_end > row0, real_end - row0, 0)
        for r in range(padding_start, rows_here):
            pos = gm_positions[row0 + r]
            mem_copy(
                tile_view(cs_ch, (1, ROPE_DIM), (r, 0)),
                tile_view(gm_cos_sin, (1, ROPE_DIM), (pos, 0)),
            )

    @jit
    def _rope_q(self, in_ch, out_ch, cs_ch, cur_rows):
        """Q head unit: 192-col passthrough + tail-64 NeoX rotate-half.

        Input is the fp32 cross-core tile; bf16 RN rounding happens at store.
        A second load at +32 puts this row's x2 in the low lanes, so every
        rotate-half pair is lane-aligned and computed with 32-lane masked
        ops -- no UB scratch round-trip, no vmem_bar, no vgather. The +32
        load of the last row spans into cv_ub's pad row (never written; its
        lanes are masked out).
        """
        with vf(mode="raw"):
            full = full_mask()
            m_lo = update_mask(HALF_ROPE_DIM, elem_bits=32)[0]

            for r in dsl_range(0, cur_rows, 1):
                base = r * HEAD_DIM
                for j in range(3):
                    off = base + j * VL
                    x = vload(in_ch, off)
                    vstore_pack(
                        out_ch,
                        off,
                        vcast(x, BF16, mask=full, rounding=RoundingMode.RN),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                # Round the fp32 MMAD result to bf16 before the RoPE input:
                # the production path materializes qkv as bf16 in GM and the
                # rope kernel reads it back (bf16 -> fp32 compute).
                x3 = vload(in_ch, base + NOPE_DIM)
                x3 = vcast(
                    vcast(x3, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                # Lanes 0-31 of the +32 load hold this row's x2; lanes 32-63
                # hold the next row's x1 (or the pad row's stale data on the
                # last row) and are masked out below.
                x3n = vload(in_ch, base + NOPE_DIM + HALF_ROPE_DIM)
                x3n = vcast(
                    vcast(x3n, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                c = vload(cs_ch, r * ROPE_DIM)  # [cos||sin]
                s = vload(cs_ch, r * ROPE_DIM + HALF_ROPE_DIM)  # [sin|..]
                t1 = vmul(x3, c, mask=m_lo)  # x1*cos
                t2 = vmul(x3n, s, mask=m_lo)  # x2*sin
                o_lo = vsub(t1, t2, mask=m_lo)
                t3 = vmul(x3, s, mask=m_lo)  # x1*sin
                t4 = vmul(x3n, c, mask=m_lo)  # x2*cos
                o_hi = vadd(t3, t4, mask=m_lo)
                vstore_pack(
                    out_ch,
                    base + NOPE_DIM,
                    vcast(o_lo, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    m_lo,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    out_ch,
                    base + NOPE_DIM + HALF_ROPE_DIM,
                    vcast(o_hi, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    m_lo,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def _norm_rope_k(
        self, in_ch, out_ch, cs_ch, gamma_ch, cur_rows, avg, epsilon
    ):
        """K head unit: per-head fp32 RMSNorm (Newton rsqrt) + tail-64 RoPE.

        RoPE uses the same dual-load low-lane scheme as _rope_q: the norm is
        applied to both halves separately (y1 from the base load, y2 from the
        +32 load), keeping all rotate-half pairs lane-aligned without
        gather/vmem_bar.
        """
        with vf(mode="raw"):
            full = full_mask()
            m_lo = update_mask(HALF_ROPE_DIM, elem_bits=32)[0]
            one = vdups(1.0, F32, mask=full)
            half = vdups(0.5, F32, mask=full)
            one_half = vdups(1.5, F32, mask=full)
            zero = vdups(0.0, F32, mask=full)
            fmax = vdups(_F32_MAX, F32, mask=full)

            gu0 = vload_unpack(gamma_ch, 0, unpack_mode=UnpackMode.B16_TO_B32)
            gu1 = vload_unpack(gamma_ch, VL, unpack_mode=UnpackMode.B16_TO_B32)
            gu2 = vload_unpack(gamma_ch, 2 * VL, unpack_mode=UnpackMode.B16_TO_B32)
            gu3 = vload_unpack(gamma_ch, NOPE_DIM, unpack_mode=UnpackMode.B16_TO_B32)
            # gamma[224:256] re-unpacked at +32 so g2 lands in the low lanes
            # (uses the 32-element tail slack on gamma_ch).
            gu3n = vload_unpack(
                gamma_ch, NOPE_DIM + HALF_ROPE_DIM, unpack_mode=UnpackMode.B16_TO_B32
            )
            g0 = vcast(gu0, F32, mask=full)
            g1 = vcast(gu1, F32, mask=full)
            g2 = vcast(gu2, F32, mask=full)
            g3 = vcast(gu3, F32, mask=full)  # lanes 0-31 = gamma[192:224]
            g3n = vcast(gu3n, F32, mask=full)  # lanes 0-31 = gamma[224:256]

            for r in dsl_range(0, cur_rows, 1):
                base = r * HEAD_DIM
                # bf16 round-trip before the RMSNorm input: the production
                # path materializes qkv as bf16 in GM and the norm kernel
                # reads it back (bf16 -> fp32 accumulate).
                x0 = vload(in_ch, base)
                x0 = vcast(
                    vcast(x0, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                x1 = vload(in_ch, base + VL)
                x1 = vcast(
                    vcast(x1, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                x2 = vload(in_ch, base + 2 * VL)
                x2 = vcast(
                    vcast(x2, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                x3 = vload(in_ch, base + NOPE_DIM)
                x3 = vcast(
                    vcast(x3, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                x3n = vload(in_ch, base + NOPE_DIM + HALF_ROPE_DIM)
                x3n = vcast(
                    vcast(x3n, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )

                t0 = vreduce_sum(vmul(x0, x0, mask=full), mask=full)
                t1 = vreduce_sum(vmul(x1, x1, mask=full), mask=full)
                t2 = vreduce_sum(vmul(x2, x2, mask=full), mask=full)
                t3 = vreduce_sum(vmul(x3, x3, mask=full), mask=full)
                total = vadd(
                    vadd(t0, t1, mask=full), vadd(t2, t3, mask=full), mask=full
                )
                var = vadds(vmuls(total, avg, mask=full), epsilon, mask=full)
                var = vmaxs(var, -99.99, mask=full)

                recip = vdiv(one, var, mask=full)
                y = vsqrt(recip, mask=full)
                t = vmuls(var, -0.5, mask=full)
                t = vmul(t, y, mask=full)
                t1n = vadd(one_half, vmul(t, y, mask=full), mask=full)
                rstd = vmul(y, t1n, mask=full)
                t3n = vmuls(var, -1.0, mask=full)
                s = vadd(one, vmul(t3n, recip, mask=full), mask=full)
                t4 = vmuls(rstd, -1.0, mask=full)
                recip = vadd(recip, vmul(t4, rstd, mask=full), mask=full)
                s = vadd(s, vmul(var, recip, mask=full), mask=full)
                s = vmul(s, rstd, mask=full)
                rstd = vadd(rstd, vmul(s, half, mask=full), mask=full)
                cmp_inf = vgts(var, _F32_MAX * 0.999, mask=full)
                rstd = vselect(zero, rstd, cond_mask=cmp_inf)
                cmp_pos = vgts(var, 0.0, mask=full)
                rstd = vselect(rstd, fmax, cond_mask=cmp_pos)
                rstd_b = vdup(rstd, mask=full)

                y0 = vmul(vmul(x0, rstd_b, mask=full), g0, mask=full)
                y1 = vmul(vmul(x1, rstd_b, mask=full), g1, mask=full)
                y2 = vmul(vmul(x2, rstd_b, mask=full), g2, mask=full)
                vstore_pack(
                    out_ch,
                    base,
                    vcast(y0, BF16, mask=full, rounding=RoundingMode.RN),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    out_ch,
                    base + VL,
                    vcast(y1, BF16, mask=full, rounding=RoundingMode.RN),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    out_ch,
                    base + 2 * VL,
                    vcast(y2, BF16, mask=full, rounding=RoundingMode.RN),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )

                # Norm applied to both rope halves in the low lanes; round
                # each back to bf16 (original two-kernel GM round-trip).
                y1r = vmul(vmul(x3, rstd_b, mask=m_lo), g3, mask=m_lo)
                y2r = vmul(vmul(x3n, rstd_b, mask=m_lo), g3n, mask=m_lo)
                y1r = vcast(
                    vcast(y1r, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    F32,
                    mask=m_lo,
                )
                y2r = vcast(
                    vcast(y2r, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    F32,
                    mask=m_lo,
                )
                c = vload(cs_ch, r * ROPE_DIM)
                s = vload(cs_ch, r * ROPE_DIM + HALF_ROPE_DIM)
                o_lo = vsub(
                    vmul(y1r, c, mask=m_lo), vmul(y2r, s, mask=m_lo), mask=m_lo
                )
                o_hi = vadd(
                    vmul(y1r, s, mask=m_lo), vmul(y2r, c, mask=m_lo), mask=m_lo
                )
                vstore_pack(
                    out_ch,
                    base + NOPE_DIM,
                    vcast(o_lo, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    m_lo,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    out_ch,
                    base + NOPE_DIM + HALF_ROPE_DIM,
                    vcast(o_hi, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    m_lo,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def _cast_rows(self, in_ch, out_ch, cur_rows):
        """V / mirror units: fp32 tile -> bf16 RN passthrough."""
        with vf(mode="raw"):
            full = full_mask()
            for r in dsl_range(0, cur_rows, 1):
                base = r * HEAD_DIM
                for j in range(4):
                    off = base + j * VL
                    x = vload(in_ch, off)
                    vstore_pack(
                        out_ch,
                        off,
                        vcast(x, BF16, mask=full, rounding=RoundingMode.RN),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )

    def __call__(
        self,
        gm_hidden: Tensor,
        gm_weight: Tensor,
        gm_gamma: Tensor,
        gm_positions: Tensor,
        gm_cos_sin: Tensor,
        gm_slot: Tensor,
        gm_q: Tensor,
        gm_k: Tensor,
        gm_v: Tensor,
        gm_mk: Tensor,
        gm_mv: Tensor,
        gm_k_cache: Tensor,
        gm_v_cache: Tensor,
        epsilon,
        gm_segment_tile_starts: Tensor,
    ):
        block_idx = get_block_idx()
        block_num = get_block_num()
        sub = get_subblock_id()

        m_rows = gm_hidden.shape[0]
        m_pairs = (m_rows + PAIR_M - 1) // PAIR_M
        # Unit count follows the projection width: 2560 = mirror-source
        # layer (10 units, Q/K/V/mirror_k/mirror_v), 2048 = plain layer
        # (8 units, Q/K/V). The per-unit dispatch below is width-agnostic;
        # mirror branches simply never fire for the 8-unit layout.
        n_units = gm_weight.shape[0] // BASE_N

        # ---- cube-side channels (L1 = 256+256 = 512KB exact; the arena
        # matmul sample runs the identical baseM=256 configuration) ----
        l1_a = Channel(MemLoc.L1, shape=(PAIR_M, K_L1), dtype=BF16, depth=2)
        l1_b = Channel(MemLoc.L1, shape=(BASE_N, K_L1), dtype=BF16, depth=2)
        l0a = Channel(MemLoc.L0A, shape=(PAIR_M, BASE_K), dtype=BF16, depth=2)
        l0b = Channel(MemLoc.L0B, shape=(BASE_N, BASE_K), dtype=BF16, depth=2)
        # 256x256 f32 == 256KB L0C: depth1 (the pair tile trades fixpipe
        # double-buffering for the doubled M; the next item's first MMAD
        # still overlaps this item's fixpipe drain).
        l0c = Channel(MemLoc.L0C, shape=(PAIR_M, BASE_N), dtype=F32, depth=1)
        # ---- cross-core handoff (fp32: dual_dst_ctl forbids narrowing cast) ----
        # One extra pad row: the rope fast path re-loads x2 at +32, whose
        # 64-lane load on the last row spans into the next row. The pad row
        # is never written; its stale lanes are masked out by the 32-lane
        # rope math. Keeping the pad as a whole row preserves the (128, 256)
        # contiguous fixpipe destination layout.
        cv_ub = Channel(
            MemLoc.UB,
            shape=(TILE_VEC_M + 1, BASE_N),
            dtype=F32,
            depth=1,
            kind=ChannelKind.CrossCore,
        )
        # ---- vec-side channels ----
        out_ch = Channel(MemLoc.UB, shape=(TILE_VEC_M, HEAD_DIM), dtype=BF16, depth=1)
        # Pad row for the rope fast path's +32 cos/sin load on the last row.
        cs_ch = Channel(
            MemLoc.UB, shape=(TILE_VEC_M + 1, ROPE_DIM), dtype=F32, depth=1
        )
        # 32-element tail slack: _norm_rope_k re-unpacks gamma[224:256] at
        # offset 224 (64 bf16 lanes) so g2 lands in the low lanes; the slack
        # lanes are written never and read only as masked-out garbage.
        gamma_ch = Channel(
            MemLoc.UB, shape=(1, HEAD_DIM + HALF_ROPE_DIM), dtype=BF16, depth=1
        )

        nd2nz_a = make_copy_engine(format_transform="nd2nz", dtype=BF16, pad_value=0.0)
        nd2nz_b = make_copy_engine(format_transform="nd2nz", dtype=BF16, pad_value=0.0)
        fixpipe = make_copy_engine(dtype=F32, dual_dst_ctl=1)

        # Work-item scheduling: granularity is (pair, unit). Whole-tile
        # assignment gives 65 pairs / 32 cores = 2.03 waves — one core runs a
        # 3rd wave while the rest idle. Item granularity spreads
        # m_pairs*n_units items evenly (<=3.4% skew).
        total_items = m_pairs * n_units
        for item in range(block_idx, total_items, block_num):
            pair = item // n_units
            u = item - pair * n_units

            cur_tokens = m_rows - pair * PAIR_M
            if cur_tokens > PAIR_M:
                cur_tokens = PAIR_M

            # Balanced split-M plan for this (possibly tail-clipped) pair
            # tile; a trailing odd tile clips to its real row count.
            q_tile0 = tile_view(gm_q, (PAIR_M, HEAD_DIM), (pair, 0))
            split_m = make_partition_tiler(q_tile0.shape, (PAIR_M, HEAD_DIM))
            half0 = partition_view(q_tile0, split_m, sub)
            rows_here = half0.shape[0]
            sub_start = select(sub == 0, 0, cur_tokens - rows_here)
            row0 = pair * PAIR_M + sub_start
            # A trailing subblock may own zero rows (e.g. M == 1 splits
            # 1/0); clamp so the scalar position load stays in bounds even
            # though every per-row loop below is empty.
            row0 = select(row0 < m_rows, row0, m_rows - 1)

            # Vec-side preloads for this item (gamma + this AIV's cos/sin);
            # item granularity re-issues them per unit (~16 KB DMA, noise).
            mem_copy(
                tile_view(gamma_ch, (1, HEAD_DIM), (0, 0)),
                tile_view(gm_gamma, (1, HEAD_DIM), (0, 0)),
            )
            # Table extent is a compile-time constant under every spec form
            # (JIT trace or AOT TensorSpec).
            table_rows = gm_cos_sin.shape[0]
            if const_expr(self.positions_segmented):
                self._load_segmented_cos_sin(
                    cs_ch, gm_cos_sin, gm_positions,
                    gm_segment_tile_starts, row0, rows_here,
                )
            elif const_expr(self.positions_contiguous):
                # Prefill: one run of TILE_VEC_M consecutive table rows.
                pos_base = gm_positions[row0]
                if rows_here > 0:
                    if pos_base + TILE_VEC_M <= table_rows:
                        # Fast path: one bulk (<=128, 64) DMA.
                        mem_copy(
                            cs_ch,
                            tile_view(
                                gm_cos_sin[
                                    pos_base : pos_base + TILE_VEC_M, 0:ROPE_DIM
                                ],
                                (TILE_VEC_M, ROPE_DIM),
                                (0, 0),
                            ),
                        )
                    else:
                        # Table-tail clamp: the bulk read would overrun the
                        # cos/sin table (e.g. a context filled to its last
                        # position). Fall back to exactly the valid rows --
                        # pos_base + rows_here <= table_rows holds because
                        # every position is range-checked.
                        for r in range(rows_here):
                            mem_copy(
                                tile_view(cs_ch, (1, ROPE_DIM), (r, 0)),
                                tile_view(
                                    gm_cos_sin, (1, ROPE_DIM), (pos_base + r, 0)
                                ),
                            )
            else:
                # Decode: arbitrary per-token positions (each request's own
                # last position). One 1x64 gather per token -- the production
                # generic rope kernel's addressing, at decode-sized M.
                for r in range(rows_here):
                    pos = gm_positions[row0 + r]
                    mem_copy(
                        tile_view(cs_ch, (1, ROPE_DIM), (r, 0)),
                        tile_view(gm_cos_sin, (1, ROPE_DIM), (pos, 0)),
                    )

            # ---- cube: K-loop MMAD accumulation for unit u ----
            for k_l1 in range(K_L1_TILES):
                a_tile = tile_view(gm_hidden, (PAIR_M, K_L1), (pair, k_l1))
                b_tile = tile_view(gm_weight, (BASE_N, K_L1), (u, k_l1))
                # l2_cache_ctl=1 (normal): W tiles stay resident in L2
                # across m-tiles; the disabled default re-fetches the
                # 10.5 MB weight from HBM for every m-tile.
                mem_copy(l1_a, a_tile, engine=nd2nz_a, l2_cache_ctl=1)
                mem_copy(l1_b, b_tile, engine=nd2nz_b, l2_cache_ctl=1)
                for k_l0 in range(K_L0_PER_L1):
                    mem_copy(
                        l0a, tile_view(l1_a, (PAIR_M, BASE_K), (0, k_l0))
                    )
                    mem_copy(
                        l0b, tile_view(l1_b, (BASE_N, BASE_K), (0, k_l0))
                    )
                    gk = k_l1 * K_L0_PER_L1 + k_l0
                    matmul(l0c, l0a, l0b, init=(gk == 0))

            # ---- handoff: L0C -> per-AIV UB halves ----
            mem_copy(
                tile_view(cv_ub, (TILE_VEC_M, BASE_N), (0, 0)),
                l0c,
                engine=fixpipe,
                partition=split_m,
            )

            # ---- vec epilogue, specialized per unit ----
            # Dense outputs use one strided block copy per item (op①
            # CopyoutProto's blockCount=rows pattern) instead of per-row
            # 512B DMAs; only the slot scatter keeps row granularity.
            out_rows = local_slice(
                out_ch, (rows_here, HEAD_DIM), stride=(HEAD_DIM, 1)
            )
            if u < NUM_Q_HEADS:
                self._rope_q(cv_ub, out_ch, cs_ch, rows_here)
                q_half = partition_view(
                    tile_view(gm_q, (PAIR_M, HEAD_DIM), (pair, u)), split_m, sub
                )
                mem_copy(q_half, out_rows)
            elif u == UNIT_K:
                self._norm_rope_k(
                    cv_ub, out_ch, cs_ch, gamma_ch, rows_here, AVG, epsilon
                )
                k_half = partition_view(
                    tile_view(gm_k, (PAIR_M, HEAD_DIM), (pair, 0)),
                    split_m,
                    sub,
                )
                # The tile drain and the slot scatter are two consumers of
                # one out_ch epoch; the channel ledger groups reads by their
                # outermost non-writing loop, so both must share one loop
                # scope. Single-shot chunk loop: rows_here <= TILE_VEC_M.
                for c in range(0, rows_here, TILE_VEC_M):
                    mem_copy(k_half, out_rows)
                    for r in range(rows_here):
                        slot = gm_slot[row0 + r]
                        if slot != -1:
                            mem_copy(
                                tile_view(
                                    gm_k_cache, (1, HEAD_DIM), (slot, 0)
                                ),
                                tile_view(out_ch, (1, HEAD_DIM), (r, 0)),
                            )
            elif u == UNIT_V:
                self._cast_rows(cv_ub, out_ch, rows_here)
                for c in range(0, rows_here, TILE_VEC_M):
                    if const_expr(self.return_v):
                        # Transitional interface: materialize v like the
                        # production scatter's transient contiguous copy.
                        v_half = partition_view(
                            tile_view(gm_v, (PAIR_M, HEAD_DIM), (pair, 0)),
                            split_m,
                            sub,
                        )
                        mem_copy(v_half, out_rows)
                    for r in range(rows_here):
                        slot = gm_slot[row0 + r]
                        if slot != -1:
                            mem_copy(
                                tile_view(
                                    gm_v_cache, (1, HEAD_DIM), (slot, 0)
                                ),
                                tile_view(out_ch, (1, HEAD_DIM), (r, 0)),
                            )
            else:
                self._cast_rows(cv_ub, out_ch, rows_here)
                gm_m = gm_mk if u == UNIT_MK else gm_mv
                if const_expr(self.has_mirror):
                    m_half = partition_view(
                        tile_view(gm_m, (PAIR_M, HEAD_DIM), (pair, 0)),
                        split_m,
                        sub,
                    )
                    mem_copy(m_half, out_rows)
                else:
                    # Plain-layer stand-ins are (1, 256); the mirror units
                    # never execute at runtime, so per-row runtime coords
                    # (which pass static verification) suffice.
                    for r in range(rows_here):
                        mem_copy(
                            tile_view(gm_m, (1, HEAD_DIM), (row0 + r, 0)),
                            tile_view(out_ch, (1, HEAD_DIM), (r, 0)),
                        )


def compile_aot(
    qkv_width, num_slots, max_pos, return_v=False,
    positions_contiguous=True, positions_segmented=False,
):
    """AOT-compile ONE dynamic-M artifact (ProviderCallable).

    The token count M is a symbolic ``Dim("M", min=1)`` — the same compiled
    artifact serves every M >= 1 (prefill chunk sizes, decode batches,
    anything in between) with no per-shape recompilation. The optional
    segment-table length is also symbolic; deployment constants are:

        qkv_width: 2560 (kv-mirror source layer) or 2048 (plain QKV layer)
        num_slots: paged KV cache rows (flattenable to (slots, 256))
        max_pos:   cos_sin_cache rows
        return_v:  additionally materialize v (M, 256)
        positions_contiguous: True = prefill artifact (bulk cos/sin DMA on a
            runtime-contiguous position run); False = decode artifact
            (per-token gather, arbitrary positions). Both share the identical
            dynamic-M Dim and epilogue math.
        positions_segmented: True = multi-request prefill artifact, using
            request-local intervals from WeLM's RoPE segment tile table.
            Takes precedence over positions_contiguous. Table entries are
            strictly increasing int32 row offsets, start at 0, end at the
            real-token count (<=M), and describe contiguous-position runs
            of at most 64 rows. Remaining rows are suffix padding.

    The returned callable takes the same 14 runtime arguments as
    ``FusedQkvProjNormRopeCache.run`` (gamma as (1, 256)). The segmented
    artifact takes a 15th argument: the device segment table, including
    its sentinel. Its dynamic length does not specialize the artifact.

    The ``Dim``-shared M slots (hidden/positions/slot/q/k/v and, for
    mirror layers, mk/mv) are runtime-equality-checked by the framework on
    every launch.
    """
    n_units = qkv_width // HEAD_DIM
    if n_units * HEAD_DIM != qkv_width or n_units not in (8, 10):
        raise ValueError(
            f"qkv_width must be 2048 or 2560, got {qkv_width}"
        )
    op = FusedQkvProjNormRopeCache(
        return_v=return_v,
        has_mirror=(n_units == 10),
        positions_contiguous=positions_contiguous,
    )
    m = Dim("M", min=1)
    mirror_rows = m if n_units == 10 else 1  # plain stand-ins stay (1, 256)
    bf, f32, i64 = BF16, F32, dtypes.int64
    specs = (
        TensorSpec((m, HIDDEN), bf),
        TensorSpec((qkv_width, HIDDEN), bf),
        TensorSpec((1, HEAD_DIM), bf),
        TensorSpec((m,), i64),
        TensorSpec((max_pos, ROPE_DIM), f32),
        TensorSpec((m,), i64),
        TensorSpec((m, NUM_Q_HEADS * HEAD_DIM), bf),
        TensorSpec((m, HEAD_DIM), bf),
        TensorSpec((m, HEAD_DIM), bf),  # gm_v (k stand-in when return_v=False)
        TensorSpec((mirror_rows, HEAD_DIM), bf),
        TensorSpec((mirror_rows, HEAD_DIM), bf),
        TensorSpec((num_slots, HEAD_DIM), bf),
        TensorSpec((num_slots, HEAD_DIM), bf),
        dtypes.float32,
    )
    if positions_segmented:
        segment_entries = Dim("segment_entries", min=2)
        return op.run_segmented.compile(
            *specs, TensorSpec((segment_entries,), dtypes.int32)
        )
    return op.run.compile(*specs)


def _aic_block_limit() -> int:
    """Effective AIC block count (device query with static fallback)."""
    try:
        from cannbotdsl.ops.info import get_platform_info

        info = get_platform_info()
        if info.available and info.cube_core_num > 0:
            return int(info.cube_core_num)
    except Exception:
        pass
    return AIC_NUM


class FusedQkvProjNormRopeCache:
    """Host-facing wrapper: projection + post-processing + cache scatter in 1 kernel.

    ``return_v=False`` (default) matches the profiled NPU dataflow: the sink
    prefill attention reads K/V from the paged cache, so v is only scattered.
    ``return_v=True`` additionally materializes v (M, 256) for transitional
    callers that need the standard attn(q, k, v) interface.
    """

    def __init__(
        self,
        return_v: bool = False,
        has_mirror: bool = True,
        positions_contiguous: bool = True,
    ):
        self._return_v = bool(return_v)
        self._has_mirror = bool(has_mirror)
        self._positions_contiguous = bool(positions_contiguous)

    @jit
    def run(
        self,
        gm_hidden,
        gm_weight,
        gm_gamma,
        gm_positions,
        gm_cos_sin,
        gm_slot,
        gm_q,
        gm_k,
        gm_v,
        gm_mk,
        gm_mv,
        gm_k_cache,
        gm_v_cache,
        eps: float,
    ):
        m_rows = gm_hidden.shape[0]
        m_pairs = (m_rows + PAIR_M - 1) // PAIR_M
        block_dim = m_pairs * (gm_weight.shape[0] // BASE_N)
        grid_cap = _aic_block_limit()
        if block_dim > grid_cap:
            block_dim = grid_cap
        op = FusedQkvProjNormRopeCacheKernel(
            self._return_v, self._has_mirror, self._positions_contiguous
        )
        op[block_dim](
            gm_hidden,
            gm_weight,
            gm_gamma,
            gm_positions,
            gm_cos_sin,
            gm_slot,
            gm_q,
            gm_k,
            gm_v,
            gm_mk,
            gm_mv,
            gm_k_cache,
            gm_v_cache,
            eps,
            # Unused stand-in: the non-segmented kernel folds this input
            # away. Preserve the existing 14-argument public run ABI.
            gm_positions,
        )

    @jit
    def run_segmented(
        self,
        gm_hidden,
        gm_weight,
        gm_gamma,
        gm_positions,
        gm_cos_sin,
        gm_slot,
        gm_q,
        gm_k,
        gm_v,
        gm_mk,
        gm_mv,
        gm_k_cache,
        gm_v_cache,
        eps: float,
        gm_segment_tile_starts,
    ):
        m_rows = gm_hidden.shape[0]
        m_pairs = (m_rows + PAIR_M - 1) // PAIR_M
        block_dim = m_pairs * (gm_weight.shape[0] // BASE_N)
        grid_cap = _aic_block_limit()
        if block_dim > grid_cap:
            block_dim = grid_cap
        op = FusedQkvProjNormRopeCacheKernel(
            self._return_v, self._has_mirror, False, True
        )
        op[block_dim](
            gm_hidden,
            gm_weight,
            gm_gamma,
            gm_positions,
            gm_cos_sin,
            gm_slot,
            gm_q,
            gm_k,
            gm_v,
            gm_mk,
            gm_mv,
            gm_k_cache,
            gm_v_cache,
            eps,
            gm_segment_tile_starts,
        )


def fused_qkv_proj_norm_rope_cache(
    hidden,
    weight,
    k_gamma,
    positions,
    cos_sin_cache,
    slot_mapping,
    k_cache,
    v_cache,
    eps=1e-6,
    return_v=False,
    validate=True,
    positions_contiguous=None,
):
    """WeLMv4 QKV 投影 + 后处理全融合前向（对外公开入口）。

    Computes (per TP rank, kv-mirror layer):
        qkv(M,2560) = hidden(M,2048) @ weight(2560,2048)^T   (bf16 MMAD, fp32 acc)
        q   = rope(qkv[:, :1536])                              -> (M,1536) bf16
        k   = rope(rmsnorm(qkv[:, 1536:1792]))                 -> (M,256)  bf16
        v   = bf16(qkv[:, 1792:2048])                          -> v_cache[slot] only
        mk  = bf16(qkv[:, 2048:2304])                          -> (M,256)  bf16
        mv  = bf16(qkv[:, 2304:2560])                          -> (M,256)  bf16
        k_cache[slot] = k, v_cache[slot] = v                   (slot == -1 skipped)

    Args:
        hidden: (M, 2048) bf16 contiguous (QKV projection input)
        weight: (2560, 2048) bf16 contiguous, nn.Linear (N, K) layout
        k_gamma: (256,) bf16, K RMSNorm weight
        positions: (M,) int64, contiguous positions (validated when validate=True)
        cos_sin_cache: (max_pos, 64) fp32, first 32 cos / last 32 sin per row
        slot_mapping: (M,) int64 cache row indices, -1 skips the scatter
        k_cache / v_cache: paged KV cache, last dim 256, flattenable to (slots, 256)
        return_v: also materialize v (M, 256) for callers needing attn(q, k, v);
            the profiled NPU sink-prefill attention reads K/V from the cache,
            so the default (False) matches the production dataflow.
        validate: host-side checks (positions contiguity/range, slot range,
            cache device/shape consistency). Each check is a small device
            reduction + sync; disable on hot paths where the caller
            guarantees the invariants.

    Returns:
        weight (2560, 2048) — kv-mirror source layer:
            return_v=False: (q, k, mirror_k, mirror_v) — v goes to cache only.
            return_v=True:  (q, k, v, mirror_k, mirror_v).
        weight (2048, 2048) — plain QKV layer (no mirror segments):
            return_v=False: (q, k).
            return_v=True:  (q, k, v).

    Note (scope): bf16 weights only. The production modelslim-mxfp8
    ``_kv_mirror_mxfp8_source_projection`` path (raw (N,K) weight + e8m0
    scales) is NOT covered by this operator.
    """
    assert hidden.dim() == 2 and hidden.shape[1] == HIDDEN
    assert hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
    assert weight.dim() == 2 and weight.shape[1] == HIDDEN
    # Two profiled widths: 2560 = kv-mirror source layer (5 head-units),
    # 2048 = plain QKV layer (q/k/v only, welmv4.py:1845-1847 branch).
    n_units = weight.shape[0] // HEAD_DIM
    assert n_units * HEAD_DIM == weight.shape[0] and n_units in (8, 10), (
        f"weight must be (2048, {HIDDEN}) or ({QKV_WIDTH}, {HIDDEN}), "
        f"got {tuple(weight.shape)}"
    )
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert k_gamma.dim() == 1 and k_gamma.shape[0] == HEAD_DIM
    assert k_gamma.dtype == torch.bfloat16 and k_gamma.is_contiguous()
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == ROPE_DIM
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.is_contiguous()

    num_tokens = hidden.shape[0]
    assert num_tokens > 0
    assert positions.shape[0] == num_tokens
    assert slot_mapping.shape[0] == num_tokens
    assert positions.device == hidden.device and slot_mapping.device == hidden.device

    if positions.dtype == torch.int32:
        positions = positions.to(torch.int64)
    assert positions.dtype == torch.int64
    if slot_mapping.dtype == torch.int32:
        slot_mapping = slot_mapping.to(torch.int64)
    assert slot_mapping.dtype == torch.int64

    def _flatten_cache(cache, name):
        assert cache.dtype == torch.bfloat16
        assert cache.shape[-1] == HEAD_DIM
        assert cache.is_contiguous()
        if cache.dim() > 2:
            cache = cache.view(-1, HEAD_DIM)
        assert cache.dim() == 2, f"{name} must be flattenable to (slots, 256)"
        return cache

    k_cache = _flatten_cache(k_cache, "k_cache")
    v_cache = _flatten_cache(v_cache, "v_cache")
    assert k_cache.device == hidden.device, "k_cache must live on hidden's device"
    assert v_cache.device == hidden.device, "v_cache must live on hidden's device"
    assert k_cache.shape == v_cache.shape, "k_cache/v_cache shape mismatch"

    # Position mode: None auto-detects (one device reduction; decode
    # callers pass it explicitly to skip the sync). An explicit
    # positions_contiguous=True with validate=True is asserted, matching
    # the historical prefill-only contract.
    if positions_contiguous is None:
        positions_contiguous = num_tokens == 1 or bool(
            (positions[1:] - positions[:-1] == 1).all()
        )
    elif validate and positions_contiguous and num_tokens > 1:
        assert bool((positions[1:] - positions[:-1] == 1).all()), (
            "positions must be contiguous"
        )
    if validate:
        assert int(positions.min()) >= 0, "negative position"
        assert int(positions.max()) < cos_sin_cache.shape[0], (
            "position beyond cos_sin_cache rows"
        )
        valid = slot_mapping != -1
        if bool(valid.any()):
            assert int(slot_mapping[valid].min()) >= 0, "negative slot"
            assert int(slot_mapping[valid].max()) < k_cache.shape[0], (
                "slot beyond cache rows"
            )

    device = hidden.device
    q = torch.empty(
        (num_tokens, NUM_Q_HEADS * HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    k = torch.empty((num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device)
    if return_v:
        v = torch.empty((num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device)
        # real destination; kernel writes it in the return_v branch
        gm_v = v
    else:
        # shape-compatible stand-in; the v-write branch is folded away at
        # trace time so this tensor is never written
        gm_v = k
    if n_units == 10:
        mirror_k = torch.empty(
            (num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        mirror_v = torch.empty(
            (num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    else:
        # Plain 8-unit layer has no mirror segments; the kernel never
        # touches gm_mk/gm_mv for u < 8, so 1-row stand-ins suffice.
        mirror_k = torch.zeros(1, HEAD_DIM, dtype=torch.bfloat16, device=device)
        mirror_v = mirror_k

    op = FusedQkvProjNormRopeCache(
        return_v=return_v,
        has_mirror=(n_units == 10),
        positions_contiguous=positions_contiguous,
    )
    op.run(
        hidden,
        weight,
        k_gamma.view(1, HEAD_DIM),
        positions,
        cos_sin_cache,
        slot_mapping,
        q,
        k,
        gm_v,
        mirror_k,
        mirror_v,
        k_cache,
        v_cache,
        float(eps),
    )
    if n_units == 10:
        if return_v:
            return q, k, v, mirror_k, mirror_v
        return q, k, mirror_k, mirror_v
    if return_v:
        return q, k, v
    return q, k
