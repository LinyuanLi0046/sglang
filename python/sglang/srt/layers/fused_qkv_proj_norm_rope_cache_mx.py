# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FusedQkvProjNormRopeCacheMX —— WeLMv4 QKV 投影全融合算子的 MXFP8 版本。

使用与 BF16 算子相同的 cannbotdsl 0.8 API；模型直接调用预量化 AOT 入口。

复刻 modelslim-mxfp8 部署路径（sglang ``_kv_mirror_mxfp8_source_projection`` /
``NPUMXFP8LinearMethod``）的前向数据流，把 profile 中 prefill 路径 kv-mirror 层
的完整序列融合为 1 个 MIX_AIC_1_2 复合核：

    网络原始序列（doc_mx profiling，prefill M=16500）：
      npu_dynamic_mx_quant(hidden)                             (A 侧在线 MX 量化)
      npu_quant_matmul(qx, W_fp8, w_scale, sa, [1,1,32])       → qkv(M,N) BF16
      Slice q/k/v/mirror_k/mirror_v ×5
      mmq_style_k_rms_norm_kernel + welmv4_inplace_rope_*      (K norm, Q/K RoPE)
      ScatterNdUpdate k_cache / v_cache ×2
    融合后：
      qkv(M,2560) = (qx(M,2048)*sa) @ (W(2560,2048)*sb)^T      (Cube, OCP MXFP8
                    MMAD：L0A/L0B 绑定 e8m0 scale，fp32 组内累加)
      ├── [0:1536]   6 x Q head   : 尾 64 维 RoPE ──────────────► q(M,1536)
      ├── [1536:1792] K head      : RMSNorm → 尾 64 RoPE ─┬─────► k(M,256)
      │                                                 └─────► k_cache[slot]
      ├── [1792:2048] V head      : fp32→bf16 RN ──────────────► v_cache[slot]
      ├── [2048:2304] mirror_k    : 直通 ──────────────────────► mirror_k(M,256)
      └── [2304:2560] mirror_v    : 直通 ──────────────────────► mirror_v(M,256)

数值语义（与 aclnnQuantMatmulV5_QuantBatchMatmulV3 + 后处理 kernel 对齐）：
    - A 侧量化由 host 调 npu_dynamic_mx_quant 完成（与网络 quantize_activation
      完全一致：per-32-group e8m0 scale = 2^ceil(log2(amax/448))，qx saturate
      RN 到 e4m3），kernel 只做 MX matmul + epilogue —— 量化结果与网络逐位相同；
    - MX MMAD 每 32 元素组以 sa[m,g]*sb[n,g] 反量化后在 fp32 累加（OCP MXFP8
      硬件路径，与 aclnn 的 QuantBatchMatmulV3 同类；组间累加顺序可能不同，
      bf16 结果允许 ±1-2 ulp）；
    - L0C fp32 经 split-M dual-destination FIXPIPE 直达 AIV 的 UB RingBuffer，
      epilogue 在进入 RMSNorm/RoPE 前先舍回 bf16 再升 fp32（复刻网络"qkv 以
      bf16 落 GM、后处理 kernel 读回"的舍入语义）。

MX 数据布局（msmodelslim 导出 + npu_dynamic_mx_quant 输出，全部 int8 字节载体）：
    gm_qx:  (M, 2048)  float8_e4m3fn 字节（int8 视图）
    gm_sa:  (M, 32, 2) e8m0 字节 —— flat 即 (M, 64) 行主序，组 g 的 scale 在
            [m, g//2, g%2]（npu_dynamic_mx_quant 的原生输出布局）
    gm_w:   (N, 2048)  float8_e4m3fn 字节（nn.Linear (N,K) 布局）
    gm_sb:  (N, 32, 2) e8m0 字节 —— flat 即 (N, 64) 行主序（msmodelslim
            weight_scale (N, K/32) 的 (N, K/64, 2) 视图）
    scale GM→L1 交接用显式布局视图：
        A 侧 flatten：GM (M, S/2, 2) → L1 (M, S_w) ZN view
        B 侧 flatten/permute：GM (N, S/2, 2) → L1 (S_w, N) NZ
    L1→L0 数据搬运绑 mx_scale=...（asc_copy_l12l0a/b_mx），matmul(mx_mode=True) 选择
    asc_mmad_mx。

结构（与 bf16 版 fused_qkv_proj_norm_rope_cache 同构，仅 cube 管线换 MX）：

    Cube:  PAIR_M=256 × BASE_N=256 / tile，BASE_K=128（2 个 64 元素 MX 块），
           K_L1=256 窗口 ×8
           L1: a(256,256 fp8) + b(256,256 fp8) depth2
               sa(256,8 e8m0 zn) + sb(8,256 e8m0 nz) depth2
           L0A/L0B(256/256,128 fp8) depth2 → L0C(256,256 f32) depth1
    交接:  mem_copy(cv_ub, l0c_live.partition(dim=0)) —— dual-destination FIXPIPE
    Vec:   与 bf16 版逐行同构（Q×6 / K / V / mirror_k / mirror_v 分派）
    多核:  grid = min(ceil(M/256) × n_units, 平台 AIC 数)，(pair, unit) 粒度
           grid-stride 均衡分发

特化条件（host 侧由调用方保证，与 bf16 版一致）：
    - positions 模式二选一（prefill 连续 / decode 逐 token gather）；
    - slot_mapping 元素为 cache 行索引或 -1（跳过）；
    - K = hidden_size = 2048、N = 2560/2048 固化；M 为 Dim 动态维。
"""

import cannbotdsl
import torch
from cannbotdsl import Dim, MemLoc, Tensor, TensorSpec, dtypes, make_layout
from cannbotdsl.lang import jit, kernel, vf
from cannbotdsl.lang import range as dsl_range
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, cv_valid_extent, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    full_mask,
    update_mask,
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
)
from cannbotdsl.ops.reg.cast import RoundingMode
from cannbotdsl.ops.sync import vec_sync_all
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import tile_slice
from cannbotdsl.types.dtypes import bfloat16 as BF16
from cannbotdsl.types.dtypes import float32 as F32

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
MX_GROUP = 32  # OCP MX group size (matches group_sizes=[1, 1, 32])
SCALE_PAIRS = HIDDEN // (2 * MX_GROUP)  # 32 e8m0 pairs per row

# ---- tiling constants (arch35: L1 512K, L0A/B 64K, L0C 256K, AIC 32) ----
PAIR_M = 256
BASE_N = 256  # exactly one head-unit per N tile (epilogue granularity)
BASE_K = 128  # two MX blocks; L0A/L0B (256,128) fp8 depth2 == 64KB (full L0)
K_L1 = 256
K_L1_TILES = HIDDEN // K_L1  # 8
K_L0_PER_L1 = K_L1 // BASE_K  # 2
S_L1 = K_L1 // MX_GROUP  # scale bytes per L1 window = 8
S_L0 = BASE_K // MX_GROUP  # scale bytes per L0 sub-block = 2
PACKED_S_L1 = K_L1 // (2 * MX_GROUP)  # scale pairs per L1 window = 4
TILE_VEC_M = PAIR_M // 2  # per-AIV rows after split-M
AIC_NUM = 32

VL = 64  # vector register elements (2048b / 32b)
AVG = 1.0 / HEAD_DIM
_F32_MAX = float.fromhex("0x1.fffffep+127")

FP8 = dtypes.float8_e4m3fn
E8M0 = dtypes.float8_e8m0


@kernel
class FusedQkvProjNormRopeCacheMxKernel:
    def __init__(
        self,
        return_v: bool = False,
        has_mirror: bool = True,
        positions_contiguous: bool = True,
        rank_chunk_layout: bool = False,
    ):
        # Trace-time constants: fold the optional v materialization, the
        # mirror block-store, and the cos/sin load mode away before IR
        # emission (same contract as the bf16 kernel).
        self.return_v = bool(return_v)
        self.has_mirror = bool(has_mirror)
        self.positions_contiguous = bool(positions_contiguous)
        self.rank_chunk_layout = bool(rank_chunk_layout)

    @jit
    def _rope_q(self, in_ch, out_ch, cs_ch, cur_rows):
        """Q head unit: 192-col passthrough + tail-64 NeoX rotate-half.

        Identical to the bf16 kernel's epilogue: input is the fp32
        cross-core tile; bf16 RN rounding happens at store. A second load
        at +32 puts this row's x2 in the low lanes, so every rotate-half
        pair is lane-aligned and computed with 32-lane masked ops.
        """
        with vf(mode="simd"):
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
                # the network materializes qkv as bf16 in GM and the rope
                # kernel reads it back (bf16 -> fp32 compute).
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
        """K head unit: per-head fp32 RMSNorm (Newton rsqrt) + tail-64 RoPE."""
        with vf(mode="simd"):
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
                # bf16 round-trip before the RMSNorm input: the network
                # materializes qkv as bf16 in GM and the norm kernel reads
                # it back (bf16 -> fp32 accumulate).
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
        with vf(mode="simd"):
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
        gm_qx: Tensor,
        gm_sa: Tensor,
        gm_w: Tensor,
        gm_sb: Tensor,
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
    ):
        """Original 16-argument entry; the helper is trace-time inlined."""
        self._run(
            gm_qx,
            gm_sa,
            gm_w,
            gm_sb,
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
            epsilon,
            0,
            0,
            0,
        )

    def rank_chunk(
        self,
        gm_qx: Tensor,
        gm_sa: Tensor,
        gm_w: Tensor,
        gm_sb: Tensor,
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
        full_local_rows,
        chunk_local_rows,
        chunk_start,
    ):
        """Packed per-rank qx/scale input, full original-order output."""
        self._run(
            gm_qx,
            gm_sa,
            gm_w,
            gm_sb,
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
            epsilon,
            full_local_rows,
            chunk_local_rows,
            chunk_start,
        )

    @jit
    def _run(
        self,
        gm_qx: Tensor,
        gm_sa: Tensor,
        gm_w: Tensor,
        gm_sb: Tensor,
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
        full_local_rows,
        chunk_local_rows,
        chunk_start,
    ):
        block_idx = get_block_idx()
        block_num = get_block_num()
        sub = get_subblock_id()

        if const_expr(self.rank_chunk_layout):
            m_rows = chunk_local_rows
            rank_count = gm_qx.shape[0] // chunk_local_rows
        else:
            m_rows = gm_qx.shape[0]
        m_pairs = (m_rows + PAIR_M - 1) // PAIR_M
        # Unit count follows the projection width: 2560 = mirror-source
        # layer (10 units), 2048 = plain QKV layer (8 units).
        n_units = gm_w.shape[0] // BASE_N

        # ---- cube-side channels ----
        # L1 data rings: a (256,256) fp8 + b (256,256) fp8 == 256KB at
        # depth 2; the paired e8m0 scale rings ride alongside in the ZN/NZ
        # layouts the l1->l0 MX binding consumes.
        l1_a = RingBuffer(MemLoc.L1, FP8, make_layout((PAIR_M, K_L1)), depth=2)
        l1_b = RingBuffer(MemLoc.L1, FP8, make_layout((BASE_N, K_L1)), depth=2)
        # ScaleA uses a transposed view of native NZ storage (effective ZN).
        # ScaleB stays NZ. GM paired E8M0 axes are flattened explicitly.
        l1_sa = RingBuffer(
            MemLoc.L1, E8M0, make_layout((S_L1, PAIR_M)), depth=2
        )
        l1_sb = RingBuffer(
            MemLoc.L1, E8M0, make_layout((S_L1, BASE_N)), depth=2
        )
        l0a = RingBuffer(MemLoc.L0A, FP8, make_layout((PAIR_M, BASE_K)), depth=2)
        l0b = RingBuffer(MemLoc.L0B, FP8, make_layout((BASE_N, BASE_K)), depth=2)
        # 256x256 f32 == 256KB L0C: depth1 (same trade as the bf16 kernel).
        l0c = RingBuffer(MemLoc.L0C, F32, make_layout((PAIR_M, BASE_N)), depth=1)
        # ---- cross-core handoff (fp32 tile, same as bf16) ----
        cv_ub = RingBuffer(
            MemLoc.UB, F32, make_layout((TILE_VEC_M + 1, BASE_N)), depth=1
        )
        # ---- vec-side channels ----
        out_ch = RingBuffer(MemLoc.UB, BF16, make_layout((TILE_VEC_M, HEAD_DIM)), depth=1)
        cs_ch = RingBuffer(
            MemLoc.UB, F32, make_layout((TILE_VEC_M + 1, ROPE_DIM)), depth=1
        )
        gamma_ch = RingBuffer(
            MemLoc.UB, BF16, make_layout((1, HEAD_DIM + HALF_ROPE_DIM)), depth=1
        )

        # Resident K-RMSNorm weight: loaded once, consumed by every item's
        # K-unit epilogue on this AIV.
        gamma_w = gamma_ch.next_write()
        mem_copy(
            tile_slice(gamma_w, (1, HEAD_DIM), (0, 0)),
            tile_slice(gm_gamma, (1, HEAD_DIM), (0, 0)),
        )

        # Work-item scheduling: granularity is (pair, unit), grid-stride.
        if const_expr(self.rank_chunk_layout):
            total_items = rank_count * m_pairs * n_units
        else:
            total_items = m_pairs * n_units
        for item in range(block_idx, total_items, block_num):
            if const_expr(self.rank_chunk_layout):
                rank_pair = item // n_units
                rank = rank_pair // m_pairs
                pair = rank_pair - rank * m_pairs
                u = item - rank_pair * n_units
                input_base = rank * chunk_local_rows
                output_base = rank * full_local_rows + chunk_start
                # Both data and paired scales have the same rank-local rows.
                # Interval aliases keep each rank's tail separate from the next.
                qx_span = gm_qx[input_base : input_base + m_rows, None]
                sa_span = gm_sa[input_base : input_base + m_rows, None, None]
                q_span = gm_q[output_base : output_base + m_rows, None]
                k_span = gm_k[output_base : output_base + m_rows, None]
                v_span = gm_v[output_base : output_base + m_rows, None]
                if const_expr(self.has_mirror):
                    mk_span = gm_mk[output_base : output_base + m_rows, None]
                    mv_span = gm_mv[output_base : output_base + m_rows, None]
                else:
                    mk_span = gm_mk
                    mv_span = gm_mv
            else:
                pair = item // n_units
                u = item - pair * n_units
                qx_span = gm_qx
                sa_span = gm_sa
                q_span = gm_q
                k_span = gm_k
                v_span = gm_v
                mk_span = gm_mk
                mv_span = gm_mv

            cur_tokens = m_rows - pair * PAIR_M
            if cur_tokens > PAIR_M:
                cur_tokens = PAIR_M

            # Balanced split-M plan for this (possibly tail-clipped) pair
            # tile; a trailing odd tile clips to its real row count.
            q_tile0 = tile_slice(q_span, (PAIR_M, HEAD_DIM), (pair, 0))
            rows_here = cv_valid_extent(q_tile0, axis=0, part_id=sub)
            sub_start = sub * (cur_tokens - rows_here)
            row0 = pair * PAIR_M + sub_start
            # A trailing subblock may own zero rows (e.g. M == 1 splits
            # 1/0); clamp so the scalar position load stays in bounds even
            # though every per-row loop below is empty.
            if row0 >= m_rows:
                row0 = m_rows - 1
            if const_expr(self.rank_chunk_layout):
                meta_row0 = output_base + row0
            else:
                meta_row0 = row0

            # Vec-side preload for this item (this AIV's cos/sin rows).
            cs_w = cs_ch.next_write()
            table_rows = gm_cos_sin.shape[0]
            if const_expr(self.positions_contiguous):
                # Prefill: one run of TILE_VEC_M consecutive table rows.
                pos_base = gm_positions[meta_row0]
                if rows_here > 0:
                    if pos_base + TILE_VEC_M <= table_rows:
                        # Fast path: one bulk (<=128, 64) DMA.
                        mem_copy(
                            tile_slice(cs_w, (TILE_VEC_M, ROPE_DIM), (0, 0)),
                            gm_cos_sin[pos_base : pos_base + TILE_VEC_M, 0:ROPE_DIM],
                        )
                    else:
                        # Table-tail clamp: fall back to exactly the valid
                        # rows -- pos_base + rows_here <= table_rows holds
                        # because every position is range-checked.
                        for r in range(rows_here):
                            mem_copy(
                                tile_slice(cs_w, (1, ROPE_DIM), (r, 0)),
                                tile_slice(
                                    gm_cos_sin, (1, ROPE_DIM), (pos_base + r, 0)
                                ),
                            )
            else:
                # Decode: arbitrary per-token positions (each request's own
                # last position). One 1x64 gather per token.
                for r in range(rows_here):
                    pos = gm_positions[meta_row0 + r]
                    mem_copy(
                        tile_slice(cs_w, (1, ROPE_DIM), (r, 0)),
                        tile_slice(gm_cos_sin, (1, ROPE_DIM), (pos, 0)),
                    )

            # ---- cube: K-loop MXFP8 MMAD accumulation for unit u ----
            # GM -> L1: data via plain ND2NZ, scales via the paired
            # engines; L1 -> L0 binds the per-sub-block e8m0 views so the
            # matmul selects asc_mmad_mx (per-32-group dequant, fp32
            # accumulate -- the same OCP MXFP8 hardware path as
            # aclnnQuantMatmulV5's QuantBatchMatmulV3).
            vec_sync_all()
            # MMAD and FIXPIPE must share the compact live-M L0 layout.
            l0c_acc = l0c.next_write(bare=True).view(
                make_layout((cur_tokens, BASE_N))
            )
            for k_l1 in range(K_L1_TILES):
                a_tile = tile_slice(qx_span, (PAIR_M, K_L1), (pair, k_l1))
                b_tile = tile_slice(gm_w, (BASE_N, K_L1), (u, k_l1))
                sa_tile = tile_slice(
                    sa_span, (PAIR_M, PACKED_S_L1, 2), (pair, k_l1, 0)
                )
                sb_tile = tile_slice(
                    gm_sb, (BASE_N, PACKED_S_L1, 2), (u, k_l1, 0)
                )
                # l2_cache_ctl=1 (normal): W tiles stay resident in L2
                # across m-tiles (same rationale as the bf16 kernel).
                l1_a_slot = l1_a.next_write()[:cur_tokens, :K_L1]
                l1_b_slot = l1_b.next_write()
                mem_copy(
                    l1_a_slot,
                    a_tile,
                    options=copy_options(l2_cache_ctl=1),
                )
                mem_copy(
                    l1_b_slot,
                    b_tile,
                    options=copy_options(l2_cache_ctl=1),
                )
                l1_sa_slot = l1_sa.next_write().permute(1, 0)[:cur_tokens, :S_L1]
                l1_sb_slot = l1_sb.next_write()
                mem_copy(l1_sa_slot, sa_tile.flatten(-2, -1))
                mem_copy(l1_sb_slot, sb_tile.flatten(-2, -1).permute(1, 0))
                for k_l0 in range(K_L0_PER_L1):
                    l0a_slot = l0a.next_write(bare=True).view(
                        make_layout((cur_tokens, BASE_K))
                    )
                    l0b_slot = l0b.next_write()
                    mem_copy(
                        l0a_slot,
                        tile_slice(l1_a_slot, (PAIR_M, BASE_K), (0, k_l0)),
                        options=copy_options(
                            mx_scale=tile_slice(
                                l1_sa_slot, (PAIR_M, S_L0), (0, k_l0)
                            )
                        ),
                    )
                    mem_copy(
                        l0b_slot,
                        tile_slice(l1_b_slot, (BASE_N, BASE_K), (0, k_l0)),
                        options=copy_options(
                            mx_scale=tile_slice(
                                l1_sb_slot, (S_L0, BASE_N), (k_l0, 0)
                            )
                        ),
                    )
                    gk = k_l1 * K_L0_PER_L1 + k_l0
                    matmul(
                        l0c_acc, l0a_slot, l0b_slot, init=(gk == 0), mx_mode=True
                    )

            # ---- handoff: L0C -> per-AIV UB halves ----
            # A partitioned live-M source selects dual-destination FIXPIPE.
            # Preserve the exact compact layout written by MMAD above.
            cv_w = cv_ub.next_write()
            mem_copy(
                tile_slice(cv_w, (TILE_VEC_M, BASE_N), (0, 0)),
                l0c_acc.partition(dim=0),
            )

            # ---- vec epilogue, specialized per unit ----
            cv_r = cv_ub.next_read()
            cs_r = cs_ch.next_read()
            out_w = out_ch.next_write()
            if u < NUM_Q_HEADS:
                self._rope_q(cv_r, out_w, cs_r, rows_here)
                q_half = tile_slice(
                    q_span, (PAIR_M, HEAD_DIM), (pair, u)
                ).partition(dim=0, part_id=sub)
                out_r = out_ch.next_read()
                if rows_here > 0:
                    mem_copy(q_half, out_r[:rows_here, :HEAD_DIM])
            elif u == UNIT_K:
                self._norm_rope_k(
                    cv_r, out_w, cs_r, gamma_w, rows_here, AVG, epsilon
                )
                out_r = out_ch.next_read()
                k_half = tile_slice(
                    k_span, (PAIR_M, HEAD_DIM), (pair, 0)
                ).partition(dim=0, part_id=sub)
                # The tile drain and the slot scatter are two readers of one
                # out_ch slot; both consume the same VF-written epoch.
                if rows_here > 0:
                    mem_copy(k_half, out_r[:rows_here, :HEAD_DIM])
                for r in range(rows_here):
                    slot = gm_slot[meta_row0 + r]
                    if slot != -1:
                        mem_copy(
                            tile_slice(gm_k_cache, (1, HEAD_DIM), (slot, 0)),
                            tile_slice(out_r, (1, HEAD_DIM), (r, 0)),
                        )
            elif u == UNIT_V:
                self._cast_rows(cv_r, out_w, rows_here)
                out_r = out_ch.next_read()
                if const_expr(self.return_v):
                    # Transitional interface: materialize v like the
                    # production scatter's transient contiguous copy.
                    v_half = tile_slice(
                        v_span, (PAIR_M, HEAD_DIM), (pair, 0)
                    ).partition(dim=0, part_id=sub)
                    if rows_here > 0:
                        mem_copy(v_half, out_r[:rows_here, :HEAD_DIM])
                for r in range(rows_here):
                    slot = gm_slot[meta_row0 + r]
                    if slot != -1:
                        mem_copy(
                            tile_slice(gm_v_cache, (1, HEAD_DIM), (slot, 0)),
                            tile_slice(out_r, (1, HEAD_DIM), (r, 0)),
                        )
            else:
                self._cast_rows(cv_r, out_w, rows_here)
                out_r = out_ch.next_read()
                gm_m = mk_span if u == UNIT_MK else mv_span
                if const_expr(self.has_mirror):
                    m_half = tile_slice(
                        gm_m, (PAIR_M, HEAD_DIM), (pair, 0)
                    ).partition(dim=0, part_id=sub)
                    if rows_here > 0:
                        mem_copy(m_half, out_r[:rows_here, :HEAD_DIM])
                else:
                    # Plain-layer stand-ins are (1, 256); the mirror units
                    # never execute at runtime, so per-row runtime coords
                    # (which pass static verification) suffice.
                    for r in range(rows_here):
                        mem_copy(
                            tile_slice(gm_m, (1, HEAD_DIM), (row0 + r, 0)),
                            tile_slice(out_r, (1, HEAD_DIM), (r, 0)),
                        )


def compile_aot_mx(
    qkv_width, num_slots, max_pos, return_v=False, positions_contiguous=True
):
    """AOT-compile ONE dynamic-M MXFP8 artifact (ProviderCallable).

    The token count M is a symbolic ``Dim("M", min=1)`` -- the same
    compiled artifact serves every M >= 1 with no per-shape recompilation.
    All other extents are static deployment constants (same contract as
    the bf16 ``compile_aot``):

        qkv_width: 2560 (kv-mirror source layer) or 2048 (plain QKV layer)
        num_slots: paged KV cache rows (flattenable to (slots, 256))
        max_pos:   cos_sin_cache rows
        return_v:  additionally materialize v (M, 256)
        positions_contiguous: True = prefill artifact; False = decode.

    The returned callable takes the same 16 runtime arguments as
    ``FusedQkvProjNormRopeCacheMx.run`` (qx/sa/w/sb as int8 byte
    carriers; the kernel interprets them as fp8/e8m0).
    """
    n_units = qkv_width // HEAD_DIM
    if n_units * HEAD_DIM != qkv_width or n_units not in (8, 10):
        raise ValueError(
            f"qkv_width must be 2048 or 2560, got {qkv_width}"
        )
    op = FusedQkvProjNormRopeCacheMx(
        return_v=return_v,
        has_mirror=(n_units == 10),
        positions_contiguous=positions_contiguous,
    )
    m = Dim("M", min=1)
    mirror_rows = m if n_units == 10 else 1  # plain stand-ins stay (1, 256)
    i8, bf, f32, i64 = dtypes.int8, BF16, F32, dtypes.int64
    return cannbotdsl.compile(
        op.run,
        TensorSpec((m, HIDDEN), i8),
        TensorSpec((m, SCALE_PAIRS, 2), i8),
        TensorSpec((qkv_width, HIDDEN), i8),
        TensorSpec((qkv_width, SCALE_PAIRS, 2), i8),
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


def compile_aot_mx_rank_chunk(
    qkv_width, num_slots, max_pos, *, positions_contiguous=True
):
    """Compile a direct-write MXFP8 pipeline artifact (19 runtime args).

    qx/sa have Mchunk=P*m rows; metadata and dense outputs have Mfull=P*L.
    L/m/a are runtime scalars with 0 <= a < a+m <= L, not compile constants.
    """
    n_units = qkv_width // HEAD_DIM
    if n_units * HEAD_DIM != qkv_width or n_units not in (8, 10):
        raise ValueError(f"qkv_width must be 2048 or 2560, got {qkv_width}")
    op = FusedQkvProjNormRopeCacheMx(
        return_v=True,
        has_mirror=(n_units == 10),
        positions_contiguous=positions_contiguous,
    )
    m_chunk = Dim("Mchunk", min=1)
    m_full = Dim("Mfull", min=1)
    mirror_rows = m_full if n_units == 10 else 1
    i8, bf, f32, i64 = dtypes.int8, BF16, F32, dtypes.int64
    return cannbotdsl.compile(
        op.run_rank_chunk,
        TensorSpec((m_chunk, HIDDEN), i8),
        TensorSpec((m_chunk, SCALE_PAIRS, 2), i8),
        TensorSpec((qkv_width, HIDDEN), i8),
        TensorSpec((qkv_width, SCALE_PAIRS, 2), i8),
        TensorSpec((1, HEAD_DIM), bf),
        TensorSpec((m_full,), i64),
        TensorSpec((max_pos, ROPE_DIM), f32),
        TensorSpec((m_full,), i64),
        TensorSpec((m_full, NUM_Q_HEADS * HEAD_DIM), bf),
        TensorSpec((m_full, HEAD_DIM), bf),
        TensorSpec((m_full, HEAD_DIM), bf),
        TensorSpec((mirror_rows, HEAD_DIM), bf),
        TensorSpec((mirror_rows, HEAD_DIM), bf),
        TensorSpec((num_slots, HEAD_DIM), bf),
        TensorSpec((num_slots, HEAD_DIM), bf),
        dtypes.float32,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )


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


def _mx_quant_activation(hidden):
    """Online MXFP8 activation quant (the network's quantize_activation).

    Returns (qx, act_scale) exactly as ``NPUMXFP8LinearMethod.quantize_activation``
    would: qx (M, K) float8_e4m3fn and act_scale (M, K/64, 2) uint8 e8m0,
    per-32-group scale = 2^ceil(log2(amax / 448)).
    """
    if hidden.dtype not in (torch.float16, torch.bfloat16):
        hidden = hidden.to(torch.bfloat16)
    x_2d = hidden.reshape(-1, hidden.shape[-1])
    return torch.ops.npu.npu_dynamic_mx_quant(x_2d, dst_type=torch.float8_e4m3fn)


class FusedQkvProjNormRopeCacheMx:
    """Host-facing MXFP8 wrapper: quant + MX matmul + post-fused epilogue in 1 kernel.

    ``return_v=False`` (default) matches the profiled NPU dataflow (v only
    scattered into the cache); ``return_v=True`` additionally materializes
    v (M, 256).
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

    @host
    def run(
        self,
        gm_qx,
        gm_sa,
        gm_w,
        gm_sb,
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
        m_rows = gm_qx.shape[0]
        m_pairs = (m_rows + PAIR_M - 1) // PAIR_M
        block_dim = m_pairs * (gm_w.shape[0] // BASE_N)
        grid_cap = _aic_block_limit()
        if block_dim > grid_cap:
            block_dim = grid_cap
        op = FusedQkvProjNormRopeCacheMxKernel(
            self._return_v, self._has_mirror, self._positions_contiguous
        )
        op[block_dim](
            gm_qx,
            gm_sa,
            gm_w,
            gm_sb,
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
        )

    @host
    def run_rank_chunk(
        self,
        gm_qx,
        gm_sa,
        gm_w,
        gm_sb,
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
        full_local_rows: int,
        chunk_local_rows: int,
        chunk_start: int,
    ):
        rank_count = gm_qx.shape[0] // chunk_local_rows
        pairs_per_rank = (chunk_local_rows + PAIR_M - 1) // PAIR_M
        block_dim = rank_count * pairs_per_rank * (gm_w.shape[0] // BASE_N)
        grid_cap = _aic_block_limit()
        if block_dim > grid_cap:
            block_dim = grid_cap
        op = FusedQkvProjNormRopeCacheMxKernel(
            self._return_v,
            self._has_mirror,
            self._positions_contiguous,
            rank_chunk_layout=True,
        )
        op[block_dim].rank_chunk(
            gm_qx,
            gm_sa,
            gm_w,
            gm_sb,
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
            full_local_rows,
            chunk_local_rows,
            chunk_start,
        )
