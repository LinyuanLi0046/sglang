# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Static-M WeLMv4 BF16 Decode/Verify QKV, K Norm, RoPE and KV scatter.

Migrated from the standalone CANNBotDSL 0.8.0 / 950DT candidate
``welmv4_fused_qkv_tp1.py``. Attention TP=1/2/4 uses already sharded weights;
this kernel performs no collective. Plain layers project Q/K/V, mirror source
layers also produce raw mirror K/V, and Q-only programs leave K/V and caches
untouched. Framework admission and cache binding belong to the caller.

The 16-row tile and NK L0B layout follow the standalone candidate. Any
positive M is specialized, including B*D Verify inputs and a partial last
tile. The live L0 layout is shared by MMAD and FIXPIPE, as in the dynamic
DSL 0.8 kernel. These extensions require device validation.
The FP32 accumulation, BF16 rounding boundaries and register epilogues follow
the standalone candidate. Compile and load programs before graph capture;
launches mutate caller-owned output and token-major cache tensors in place.
"""

import cannbotdsl
from cannbotdsl import MemLoc, Tensor, TensorSpec, dtypes, make_layout
from cannbotdsl.lang import jit, kernel, vf
from cannbotdsl.lang import range as dsl_range
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
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
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import tile_slice
from cannbotdsl.types.dtypes import bfloat16 as BF16
from cannbotdsl.types.dtypes import float32 as F32

HIDDEN = 2048
HEAD_DIM = 256
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE_DIM = ROPE_DIM // 2
GLOBAL_Q_HEADS = 24
GLOBAL_KV_HEADS = 2
TP_SIZE_CHOICES = (1, 2, 4)
MAX_TILE_M = 16
BASE_N = HEAD_DIM
BASE_K = 64
K_L1 = 256
K_L1_TILES = HIDDEN // K_L1
K_L0_PER_L1 = K_L1 // BASE_K
VL = 64
AVG = 1.0 / HEAD_DIM
_F32_MAX = float.fromhex("0x1.fffffep+127")


@kernel
class FusedQkvProjNormRopeCacheTP1Kernel:
    # Keep the trace-visible name for existing TP1 profiling scripts.
    def __init__(
        self,
        fixed_m: int,
        has_mirror: bool,
        q_heads: int,
        kv_heads: int,
        q_only: bool,
    ):
        self.has_mirror = has_mirror
        self.fixed_m = fixed_m
        self.tile_m = min(fixed_m, MAX_TILE_M)
        self.m_tiles = (fixed_m + self.tile_m - 1) // self.tile_m
        self.q_heads = q_heads
        self.unit_v = q_heads + kv_heads
        self.unit_mk = q_heads + 2 * kv_heads
        self.unit_mv = q_heads + 3 * kv_heads
        self.q_only = q_only

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
    ):
        pair_m = self.tile_m
        m_tiles = self.m_tiles
        n_units = gm_weight.shape[0] // BASE_N
        block_idx = get_block_idx()
        block_num = get_block_num()
        sub = get_subblock_id()

        # Format defaults are NZ for L1/L0 and ND for UB. W remains [N,K]
        # in GM, L1 and L0B, following the validated BF16 MatMul layout.
        l1_a = RingBuffer(MemLoc.L1, BF16, make_layout((pair_m, K_L1)), depth=2)
        l1_b = RingBuffer(MemLoc.L1, BF16, make_layout((BASE_N, K_L1)), depth=2)
        l0a = RingBuffer(MemLoc.L0A, BF16, make_layout((pair_m, BASE_K)), depth=2)
        l0b_layout = make_layout((BASE_N, BASE_K))
        l0b = RingBuffer(MemLoc.L0B, BF16, l0b_layout, depth=2)
        l0c = RingBuffer(MemLoc.L0C, F32, make_layout((pair_m, BASE_N)), depth=1)
        # AIV0 handles all rows. The extra row and gamma slack preserve
        # the TP4 epilogue's masked +32 vector loads.
        cv_ub = RingBuffer(MemLoc.UB, F32, make_layout((pair_m + 1, BASE_N)), depth=1)
        out_ch = RingBuffer(MemLoc.UB, BF16, make_layout((pair_m, HEAD_DIM)), depth=1)
        cs_ch = RingBuffer(MemLoc.UB, F32, make_layout((pair_m + 1, ROPE_DIM)), depth=1)
        gamma_ch = RingBuffer(
            MemLoc.UB, BF16, make_layout((1, HEAD_DIM + HALF_ROPE_DIM)), depth=1
        )

        for item in range(block_idx, m_tiles * n_units, block_num):
            # Keep full tiles on the original static extent. Verify can end
            # in a partial tile; every consumer uses that same live layout.
            if const_expr(self.fixed_m <= MAX_TILE_M):
                pair = 0
                u = item
                row0 = 0
            else:
                pair = item // n_units
                u = item - pair * n_units
                row0 = pair * pair_m
            cur_tokens = self.tile_m
            if const_expr(self.fixed_m % self.tile_m != 0):
                cur_tokens = self.fixed_m - row0
                if cur_tokens > self.tile_m:
                    cur_tokens = self.tile_m

            l0c_acc = l0c.next_write(bare=True).view(
                make_layout((cur_tokens, BASE_N))
            )
            for k_l1 in range(K_L1_TILES):
                a_tile = tile_slice(gm_hidden, (pair_m, K_L1), (pair, k_l1))
                b_tile = tile_slice(gm_weight, (BASE_N, K_L1), (u, k_l1))
                l1_a_write = l1_a.next_write()[:cur_tokens, :K_L1]
                mem_copy(
                    l1_a_write,
                    a_tile,
                    options=copy_options(l2_cache_ctl=1),
                )
                mem_copy(
                    l1_b.next_write(), b_tile, options=copy_options(l2_cache_ctl=1)
                )
                l1_a_slot = l1_a_write
                l1_b_slot = l1_b.next_read()
                for k_l0 in range(K_L0_PER_L1):
                    l0a_write = l0a.next_write(bare=True).view(
                        make_layout((cur_tokens, BASE_K))
                    )
                    mem_copy(
                        l0a_write,
                        tile_slice(l1_a_slot, (pair_m, BASE_K), (0, k_l0)),
                    )
                    l0b_source = tile_slice(l1_b_slot, (BASE_N, BASE_K), (0, k_l0))
                    mem_copy(l0b.next_write(), l0b_source)
                    l0a_slot = l0a_write
                    l0b_slot = l0b.next_read()
                    gk = k_l1 * K_L0_PER_L1 + k_l0
                    matmul(l0c_acc, l0a_slot, l0b_slot, init=(gk == 0))

            # 0.8 names AIV0 explicitly. No old split_axis/actual keywords
            # or split-M padding/partition semantics are invoked.
            cv_w = cv_ub.next_write()
            mem_copy(
                cv_w[:cur_tokens, None],
                l0c_acc,
                options=copy_options(dst_subblock=0),
            )

            if sub == 0:
                gamma_w = gamma_ch.next_write()
                mem_copy(
                    tile_slice(gamma_w, (1, HEAD_DIM), (0, 0)),
                    tile_slice(gm_gamma, (1, HEAD_DIM), (0, 0)),
                )
                cs_w = cs_ch.next_write()
                for r in range(cur_tokens):
                    pos = gm_positions[row0 + r]
                    mem_copy(
                        tile_slice(cs_w, (1, ROPE_DIM), (r, 0)),
                        tile_slice(gm_cos_sin, (1, ROPE_DIM), (pos, 0)),
                    )

                cv_r = cv_ub.next_read()
                cs_r = cs_ch.next_read()
                out_w = out_ch.next_write()
                if u < self.q_heads:
                    self._rope_q(cv_r, out_w, cs_r, cur_tokens)
                    dst = tile_slice(gm_q, (pair_m, HEAD_DIM), (pair, u))
                    mem_copy(dst, out_ch.next_read()[:cur_tokens, None])
                else:
                    if const_expr(not self.q_only):
                        if u < self.unit_v:
                            h = u - self.q_heads
                            self._norm_rope_k(
                                cv_r, out_w, cs_r, gamma_w, cur_tokens, AVG, epsilon
                            )
                            out_r = out_ch.next_read()[:cur_tokens, None]
                            dst = tile_slice(gm_k, (pair_m, HEAD_DIM), (pair, h))
                            mem_copy(dst, out_r)
                            for r in range(cur_tokens):
                                slot = gm_slot[row0 + r]
                                # SGLang's native pool reserves slot 0 for
                                # graph/DP padding. Never race dummy writes.
                                if slot > 0:
                                    mem_copy(
                                        tile_slice(
                                            gm_k_cache, (1, HEAD_DIM), (slot, h)
                                        ),
                                        tile_slice(out_r, (1, HEAD_DIM), (r, 0)),
                                    )
                        elif u < self.unit_mk:
                            h = u - self.unit_v
                            self._cast_rows(cv_r, out_w, cur_tokens)
                            out_r = out_ch.next_read()[:cur_tokens, None]
                            dst = tile_slice(gm_v, (pair_m, HEAD_DIM), (pair, h))
                            mem_copy(dst, out_r)
                            for r in range(cur_tokens):
                                slot = gm_slot[row0 + r]
                                if slot > 0:
                                    mem_copy(
                                        tile_slice(
                                            gm_v_cache, (1, HEAD_DIM), (slot, h)
                                        ),
                                        tile_slice(out_r, (1, HEAD_DIM), (r, 0)),
                                    )
                        else:
                            if const_expr(self.has_mirror):
                                # Mirror K/V is raw projection output, not Norm/RoPE.
                                self._cast_rows(cv_r, out_w, cur_tokens)
                                out_r = out_ch.next_read()[:cur_tokens, None]
                                if u < self.unit_mv:
                                    mem_copy(
                                        tile_slice(
                                            gm_mk,
                                            (pair_m, HEAD_DIM),
                                            (pair, u - self.unit_mk),
                                        ),
                                        out_r,
                                    )
                                else:
                                    mem_copy(
                                        tile_slice(
                                            gm_mv,
                                            (pair_m, HEAD_DIM),
                                            (pair, u - self.unit_mv),
                                        ),
                                        out_r,
                                    )


class _FusedQkv:
    def __init__(
        self,
        fixed_m: int,
        has_mirror: bool,
        max_blocks: int,
        q_heads: int,
        kv_heads: int,
        q_only: bool,
    ):
        self.has_mirror = has_mirror
        self.max_blocks = max_blocks
        self.fixed_m = fixed_m
        self.m_tiles = (fixed_m + MAX_TILE_M - 1) // MAX_TILE_M
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.q_only = q_only

    @host
    def run(
        self,
        hidden,
        weight,
        gamma,
        positions,
        cos_sin,
        slots,
        q,
        k,
        v,
        mirror_k,
        mirror_v,
        k_cache,
        v_cache,
        eps: float,
    ):
        blocks = self.m_tiles * (weight.shape[0] // BASE_N)
        if blocks > self.max_blocks:
            blocks = self.max_blocks
        op = FusedQkvProjNormRopeCacheTP1Kernel(
            self.fixed_m, self.has_mirror, self.q_heads, self.kv_heads, self.q_only
        )
        op[blocks](
            hidden,
            weight,
            gamma,
            positions,
            cos_sin,
            slots,
            q,
            k,
            v,
            mirror_k,
            mirror_v,
            k_cache,
            v_cache,
            eps,
        )


def compile_qkv(
    num_slots: int,
    max_pos: int,
    *,
    tp_size: int,
    has_mirror: bool,
    static_m: int,
    max_blocks: int,
    q_only: bool = False,
):
    """Compile one static-M, NK-layout rank-local BF16 Decode/Verify program.

    TP1/2/4 use (Q heads, KV heads) = (24,2)/(12,1)/(6,1). The caller has
    already placed/replicated KV shards (TP4), and supplies all physical input
    rows, including graph padding. M is any positive compile-time integer.
    Larger M uses 16-row tiles and a clipped tail in a single launch; per-block
    buffers do not grow beyond the M=16 specialization. Compilation
    and program loading must finish before formal graph capture. The caller
    supplies the verified launch core budget as max_blocks; this module does
    not query the device or substitute a default budget.

    All tensors are contiguous and on the same device. The returned callable
    takes these 14 arguments in order and mutates outputs/caches in place:
      hidden [M,2048], weight [N,2048], gamma [1,256] all BF16;
      positions [M] int64, cos_sin [max_pos,64] FP32 (cos||sin);
      slots [M] int64: non-positive slots skip writes, otherwise slot < num_slots;
      q [M,QW], k/v [M,KVW] except q_only (untouched [1,KVW]);
      mirror_k/v [M,KVW] for mirror source, otherwise untouched [1,KVW];
      k_cache/v_cache [num_slots,KVW] BF16 token-major; eps Python float.
      N = QW + (0/2/4) * KVW for q_only/plain/mirror source respectively.

    All output tensors are BF16. Positions, including padded rows, must stay
    in table range. Writable slots must be unique, and output/cache tensors
    must not alias. The caller must bind the real physical cache, translate
    Full/SWA slots. The native pool's dummy slot 0 and -1 both skip writes.
    Q-only programs do not postprocess or store external K/V.
    """
    if type(num_slots) is not int or num_slots < 1:
        raise ValueError("num_slots must be a positive integer")
    if type(max_pos) is not int or max_pos < 1:
        raise ValueError("max_pos must be a positive integer")
    if type(tp_size) is not int or tp_size not in TP_SIZE_CHOICES:
        raise ValueError(f"tp_size must be one of {TP_SIZE_CHOICES}, got {tp_size!r}")
    if q_only and has_mirror:
        raise ValueError("q_only and has_mirror are mutually exclusive")
    if type(static_m) is not int or static_m < 1:
        raise ValueError(f"static_m must be a positive integer, got {static_m}")
    if type(max_blocks) is not int or max_blocks < 1:
        raise ValueError("max_blocks must be a positive integer")
    m = static_m
    q_heads = GLOBAL_Q_HEADS // tp_size
    kv_heads = max(1, GLOBAL_KV_HEADS // tp_size)
    q_width, kv_width = q_heads * HEAD_DIM, kv_heads * HEAD_DIM
    op = _FusedQkv(
        m, bool(has_mirror), max_blocks, q_heads, kv_heads, bool(q_only)
    )
    n = q_width + (0 if q_only else (4 if has_mirror else 2)) * kv_width
    kv_rows = 1 if q_only else m
    mirror_rows = m if has_mirror else 1
    specs = (
        TensorSpec((m, HIDDEN), BF16),
        TensorSpec((n, HIDDEN), BF16),
        TensorSpec((1, HEAD_DIM), BF16),
        TensorSpec((m,), dtypes.int64),
        TensorSpec((max_pos, ROPE_DIM), F32),
        TensorSpec((m,), dtypes.int64),
        TensorSpec((m, q_width), BF16),
        TensorSpec((kv_rows, kv_width), BF16),
        TensorSpec((kv_rows, kv_width), BF16),
        TensorSpec((mirror_rows, kv_width), BF16),
        TensorSpec((mirror_rows, kv_width), BF16),
        TensorSpec((num_slots, kv_width), BF16),
        TensorSpec((num_slots, kv_width), BF16),
        F32,
    )
    return cannbotdsl.compile(op.run, *specs)
