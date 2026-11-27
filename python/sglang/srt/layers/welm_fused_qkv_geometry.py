"""Rank-local BF16 WeLM QKV geometry, independent of device libraries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedQkvGeometry:
    num_q_heads: int = 6
    num_kv_heads: int = 1
    head_dim: int = 256
    has_mirror: bool = False

    def __post_init__(self):
        if (
            (self.num_q_heads, self.num_kv_heads) not in ((6, 1), (12, 1), (24, 2))
            or self.head_dim != 256
        ):
            raise ValueError(
                "WeLM fused BF16 QKV supports Q6/KV1, Q12/KV1 or Q24/KV2, "
                "head_dim=256"
            )

    @property
    def q_width(self):
        return self.num_q_heads * self.head_dim

    @property
    def kv_width(self):
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_width(self):
        return self.q_width + (2 + 2 * self.has_mirror) * self.kv_width

    @property
    def units(self):
        q, kv = self.num_q_heads, self.num_kv_heads
        return dict(k=q, v=q + kv, mirror_k=q + 2 * kv, mirror_v=q + 3 * kv)

    @classmethod
    def from_width(cls, width):
        layouts = {
            2048: (6, 1, False),
            2560: (6, 1, True),
            3584: (12, 1, False),
            4096: (12, 1, True),
            7168: (24, 2, False),
            8192: (24, 2, True),
        }
        if width not in layouts:
            raise ValueError(f"Unsupported WeLM BF16 QKV projection width: {width}")
        q_heads, kv_heads, mirror = layouts[width]
        return cls(num_q_heads=q_heads, num_kv_heads=kv_heads, has_mirror=mirror)
