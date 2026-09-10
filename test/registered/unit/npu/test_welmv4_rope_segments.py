from sglang.srt.layers.welmv4_npu_op import (
    build_welmv4_rope_segment_tile_starts,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


def test_builds_segment_tiles_without_crossing_request_boundaries():
    tile_starts = build_welmv4_rope_segment_tile_starts(
        [65, 576],
        batch_size=2,
        num_position_tokens=641,
        ordinary_prefill=True,
    )

    assert tile_starts == [
        0,
        64,
        65,
        129,
        193,
        257,
        321,
        385,
        449,
        513,
        577,
        641,
    ]


def test_aligned_segment_boundary_is_shared_with_next_tile():
    tile_starts = build_welmv4_rope_segment_tile_starts(
        [320, 384],
        batch_size=2,
        num_position_tokens=704,
        ordinary_prefill=True,
    )

    assert tile_starts == list(range(0, 705, 64))


def test_builds_short_prefill_segment_metadata():
    for lengths in ([0, 1], [1, 1], [31, 34], [320, 320]):
        expected = []
        offset = 0
        for length in lengths:
            expected.extend(range(offset, offset + length, 64))
            offset += length
        expected.append(offset)
        assert build_welmv4_rope_segment_tile_starts(
            lengths,
            batch_size=len(lengths),
            num_position_tokens=sum(lengths),
            ordinary_prefill=True,
        ) == expected


def test_rejects_unsafe_segment_metadata():
    common = dict(
        segment_lengths=[321, 320],
        batch_size=2,
        num_position_tokens=641,
        ordinary_prefill=True,
    )

    assert build_welmv4_rope_segment_tile_starts(**common) is not None
    assert (
        build_welmv4_rope_segment_tile_starts(
            **(common | {"ordinary_prefill": False})
        )
        is None
    )
    assert (
        build_welmv4_rope_segment_tile_starts(
            **(common | {"batch_size": 1, "segment_lengths": [641]})
        )
        is None
    )
    assert (
        build_welmv4_rope_segment_tile_starts(
            **(common | {"num_position_tokens": 704})
        )
        is None
    )
    assert (
        build_welmv4_rope_segment_tile_starts(
            **(common | {"segment_lengths": [642, -1]})
        )
        is None
    )


if __name__ == "__main__":
    test_builds_segment_tiles_without_crossing_request_boundaries()
    test_aligned_segment_boundary_is_shared_with_next_tile()
    test_builds_short_prefill_segment_metadata()
    test_rejects_unsafe_segment_metadata()
