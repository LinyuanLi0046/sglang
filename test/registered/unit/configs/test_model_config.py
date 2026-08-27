"""Unit tests for hybrid attention model configuration."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    get_welmv4_layerwise_sliding_windows,
    is_embedding_gemma,
    is_hybrid_swa_model,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridLayerIds(CustomTestCase):
    def test_welm_draft_reuses_num_target_hidden_layers(self):
        model_config = object.__new__(ModelConfig)
        model_config.is_draft_model = True
        model_config.hf_config = SimpleNamespace(
            architectures=["WeLMV4MoeForCausalLM"],
            num_hidden_layers=48,
            num_nextn_predict_layers=1,
        )

        model_config._config_draft_model()

        self.assertEqual(
            model_config.hf_config.architectures,
            ["WeLMV4MoeForCausalLMNextN"],
        )
        self.assertEqual(model_config.hf_config.num_target_hidden_layers, 48)
        self.assertEqual(model_config.hf_config.num_hidden_layers, 1)

    def test_welm_draft_preserves_existing_num_target_hidden_layers(self):
        model_config = object.__new__(ModelConfig)
        model_config.is_draft_model = True
        model_config.hf_config = SimpleNamespace(
            architectures=["WeLMV4MoeForCausalLM"],
            num_hidden_layers=49,
            num_target_hidden_layers=48,
            num_nextn_predict_layers=1,
        )

        model_config._config_draft_model()

        self.assertEqual(model_config.hf_config.num_target_hidden_layers, 48)
        self.assertEqual(model_config.hf_config.num_hidden_layers, 1)

    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )

    def test_welm_layerwise_windows_partition_full_and_swa(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            # The fifth value represents a future MTP/config-space layer and
            # must not enter the target model's runtime KV-pool partition.
            sliding_window_size_layerwise=[1024, 512, 1024, 512, 256],
            sliding_window=1024,
            max_position_embeddings=2048,
        )

        self.assertTrue(is_hybrid_swa_model(["WeLMV4MoeForCausalLM"], config))
        self.assertEqual(
            get_hybrid_layer_ids(
                ["WeLMV4MoeForCausalLM"], config, context_len=1024
            ),
            ([1, 3], [0, 2]),
        )
        self.assertEqual(
            get_welmv4_layerwise_sliding_windows(config, context_len=1024),
            [-1, 511, -1, 511],
        )

    def test_welm_full_window_boundary_and_invalid_values(self):
        config = SimpleNamespace(
            num_hidden_layers=5,
            sliding_window_size_layerwise=[None, 0, -1, 1024, 512],
            sliding_window=1024,
            max_position_embeddings=2048,
        )

        self.assertEqual(
            get_hybrid_layer_ids(
                ["WeLMV4MoeForCausalLM"], config, context_len=1024
            ),
            ([4], [0, 1, 2, 3]),
        )
        self.assertEqual(
            get_welmv4_layerwise_sliding_windows(config, context_len=1024),
            [-1, -1, -1, -1, 511],
        )

        config.num_hidden_layers = 1
        config.sliding_window_size_layerwise = [1]
        with self.assertRaisesRegex(ValueError, "zero history"):
            get_welmv4_layerwise_sliding_windows(config, context_len=1024)

    def test_welm_runtime_context_participates_in_full_limit(self):
        config = SimpleNamespace(
            num_hidden_layers=2,
            sliding_window_size_layerwise=[768, 512],
            sliding_window=4096,
            max_position_embeddings=8192,
        )

        self.assertEqual(
            get_hybrid_layer_ids(
                ["WeLMV4MoeForCausalLM"], config, context_len=768
            ),
            ([1], [0]),
        )
        self.assertEqual(
            get_hybrid_layer_ids(
                ["WeLMV4MoeForCausalLM"], config, context_len=512
            ),
            ([], [0, 1]),
        )

    def test_welm_nonempty_layerwise_windows_must_cover_active_layers(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            sliding_window_size_layerwise=[1024, 512, 1024],
            sliding_window=1024,
            max_position_embeddings=1024,
        )

        with self.assertRaisesRegex(ValueError, "does not cover all active layers"):
            get_hybrid_layer_ids(
                ["WeLMV4MoeForCausalLM"], config, context_len=1024
            )

    def test_welm_nextn_uses_logical_layer48_config_with_physical_layer0(self):
        config = SimpleNamespace(
            num_hidden_layers=1,
            num_target_hidden_layers=48,
            sliding_window_size_layerwise=[1024] * 48 + [512],
            sliding_window=1024,
            max_position_embeddings=4096,
        )

        # The returned IDs belong to the one-layer draft KV pool, but the
        # window decision comes from logical/checkpoint layer 48.
        self.assertEqual(
            get_hybrid_layer_ids(
                ["WeLMV4MoeForCausalLMNextN"], config, context_len=1024
            ),
            ([0], []),
        )


class TestEmbeddingGemmaConfig(CustomTestCase):
    def test_detects_bidirectional_gemma3_text_config(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=True
        )
        self.assertTrue(is_embedding_gemma(config))

    def test_does_not_misclassify_causal_gemma3(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=False
        )
        self.assertFalse(is_embedding_gemma(config))


if __name__ == "__main__":
    unittest.main()
