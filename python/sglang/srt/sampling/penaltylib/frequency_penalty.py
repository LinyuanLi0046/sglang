import json
import os

import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer


def _read_int_env(name: str, minimum: int):
    value = os.environ.get(name, "")
    if not value:
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _read_exclude_tokens():
    value = os.environ.get("FREQUENCY_PENALTY_EXCLUDE_TOKENS", "")
    if not value:
        return []
    try:
        tokens = json.loads(value)
    except ValueError as exc:
        raise ValueError(
            "FREQUENCY_PENALTY_EXCLUDE_TOKENS must be a JSON list of token IDs"
        ) from exc
    if not isinstance(tokens, list) or any(
        type(token) is not int or token < 0 for token in tokens
    ):
        raise ValueError(
            "FREQUENCY_PENALTY_EXCLUDE_TOKENS must be a JSON list of nonnegative integers"
        )
    return tokens


class BatchedFrequencyPenalizer(_BatchedPenalizer):
    """
    Frequency penalizer penalizes tokens based on their frequency in the output.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.frequency_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.frequency_penalty_factor_start = _read_int_env(
            "FREQUENCY_PENALTY_FACTOR_START", minimum=0
        )
        self.frequency_penalty_factor_offset = _read_int_env(
            "FREQUENCY_PENALTY_FACTOR_OFFSET", minimum=1
        )
        excluded = _read_exclude_tokens()
        self.token_penalty_mask = None
        if excluded:
            # Allocate on the batch's device, including NPU. Do not initialize
            # CUDA contexts during module import or on other workers' devices.
            self.token_penalty_mask = torch.ones(
                self.orchestrator.vocab_size,
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
            excluded_ids = torch.tensor(
                [token for token in excluded if token < self.orchestrator.vocab_size],
                dtype=torch.long,
                device=self.orchestrator.device,
            )
            self.token_penalty_mask.index_fill_(0, excluded_ids, 0)

        # Each request has its own age across continuous-batch merges/filters.
        self.generation_lengths = torch.zeros(
            (len(self.orchestrator.reqs()), 1),
            dtype=torch.long,
            device=self.orchestrator.device,
        )
        self.cumulated_frequency_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        self.frequency_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.frequency_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        penalties = self.frequency_penalties
        if self.token_penalty_mask is not None:
            penalties = penalties * self.token_penalty_mask[output_ids].unsqueeze(1)

        if (
            self.frequency_penalty_factor_start is not None
            and self.frequency_penalty_factor_offset is not None
        ):
            self.generation_lengths.add_(1)
            factor = (
                self.generation_lengths - self.frequency_penalty_factor_start
            ).clamp_min(0).div(
                self.frequency_penalty_factor_offset, rounding_mode="floor"
            ) + 1
            penalties = penalties * factor

        self.cumulated_frequency_penalties.scatter_add_(
            dim=1,
            index=output_ids.unsqueeze(1),
            src=penalties,
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits.sub_(self.cumulated_frequency_penalties)

    def _filter(self, keep_indices: torch.Tensor):
        self.generation_lengths = self.generation_lengths[keep_indices]
        self.frequency_penalties = self.frequency_penalties[keep_indices]
        self.cumulated_frequency_penalties = self.cumulated_frequency_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedFrequencyPenalizer"):
        self.generation_lengths = torch.cat(
            [self.generation_lengths, their.generation_lengths], dim=0
        )
        self.frequency_penalties = torch.cat(
            [self.frequency_penalties, their.frequency_penalties], dim=0
        )
        self.cumulated_frequency_penalties = torch.cat(
            [self.cumulated_frequency_penalties, their.cumulated_frequency_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in (
            "frequency_penalties",
            "cumulated_frequency_penalties",
            "generation_lengths",
            "token_penalty_mask",
        ):
            if hasattr(self, name):
                delattr(self, name)
