# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rope overrides must reach same-checkpoint draft configs.

The drafter's rotary cos/sin cache is position-indexed; a draft config that
misses the target's yarn rope override builds a native-sized cache and
device-asserts (OOB gather inside the captured draft CUDA graph) once the
yarn-extended target runs past the native max position. Dict hf_overrides
are target-only by design, so the draft override composition selects the
rope-relevant keys back out (#58080) while leaving target-only keys
(`max_position_embeddings`) behind.
"""

from transformers import Qwen3Config

from vllm.config.speculative import SpeculativeConfig

YARN = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 8192}


def _draft_config() -> Qwen3Config:
    draft = Qwen3Config()
    draft.architectures = ["Qwen3ForCausalLM"]
    return draft


def test_target_rope_reaches_draft():
    composed = SpeculativeConfig.compose_draft_hf_overrides(
        {"rope_scaling": dict(YARN)}
    )
    draft = composed(_draft_config())
    assert draft.rope_scaling == YARN


def test_draft_without_target_rope_is_untouched():
    draft = _draft_config()
    before = draft.rope_scaling
    composed = SpeculativeConfig.compose_draft_hf_overrides(
        {"max_position_embeddings": 12345}
    )
    composed(draft)
    assert draft.rope_scaling == before
    # max_position_embeddings stays target-only so the draft max-model-len
    # clamp keeps holding (#63909).
    assert draft.max_position_embeddings != 12345


def test_rope_parameters_family_propagates():
    # Qwen4Exp-style configs carry rope_parameters instead of rope_scaling
    composed = SpeculativeConfig.compose_draft_hf_overrides(
        {"rope_parameters": {"rope_type": "yarn", "factor": 2.0}}
    )
    draft = composed(_draft_config())
    assert draft.rope_parameters == {"rope_type": "yarn", "factor": 2.0}


def test_text_config_nesting_respected():
    # multimodal-style configs wrap fields in text_config
    draft = _draft_config()
    draft.text_config = Qwen3Config()
    composed = SpeculativeConfig.compose_draft_hf_overrides(
        {"text_config": {"rope_scaling": dict(YARN)}}
    )
    composed(draft)
    # nested configs merge into the existing rope_scaling dict rather than
    # replacing it (the draft's default rope_theta survives)
    assert draft.text_config.rope_scaling["rope_type"] == "yarn"
    assert draft.text_config.rope_scaling["factor"] == 2.0
    assert draft.rope_scaling["rope_type"] != "yarn"  # outer config untouched
