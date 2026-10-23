# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rope overrides must reach same-checkpoint draft configs.

The drafter's rotary cos/sin cache is position-indexed; a draft config that
misses the target's yarn rope override builds a native-sized cache and
device-asserts (OOB gather inside the captured draft CUDA graph) once the
yarn-extended target runs past the native max position.
"""

from transformers import Qwen3Config

from vllm.config.speculative import _propagate_rope_to_draft

YARN = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 8192}


def test_target_rope_reaches_draft():
    target, draft = Qwen3Config(), Qwen3Config()
    target.rope_scaling = dict(YARN)
    _propagate_rope_to_draft(target, draft)
    assert draft.rope_scaling == YARN


def test_draft_without_target_rope_is_untouched():
    target, draft = Qwen3Config(), Qwen3Config()
    before = dict(draft.rope_scaling or {})
    _propagate_rope_to_draft(target, draft)
    assert draft.rope_scaling == before


def test_rope_parameters_family_propagates():
    # Qwen4Exp-style configs carry rope_parameters instead of rope_scaling
    target, draft = Qwen3Config(), Qwen3Config()
    target.rope_parameters = {"rope_type": "yarn", "factor": 2.0}
    _propagate_rope_to_draft(target, draft)
    assert draft.rope_parameters == {"rope_type": "yarn", "factor": 2.0}


def test_text_config_nesting_respected():
    # multimodal-style configs wrap fields in text_config
    target, draft = Qwen3Config(), Qwen3Config()
    target.text_config = Qwen3Config()
    draft.text_config = Qwen3Config()
    target.text_config.rope_scaling = dict(YARN)
    _propagate_rope_to_draft(target, draft)
    assert draft.text_config.rope_scaling == YARN
    assert draft.rope_scaling != YARN  # outer config untouched
