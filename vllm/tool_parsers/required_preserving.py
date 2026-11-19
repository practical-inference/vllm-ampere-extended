# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Required-preserving expansion for strict tool grammars (fork delta).

Deployed context (2026-11-02 adversarial review): `any_order=True` on the
strict GLM-4.7 path fixed the declared-order repetition basin but SILENTLY
DROPPED the strict contract — xgrammar 0.2.8's any_order does not check key
presence or uniqueness (missing required keys and duplicate keys are
accepted; confirmed empirically, JSON style + the deployed builtin path).

This module re-tightens: it rewrites the built structural tag, expanding
every `any_order: true` json_schema node into an **or-format over key
permutations** — one fully-validated branch per ordering (declared order
within the branch, required keys present, no duplicates, value schemas
pinned). The union accepts exactly: every required key present, no key
twice, only schema keys, each value matching its schema — in any order.

The node's `max_whitespace_cnt` and `excludes` are threaded into EVERY
branch leaf — the C++ core's or-format reads only `elements`, so constraint
material on the or node itself is dropped (2026-11-03 review finding F1,
fixed 21/21 in the harness probe).

k! branch growth is capped (REQUIRED_PRESERVING_MAX_KEYS); above the cap the
node keeps the loose semantics — carried onto the format node (`any_order:
true`, the warning fires) so the fallback is delivered, not silent.

Upstream path: the same expansion is proposed to xgrammar upstream as
`any_order="required_preserving"` (issue-first, 2026-11-02; rebuilt
2026-11-03); when a release carries it, this module becomes a shim over the
native mode.
"""

from __future__ import annotations

import itertools
import json
import warnings
from typing import Any

REQUIRED_PRESERVING_MAX_KEYS = 6
_MARKER_KEY = "x-required-preserving-marker"
_FALLBACK_KEY = "x-any-order-loose-fallback"


def _reordered(schema: dict[str, Any], order: list[str]) -> dict[str, Any]:
    out = dict(schema)
    props = schema.get("properties", {})
    out["properties"] = {k: props[k] for k in order if k in props}
    return out


def _object_keys(schema: dict[str, Any]) -> list[str]:
    props = schema.get("properties", {})
    return list(props.keys()) if isinstance(props, dict) else []


def _node_to_format(
    leaf_schema: dict[str, Any],
    style: str,
    any_order: bool,
    max_whitespace_cnt: Any,
    excludes: list[str],
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": leaf_schema,
        "style": style,
        "any_order": any_order,
        "max_whitespace_cnt": max_whitespace_cnt,
        "excludes": list(excludes or []),
    }


def _expand_format_node(node: dict[str, Any]) -> dict[str, Any]:
    """Expand every marked (loose any_order:true) json_schema node.

    Recursion covers every parent shape that can nest a json_schema node:
    or/sequence (elements), triggered tags & separator tags (tags), dispatch
    rule sets (rules — serialized as [pattern, format] pairs).
    """
    if not isinstance(node, dict):
        return node
    if node.get("type") == "json_schema":
        schema = node.get("json_schema")
        if isinstance(schema, dict) and node.get("any_order") is True:
            style = node.get("style", "json")
            ws = node.get("max_whitespace_cnt")
            excludes = node.get("excludes") or []
            marked = {**schema, _MARKER_KEY: True}
            keys = _object_keys(marked)
            if len(keys) > REQUIRED_PRESERVING_MAX_KEYS:
                warnings.warn(
                    f"required-preserving expansion capped: {len(keys)} keys > "
                    f"{REQUIRED_PRESERVING_MAX_KEYS}; node keeps loose "
                    "any_order semantics (required keys may be missing, "
                    "duplicates accepted)",
                    stacklevel=2,
                )
                leaf = {k: v for k, v in marked.items() if k != _MARKER_KEY}
                out = _node_to_format(leaf, style, True, ws, excludes)
            else:
                stripped = {k: v for k, v in marked.items() if k != _MARKER_KEY}
                out = {
                    "type": "or",
                    "elements": [
                        _node_to_format(
                            _reordered(stripped, list(perm)), style, False, ws, excludes
                        )
                        for perm in itertools.permutations(keys)
                    ],
                }
            return out
        return node
    for attr in ("elements", "content", "tags"):
        v = node.get(attr)
        if isinstance(v, list):
            node[attr] = [_expand_format_node(x) for x in v]
        elif isinstance(v, dict):
            node[attr] = _expand_format_node(v)
    rules = node.get("rules")
    if isinstance(rules, list):
        node["rules"] = [
            [rule[0], _expand_format_node(rule[1])]
            if isinstance(rule, list) and len(rule) == 2
            else rule
            for rule in rules
        ]
    return node


def expand_loose_to_required_preserving(structural_tag_json: str) -> str:
    """Expand every loose `any_order: true` json_schema node in a serialized
    tag into the required-preserving or-permutation shape.

    Input/output: the serialized StructuralTag (model_dump_json shape). The
    C++ core receives only or-format/json_schema nodes it already
    understands; the loose semantics survive only above the cap (warned,
    any_order:true on the fallback node for structural auditors).
    """
    parsed = json.loads(structural_tag_json)
    if parsed.get("type") == "structural_tag" and isinstance(
        parsed.get("format"), dict
    ):
        parsed["format"] = _expand_format_node(parsed["format"])
    return json.dumps(parsed)
