# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blessed-hash gate around `uv pip compile` (the pip-compile hooks).

The naive hook re-resolves the lockfile on every run, so every CI push
needs live index access — and when the index host degrades (the xpu index
is served from GitHub Pages, i.e. GitHub itself), the gate goes red for
reasons that have nothing to do with our files.

This wrapper makes the gate honest under vendor degradation:

- Offline fast path: a sha256 over the hook args, the source .in, every
  `-c` constraint, every `--override` file, and the current lockfile is
  compared against the blessed store (requirements/.pip-compile-blessed.json,
  committed). Match -> exit 0 without touching the network.
- Fail-closed on drift: on a hash miss the wrapper resolves. If the
  resolution rewrites the lockfile, pre-commit reports the modification
  and the hook fails — a red gate means our files drifted.
- Fail-open on vendor outage: if the resolve fails with a network-class
  error (index unreachable), the wrapper warns and exits 0 — the drift
  check is deferred, not faked. A vendor outage must never be the reason
  a gate is red.

Set PIP_COMPILE_BLESSED=force to skip the fast path (the daily drift
workflow uses this to re-resolve unconditionally).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = 1
STORE = Path("requirements/.pip-compile-blessed.json")
FORCE_ENV = "PIP_COMPILE_BLESSED"

# argv tokens that take a value (everything else starting with "-" is
# treated as boolean). Only -o/-c/--override plus the source positional
# matter for hashing; the rest just needs to be skipped correctly.
VALUE_FLAGS = {
    "-o",
    "-c",
    "--override",
    "--no-emit-package",
    "--index-strategy",
    "--torch-backend",
    "--python-platform",
    "--python-version",
    "--annotation-style",
    "--extra-index-url",
    "--index-url",
}
BOOLEAN_FLAGS = {"--generate-hashes", "--strip-extras", "--allow-unsafe", "--quiet"}

# uv failure text that means "the index host is unreachable", not "the
# lockfile is wrong". Matched case-insensitively against the combined
# output.
NETWORK_PATTERNS = (
    "failed to fetch",
    "http status server error",
    "connection error",
    "connection refused",
    "connection reset",
    "timed out",
    "error sending request",
    "temporary failure",
    "no such host",
)


def parse_args(argv: list[str]) -> tuple[str, str, list[str], list[str]]:
    src = out = None
    constraints: list[str] = []
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-o":
            out = argv[i + 1]
            i += 2
        elif arg == "-c":
            constraints.append(argv[i + 1])
            i += 2
        elif arg == "--override":
            overrides.append(argv[i + 1])
            i += 2
        elif arg.startswith("-"):
            i += 1 if arg in BOOLEAN_FLAGS or arg.startswith("--") and "=" in arg else 2
        else:
            src = arg
            i += 1
    if src is None or out is None:
        print(
            f"pip-compile-blessed: cannot parse -o/<source> out of {argv!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    return src, out, constraints, overrides


def content_or_missing(path: str) -> str:
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return "<missing>"


def compute_hash(
    argv: list[str],
    src: str,
    out: str,
    constraints: list[str],
    overrides: list[str],
) -> str:
    h = hashlib.sha256()
    h.update(f"v{VERSION}\0".encode())
    h.update(json.dumps(argv).encode())
    for path in [src, *constraints, *overrides, out]:
        h.update(f"{path}\0{content_or_missing(path)}\0".encode())
    return h.hexdigest()


def load_store() -> dict:
    try:
        return json.loads(STORE.read_text())
    except FileNotFoundError:
        return {}


def is_network_error(output: str) -> bool:
    lowered = output.lower()
    return any(pattern in lowered for pattern in NETWORK_PATTERNS)


def main() -> int:
    argv = sys.argv[1:]
    src, out, constraints, overrides = parse_args(argv)
    digest = compute_hash(argv, src, out, constraints, overrides)
    store = load_store()
    forced = os.environ.get(FORCE_ENV) == "force"

    if not forced and store.get(out) == digest:
        print(f"pip-compile-blessed: {out} matches its blessed resolution (offline)")
        return 0

    uv = shutil.which("uv")
    if uv is None:
        print(
            "pip-compile-blessed: uv not found in the hook env; cannot resolve "
            f"{out}. Install uv into the hook (additional_dependencies) or "
            "bless the store.",
            file=sys.stderr,
        )
        return 1

    result = subprocess.run(
        [uv, "pip", "compile", *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        if is_network_error(output):
            print(
                f"pip-compile-blessed: WARNING index unreachable while "
                f"resolving {out} — drift check deferred, gate stays green. "
                "The scheduled drift workflow re-checks when the index "
                "returns.",
                file=sys.stderr,
            )
            return 0
        print(output, file=sys.stderr)
        return result.returncode

    # Resolution succeeded: bless the post-resolve state. pre-commit fails
    # the hook on its own when the lockfile (or the store) changed, which
    # is the drift-fix contract.
    new_digest = compute_hash(argv, src, out, constraints, overrides)
    if store.get(out) != new_digest:
        store[out] = new_digest
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
