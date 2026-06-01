#!/usr/bin/env python3
"""Patch OpenClaw's image tool to ignore bogus model overrides from the LLM."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


OLD_SNIPPET = """function resolvePromptAndModelOverride(args, defaultPrompt) {
\treturn {
\t\tprompt: normalizeOptionalString(args.prompt) ?? defaultPrompt,
\t\tmodelOverride: normalizeOptionalString(args.model)
\t};
}
"""


NEW_SNIPPET = """function sanitizeImageToolModelOverride(value) {
\tconst normalized = normalizeOptionalString(value);
\tif (!normalized) return void 0;
\tconst lowered = normalized.toLowerCase();
\tif (lowered.includes("agents.defaults.imagemodel")) return void 0;
\tif (/(?:^|\\/)(null|undefined|none|default)$/.test(lowered)) return void 0;
\treturn normalized;
}
function resolvePromptAndModelOverride(args, defaultPrompt) {
\treturn {
\t\tprompt: normalizeOptionalString(args.prompt) ?? defaultPrompt,
\t\tmodelOverride: sanitizeImageToolModelOverride(args.model)
\t};
}
"""


def patch_file(target: Path) -> str:
    text = target.read_text(encoding="utf-8")
    if NEW_SNIPPET in text:
        return "already-patched"
    if OLD_SNIPPET not in text:
        raise RuntimeError(f"expected snippet not found in {target}")
    backup = target.with_suffix(target.suffix + ".bak")
    if not backup.exists():
        shutil.copyfile(target, backup)
    target.write_text(text.replace(OLD_SNIPPET, NEW_SNIPPET, 1), encoding="utf-8")
    return f"patched (backup: {backup})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/usr/lib/node_modules/openclaw/dist/openclaw-tools-BIHCDPUL.js",
        help="Path to the bundled OpenClaw tool runtime JS file.",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        raise FileNotFoundError(f"target not found: {target}")

    status = patch_file(target)
    print(f"{target}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
