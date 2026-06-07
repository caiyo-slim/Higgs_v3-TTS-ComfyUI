"""Dependency check/install helper for Higgs_v3-TTS-ComfyUI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys


CRITICAL_IMPORTS = ["torch", "torchaudio", "transformers"]

LIGHTWEIGHT_IMPORTS = {
    "huggingface_hub": "huggingface_hub",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "accelerate": "accelerate",
    "numpy": "numpy",
    "tqdm": "tqdm",
}


def _missing() -> list[str]:
    return [package for module, package in LIGHTWEIGHT_IMPORTS.items() if importlib.util.find_spec(module) is None]


def main() -> int:
    missing_critical = [name for name in CRITICAL_IMPORTS if importlib.util.find_spec(name) is None]
    if missing_critical:
        print("Missing ComfyUI/runtime dependency:", ", ".join(missing_critical))
        print("Install this nodepack inside a working ComfyUI environment; this helper will not modify torch, torchaudio, or transformers.")
        return 1

    missing = _missing()
    if not missing:
        print("Higgs_v3-TTS-ComfyUI dependencies are already present.")
        return 0

    print("Installing missing lightweight dependencies:", ", ".join(missing))
    print("Torch/torchaudio are not modified by this installer.")
    cmd = [sys.executable, "-m", "pip", "install", *missing]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
