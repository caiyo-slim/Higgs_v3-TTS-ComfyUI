"""Higgs v3 model loading, assets, and ComfyUI memory registration."""

from __future__ import annotations

import gc
import importlib.util
import logging
import math
import shutil
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .native import (
    HiggsAudioCodec,
    build_native_model,
    load_native_weights,
    load_tokenizer,
    read_config,
)

logger = logging.getLogger("Higgs_v3-TTS-ComfyUI")

MODEL_FOLDER_NAME = "higgsv3tts"
OFFICIAL_REPO_ID = "bosonai/higgs-audio-v3-tts-4b"
OFFICIAL_MODEL_NAME = "Higgs Audio v3 TTS 4B - bosonai (auto-download)"
HF_ENDPOINT = "https://huggingface.co"
DTYPE_OPTIONS = ["auto", "bf16"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
ATTENTION_OPTIONS = ["auto", "sdpa", "flash_attention", "sageattention"]
SMALL_ASSET_PATTERNS = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "assets/model_architecture.png",
]

_ACTIVE_BUNDLE: "HiggsV3Bundle | None" = None
_ACTIVE_LOAD_KEY: tuple[Any, ...] | None = None


@dataclass
class HiggsV3Bundle:
    model: torch.nn.Module
    codec: HiggsAudioCodec
    tokenizer: Any
    model_dir: Path
    device: torch.device
    torch_dtype: torch.dtype
    dtype_name: str
    attention: str
    patchers: list[Any] = field(default_factory=list)


def _module_unique_tensors(module: torch.nn.Module) -> list[torch.Tensor]:
    seen: set[int] = set()
    tensors: list[torch.Tensor] = []
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        ident = id(tensor)
        if ident in seen:
            continue
        seen.add(ident)
        tensors.append(tensor)
    return tensors


def _same_device(a: torch.device, b: torch.device) -> bool:
    a = torch.device(a)
    b = torch.device(b)
    return a.type == b.type and (a.index or 0) == (b.index or 0)


class HiggsV3VBar:
    page_size: int = 32 * 1024 * 1024

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = torch.device(device)
        self.tensors: list[torch.Tensor] = []
        self.total_size = 0
        self.total_pages = 1
        self.watermark = 0
        self._refresh_tensors()

    @property
    def offset(self) -> int:
        return self.total_size

    def _refresh_tensors(self) -> None:
        self.tensors = _module_unique_tensors(self.model)
        self.total_size = sum(t.nelement() * t.element_size() for t in self.tensors if t.device.type != "meta")
        self.total_pages = max(1, math.ceil(self.total_size / self.page_size)) if self.total_size > 0 else 0

    def loaded_size(self) -> int:
        self._refresh_tensors()
        return sum(
            t.nelement() * t.element_size()
            for t in self.tensors
            if t.device.type != "meta" and _same_device(t.device, self.device)
        )

    def get_residency(self) -> list[int]:
        self._refresh_tensors()
        if self.total_size <= 0:
            return []
        residency = [0 for _ in range(self.total_pages)]
        cursor = 0
        for tensor in self.tensors:
            if tensor.device.type == "meta":
                continue
            size = tensor.nelement() * tensor.element_size()
            if size <= 0:
                continue
            if _same_device(tensor.device, self.device):
                start_page = cursor // self.page_size
                end_page = min(self.total_pages - 1, (cursor + size - 1) // self.page_size)
                for page in range(start_page, end_page + 1):
                    residency[page] |= 1
            cursor += size
        return residency

    def get_watermark(self) -> int:
        self.watermark = max(self.watermark, self.loaded_size())
        return self.watermark

    def prioritize(self) -> None:
        self.watermark = self.loaded_size()


try:
    import comfy.model_patcher as _model_patcher

    class HiggsV3Patcher(_model_patcher.ModelPatcher):
        def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
            super().__init__(model, load_device, offload_device, size, weight_inplace_update)
            self._ensure_dynamic_state(load_device)

        def is_dynamic(self):
            return True

        def _ensure_dynamic_state(self, device):
            device = torch.device(device)
            if not hasattr(self.model, "dynamic_vbars"):
                self.model.dynamic_vbars = {}
            if not hasattr(self.model, "dynamic_pins"):
                self.model.dynamic_pins = {}
            if device not in self.model.dynamic_pins:
                try:
                    import comfy_aimdo.host_buffer

                    empty_hostbuf = comfy_aimdo.host_buffer.HostBuffer(0, 0, 0)
                except Exception:
                    empty_hostbuf = None
                self.model.dynamic_pins[device] = {
                    "weights": (empty_hostbuf, [], [-1], [0], [0], {}),
                    "patches": (empty_hostbuf, [], [-1], [0], [0], {}),
                    "hostbufs_initialized": False,
                    "failed": False,
                    "active": False,
                }

        def _vbar_get(self):
            vbars = getattr(self.model, "dynamic_vbars", {})
            if vbars:
                return next(iter(vbars.values()))
            return None

        def loaded_size(self):
            vbar = self._vbar_get()
            if vbar is not None:
                return vbar.loaded_size()
            return getattr(self.model, "model_loaded_weight_memory", 0)

        def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            self._ensure_dynamic_state(torch.device(device_to))
            before = self.loaded_size()
            self.model.to(device_to)
            self.model.model_loaded_weight_memory = self.model_size()
            return max(0, self.loaded_size() - before)

        def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
            before = self.loaded_size()
            self.detach()
            return before

        def detach(self, unpatch_all=True):
            try:
                self.model.to(self.offload_device)
                self.model.model_loaded_weight_memory = 0
            except Exception:
                pass
            _empty_accelerator_cache()
            return self.model

        def current_loaded_device(self):
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                return self.offload_device

        def loaded_ram_size(self):
            return 0

        def pinned_memory_size(self):
            return 0

        def unregister_inactive_pins(self, ram_to_unload, subsets=["weights", "patches"]):
            return 0

        def partially_unload_ram(self, ram_to_unload, subsets=["weights", "patches"]):
            return 0

    del _model_patcher
except Exception:
    HiggsV3Patcher = None


def _empty_accelerator_cache() -> None:
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
        return
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def model_dir() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    except Exception:
        base = Path(__file__).resolve().parent / "models" / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_model_folder() -> None:
    try:
        import folder_paths

        base = str(model_dir())
        if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, base)
        logger.info("Higgs v3 model folder registered: %s", base)
    except Exception:
        pass


def assets_dir() -> Path:
    path = Path(__file__).resolve().parent / "assets" / "higgs-audio-v3-tts-4b"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_small_assets(download_if_missing: bool) -> Path:
    dest = assets_dir()
    required = [dest / "config.json", dest / "tokenizer.json", dest / "model.safetensors.index.json"]
    if all(path.is_file() for path in required):
        return dest
    if not download_if_missing:
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"Missing Higgs v3 small assets: {missing}. Enable download_if_missing.")

    from huggingface_hub import snapshot_download

    logger.info("Downloading Higgs v3 small assets to %s", dest)
    kwargs = {
        "repo_id": OFFICIAL_REPO_ID,
        "local_dir": str(dest),
        "allow_patterns": SMALL_ASSET_PATTERNS,
        "ignore_patterns": ["model.safetensors"],
        "endpoint": HF_ENDPOINT,
    }
    snapshot_download(**kwargs)
    return dest


def _copy_small_assets_to_model_dir(runtime_dir: Path, asset_dir: Path) -> None:
    for rel in SMALL_ASSET_PATTERNS:
        src = asset_dir / rel
        if not src.is_file():
            continue
        dst = runtime_dir / rel
        if dst.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _has_model_file(path: Path) -> bool:
    return (path / "model.safetensors").is_file()


def get_model_choices() -> list[str]:
    base = model_dir()
    choices = [OFFICIAL_MODEL_NAME]
    if _has_model_file(base):
        choices.append("local root: ComfyUI/models/higgsv3tts")
    try:
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and _has_model_file(entry):
                choices.append(entry.name)
    except OSError:
        pass
    return choices


def _download_full_model(target_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    logger.info("Downloading Higgs v3 model.safetensors to %s. This is a large download.", target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=OFFICIAL_REPO_ID,
        filename="model.safetensors",
        local_dir=str(target_dir),
        endpoint=HF_ENDPOINT,
    )
    return target_dir


def resolve_model_dir(model_choice: str, download_if_missing: bool) -> Path:
    base = model_dir()
    if model_choice == "local root: ComfyUI/models/higgsv3tts" and _has_model_file(base):
        path = base
    elif model_choice == OFFICIAL_MODEL_NAME:
        official_dir = base / "higgs-audio-v3-tts-4b"
        path = official_dir if _has_model_file(official_dir) else base if _has_model_file(base) else official_dir
        if not _has_model_file(path):
            if not download_if_missing:
                raise FileNotFoundError(
                    f"Missing model.safetensors. Put it in {official_dir} or {base}, or enable download_if_missing."
                )
            path = _download_full_model(official_dir)
    else:
        path = base / model_choice
        if not _has_model_file(path):
            raise FileNotFoundError(f"Higgs model folder does not contain model.safetensors: {path}")

    asset_dir = ensure_small_assets(download_if_missing)
    _copy_small_assets_to_model_dir(path, asset_dir)
    return path


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        try:
            import comfy.model_management as mm

            return torch.device(mm.get_torch_device())
        except Exception:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but torch.cuda is not available.")
    return device


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            except Exception:
                return torch.float32
        return torch.float32
    if dtype_name == "bf16":
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("bf16 was selected, but this CUDA device does not report bf16 support. Use dtype=auto to fall back to fp32.")
            except RuntimeError:
                raise
            except Exception:
                pass
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_attention(attention: str) -> tuple[str, str | None]:
    if attention in {"auto", "sdpa"}:
        return "sdpa", "sdpa"
    if attention == "flash_attention":
        if importlib.util.find_spec("flash_attn") is None:
            raise ImportError("flash_attention was selected, but flash_attn is not installed.")
        return "flash_attention", "flash_attention_2"
    if attention == "sageattention":
        if importlib.util.find_spec("sageattention") is None:
            raise ImportError("sageattention was selected, but sageattention is not installed.")
        return "sageattention", "sdpa"
    raise ValueError(f"Unsupported attention mode: {attention}")


def _register_with_comfy(patcher: Any) -> None:
    if patcher is None:
        return
    try:
        import comfy.model_management as mm

        if patcher.load_device.type == "cpu":
            return
        if any(loaded.model is patcher for loaded in mm.current_loaded_models):
            return
        raw = patcher.model
        if hasattr(patcher, "_ensure_dynamic_state"):
            patcher._ensure_dynamic_state(patcher.load_device)
        raw.model_loaded_weight_memory = patcher.loaded_size()
        raw.dynamic_vbars = {patcher.load_device: HiggsV3VBar(raw, patcher.load_device)}
        loaded = mm.LoadedModel(patcher)
        loaded.real_model = weakref.ref(raw)
        loaded.model_finalizer = weakref.finalize(raw, mm.cleanup_models)
        loaded.model_finalizer.atexit = False
        loaded.currently_used = True
        mm.current_loaded_models.insert(0, loaded)
        logger.info("Registered %s with ComfyUI/AIMDO memory tracking.", raw.__class__.__name__)
    except Exception as exc:
        logger.warning("Could not register Higgs module with ComfyUI memory tracking: %s", exc)


def _unregister_from_comfy(patcher: Any) -> None:
    try:
        import comfy.model_management as mm

        survivors = []
        for loaded in mm.current_loaded_models:
            if loaded.model is patcher:
                try:
                    if loaded.model_finalizer is not None:
                        loaded.model_finalizer.detach()
                    loaded.model_finalizer = None
                    loaded.real_model = None
                except Exception:
                    pass
                try:
                    finalizer = getattr(loaded, "_patcher_finalizer", None)
                    if finalizer is not None:
                        finalizer.detach()
                    loaded._patcher_finalizer = None
                except Exception:
                    pass
                continue
            survivors.append(loaded)
        mm.current_loaded_models[:] = survivors
    except Exception:
        pass


def register_runtime_module(module: torch.nn.Module, device: torch.device) -> Any:
    if HiggsV3Patcher is None or torch.device(device).type == "cpu":
        module.to(device)
        return None
    patcher = HiggsV3Patcher(module, load_device=torch.device(device), offload_device=torch.device("cpu"))
    module.model_loaded_weight_memory = patcher.model_size()
    _register_with_comfy(patcher)
    return patcher


def resume_runtime_module(patcher: Any, device: torch.device) -> None:
    if patcher is None:
        return
    patcher.partially_load(torch.device(device))
    _register_with_comfy(patcher)


def unload_runtime_module(patcher: Any) -> None:
    if patcher is None:
        return
    _unregister_from_comfy(patcher)
    try:
        patcher.detach()
    except Exception:
        pass


def resume_bundle_to_device(bundle: HiggsV3Bundle) -> None:
    for patcher in bundle.patchers:
        resume_runtime_module(patcher, bundle.device)


def unload_higgs_bundle(bundle: HiggsV3Bundle | None, reason: str = "manual unload", hard: bool = True) -> None:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY
    if bundle is None:
        return
    logger.info("Unloading Higgs v3 bundle (%s).", reason)
    for patcher in list(bundle.patchers):
        unload_runtime_module(patcher)
    try:
        bundle.model.to("cpu")
    except Exception:
        pass
    try:
        bundle.codec.model.to("cpu")
    except Exception:
        pass
    for module in (bundle.model, bundle.codec.model):
        try:
            module.model_loaded_weight_memory = 0
            if hasattr(module, "dynamic_vbars"):
                module.dynamic_vbars.clear()
            if hasattr(module, "dynamic_pins"):
                module.dynamic_pins.clear()
            if hard and hasattr(module, "to_empty"):
                module.to_empty(device=torch.device("meta"))
        except Exception:
            pass
    bundle.patchers.clear()
    if hard:
        try:
            bundle.model = None
            bundle.codec = None
            bundle.tokenizer = None
        except Exception:
            pass
    gc.collect()
    _empty_accelerator_cache()
    if _ACTIVE_BUNDLE is bundle:
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None


def load_higgs_bundle(
    model_choice: str,
    dtype_name: str,
    device_name: str,
    attention: str,
    download_if_missing: bool,
) -> HiggsV3Bundle:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    register_model_folder()
    runtime_dir = resolve_model_dir(model_choice, download_if_missing)
    device = resolve_device(device_name)
    torch_dtype = resolve_dtype(dtype_name, device)
    runtime_attention, hf_attention = resolve_attention(attention)

    model_file = runtime_dir / "model.safetensors"
    load_key = (
        str(runtime_dir.resolve()),
        model_file.stat().st_mtime_ns,
        str(device),
        str(torch_dtype),
        runtime_attention,
    )
    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        resume_bundle_to_device(_ACTIVE_BUNDLE)
        return _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        unload_higgs_bundle(_ACTIVE_BUNDLE, reason="load settings changed")

    config = read_config(runtime_dir)
    logger.info("Loading Higgs v3 from %s on %s with dtype=%s attention=%s", runtime_dir, device, torch_dtype, runtime_attention)
    model = build_native_model(config, torch_dtype, hf_attention)
    load_native_weights(model, runtime_dir, device, torch_dtype)
    codec = HiggsAudioCodec.from_pretrained(runtime_dir, device=device, dtype=torch_dtype)
    tokenizer = load_tokenizer(runtime_dir)

    patchers: list[Any] = []
    model_patcher = register_runtime_module(model, device)
    if model_patcher is not None:
        patchers.append(model_patcher)
    codec_patcher = register_runtime_module(codec.model, device)
    if codec_patcher is not None:
        patchers.append(codec_patcher)

    bundle = HiggsV3Bundle(
        model=model,
        codec=codec,
        tokenizer=tokenizer,
        model_dir=runtime_dir,
        device=device,
        torch_dtype=torch_dtype,
        dtype_name=dtype_name,
        attention=runtime_attention,
        patchers=patchers,
    )
    _ACTIVE_BUNDLE = bundle
    _ACTIVE_LOAD_KEY = load_key
    _empty_accelerator_cache()
    return bundle


def unload_active_bundle() -> None:
    unload_higgs_bundle(_ACTIVE_BUNDLE, reason="active unload")
