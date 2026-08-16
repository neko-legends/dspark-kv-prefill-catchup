#!/usr/bin/env python3
"""Enable clamp-preserving FlashInfer B12X NVFP4 MoE in the pinned vLLM build.

FlashInfer 0.6.15 already exposes the SM120/SM121 B12X kernel with
``swigluoai_uninterleave`` and explicit alpha/beta/limit parameters.  The
pinned vLLM adapter predates that wiring and rejects B12X whenever a model sets
``swiglu_limit``.  It also creates one max-batch CUDA-graph workspace per MoE
layer, which exhausts unified memory on a two-Spark DeepSeek deployment.  This
patch bridges the activation APIs and shares a geometry-identical wrapper
inside each GPU process.  It does not modify CUDA kernels or checkpoint tensors.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import py_compile
from pathlib import Path


B12X_MARKER = "# NEKO_B12X_CLAMP_ADAPTER_V1"
SHARED_MARKER = "# NEKO_B12X_SHARED_WRAPPER_V1"
ORACLE_MARKER = "# NEKO_B12X_CLAMP_ORACLE_V1"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def locate_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vllm package was not found")
    return Path(spec.origin).resolve().parent


def verify_flashinfer_api() -> None:
    from flashinfer.fused_moe import B12xMoEWrapper

    params = inspect.signature(B12xMoEWrapper.__init__).parameters
    required = {"activation", "swiglu_alpha", "swiglu_beta", "swiglu_limit"}
    missing = sorted(required.difference(params))
    if missing:
        raise RuntimeError(
            "Installed FlashInfer B12xMoEWrapper lacks required clamp API: "
            + ", ".join(missing)
        )


def patch_b12x_adapter(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False

    if B12X_MARKER not in text:
        old_activation = """        self._activation_str = self._ACTIVATION_MAP[activation]\n\n        # Lazily created on first apply() call.\n"""
        new_activation = """        self._activation_str = self._ACTIVATION_MAP[activation]\n\n        # NEKO_B12X_CLAMP_ADAPTER_V1\n        # FlashInfer 0.6.15 supports the packed-layout SwiGLU-OAI form with\n        # explicit clamp parameters.  DeepSeek V4 presents this through vLLM\n        # as SILU + swiglu_limit; alpha=1 and beta=0 reproduce the existing\n        # silu_and_mul_with_clamp / FlashInfer-CUTLASS math exactly.\n        self._swiglu_limit = moe_config.swiglu_limit\n        self._swiglu_alpha = (\n            float(moe_config.swiglu_alpha)\n            if moe_config.swiglu_alpha is not None\n            else 1.0\n        )\n        self._swiglu_beta = (\n            float(moe_config.swiglu_beta)\n            if moe_config.swiglu_beta is not None\n            else 0.0\n        )\n        if self._swiglu_limit is not None:\n            if activation != MoEActivation.SILU:\n                raise ValueError(\n                    \"Clamp-enabled B12X adapter currently requires SILU; \"\n                    f\"received {activation!r}\"\n                )\n            self._activation_str = \"swigluoai_uninterleave\"\n\n        # Lazily created on first apply() call.\n"""
        text = replace_once(
            text, old_activation, new_activation, label="B12X activation adapter"
        )

        old_wrapper = """            num_local_experts=self.num_local_experts,\n            activation=self._activation_str,\n        )\n"""
        new_wrapper = """            num_local_experts=self.num_local_experts,\n            activation=self._activation_str,\n            swiglu_alpha=self._swiglu_alpha,\n            swiglu_beta=self._swiglu_beta,\n            swiglu_limit=self._swiglu_limit,\n        )\n"""
        text = replace_once(
            text, old_wrapper, new_wrapper, label="B12X wrapper arguments"
        )
        changed = True

    if SHARED_MARKER not in text:
        old_class = """class FlashInferB12xExperts(mk.FusedMoEExpertsModular):\n"""
        new_class = """# NEKO_B12X_SHARED_WRAPPER_V1\n# Every DeepSeek MoE layer has identical geometry.  A separate wrapper per\n# layer duplicates the max-batch static/dynamic workspaces dozens of times.\n# The GPU stream executes layers serially, so one scratch wrapper per geometry\n# and GPU process is sufficient and remains CUDA-graph safe.\n_SHARED_B12X_WRAPPERS: dict[tuple[Any, ...], Any] = {}\n\n\nclass FlashInferB12xExperts(mk.FusedMoEExpertsModular):\n"""
        text = replace_once(text, old_class, new_class, label="B12X shared registry")

        old_ensure = """        from flashinfer.fused_moe import B12xMoEWrapper\n\n        self._wrapper = B12xMoEWrapper(\n            num_experts=self.global_num_experts,\n            top_k=self.topk,\n            hidden_size=self.hidden_dim,\n            intermediate_size=self.intermediate_size_per_partition,\n            use_cuda_graph=True,\n            max_num_tokens=self.max_num_tokens,\n            num_local_experts=self.num_local_experts,\n            activation=self._activation_str,\n            swiglu_alpha=self._swiglu_alpha,\n            swiglu_beta=self._swiglu_beta,\n            swiglu_limit=self._swiglu_limit,\n        )\n"""
        new_ensure = """        from flashinfer.fused_moe import B12xMoEWrapper\n\n        key = (\n            torch.cuda.current_device(),\n            self.global_num_experts,\n            self.topk,\n            self.hidden_dim,\n            self.intermediate_size_per_partition,\n            self.max_num_tokens,\n            self.num_local_experts,\n            self._activation_str,\n            self._swiglu_alpha,\n            self._swiglu_beta,\n            self._swiglu_limit,\n            self.out_dtype,\n        )\n        wrapper = _SHARED_B12X_WRAPPERS.get(key)\n        if wrapper is None:\n            wrapper = B12xMoEWrapper(\n                num_experts=self.global_num_experts,\n                top_k=self.topk,\n                hidden_size=self.hidden_dim,\n                intermediate_size=self.intermediate_size_per_partition,\n                use_cuda_graph=True,\n                max_num_tokens=self.max_num_tokens,\n                num_local_experts=self.num_local_experts,\n                output_dtype=self.out_dtype,\n                activation=self._activation_str,\n                swiglu_alpha=self._swiglu_alpha,\n                swiglu_beta=self._swiglu_beta,\n                swiglu_limit=self._swiglu_limit,\n            )\n            _SHARED_B12X_WRAPPERS[key] = wrapper\n        self._wrapper = wrapper\n"""
        text = replace_once(
            text, old_ensure, new_ensure, label="B12X shared wrapper construction"
        )
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_nvfp4_oracle(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if ORACLE_MARKER in text:
        return False

    old = """        NvFp4MoeBackend.FLASHINFER_CUTLASS,\n        NvFp4MoeBackend.MARLIN,\n"""
    new = """        NvFp4MoeBackend.FLASHINFER_CUTLASS,\n        NvFp4MoeBackend.FLASHINFER_B12X,  # NEKO_B12X_CLAMP_ORACLE_V1\n        NvFp4MoeBackend.MARLIN,\n"""
    text = replace_once(text, old, new, label="NVFP4 clamp-safe backend set")
    path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate APIs only")
    args = parser.parse_args()

    verify_flashinfer_api()
    root = locate_vllm_root()
    b12x = root / "model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"
    oracle = root / "model_executor/layers/fused_moe/oracle/nvfp4.py"
    for path in (b12x, oracle):
        if not path.is_file():
            raise RuntimeError(f"expected vLLM source file is missing: {path}")

    if args.check:
        print("B12X clamp adapter prerequisites satisfied")
        return

    changed = [patch_b12x_adapter(b12x), patch_nvfp4_oracle(oracle)]
    py_compile.compile(str(b12x), doraise=True)
    py_compile.compile(str(oracle), doraise=True)
    print(
        "B12X clamp adapter ready: "
        f"adapter={'patched' if changed[0] else 'already-patched'}, "
        f"oracle={'patched' if changed[1] else 'already-patched'}"
    )


if __name__ == "__main__":
    main()
