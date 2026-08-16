#!/usr/bin/env python3
"""Precompile SM121 B12X NVFP4 MoE kernels without loading DeepSeek weights.

This uses the exact DeepSeek V4 expert geometry and clamp-preserving activation,
but synthetic packed FP4 weights.  It exercises the CUDA-graph decode sizes and
the full 16,384-token dynamic workspace while the node has its full memory
budget available.  CuTe DSL persists the resulting file cache under /tmp,
which the DSpark Compose deployment maps to the host cache.
"""

from __future__ import annotations

import gc
import time

import torch

from flashinfer import B12xMoEWrapper
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout


NUM_EXPERTS = 256
TOP_K = 6
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
MAX_NUM_TOKENS = 16384
GRAPH_TOKEN_SIZES = (1, 2, 4, 8, 16, 24, 32, 40, 48)


def report(label: str) -> None:
    allocated = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        f"{label}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB, "
        f"device_free={free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB",
        flush=True,
    )


def make_mma_scales(rows: int, cols: int) -> torch.Tensor:
    linear = torch.ones(
        (NUM_EXPERTS * rows, cols // 16),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    converted = convert_sf_to_mma_layout(
        linear,
        m=rows,
        k=cols,
        num_groups=NUM_EXPERTS,
        sf_vec_size=16,
    )
    del linear
    return converted


def run_shape(wrapper: B12xMoEWrapper, num_tokens: int, tensors: dict) -> None:
    x = torch.zeros((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
    expert_row = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
    topk_ids = expert_row.repeat(num_tokens, 1)
    topk_weights = torch.full(
        (num_tokens, TOP_K),
        1.0 / TOP_K,
        dtype=torch.float32,
        device="cuda",
    )

    started = time.perf_counter()
    output = wrapper.run(
        x=x,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        **tensors,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if output.shape != (num_tokens, HIDDEN_SIZE):
        raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all().item():
        raise RuntimeError(f"non-finite output for token shape {num_tokens}")
    print(f"compiled/executed token shape {num_tokens} in {elapsed:.2f}s", flush=True)
    del x, topk_ids, topk_weights, output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # Keep at least ~36 GiB outside PyTorch on a 128 GiB Spark so CuTe/LLVM,
    # SSH, and the operating system remain responsive even if allocation fails.
    torch.cuda.set_per_process_memory_fraction(0.70)
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"expected SM121, found capability {capability}")
    print(f"device={torch.cuda.get_device_name()} capability={capability}", flush=True)
    report("initial")

    w1_weight = torch.zeros(
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2_weight = torch.zeros(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w1_weight_sf = make_mma_scales(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    w2_weight_sf = make_mma_scales(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    w1_alpha = torch.ones(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    w2_alpha = torch.ones(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    fc2_input_scale = torch.ones(
        NUM_EXPERTS, dtype=torch.float32, device="cuda"
    )
    tensors = {
        "w1_weight": w1_weight,
        "w1_weight_sf": w1_weight_sf,
        "w1_alpha": w1_alpha,
        "fc2_input_scale": fc2_input_scale,
        "w2_weight": w2_weight,
        "w2_weight_sf": w2_weight_sf,
        "w2_alpha": w2_alpha,
    }
    torch.cuda.empty_cache()
    report("synthetic weights ready")

    wrapper = B12xMoEWrapper(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        use_cuda_graph=True,
        max_num_tokens=MAX_NUM_TOKENS,
        num_local_experts=NUM_EXPERTS,
        output_dtype=torch.bfloat16,
        activation="swigluoai_uninterleave",
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
        quant_mode="nvfp4",
        source_format="modelopt",
    )
    report("shared max-token workspace ready")

    for num_tokens in GRAPH_TOKEN_SIZES:
        run_shape(wrapper, num_tokens, tensors)

    # The startup memory-profile pass exercises the full scheduler capacity and
    # selects the dynamic backend. Compile that exact shape while memory is free.
    run_shape(wrapper, MAX_NUM_TOKENS, tensors)
    report("all paths complete")
    gc.collect()
    torch.cuda.empty_cache()
    print("B12X DeepSeek V4 SM121 precompile completed successfully", flush=True)


if __name__ == "__main__":
    main()
