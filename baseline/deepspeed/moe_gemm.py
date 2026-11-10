# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVIDIA baseline for moe_gemm.

The baseline is DeepSpeed's CUTLASS MoE GEMM:

    deepspeed.ops.op_builder.InferenceCutlassBuilder().load().moe_gemm

The operator computes a separate GEMM for each expert using its assigned
hidden states and weights, then applies the specified activation function.

The input tensors use 16-bit floating-point data types (FP16/BF16):

* hidden_states: [total_tokens, K], containing the input tokens for all experts;
* weights: [E, K, N], containing the weight matrix for each expert;
* biases: [E, N], containing the bias vector for each expert;
* total_rows_before_expert: [E], containing the cumulative row boundaries
  used to identify the token range assigned to each expert;
* output: [total_tokens, N], storing the computed results.

The benchmark generates random inputs with fixed shapes and data types.
The same input construction is used for the different implementations
so that their execution latency can be compared under identical conditions.

The benchmark parameterizes the expert count, tokens per expert, K, N,
and activation type. The activation ID is passed directly to the
DeepSpeed binding.
"""

import torch

OP_NAME = "moe_gemm"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True  # the operator writes the preallocated `output`

# Each tuple represents (experts, tokens_per_expert, K, N).
# These shapes cover different expert counts and token distributions.
_SHAPES = [
    (1, 128, 2048, 2048),
    (8, 16, 1024, 2048),
    (64, 1, 1024, 1024),
    (8, 128, 1024, 2048),
]

# Activation IDs: 6 = identity, 2 = silu.
_ACTIVATIONS = (6, 2)


def native():
    """Load and return DeepSpeed's native CUTLASS moe_gemm binding."""
    from deepspeed.ops.op_builder import InferenceCutlassBuilder

    try:
        inf_module = InferenceCutlassBuilder().load()
        return inf_module.moe_gemm
    except Exception as e:
        raise RuntimeError("Failed to load DeepSpeed moe_gemm") from e


def grid():
    """Generate all shape and activation combinations for the benchmark."""
    for shape in _SHAPES:
        for activation in _ACTIVATIONS:
            yield {
                "shape": shape,
                "activation": activation,
            }


def build_inputs(binding, dtype, device):
    """Build random inputs matching the selected benchmark configuration."""
    torch.manual_seed(0)

    experts, tokens, k, n = binding["shape"]
    activation = binding["activation"]

    # Total number of tokens processed by all experts.
    total_tokens = experts * tokens

    # Input hidden states for all experts, with shape [E * tokens, K].
    hidden_states = torch.randn((total_tokens, k), dtype=dtype, device=device)

    # Each expert has an independent weight matrix of shape [K, N].
    weights = torch.randn((experts, k, n), dtype=dtype, device=device)

    # One bias vector of length N for each expert.
    biases = torch.randn((experts, n), dtype=dtype, device=device)

    # Cumulative row boundaries for each expert.
    # For uniformly distributed tokens, the values are:
    # [tokens, 2 * tokens, ..., experts * tokens].
    total_rows_before_expert = (
        torch.arange(1, experts + 1, dtype=torch.int64, device=device) * tokens
    )

    # Preallocated output tensor; the operator writes results into it.
    output = torch.empty((total_tokens, n), dtype=dtype, device=device)

    # Arguments passed to the native moe_gemm binding.
    args = (
        output,
        hidden_states,
        weights,
        biases,
        total_rows_before_expert,
        activation,
    )

    return args, {}


def key_shape(binding):
    """Build a unique shape key for the benchmark configuration."""
    experts, tokens, k, n = binding["shape"]

    return f"{experts}x{tokens}x{k}x{n}" f"_act{binding['activation']}"


def config(binding, dtype):
    """Describe the input tensor shapes, data types, and GEMM dimensions."""
    experts, tokens, k, n = binding["shape"]

    return {
        "inputs": {
            "output": {
                "shape": [experts * tokens, n],
                "dtype": str(dtype),
            },
            "hidden_states": {
                "shape": [experts * tokens, k],
                "dtype": str(dtype),
            },
            "weights": {
                "shape": [experts, k, n],
                "dtype": str(dtype),
            },
            "biases": {
                "shape": [experts, n],
                "dtype": str(dtype),
            },
            "total_rows_before_expert": {
                "shape": [experts],
                "dtype": "torch.int64",
            },
            "activation": {
                "scalar": binding["activation"],
            },
        },
        "dims": {
            "experts": experts,
            "tokens_per_expert": tokens,
            "K": k,
            "N": n,
        },
    }
