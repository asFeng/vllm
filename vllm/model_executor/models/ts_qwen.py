# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM-compatible Timeseries + Qwen Model (TS2QwenForCausalLM).

This module provides a multimodal model that combines:
- MultichannelTSEncoder: Encodes OHLCV timeseries data
- Qwen2 LLM: Text generation with timeseries context

The timeseries data is encoded to "soft tokens" that are prepended to
text embeddings, providing financial context for text generation.
"""

import abc
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from math import sqrt
from typing import Annotated, Any, Literal, Optional, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class TSEncoderConfig:
    """Configuration for the timeseries encoder."""
    ts_input_len: int = 90         # Input sequence length (days)
    ts_in_channels: int = 5        # Number of channels (OHLCV)
    ts_d_model: int = 512          # Transformer hidden dimension
    ts_heads: int = 8              # Number of attention heads
    ts_num_layers: int = 8         # Number of transformer layers
    ts_patch_len: int = 5          # Patch size for tokenization
    ts_dropout: float = 0.1        # Dropout rate

    @property
    def num_patches(self) -> int:
        return self.ts_input_len // self.ts_patch_len

    @property
    def flatten_dim(self) -> int:
        return self.num_patches * self.ts_d_model

    @property
    def num_soft_tokens(self) -> int:
        """Number of soft tokens = number of channels."""
        return self.ts_in_channels


# ============================================================================
# Timeseries Input Schema
# ============================================================================
class TS2QwenTimeseriesInputs(TensorSchema):
    """
    Timeseries input tensor schema.

    Dimensions:
        - b: Batch size
        - l: Sequence length (ts_input_len, default 90)
        - c: Number of channels (ts_in_channels, default 5 for OHLCV)
    """
    type: Literal["timeseries"] = "timeseries"
    data: Annotated[torch.Tensor, TensorShape("b", "l", "c")]


class TS2QwenEmbeddingInputs(TensorSchema):
    """
    Pre-computed timeseries embedding inputs.

    Dimensions:
        - b: Batch size
        - n: Number of soft tokens
        - h: Hidden size (matches LLM hidden size)
    """
    type: Literal["ts_embeds"] = "ts_embeds"
    data: Annotated[torch.Tensor, TensorShape("b", "n", "h")]


TS2QwenInputs: TypeAlias = TS2QwenTimeseriesInputs | TS2QwenEmbeddingInputs


# ============================================================================
# Attention Projection Layers
# ============================================================================
class Projection(nn.Module, abc.ABC):
    """Base class for query/key projections."""

    def __init__(self, proj_width: int, num_heads: int, **kwargs):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads

    @abc.abstractmethod
    def forward(self, x, seq_id):
        ...


class RotaryProjection(Projection):
    """Rotary Position Embedding (RoPE) projection."""

    def __init__(self, *, proj_width: int, num_heads: int, max_len: int = 512, base: int = 10000):
        super().__init__(proj_width, num_heads)
        assert self.proj_width % 2 == 0, f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(
            "theta",
            1.0 / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float) / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(max_len, device=self.theta.device, dtype=self.theta.dtype)
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x):
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)

    def forward(self, x, seq_id):
        self._init_freq(max_len=seq_id.max() + 1)
        if self.cos.device != seq_id.device:
            self.cos = self.cos.to(seq_id.device)
        if self.sin.device != seq_id.device:
            self.sin = self.sin.to(seq_id.device)
        rot_cos = self.cos[seq_id]
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)


class QueryKeyProjection(nn.Module):
    """Query-Key projection with optional rotary embeddings."""

    def __init__(self, dim: int, num_heads: int, proj_layer, kwargs=None, partial_factor=None):
        super().__init__()
        if partial_factor is not None:
            assert 0.0 <= partial_factor[0] < partial_factor[1] <= 1.0
        assert num_heads > 0 and dim % num_heads == 0

        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            **(kwargs or {}),
        )
        self.key_proj = self.query_proj

    @cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self):
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(self, query, key, query_id, kv_id):
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_id=query_id)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id)
            key = self.key_proj(key, seq_id=kv_id)
        return query, key


# ============================================================================
# Time-Series Attention Layers
# ============================================================================
class TimerMultivariateMask:
    """Creates attention mask for multivariate time-series with causal structure."""

    def __init__(self, B, n_vars, n_tokens, device="cpu"):
        mask_shape = [B, 1, n_tokens, n_tokens]
        with torch.no_grad():
            self._mask1 = torch.ones((n_vars, n_vars), dtype=torch.bool).to(device)
            self._mask2 = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)
            self._mask = torch.kron(self._mask1, self._mask2)

    @property
    def mask(self):
        return self._mask


class BinaryAttentionBias(nn.Module):
    """Learned attention bias based on variable identity."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

    def forward(self, query_id, kv_id):
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = ~ind * weight[:1] + ind * weight[1:]
        return bias


class TimeAttention(nn.Module):
    """Time-aware attention mechanism for multivariate time-series."""

    def __init__(
        self,
        scale=None,
        attention_dropout=0.1,
        d_model=512,
        num_heads=8,
        max_len=100,
        flash_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(
            dim=d_model,
            num_heads=num_heads,
            proj_layer=RotaryProjection,
            kwargs=dict(max_len=max_len),
            partial_factor=(0.0, 0.5),
        )
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, n_vars, n_tokens):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        dtype = queries.dtype  # Preserve input dtype

        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        full_seq = torch.cat([torch.arange(0, n_tokens, device=queries.device).repeat(n_vars)])
        seq_id = repeat(full_seq, "n -> b h n", b=B, h=H)

        queries, keys = self.qk_proj(queries, keys, query_id=seq_id, kv_id=seq_id)

        # Use tensor for scale to preserve dtype
        scale = self.scale or (1.0 / sqrt(E))
        scale = torch.tensor(scale, dtype=dtype, device=queries.device)

        var_id = repeat(
            torch.arange(n_vars, device=queries.device),
            "C -> (C n_tokens)",
            n_tokens=n_tokens,
        )
        var_id = repeat(var_id, "L -> b h L", b=B, h=1)

        attn_bias = self.attn_bias(var_id, var_id)
        attn_mask = TimerMultivariateMask(B, n_vars, n_tokens, device=queries.device)
        # Use dtype-specific infinity
        neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=queries.device)
        attn_mask = attn_bias.masked_fill(attn_mask.mask, neg_inf)

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores = scores + attn_mask.to(scores.dtype)
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            # Ensure A matches values dtype for einsum
            A = A.to(values.dtype)
            V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class TSAttentionLayer(nn.Module):
    """Full attention layer with projections for timeseries."""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, n_vars, n_tokens):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values, n_vars, n_tokens)
        out = out.view(B, L, -1)

        return self.out_projection(out)


# ============================================================================
# Time-Series Encoder Components
# ============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    """SwiGLU-style feed-forward network."""

    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer block with TimeAttention and SwiGLU FFN."""

    def __init__(self, d_model, n_heads, ff_dim=1024, ffn_dropout_p=0.0, attn_dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = TSAttentionLayer(
            TimeAttention(attention_dropout=attn_dropout_p, d_model=d_model, num_heads=n_heads),
            d_model,
            n_heads,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, in_channels, n_tokens):
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x, x, x, in_channels, n_tokens)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x


class MultichannelTSEncoder(nn.Module):
    """
    Multi-channel Time-Series Encoder for LLM Integration.

    Encodes OHLCV time-series using patch-based tokenization and
    transformer layers with time-aware attention.
    """

    def __init__(self, config: TSEncoderConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.ts_in_channels
        self.d_model = config.ts_d_model
        self.n_heads = config.ts_heads
        self.enc_layers = config.ts_num_layers
        self.patch_len = config.ts_patch_len
        self.dropout = config.ts_dropout
        self.num_patches = config.num_patches
        self.flatten_dim = config.flatten_dim

        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)

        # Transformer encoder layers
        self.encoder = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ffn_dropout_p=self.dropout,
                attn_dropout_p=self.dropout,
            )
            for _ in range(self.enc_layers)
        ])

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time-series to patch embeddings.

        Args:
            x: Input tensor [B, L, C] where L=timesteps, C=channels

        Returns:
            Encoded tensor [B, C, N, D] where N=num_patches, D=d_model
        """
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # [B, C, N, P]
        x = self.patch_embedding(x)  # [B, C, N, D]
        N = x.shape[2]

        assert N == self.num_patches, f"Patches {N} != expected {self.num_patches}"
        x = x.reshape(B, C * N, -1)  # [B, C * N, D]

        for layer in self.encoder:
            x = layer(x, self.in_channels, N)

        x = x.reshape(B, C, N, -1)
        return x  # [B, C, N, D]


# ============================================================================
# vLLM Multimodal Processing
# ============================================================================
def _get_ts_config(hf_config) -> TSEncoderConfig:
    """Extract TSEncoderConfig from HuggingFace config."""
    ts_config_dict = getattr(hf_config, "ts_config", {})
    if isinstance(ts_config_dict, dict) and ts_config_dict:
        return TSEncoderConfig(**ts_config_dict)
    return TSEncoderConfig()


class TS2QwenProcessingInfo(BaseProcessingInfo):
    """Processing info for TS2Qwen model."""

    # The special token used for timeseries placeholder
    TIMESERIES_TOKEN = "<|timeseries_pad|>"

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_ts_config(self) -> TSEncoderConfig:
        return _get_ts_config(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"timeseries": 1}  # One timeseries per prompt

    def get_timeseries_token_id(self) -> int:
        """
        Get the token ID for the timeseries placeholder.

        This adds the special token `<|timeseries_pad|>` to the tokenizer if not
        already present, following the same pattern as Qwen3-VL's `<|image_pad|>`.

        Returns:
            The token ID for `<|timeseries_pad|>`
        """
        tokenizer = self.get_tokenizer()
        ts_token = self.TIMESERIES_TOKEN

        # Check if token already exists in vocabulary
        vocab = tokenizer.get_vocab()
        if ts_token in vocab:
            return vocab[ts_token]

        # Add the special token to the tokenizer
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [ts_token]}
        )
        if num_added > 0:
            logger.info(f"Added special token {ts_token} to tokenizer")

        return tokenizer.convert_tokens_to_ids(ts_token)


class TS2QwenDummyInputsBuilder(BaseDummyInputsBuilder[TS2QwenProcessingInfo]):
    """Dummy inputs builder for memory profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """
        Build dummy text containing placeholder tokens for memory profiling.

        Uses the special token `<|timeseries_pad|>` which is registered as a
        single token in the tokenizer. This follows the Qwen3-VL pattern where
        special tokens like `<|image_pad|>` are single tokens in the vocabulary.

        Using a single special token avoids BPE tokenization boundary issues
        that occur with multi-token sequences like `<timeseries>`.
        """
        num_ts = mm_counts.get("timeseries", 0)
        ts_token = TS2QwenProcessingInfo.TIMESERIES_TOKEN
        placeholders = f"{ts_token} " * num_ts
        return f"{placeholders}Analyze the following stock data."

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        ts_config = self.info.get_ts_config()
        num_ts = mm_counts.get("timeseries", 0)

        if num_ts == 0:
            return {}

        dummy_ts = torch.zeros(
            ts_config.ts_input_len,
            ts_config.ts_in_channels,
            dtype=torch.float32,
        )

        return {"timeseries": dummy_ts}


def _ts2qwen_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    """Field configuration for timeseries inputs."""
    return dict(
        timeseries=MultiModalFieldConfig.batched("timeseries"),
        ts_embeds=MultiModalFieldConfig.batched("timeseries"),
    )


class TS2QwenMultiModalProcessor(BaseMultiModalProcessor[TS2QwenProcessingInfo]):
    """Multimodal processor for TS2Qwen."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Ensure the special token is added BEFORE tokenization
        # This is critical - the token must exist in the vocab before encode()
        _ = self.info.get_timeseries_token_id()

        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt)

        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        if "timeseries" in mm_data:
            ts_data = mm_data["timeseries"]
            if isinstance(ts_data, torch.Tensor):
                # Single tensor - ensure it has batch dimension
                if ts_data.ndim == 2:
                    ts_data = ts_data.unsqueeze(0)
                result["timeseries"] = ts_data
            elif isinstance(ts_data, list):
                # List of tensors - stack them into a batch
                tensors = []
                for item in ts_data:
                    if isinstance(item, torch.Tensor):
                        tensors.append(item)
                    else:
                        tensors.append(torch.tensor(item, dtype=torch.float32))
                if tensors:
                    result["timeseries"] = torch.stack(tensors, dim=0)
            else:
                result["timeseries"] = torch.tensor(ts_data, dtype=torch.float32)

        return result

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _ts2qwen_field_config(hf_inputs)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """
        Return False because our _call_hf_processor just tokenizes the prompt
        without applying any placeholder replacements.

        This tells vLLM to apply the target→replacement substitution itself
        using the PromptReplacement rules from _get_prompt_updates.
        """
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """
        Define how `<|timeseries_pad|>` placeholders in the prompt are replaced
        with feature placeholder tokens.

        Single Token Strategy (following Qwen3-VL pattern)
        --------------------------------------------------
        We use a single special token `<|timeseries_pad|>` as the placeholder,
        similar to how Qwen3-VL uses `<|image_pad|>` (token ID 151655).

        This approach has several advantages over multi-token sequences:
        1. Single token is reliably detected regardless of surrounding context
        2. No BPE tokenization boundary issues
        3. Follows the established vLLM multimodal pattern

        The replacement consists of `num_soft_tokens` copies of the same token,
        which will be replaced with actual timeseries embeddings by
        `embed_multimodal()` during the forward pass.
        """
        # Get the number of soft tokens (= number of channels in timeseries)
        ts_config = self.info.get_ts_config()
        num_soft_tokens = ts_config.num_soft_tokens

        # Get the single special token ID for timeseries
        # This also adds the token to the tokenizer if not present
        ts_token_id = self.info.get_timeseries_token_id()

        def get_replacement(item_idx: int) -> list[int]:
            # Each timeseries item maps to num_soft_tokens placeholder tokens
            # We reuse the same timeseries token ID for the replacements
            return [ts_token_id] * num_soft_tokens

        return [
            PromptReplacement(
                modality="timeseries",
                target=[ts_token_id],  # Single token ID
                replacement=get_replacement,
            ),
        ]


# ============================================================================
# Main Model Class
# ============================================================================
@MULTIMODAL_REGISTRY.register_processor(
    TS2QwenMultiModalProcessor,
    info=TS2QwenProcessingInfo,
    dummy_inputs=TS2QwenDummyInputsBuilder,
)
class TS2QwenForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    """
    vLLM-compatible Timeseries + Qwen Model.

    This model wraps Qwen2 and adds timeseries processing capability.
    Time-series data is encoded and prepended as soft tokens.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "timeseries":
            return None
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.ts_config = _get_ts_config(config)
        self.llm_hidden_size = getattr(config, "hidden_size", 4096)

        # Initialize TS encoder
        self.ts_encoder = MultichannelTSEncoder(self.ts_config)
        self.ts_soft_token_layer = nn.Linear(
            self.ts_config.flatten_dim,
            self.llm_hidden_size,
        )

        for p in self.ts_encoder.parameters():
            p.requires_grad = False

        # Initialize the language model
        # Create a modified config that points to the base Qwen model,
        # not TS2QwenForCausalLM (to avoid infinite recursion)
        import copy
        llm_config = copy.deepcopy(config)
        # Determine the base model architecture from the config type
        config_class_name = type(config).__name__
        if "Qwen3" in config_class_name or "Qwen3" in str(getattr(config, "model_type", "")):
            llm_config.architectures = ["Qwen3ForCausalLM"]
        elif "Qwen2" in config_class_name or "Qwen2" in str(getattr(config, "model_type", "")):
            llm_config.architectures = ["Qwen2ForCausalLM"]
        else:
            # Default to Qwen2
            llm_config.architectures = ["Qwen2ForCausalLM"]

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=llm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        logger.info(
            f"Initialized TS2QwenForCausalLM: "
            f"ts_input_len={self.ts_config.ts_input_len}, "
            f"channels={self.ts_config.ts_in_channels}, "
            f"soft_tokens={self.ts_config.num_soft_tokens}"
        )

    def _encode_timeseries(self, timeseries: torch.Tensor) -> torch.Tensor:
        """Encode time-series data to soft tokens."""
        B = timeseries.shape[0]
        # Match input dtype to encoder's dtype (vLLM may convert model to float16)
        encoder_dtype = next(self.ts_encoder.parameters()).dtype
        if timeseries.dtype != encoder_dtype:
            timeseries = timeseries.to(encoder_dtype)
        ts_embeddings = self.ts_encoder.embedding(timeseries)
        _, C, N, D = ts_embeddings.shape
        ts_embeddings = ts_embeddings.reshape(B, C, -1)
        ts_embeddings = F.gelu(ts_embeddings)
        soft_tokens = self.ts_soft_token_layer(ts_embeddings)
        return soft_tokens

    def get_language_model(self) -> nn.Module:
        """Returns the underlying language model for text generation."""
        return self.language_model

    def _parse_and_validate_timeseries_input(
        self, **kwargs: object
    ) -> torch.Tensor | None:
        """Parse and validate timeseries input from kwargs."""
        timeseries = kwargs.pop("timeseries", None)
        ts_embeds = kwargs.pop("ts_embeds", None)

        if timeseries is None and ts_embeds is None:
            return None

        # If pre-computed embeddings are provided, return them directly
        if ts_embeds is not None:
            if isinstance(ts_embeds, torch.Tensor):
                return ts_embeds
            return torch.tensor(ts_embeds, dtype=torch.float32)

        # Process raw timeseries data
        if isinstance(timeseries, torch.Tensor):
            return timeseries
        return torch.tensor(timeseries, dtype=torch.float32)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """
        Returns multimodal embeddings (soft tokens) generated from timeseries data.

        The returned embeddings are prepended to text embeddings during forward pass.
        """
        ts_input = self._parse_and_validate_timeseries_input(**kwargs)
        if ts_input is None:
            return []

        # Handle pre-computed embeddings
        if ts_input.dim() == 3 and ts_input.shape[-1] == self.llm_hidden_size:
            # Already embedded (ts_embeds case)
            return ts_input

        # Encode raw timeseries to soft tokens
        ts_input = ts_input.to(dtype=torch.float32)
        if ts_input.dim() == 2:
            ts_input = ts_input.unsqueeze(0)

        soft_tokens = self._encode_timeseries(ts_input)
        return soft_tokens

    def _process_timeseries_input(
        self,
        timeseries: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Process timeseries and prepend soft tokens to inputs_embeds."""
        if timeseries is None:
            return inputs_embeds

        timeseries = timeseries.to(dtype=torch.float32, device=inputs_embeds.device)
        if timeseries.dim() == 2:
            timeseries = timeseries.unsqueeze(0)

        soft_tokens = self._encode_timeseries(timeseries)
        soft_tokens = soft_tokens.to(dtype=inputs_embeds.dtype)

        return torch.cat([soft_tokens, inputs_embeds], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass."""
        timeseries = kwargs.pop("timeseries", None)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if timeseries is not None:
            inputs_embeds = self._process_timeseries_input(timeseries, inputs_embeds)
            num_soft_tokens = self.ts_config.num_soft_tokens
            positions = positions + num_soft_tokens

        return self.language_model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    # NOTE: We inherit embed_input_ids from SupportsMultiModal which handles
    # merging multimodal embeddings. It uses get_language_model().embed_input_ids
    # for text tokens and _merge_multimodal_embeddings for multimodal tokens.

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with proper mapping.

        The Qwen checkpoint has weights like 'model.layers...' and 'lm_head...'
        but our model structure has these under 'language_model.', so we
        need to remap the names.

        TS encoder weights are randomly initialized (not from checkpoint).
        """
        # Remap weight names to add 'language_model.' prefix for LLM weights
        def remap_weights():
            for name, tensor in weights:
                # TS encoder weights (if present in checkpoint)
                if name.startswith("ts_encoder.") or name.startswith("ts_soft_token_layer."):
                    yield name, tensor
                else:
                    # LLM weights - add language_model prefix
                    yield f"language_model.{name}", tensor

        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(remap_weights())

        # Add TS encoder and ts_soft_token_layer parameter names to indicate
        # they are "handled" (randomly initialized, not loaded from checkpoint)
        # This prevents vLLM from raising "weights not initialized" error
        ts_param_names = {
            name for name, _ in self.named_parameters()
            if name.startswith("ts_encoder.") or name.startswith("ts_soft_token_layer.")
        }
        loaded_weights.update(ts_param_names)

        return loaded_weights
