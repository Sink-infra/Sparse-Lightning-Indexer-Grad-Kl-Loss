"""Logical dtype conversion shared by generators, references, and verifiers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def float32_to_bfloat16_bits(value: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round-to-nearest-even), returned as uint16."""

    source = np.asarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def bfloat16_bits_to_float32(value: np.ndarray) -> np.ndarray:
    """Decode uint16 bfloat16 bit patterns into float32."""

    bits = np.asarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def encode_logical(value: np.ndarray, logical_dtype: str) -> np.ndarray:
    if logical_dtype == "float16":
        return np.asarray(value, dtype=np.float16)
    if logical_dtype == "bfloat16":
        return float32_to_bfloat16_bits(value)
    if logical_dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    raise ValueError(f"unsupported logical dtype: {logical_dtype}")


def decode_logical(value: np.ndarray, logical_dtype: str) -> np.ndarray:
    if logical_dtype == "bfloat16":
        return bfloat16_bits_to_float32(value)
    return np.asarray(value, dtype=np.float32)


def zeros_logical(shape: Tuple[int, ...], logical_dtype: str) -> np.ndarray:
    if logical_dtype == "float16":
        return np.zeros(shape, dtype=np.float16)
    if logical_dtype == "bfloat16":
        return np.zeros(shape, dtype=np.uint16)
    if logical_dtype == "float32":
        return np.zeros(shape, dtype=np.float32)
    raise ValueError(f"unsupported logical dtype: {logical_dtype}")
