#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_shape(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x)


def metrics(ref: np.ndarray, other: np.ndarray) -> dict[str, object]:
    diff = np.abs(ref.astype(np.float64) - other.astype(np.float64))
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-12)
    return {
        "shape_match": list(ref.shape) == list(other.shape),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "max_rel_diff": float((diff / denom).max()) if diff.size else 0.0,
        "cosine_similarity": float(np.dot(ref.reshape(-1), other.reshape(-1)) / ((np.linalg.norm(ref) * np.linalg.norm(other)) + 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Android debug .bin tensors with NumPy references.")
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--android", type=Path, required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--name", default="tensor")
    args = parser.parse_args()

    shape = parse_shape(args.shape)
    ref = np.load(args.ref).reshape(shape)
    android = np.fromfile(args.android, dtype=np.dtype(args.dtype)).reshape(shape)
    report = {"name": args.name, "metrics": metrics(ref, android)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
