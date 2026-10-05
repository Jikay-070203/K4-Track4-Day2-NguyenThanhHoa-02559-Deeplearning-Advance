"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (fp32/amp/fp16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý: chỉ đo phần model (forward) trên tensor đã nằm trên thiết bị (ghi rõ trong báo cáo)
"""
from __future__ import annotations

import contextlib
import copy
import time

import numpy as np
import torch

STRICT = True   # True: bắt buộc warmup >= 10 và >= 50 lần đo (GUIDE mục 4.1). Chỉ đặt False khi chạy thử code.


def _amp(device: str, enabled: bool):
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây: p50, p95, p99, mean, n."""
    if STRICT:
        assert warmup >= 10 and iters >= 50, "cần warmup >= 10 và iters >= 50 (GUIDE mục 4.1)"
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()), "n": int(iters)}


def _gpu_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "CPU"


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 20, iters: int = 100, bn_fused: bool = False, label: str = "") -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast FP16) | "fp16" (model.half() trên bản sao).
    Trả về dict ghi thẳng được vào sheet `Latency`.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    m = copy.deepcopy(model).eval().to(device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        m, x = m.half(), x.half()
    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    use_amp = dtype == "amp" and device.startswith("cuda")

    def fn():
        with torch.inference_mode(), _amp(device, use_amp):
            m(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    del m
    return {"label": label, "gpu": _gpu_name(device), "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "bn_fused": bool(bn_fused), **r, "images_per_s": batch_size / (r["p50"] / 1000.0),
            "torch": torch.__version__}


def tta_latency(model, k_views: int, batch_size: int = 1, img_size: int = 224, dtype: str = "fp32",
                device: str = "cuda", warmup: int = 10, iters: int = 50, label: str = "") -> dict:
    """Độ trễ của TTA K view chạy tuần tự (K lượt forward); trả kèm K * p50 của một lượt để đối chiếu."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    m = copy.deepcopy(model).eval().to(device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        m, x = m.half(), x.half()
    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    use_amp = dtype == "amp" and device.startswith("cuda")

    def fn():
        with torch.inference_mode(), _amp(device, use_amp):
            for _ in range(k_views):
                m(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    one = latency_report(model, batch_size, img_size, dtype, device, warmup=warmup, iters=max(iters, 50))
    del m
    return {"label": label, "gpu": _gpu_name(device), "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "k_views": k_views, **r, "k_times_single_p50": k_views * one["p50"], "single_p50": one["p50"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__}
