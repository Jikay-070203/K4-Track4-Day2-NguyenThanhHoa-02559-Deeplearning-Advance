"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện:
    build_criterion(kind, **kw)            -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)            -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)           -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets) -> loss scalar
Các kiểm tra tự viết (focal gamma=0 == CE, LS eps=0 == CE, CutMix lam == diện tích thật) nằm ở `self_test()`
và được notebook gọi trước khi chạy thí nghiệm.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Cài đặt tay: loss = (1 - eps) * NLL + eps * mean_k(-log p_k). eps = 0 cho đúng cross-entropy.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp (độ dài K). gamma = 0 và alpha = None cho đúng cross-entropy.
    Trung bình theo batch (không chia thêm cho tổng alpha).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t).clamp(min=0.0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (không dùng val/test).

    - beta = 0: w_c = 1 / n_c, chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng (Cui et al., arXiv:1901.05555):
                w_c = (1 - beta) / (1 - beta ** n_c), chuẩn hoá tổng trọng số về số lớp
    """
    n = np.asarray(counts, dtype=np.float64)
    assert (n > 0).all(), "mọi lớp phải có ít nhất một ảnh trong train"
    if beta == 0.0:
        w = 1.0 / n
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w * len(n) / w.sum()
    return torch.as_tensor(w, dtype=torch.float32)


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    kw: smoothing=0.1, gamma=2.0, alpha=None (cho focal), weight=tensor (cho ce_weighted).
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần weight=class_weights(...)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(kw["weight"], dtype=torch.float32))
    raise ValueError(f"loss không hợp lệ: {kind!r}")


def _rand_box(h: int, w: int, lam: float):
    cut = math.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = int(np.clip(cy - ch // 2, 0, h)), int(np.clip(cy + ch // 2, 0, h))
    x1, x2 = int(np.clip(cx - cw // 2, 0, w)), int(np.clip(cx + cw // 2, 0, w))
    return y1, y2, x1, x2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh (N, C, H, W) và nhãn. lam ~ Beta(alpha, alpha).

    - mixup : x_mix = lam * x + (1 - lam) * x[perm]
    - cutmix: dán một hộp chữ nhật của x[perm] vào x; lam được TÍNH LẠI theo diện tích thật của hộp
              sau khi cắt ra ngoài biên (slide trang 48)
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]. Dùng RNG của numpy/torch đã được set_seed.
    """
    if alpha <= 0:
        raise ValueError("alpha phải > 0")
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = _rand_box(h, w, lam)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / float(h * w)
    else:
        raise ValueError(f"mode không hợp lệ: {mode!r}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits: torch.Tensor, targets) -> torch.Tensor:
    """lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b) cho batch đã trộn."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def self_test(verbose: bool = True) -> dict:
    """Các kiểm tra tự viết cho những phần dễ sai (RUBRIC mục H). Ném AssertionError nếu sai."""
    torch.manual_seed(0)
    np.random.seed(0)
    logits = torch.randn(64, 9) * 3
    y = torch.randint(0, 9, (64,))
    ce = F.cross_entropy(logits, y)
    res = {}
    res["focal(gamma=0) - CE"] = abs(FocalLoss(0.0)(logits, y) - ce).item()
    res["LS(eps=0) - CE"] = abs(LabelSmoothingCE(0.0)(logits, y) - ce).item()
    res["LS(eps=0.1) - torch"] = abs(LabelSmoothingCE(0.1)(logits, y)
                                     - F.cross_entropy(logits, y, label_smoothing=0.1)).item()
    cw = class_weights([100, 100, 100, 100, 100, 100, 100, 100, 100])
    res["ce_weighted(đều) - CE"] = abs(build_criterion("ce_weighted", weight=cw)(logits, y) - ce).item()
    res["focal(gamma=2) <= CE"] = float(FocalLoss(2.0)(logits, y) <= ce)
    x = torch.randn(8, 3, 32, 32)
    yy = torch.arange(8)
    _, (_, _, lam) = mix_batch(x, yy, 1.0, "cutmix")
    # diện tích thật: dán x[perm] -> số điểm ảnh khác x phải <= (1 - lam) * H * W
    np.random.seed(1)
    torch.manual_seed(1)
    xm, (ya, yb, lam) = mix_batch(x, yy, 1.0, "cutmix")
    changed = (xm != x).any(1).float().mean((1, 2))            # tỉ lệ điểm ảnh đổi, mỗi ảnh
    res["cutmix: max(đổi) <= 1 - lam"] = float(changed.max().item() <= (1 - lam) + 1e-6)
    xm, (ya, yb, lam) = mix_batch(x, yy, 1.0, "mixup")
    res["mixup: 0 <= lam <= 1"] = float(0.0 <= lam <= 1.0)
    crit = build_criterion("ce")
    res["mixed_loss(lam=1) - CE"] = abs(mixed_loss(crit, logits, (y, y.roll(1), 1.0)) - ce).item()
    ok = (res["focal(gamma=0) - CE"] < 1e-6 and res["LS(eps=0) - CE"] < 1e-6 and res["LS(eps=0.1) - torch"] < 1e-6
          and res["ce_weighted(đều) - CE"] < 1e-6 and res["focal(gamma=2) <= CE"] == 1.0
          and res["cutmix: max(đổi) <= 1 - lam"] == 1.0 and res["mixup: 0 <= lam <= 1"] == 1.0
          and res["mixed_loss(lam=1) - CE"] < 1e-6)
    assert ok, res
    if verbose:
        for k, v in res.items():
            print(f"  {k:<32} {v:.2e}" if v not in (0.0, 1.0) or "<=" not in k else f"  {k:<32} OK")
        print("self_test losses: OK")
    return res
