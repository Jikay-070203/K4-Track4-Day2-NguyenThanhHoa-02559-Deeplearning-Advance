"""model.py - tạo backbone (timm), đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init, drop_path_rate) -> nn.Module
    freeze_backbone(model) / set_train_mode(model)
    param_groups(model, lr_backbone, lr_head, weight_decay)                    -> list[dict] cho optimizer
    count_params(model) -> float (triệu)        count_gmacs(model, img_size) -> float
Model có thêm thuộc tính: `weight_tag` (tag trọng số thực sự được tải), `norm_mean`, `norm_std`, `frozen`.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản: ghi lại tag thực tế.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}
# Mạng chấp nhận ảnh có kích thước khác lúc train (có global pooling, không có embedding vị trí cố định).
VARIABLE_RES_KEYS = ("resnet", "resnext", "convnext", "efficientnet", "mobilenet", "regnet")
LIGHT_KEYS = ("efficientnet", "mobilenet")


def supports_variable_res(name: str) -> bool:
    return any(k in name for k in VARIABLE_RES_KEYS)


def is_light(name: str) -> bool:
    return any(k in name for k in LIGHT_KEYS)


def _weight_tag(model, name: str, pretrained: bool) -> str:
    if not pretrained:
        return f"{name} (khởi tạo ngẫu nhiên, không tiền huấn luyện)"
    cfg = dict(getattr(model, "pretrained_cfg", None) or {})
    tag = cfg.get("hf_hub_id") or cfg.get("url") or cfg.get("file") or cfg.get("tag") or "không rõ tag"
    return f"{name} ({tag})"


def build_model(name: str, pretrained: bool = True, num_classes: int = 9, drop_rate: float = 0.0,
                init: str = "finetune", drop_path_rate: float = 0.0):
    """Tạo model phân loại 9 lớp bằng timm (head mới khởi tạo ngẫu nhiên).

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : không tiền huấn luyện, huấn luyện toàn bộ
      - "frozen"   : tiền huấn luyện ImageNet, đóng băng backbone, chỉ train head
      - "finetune" : tiền huấn luyện ImageNet, train toàn bộ (công thức nền)
    `pretrained=False` buộc không tải trọng số (dùng khi nạp checkpoint đã huấn luyện, hoặc chạy thử).
    """
    import timm
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init!r}")
    use_pre = bool(pretrained) and init != "scratch"
    kw = dict(pretrained=use_pre, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate and drop_path_rate > 0:
        kw["drop_path_rate"] = drop_path_rate
    model = timm.create_model(name, **kw)
    cfg = dict(getattr(model, "pretrained_cfg", None) or {})
    model.norm_mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
    model.norm_std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    model.weight_tag = _weight_tag(model, name, use_pre)
    model.backbone_name = name
    model.frozen = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def head_params(model) -> list:
    return list(model.get_classifier().parameters())


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    Backbone đóng băng thì BatchNorm cũng phải ở eval (thống kê chạy không được cập nhật): xem
    `set_train_mode`, gọi ở đầu MỖI epoch thay cho model.train().
    """
    head_ids = {id(p) for p in head_params(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.frozen = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone bị đóng băng thì giữ backbone ở eval và chỉ để head ở train."""
    if getattr(model, "frozen", False):
        model.eval()
        model.get_classifier().train()
    else:
        model.train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành các nhóm như slide Day 2, trang 52 (bỏ tham số requires_grad == False):

    - backbone ndim > 1          : lr_backbone, weight_decay
    - norm và bias (ndim <= 1)   : lr_backbone, weight_decay = 0   (cả backbone lẫn head)
    - head ndim > 1              : lr_head,     weight_decay
    - bias của head              : lr_head,     weight_decay = 0
    """
    head_ids = {id(p) for p in head_params(model)}
    buckets = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        side = "head" if id(p) in head_ids else "backbone"
        buckets[f"{side}_{'decay' if p.ndim > 1 else 'no_decay'}"].append(p)
    spec = [("backbone_decay", lr_backbone, weight_decay), ("backbone_no_decay", lr_backbone, 0.0),
            ("head_decay", lr_head, weight_decay), ("head_no_decay", lr_head, 0.0)]
    return [{"params": buckets[k], "lr": lr, "weight_decay": wd, "name": k} for k, lr, wd in spec if buckets[k]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (MAC, không phải FLOPs = 2 x MAC).

    Công cụ: torch.utils.flop_counter.FlopCounterMode (có sẵn trong PyTorch >= 2.1) trên bản sao CPU của
    model; tắt fused attention để các phép matmul của attention được đếm. Chỉ đếm conv/matmul/linear
    (bỏ qua các phép phần tử và softmax), nên có thể lệch vài phần trăm so với fvcore/ptflops.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
        m = copy.deepcopy(model).cpu().eval().float()
        for mod in m.modules():
            if hasattr(mod, "fused_attn"):
                mod.fused_attn = False
        x = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(x)
        return float(fc.get_total_flops()) / 2.0 / 1e9
    except Exception as e:  # noqa: BLE001 - không để việc đếm GMAC làm hỏng cả lần chạy
        print(f"[count_gmacs] không đếm được GMAC: {type(e).__name__}: {e}")
        return float("nan")
