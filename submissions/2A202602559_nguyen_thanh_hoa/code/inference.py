"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md) và dự đoán test đúng một lần (Bước 4).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra (trang 68),
temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val; nhiệt độ T khớp trên VAL rồi
áp dụng sang test (README.md, S2 và S4).

Quy ước "view": hàm biến đổi một batch ảnh ĐÃ CHUẨN HOÁ (N, 3, 256, 256) thành (N, 3, h, w). Center-crop,
lật, 5-crop, resize đều giao hoán với phép chuẩn hoá theo kênh nên làm sau chuẩn hoá vẫn chính xác.

Giao diện chính:
    predict_logits(model, X_u8, device, normalizer, view, ...) -> logits[N, 9] (numpy)
    make_views(kind, base)         -> list[(tên, view)]
    aggregate_views(list_logits, space) -> probs[N, 9]
    fit_temperature(val_logits, val_labels) / apply_temperature(logits, T)
    ensemble_probs(list_of_probs)
    fuse_conv_bn(model)            -> model đã gộp BN vào conv (kiểm tra sai số)
    load_run_model(cfg, device)    -> (model, normalizer) từ checkpoint tốt nhất của một run
    test_views_once(cfg, ...)      -> logits test theo view; mỗi checkpoint chỉ được quét test MỘT lần
"""
from __future__ import annotations

import contextlib
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def amp_ctx(device, enabled: bool):
    """autocast FP16 trên CUDA khi enabled; ngược lại không làm gì (CPU không hỗ trợ autocast FP16)."""
    if enabled and getattr(device, "type", str(device)) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------- các view (trên GPU)
def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_center(size: int):
    """Center-crop về size x size (nếu size >= kích thước ảnh thì giữ nguyên)."""
    def f(x):
        h, w = x.shape[-2:]
        if size >= min(h, w):
            return x
        t, l = (h - size) // 2, (w - size) // 2
        return x[..., t:t + size, l:l + size]
    f.__name__ = f"center{size}"
    return f


def view_hflip(size: int | None = None):
    """Center-crop (nếu có size) rồi lật ngang bằng torch.flip trên chiều rộng (slide trang 75)."""
    base = view_center(size) if size else view_identity

    def f(x):
        return torch.flip(base(x), dims=[-1])
    f.__name__ = f"hflip{size or ''}"
    return f


def view_corner(size: int, corner: str, flip: bool = False):
    """Crop size x size ở một góc (tl, tr, bl, br) hoặc giữa (c), tuỳ chọn lật ngang."""
    def f(x):
        h, w = x.shape[-2:]
        s = min(size, h, w)
        top = {"tl": 0, "tr": 0, "bl": h - s, "br": h - s, "c": (h - s) // 2}[corner]
        left = {"tl": 0, "tr": w - s, "bl": 0, "br": w - s, "c": (w - s) // 2}[corner]
        out = x[..., top:top + s, left:left + s]
        return torch.flip(out, dims=[-1]) if flip else out
    f.__name__ = f"crop_{corner}{'_flip' if flip else ''}"
    return f


def view_resize(size: int):
    """Resize toàn bộ ảnh về size x size (nội suy song tuyến tính; antialias khi thu nhỏ)."""
    def f(x):
        h = x.shape[-1]
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=size < h)
    f.__name__ = f"resize{size}"
    return f


def views_multicrop(x: torch.Tensor, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`; flip=True thêm bản lật (10 view). Trả về list batch."""
    out = [view_corner(crop, c)(x) for c in ("tl", "tr", "bl", "br", "c")]
    if flip:
        out += [view_corner(crop, c, True)(x) for c in ("tl", "tr", "bl", "br", "c")]
    return out


def views_multiscale(x: torch.Tensor, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list batch.

    CNN có global pooling chấp nhận ảnh khác kích thước lúc train; ViT/Swin có embedding vị trí/cửa sổ cố định
    nên KHÔNG áp dụng (xem model.supports_variable_res).
    """
    return [view_resize(s)(x) for s in sizes]


def make_views(kind: str, base: int = 224):
    """Danh sách (tên, view) cho từng kiểu TTA (đều là view 1 ảnh -> 1 ảnh, lấy từ ảnh 256x256 gốc):

      "identity"  : [center{base}]                               K = 1 (mốc I00)
      "hflip"     : [center{base}, hflip{base}]                  K = 2 (I01)
      "fivecrop"  : 4 góc + giữa, mỗi crop base x base           K = 5 (I02)
      "tencrop"   : fivecrop + bản lật                           K = 10
    """
    if kind == "identity":
        return [(f"center{base}", view_center(base))]
    if kind == "hflip":
        return [(f"center{base}", view_center(base)), (f"hflip{base}", view_hflip(base))]
    if kind in ("fivecrop", "tencrop"):
        flips = (False, True) if kind == "tencrop" else (False,)
        return [(f"crop_{c}{'_flip' if fl else ''}", view_corner(base, c, fl))
                for fl in flips for c in ("tl", "tr", "bl", "br", "c")]
    raise ValueError(f"kiểu view không hợp lệ: {kind!r}")


# --------------------------------------------------------------------------- dự đoán logit
@torch.inference_mode()
def predict_logits(model, X: torch.Tensor, device, normalizer, view=None, bs: int = 128, amp: bool = False,
                   half: bool = False, channels_last: bool = False) -> np.ndarray:
    """Chạy model trên X (uint8, N x 3 x 256 x 256, giữ thứ tự file) và gom logit float32 (N, 9).

    view: hàm biến đổi batch đã chuẩn hoá (hoặc None). amp: autocast FP16; half: model đã .half() và đầu vào
    ép half. model.eval() được gọi ở đây.
    """
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = normalizer(X[i:i + bs].to(device, non_blocking=True))
        if view is not None:
            xb = view(xb)
        if half:
            xb = xb.half()
        if channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
        with amp_ctx(device, bool(amp)):
            lg = model(xb)
        out.append(lg.float().cpu())
    return torch.cat(out).numpy()


def softmax_np(logits, T: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob", T: float = 1.0) -> np.ndarray:
    """Gộp K lượt chạy của TTA thành xác suất (N, 9) đã chuẩn hoá (slide trang 62).

      - space="prob":  trung bình softmax(logit / T) của từng view
      - space="logit": trung bình logit rồi softmax(. / T)
    """
    L = [np.asarray(l, dtype=np.float64) for l in logits_per_view]
    if space == "prob":
        return np.mean([softmax_np(l, T) for l in L], axis=0)
    if space == "logit":
        return softmax_np(np.mean(L, axis=0), T)
    raise ValueError(f"space phải là 'prob' hoặc 'logit', nhận {space!r}")


def ensemble_probs(list_of_probs) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình trên CÙNG tập ảnh và cùng thứ tự file."""
    return np.mean([np.asarray(p, dtype=np.float64) for p in list_of_probs], axis=0)


# --------------------------------------------------------------------------- temperature scaling
def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())


def _fit_scalar(nll_of_T, lo: float = 0.05, hi: float = 20.0) -> float:
    """Cực tiểu hoá nll_of_T(T) trên [lo, hi]: lưới thô theo log rồi tìm tam phân tinh (không cần scipy)."""
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 80))
    vals = [nll_of_T(t) for t in grid]
    k = int(np.argmin(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    for _ in range(60):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        if nll_of_T(m1) < nll_of_T(m2):
            b = m2
        else:
            a = m1
    return float((a + b) / 2)


def fit_temperature(val_logits, val_labels) -> float:
    """Nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    L = np.asarray(val_logits, dtype=np.float64)
    y = np.asarray(val_labels, dtype=np.int64)
    return _fit_scalar(lambda t: _nll(softmax_np(L, t), y))


def fit_temperature_views(val_logits_per_view, val_labels, space: str = "logit") -> float:
    """Như fit_temperature nhưng cho dự đoán đã gộp nhiều view (tối thiểu NLL của xác suất đã gộp)."""
    y = np.asarray(val_labels, dtype=np.int64)
    return _fit_scalar(lambda t: _nll(aggregate_views(val_logits_per_view, space, t), y))


def apply_temperature(logits, T: float) -> np.ndarray:
    """softmax(logits / T)."""
    return softmax_np(logits, T)


# --------------------------------------------------------------------------- gộp BatchNorm vào conv
def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.size(0), device=w.device, dtype=w.dtype)
    inv = torch.rsqrt(bn.running_var + bn.eps)
    gamma = bn.weight.detach() if bn.weight is not None else torch.ones_like(inv)
    beta = bn.bias.detach() if bn.bias is not None else torch.zeros_like(inv)
    scale = gamma * inv
    new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode).to(w.device, w.dtype)
    new.weight.data.copy_(w * scale.reshape(-1, 1, 1, 1))
    new.bias.data.copy_(beta + (b - bn.running_mean) * scale)
    return new


def _fuse_children(module: nn.Module) -> int:
    """Tìm cặp (Conv2d, BatchNorm2d) liền kề trong cùng module cha và gộp lại; đệ quy xuống module con."""
    count = 0
    keys = list(module._modules.keys())
    i = 0
    while i < len(keys) - 1:
        a, b = module._modules[keys[i]], module._modules[keys[i + 1]]
        if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d) and a.out_channels == b.num_features:
            module._modules[keys[i]] = _fold(a, b)
            act = getattr(b, "act", None)        # timm BatchNormAct2d: giữ lại phần kích hoạt
            module._modules[keys[i + 1]] = act if (act is not None and not isinstance(act, nn.Identity)) else nn.Identity()
            count += 1
            i += 2
            continue
        i += 1
    for child in list(module.children()):
        count += _fuse_children(child)
    return count


def fuse_conv_bn(model, img_size: int = 224, tol: float = 1e-3, verbose: bool = True):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Trả về BẢN SAO model đã gộp (`model.n_fused` = số cặp). Kiểm tra: đầu ra trước/sau gộp lệch tối đa
    `err` trên đầu vào ngẫu nhiên; nếu err > tol thì ném lỗi. Kiến trúc không có BN (ViT, Swin, ConvNeXt dùng
    LayerNorm) cho n_fused = 0: mục này không áp dụng.
    """
    m = copy.deepcopy(model).eval()
    dev = next(m.parameters()).device
    x = torch.randn(2, 3, img_size, img_size, device=dev)
    with torch.inference_mode():
        ref = m(x)
    n = _fuse_children(m)
    with torch.inference_mode():
        out = m(x)
    err = float((ref - out).abs().max())
    if verbose:
        print(f"[fuse_conv_bn] gộp {n} cặp Conv+BN, sai số lớn nhất của logit = {err:.2e}")
    if err > tol:
        raise RuntimeError(f"gộp BN lệch quá lớn ({err:.2e} > {tol}); kiến trúc này không gộp an toàn bằng cách này")
    m.n_fused = n
    return m


# --------------------------------------------------------------------------- checkpoint và test một lần
def load_run_model(cfg, device):
    """Dựng lại model của một run (không tải trọng số tiền huấn luyện) và nạp checkpoint tốt nhất."""
    import model as mdl
    import dataset as ds
    from train import run_dir
    m = mdl.build_model(cfg.backbone, pretrained=False, num_classes=ds.NUM_CLASSES, drop_rate=cfg.drop_rate,
                        init="finetune", drop_path_rate=cfg.drop_path_rate)
    state = torch.load(run_dir(cfg) / "best.pt", map_location="cpu")
    m.load_state_dict(state)
    m.to(device).eval()
    return m, ds.Normalizer(m.norm_mean, m.norm_std)


def _prepare(model, prec: str):
    """Trả về (model_dùng_để_chạy, half, amp, channels_last) cho một kiểu số học."""
    if prec == "fp32":
        return model, False, False, False
    if prec == "amp":
        return model, False, True, False
    if prec == "fp16":
        return copy.deepcopy(model).half(), True, False, False
    if prec == "fused32":
        return fuse_conv_bn(model, verbose=False), False, False, False
    if prec == "fused16":
        return fuse_conv_bn(model, verbose=False).half(), True, False, False
    raise ValueError(f"prec không hợp lệ: {prec!r}")


def test_views_once(cfg, model, X_test: torch.Tensor, device, normalizer, view_specs: dict,
                    bs: int = 128) -> dict:
    """Quét tập TEST đúng MỘT lần cho checkpoint của `cfg` và trả về logit theo từng view.

    view_specs: {tên: (view, prec)} với prec thuộc fp32 | amp | fp16 | fused32 | fused16. Toàn bộ view cần dùng
    (ví dụ center, hflip, và biến thể số học của cấu hình thời gian thực) phải được khai báo ngay lúc gọi này.
    Kết quả lưu vào run_dir/test_views.npz; file cờ TEST_USED.json chặn mọi lần quét thứ hai (README S4).
    Gọi lại hàm này khi đã có kết quả thì chỉ ĐỌC file đã lưu, không chạy lại model trên test.
    """
    from train import run_dir
    rd = run_dir(cfg)
    flag, npz = rd / "TEST_USED.json", rd / "test_views.npz"
    if npz.exists():
        data = np.load(npz)
        missing = [k for k in view_specs if k not in data.files]
        if missing:
            raise RuntimeError(f"test đã được quét cho {rd} nhưng thiếu view {missing}; không được quét lại (S4)")
        return {k: data[k] for k in view_specs}
    if flag.exists():
        raise RuntimeError(f"{flag} đã tồn tại mà không có {npz.name}: không quét test lần hai (S4)")
    out = {}
    for name, (view, prec) in view_specs.items():
        m, half, amp, cl = _prepare(model, prec)
        out[name] = predict_logits(m, X_test, device, normalizer, view, bs=bs, amp=amp, half=half, channels_last=cl)
    flag.write_text(json.dumps({"exp_id": cfg.exp_id, "seed": cfg.seed, "views": list(view_specs),
                                "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    np.savez_compressed(npz, **out)
    return out
