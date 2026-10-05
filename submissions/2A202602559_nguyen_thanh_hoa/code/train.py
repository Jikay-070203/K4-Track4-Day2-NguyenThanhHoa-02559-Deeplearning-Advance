"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F): MỘT hàm `run(cfg)`, đổi thí nghiệm bằng Config.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc, để cùng định
nghĩa với lúc chấm (cần thư mục chứa eval.py nằm trong sys.path).

Quy ước lưu trữ (mỗi lần chạy): <out_dir>/<exp_id>/seed<k>/
    config.json, history.csv, lr_steps.npy, curves.png, best.pt (checkpoint tốt nhất theo macro-F1 val),
    val_logits.npy (FP32, tại epoch tốt nhất), summary.json (đánh dấu lần chạy đã xong -> có thể nối tiếp).
Test KHÔNG được dùng ở đây (S4); chỉ inference.test_views_once đụng test, đúng một lần cho mỗi checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import random
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import dataset as ds
import inference as inf
import losses as ls
import model as mdl
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    mota: str = ""                    # mô tả ngắn cho tên file ảnh curves/<exp_id>_<mota>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    pretrained: bool = True           # False: không tải trọng số (chỉ dùng khi chạy thử code)
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0       # stochastic depth (trục F)
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | vflip | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.1      # chỉ dùng khi loss == "ls"
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None   # loss == "ce_weighted": 0 -> 1/n_c; >0 -> class-balanced
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    cache_dir: str = "cache"          # mảng ảnh uint8 giải mã sẵn (ngoài git)
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ để chạy thử nhanh ---
    max_train_images: int | None = None
    max_eval_images: int | None = None
    # --- chỉ bật ở Bước 4 (chung kết). Mặc định TẮT (quy tắc S4): ghi dự đoán test 1 view đúng một lần ---
    save_test_predictions: bool = False
    save_val_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def cfg_hash(cfg: Config) -> str:
    """Băm các trường ảnh hưởng tới kết quả huấn luyện (bỏ đường dẫn, tên, cờ ghi file)."""
    skip = {"mota", "images_dir", "labels_dir", "cache_dir", "out_dir", "pred_dir", "curves_dir",
            "save_test_predictions", "save_val_predictions", "num_workers"}
    d = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in skip}
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU và CUDA).

    cudnn.benchmark = True để nhanh hơn nên KHÔNG tái lập từng bit trên GPU (các thuật toán conv có thể khác
    giữa các lần chạy); seed cố định khởi tạo head, thứ tự batch và augmentation. Ghi rõ trong báo cáo.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    """AdamW với các nhóm tham số của model.param_groups (norm/bias không weight decay)."""
    groups = mdl.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups, lr=cfg.lr_backbone, betas=(0.9, 0.999))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0 (slide trang 55), cập nhật theo TỪNG BƯỚC (iteration)."""
    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if warm > 0 and step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    d tăng dần theo số lần cập nhật: d_t = min(decay, (1 + t) / (10 + t)) để EMA không bị kéo về trọng số khởi tạo.
    Cả tham số lẫn buffer BatchNorm (running_mean/var) đều được lấy trung bình; buffer số nguyên được chép.
    Đánh giá bằng `ema.module` (bản sao ở chế độ eval).
    """

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.n = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                e.copy_(m)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)       # PyTorch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_train_criterion(cfg: Config, train_df: pd.DataFrame, device):
    """Hàm loss theo cfg; trọng số lớp chỉ tính từ số ảnh TRAIN."""
    kw: dict = {}
    if cfg.loss == "ls":
        kw["smoothing"] = cfg.label_smoothing
    elif cfg.loss == "focal":
        kw["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"].to_numpy(), minlength=ds.NUM_CLASSES)
        kw["weight"] = ls.class_weights(counts, cfg.class_weight_beta or 0.0)
    return ls.build_criterion(cfg.loss, **kw).to(device)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config, device,
                    ema: EMA | None = None, normalizer=None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "lr_steps": [(lr_nhóm_đầu, lr_nhóm_cuối), ...]}."""
    mdl.set_train_mode(model)                       # backbone đóng băng -> giữ BN ở eval
    on_cuda = device.type == "cuda"
    loss_sum = torch.zeros((), device=device)
    n = 0
    lr_steps = []
    for xb, yb, _ in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        x = normalizer(xb)
        if cfg.channels_last and on_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        targets = yb
        if cfg.mix:
            x, targets = ls.mix_batch(x, yb, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with inf.amp_ctx(device, bool(cfg.amp)):
            logits = model(x)
        logits = logits.float()                      # loss luôn tính ở FP32
        loss = ls.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        loss_sum += loss.detach() * xb.size(0)
        n += xb.size(0)
        g = optimizer.param_groups
        lr_steps.append((g[0]["lr"], g[-1]["lr"]))
    return {"train_loss": float(loss_sum.item() / max(n, 1)), "lr_steps": lr_steps}


def evaluate(model, evalset: ds.EvalSet, device, normalizer, img_size: int = 224, amp: bool = False,
             view=None, bs: int = 128):
    """Chạy model trên một EvalSet ở chế độ eval, KHÔNG gradient.

    Mặc định center-crop img_size từ ảnh 256 (img_size >= 256: dùng ảnh nguyên). Trả về
    (filenames, y_true, logits[N, 9] float32, loss CE trung bình) theo đúng thứ tự file.
    """
    view = view if view is not None else inf.view_center(img_size)
    logits = inf.predict_logits(model, evalset.X, device, normalizer, view, bs=bs, amp=amp)
    loss = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(evalset.y)).item())
    return evalset.names, evalset.y, logits, loss


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> dict:
    probs = inf.softmax_np(logits)
    return compute_metrics(y_true, probs.argmax(1), probs)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps=None) -> None:
    """Vẽ đường cong training -> curves/<exp_id>_<mota>.png: loss train/val, macro-F1 val (và top-1), LR theo bước."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(h["epoch"], h["train_loss"], "o-", ms=3, label="train loss (huấn luyện)")
    ax[0].plot(h["epoch"], h["val_loss"], "s-", ms=3, label="val loss (CE)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss train / val")
    ax[1].plot(h["epoch"], h["val_f1"], "o-", ms=3, label="val macro-F1")
    ax[1].plot(h["epoch"], h["val_top1"], "s--", ms=3, label="val top-1")
    if "val_f1_raw" in h and h["val_f1_raw"].notna().any():
        ax[1].plot(h["epoch"], h["val_f1_raw"], "^:", ms=3, label="val macro-F1 (trọng số thô, không EMA)")
    best = int(h["val_f1"].values.argmax())
    ax[1].axvline(h["epoch"].iloc[best], color="gray", ls="--", lw=1, label=f"best epoch = {int(h['epoch'].iloc[best])}")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("điểm"); ax[1].set_title("Val macro-F1 / top-1")
    ax[1].set_ylim(max(0.0, float(h[["val_f1", "val_top1"]].min().min()) - 0.05), 1.0)
    if lr_steps is not None and len(lr_steps):
        a = np.asarray(lr_steps)
        ax[2].plot(a[:, 0], label="nhóm backbone")
        if not np.allclose(a[:, 0], a[:, 1]):
            ax[2].plot(a[:, 1], label="nhóm head")
        ax[2].legend(fontsize=8)
    ax[2].set_xlabel("bước"); ax[2].set_ylabel("learning rate"); ax[2].set_title("LR theo bước (warmup + cosine)")
    for a_ in ax[:2]:
        a_.grid(alpha=0.3); a_.legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


_GMAC_CACHE: dict = {}


def run(cfg: Config, force: bool = False, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình, lưu mọi thứ cần thiết, trả về dict tóm tắt (đọc lại từ summary.json nếu đã xong).

    Thứ tự: set_seed -> split -> loader -> model/optimizer/scheduler/EMA -> mỗi epoch (train, eval val,
    ghi history, lưu checkpoint khi macro-F1 val tăng, hòa thì giữ epoch sớm hơn) -> nạp checkpoint tốt nhất,
    tính lại logit val bằng FP32 -> ghi file. Không đụng test (trừ khi bật save_test_predictions ở Bước 4).
    """
    rd = run_dir(cfg)
    summ_path = rd / "summary.json"
    h = cfg_hash(cfg)
    if summ_path.exists() and not force:
        s = json.loads(summ_path.read_text())
        if s.get("config_hash") == h and (rd / "best.pt").exists():
            if verbose:
                print(f"[{cfg.exp_id} seed{cfg.seed}] đã có kết quả, nạp lại: val F1 {s['val_f1']:.4f}")
            return s
    rd.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    (rd / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1, default=str))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, cfg.fold)
    all_names = pd.concat([train_df, val_df, test_df])["Filename"]
    store = ds.get_store(cfg.images_dir, cfg.cache_dir, all_names)
    if cfg.max_train_images:
        train_df = train_df.sample(n=min(cfg.max_train_images, len(train_df)), random_state=0).reset_index(drop=True)
    val_set = ds.get_evalset("val", val_df, store, device, cfg.max_eval_images)

    train_loader = ds.make_loader(train_df, cfg.images_dir, ds.build_transforms(True, cfg.img_size, cfg.aug),
                                  cfg.batch_size, True, cfg.sampler, cfg.num_workers, store, cfg.seed)
    model = mdl.build_model(cfg.backbone, cfg.pretrained, ds.NUM_CLASSES, cfg.drop_rate, cfg.init, cfg.drop_path_rate)
    normalizer = ds.Normalizer(model.norm_mean, model.norm_std)
    model.to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = build_train_criterion(cfg, train_df, device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = make_scaler(bool(cfg.amp) and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.module if ema is not None else model

    params_m = mdl.count_params(model)
    key = (cfg.backbone, cfg.img_size)
    if key not in _GMAC_CACHE:
        _GMAC_CACHE[key] = mdl.count_gmacs(model, cfg.img_size)
    gmac = _GMAC_CACHE[key]

    history, lr_all = [], []
    best_f1, best_epoch, best_state = -1.0, 0, None
    t_train = []
    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema, normalizer)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_ep = time.perf_counter() - t0
        t_train.append(t_ep)
        lr_all += tr["lr_steps"]
        _, y, logits, vloss = evaluate(eval_model, val_set, device, normalizer, cfg.img_size, amp=bool(cfg.amp))
        m = metrics_from_logits(y, logits)
        row = {"epoch": epoch, "train_loss": tr["train_loss"], "val_loss": vloss, "val_f1": m["macro_f1"],
               "val_top1": m["top1"], "val_f1_raw": np.nan, "lr": tr["lr_steps"][-1][0], "time_s": t_ep}
        if ema is not None:                          # so sánh EMA với trọng số thô (I06)
            _, y2, lg2, _ = evaluate(model, val_set, device, normalizer, cfg.img_size, amp=bool(cfg.amp))
            row["val_f1_raw"] = metrics_from_logits(y2, lg2)["macro_f1"]
        history.append(row)
        if m["macro_f1"] > best_f1 + 1e-12:          # hòa -> giữ epoch sớm hơn
            best_f1, best_epoch = m["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        if verbose:
            print(f"[{cfg.exp_id} s{cfg.seed}] ep {epoch:>2}/{cfg.epochs} train {tr['train_loss']:.4f} "
                  f"val {vloss:.4f} F1 {m['macro_f1']:.4f} top1 {m['top1']:.4f} ({t_ep:.0f}s)")

    # --- nạp checkpoint tốt nhất, tính lại logit val bằng FP32 (đây là logit dùng cho mọi bước sau)
    torch.save(best_state, rd / "best.pt")
    eval_model.load_state_dict(best_state)
    names, y, logits, vloss = evaluate(eval_model, val_set, device, normalizer, cfg.img_size, amp=False)
    m = metrics_from_logits(y, logits)
    np.save(rd / "val_logits.npy", logits)
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    np.save(rd / "lr_steps.npy", np.asarray(lr_all))
    title = f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed} | init={cfg.init} aug={cfg.aug} mix={cfg.mix} loss={cfg.loss} " \
            f"sampler={cfg.sampler} ema={cfg.ema_decay} res={cfg.img_size}"
    plot_curves(history, rd / "curves.png", title, lr_all)
    if cfg.seed == 0 and cfg.curves_dir:
        plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.mota or cfg.backbone}.png", title, lr_all)
    if cfg.save_val_predictions:
        save_predictions(pred_path(cfg, "val"), names, y, inf.softmax_np(logits))
    if cfg.save_test_predictions:                    # chỉ Bước 4: test đúng MỘT lần, 1 view center
        test_set = ds.get_evalset("test", test_df, store, device, cfg.max_eval_images)
        views = inf.test_views_once(cfg, eval_model, test_set.X, device, normalizer,
                                    {"center": (inf.view_center(cfg.img_size), "fp32")})
        save_predictions(pred_path(cfg, "test"), test_set.names, test_set.y, inf.softmax_np(views["center"]))

    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": model.weight_tag,
        "params_m": params_m, "gmac": gmac, "img_size": cfg.img_size, "epochs": cfg.epochs,
        "best_epoch": best_epoch, "val_f1": m["macro_f1"], "val_top1": m["top1"], "val_loss": vloss,
        "val_ece": m["ece"], "val_f1_amp_selection": best_f1,
        "time_per_epoch_s": float(np.mean(t_train)), "train_time_s": float(np.sum(t_train)),
        "val_recall": [float(v) for v in m["recall"]], "val_f1_per_class": [float(v) for v in m["f1"]],
        "config_hash": h, "torch": torch.__version__,
        "val_f1_raw_best": (float(history[best_epoch - 1]["val_f1_raw"]) if ema is not None else None),
    }
    summ_path.write_text(json.dumps(summary, indent=1))
    if verbose:
        print(f"[{cfg.exp_id} s{cfg.seed}] xong: best epoch {best_epoch}, val macro-F1 {m['macro_f1']:.4f}, "
              f"top-1 {m['top1']:.4f}, {np.mean(t_train):.0f}s/epoch")
    del model, optimizer, ema, eval_model, train_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    hints = typing.get_type_hints(Config)
    default = Config()
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"cần dạng KEY=VALUE, nhận {p!r}")
        k, v = p.split("=", 1)
        if k not in hints:
            raise ValueError(f"key không có trong Config: {k!r}")
        if v.lower() in ("none", "null"):
            out[k] = None
            continue
        ref = getattr(default, k)
        if isinstance(ref, bool):
            out[k] = v.lower() in ("1", "true", "yes", "y")
        elif isinstance(ref, int):
            out[k] = int(v)
        elif isinstance(ref, float):
            out[k] = float(v)
        elif ref is None:                       # field Optional mặc định None: đoán kiểu
            for conv in (int, float, str):
                try:
                    out[k] = conv(v)
                    break
                except ValueError:
                    continue
        else:
            out[k] = v
    return out


def main(argv=None) -> None:
    """Điểm vào dòng lệnh: python train.py --set exp_id=B01 backbone=resnet50 seed=0"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = Config(**parse_overrides(a.set))
    print(json.dumps(run(cfg, force=a.force), indent=1))


if __name__ == "__main__":
    main()
