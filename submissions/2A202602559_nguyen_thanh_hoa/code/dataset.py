"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1: dùng fold 0 nguyên bản của tác giả,
không sửa / lọc / chia lại; val để chọn cấu hình, test chỉ chạy một lần mỗi seed ở Bước 4.

Thiết kế để chạy nhanh trên Colab (2 vCPU):
  * Giải mã JPEG MỘT lần, lưu thành mảng uint8 (N, 256, 256, 3) dạng memmap (`build_image_cache`);
    mọi DataLoader / EvalSet đọc từ mảng này nên không còn nghẽn giải mã ảnh.
  * Transform ở worker chỉ trả về tensor uint8; việc chuẩn hoá (mean/std) làm trên GPU bằng `Normalizer`.
  * Tập val/test (không augmentation) nạp thẳng lên thiết bị dưới dạng `EvalSet`; center-crop, lật, resize
    ... là các "view" áp dụng trên GPU (xem inference.py).

Giao diện chính:
    load_split(labels_dir, fold=0)                     -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)             -> torchvision transform (trả về tensor uint8)
    DeepWeedsDataset[i]                                -> (uint8 tensor CxHxW, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers, store, seed)
    EvalSet(df, store, device)                         -> X uint8 (N,3,256,256) trên thiết bị, y, names
"""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
N_TOTAL = 17_509
IMG_SIDE = 256
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # model.py ghi đè bằng mean/std đúng của trọng số timm đang dùng
IMAGENET_STD = (0.229, 0.224, 0.225)
RRC_SCALE = (0.25, 1.0)                # RandomResizedCrop: ảnh cỏ dại chiếm gần cả khung, không cắt quá nhỏ


# --------------------------------------------------------------------------- chia dữ liệu
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Các file của tác giả có cột `Filename, Label` (tên loài nằm ở labels.csv). KHÔNG sửa/lọc/chia lại.
    """
    d = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(d / f"{split}_subset{fold}.csv")
        if not {"Filename", "Label"} <= set(df.columns):
            raise ValueError(f"{split}_subset{fold}.csv: cần cột Filename và Label, có {list(df.columns)}")
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True, expected_total: int = N_TOTAL) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập, mỗi lớp (xấp xỉ 60/20/20, lệch quá 1 điểm phần trăm thì dừng);
    2. giao của từng cặp tập theo Filename phải RỖNG;
    3. hợp ba tập đúng 17.509 ảnh, không trùng tên trong một tập;
    4. mọi Filename đều tồn tại trong images_dir.
    """
    sets = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: int(len(v)) for k, v in sets.items()}
    total = sum(n.values())
    frac = {k: n[k] / total for k in n}
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(frac[k] - target) <= 0.01, (
            f"tỉ lệ {k} = {frac[k]:.3%} lệch hơn 1 điểm phần trăm khỏi {target:.0%}: báo giảng viên trước khi chạy tiếp")
    for k, v in sets.items():
        assert v["Filename"].is_unique, f"{k}: có Filename bị trùng"
    names = {k: set(v["Filename"]) for k, v in sets.items()}
    overlap = {f"{a}&{b}": len(names[a] & names[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(v == 0 for v in overlap.values()), f"giao các tập khác rỗng: {overlap}"
    union = names["train"] | names["val"] | names["test"]
    assert total == expected_total and len(union) == expected_total, (
        f"hợp ba tập = {len(union)} (tổng dòng {total}), kỳ vọng {expected_total}")
    images_dir = Path(images_dir)
    missing = [f for f in sorted(union) if not (images_dir / f).exists()]
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:3]}"
    per_class = pd.DataFrame({k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                              for k, v in sets.items()})
    per_class.index = CLASS_NAMES
    per_class["total"] = per_class.sum(1)
    info = {"n": n, "fraction": frac, "per_class": per_class.to_dict(), "overlap": overlap,
            "union": len(union), "missing_files": 0}
    if verbose:
        print("số ảnh:", n, "| tổng", total, "| tỉ lệ", {k: f"{v:.2%}" for k, v in frac.items()})
        print("giao các cặp tập:", overlap, "| hợp ba tập =", len(union), "| thiếu file = 0")
        print(per_class.to_string())
        mx, mn = per_class["total"].max(), per_class["total"].min()
        print(f"lớp lớn nhất / nhỏ nhất = {mx} / {mn} = {mx / mn:.2f}×")
    return info


# --------------------------------------------------------------------------- bộ nhớ đệm ảnh
def build_image_cache(images_dir: str | Path, filenames, cache_dir: str | Path, side: int = IMG_SIDE,
                      workers: int | None = None, force: bool = False) -> Path:
    """Giải mã mọi ảnh một lần, ghi mảng uint8 (N, side, side, 3) + danh sách tên vào cache_dir."""
    images_dir, cache_dir = Path(images_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy, names_json = cache_dir / "images_u8.npy", cache_dir / "names.json"
    filenames = list(filenames)
    if not force and npy.exists() and names_json.exists():
        if json.loads(names_json.read_text()) == filenames:
            return npy
    arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(len(filenames), side, side, 3))

    def _load(i: int) -> None:
        with Image.open(images_dir / filenames[i]) as im:
            im = im.convert("RGB")
            if im.size != (side, side):
                im = im.resize((side, side), Image.BILINEAR)
            arr[i] = np.asarray(im)

    with ThreadPoolExecutor(max_workers=workers or max(2, (os.cpu_count() or 2))) as ex:
        list(ex.map(_load, range(len(filenames))))
    arr.flush()
    names_json.write_text(json.dumps(filenames))
    return npy


class ImageStore:
    """Mảng ảnh uint8 dạng memmap + chỉ mục tên file -> dòng."""

    def __init__(self, cache_dir: str | Path):
        cache_dir = Path(cache_dir)
        self.arr = np.load(cache_dir / "images_u8.npy", mmap_mode="r")
        self.names = json.loads((cache_dir / "names.json").read_text())
        self.index = {n: i for i, n in enumerate(self.names)}

    def rows(self, filenames) -> np.ndarray:
        return np.array([self.index[f] for f in filenames], dtype=np.int64)


_STORES: dict = {}


def get_store(images_dir, cache_dir, filenames) -> ImageStore:
    """Tạo (nếu chưa có) và trả về ImageStore, nhớ trong tiến trình để các run dùng chung."""
    key = str(Path(cache_dir).resolve())
    if key not in _STORES:
        build_image_cache(images_dir, sorted(set(filenames)), cache_dir)
        _STORES[key] = ImageStore(cache_dir)
    return _STORES[key]


# --------------------------------------------------------------------------- transform
def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Transform trả về tensor uint8 (C, H, W); chuẩn hoá làm trên GPU bằng `Normalizer`.

    Train (mọi mức): RandomResizedCrop(img_size, scale=(0.25, 1)) + lật ngang. `aug` thêm:
        "basic"  : không thêm gì (công thức nền)
        "vflip"  : + lật dọc
        "color"  : + ColorJitter(0.3, 0.3, 0.3, 0.05)
        "trivial": + TrivialAugmentWide
        "randaug": + RandAugment(2, 9)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.
    Val/test: chỉ PILToTensor (ảnh gốc 256x256); center-crop / resize là "view" trên GPU (inference.py),
    KHÔNG có augmentation ngẫu nhiên khi đánh giá.
    """
    if not train:
        return T.PILToTensor()
    ops = [T.RandomResizedCrop(img_size, scale=RRC_SCALE, interpolation=T.InterpolationMode.BILINEAR),
           T.RandomHorizontalFlip()]
    if aug == "vflip":
        ops.append(T.RandomVerticalFlip())
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(2, 9))
    elif aug != "basic":
        raise ValueError(f"aug không hợp lệ: {aug!r}")
    ops.append(T.PILToTensor())
    return T.Compose(ops)


class Normalizer:
    """(x_uint8 - mean*255) / (std*255) trên thiết bị của x. Gọi như hàm; trả float32."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean, self.std = tuple(mean), tuple(std)
        self._cache: dict = {}

    def _stats(self, device):
        key = str(device)
        if key not in self._cache:
            m = torch.tensor(self.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1) * 255.0
            s = torch.tensor(self.std, dtype=torch.float32, device=device).view(1, 3, 1, 1) * 255.0
            self._cache[key] = (m, s)
        return self._cache[key]

    def __call__(self, x_u8: torch.Tensor) -> torch.Tensor:
        m, s = self._stats(x_u8.device)
        return (x_u8.float() - m) / s

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Đảo chuẩn hoá để vẽ ảnh (trả về float trong [0, 1])."""
        m, s = self._stats(x.device)
        return ((x * s + m) / 255.0).clamp(0, 1)


# --------------------------------------------------------------------------- Dataset / DataLoader
class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh theo DataFrame (Filename, Label); trả về (tensor, nhãn int, tên file str).

    Nếu có `store` (ImageStore) thì lấy ảnh từ mảng uint8 trong RAM/memmap; nếu không thì mở file JPEG.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path | None = None, transform=None, store=None):
        self.names = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.transform = transform
        self.store = store
        self.images_dir = Path(images_dir) if images_dir is not None else None
        self.rows = store.rows(self.names) if store is not None else None
        if store is None and self.images_dir is None:
            raise ValueError("cần images_dir hoặc store")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        if self.store is not None:
            img = Image.fromarray(np.asarray(self.store.arr[self.rows[i]]))
        else:
            with Image.open(self.images_dir / self.names[i]) as im:
                img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.names[i]


def _seed_worker(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir, transform, batch_size: int, train: bool,
                sampler: str | None = None, num_workers: int = 2, store=None, seed: int = 0) -> DataLoader:
    """DataLoader. train: xáo (hoặc sampler cân bằng), drop_last; eval: giữ nguyên thứ tự của df.

    sampler="balanced": WeightedRandomSampler với trọng số 1/(số ảnh của lớp) tính TRÊN df truyền vào
    (chỉ gọi với train_df).
    """
    dataset = DeepWeedsDataset(df, images_dir, transform, store)
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              drop_last=bool(train), generator=g)
    if num_workers > 0:
        kw.update(worker_init_fn=_seed_worker, persistent_workers=bool(train), prefetch_factor=4)
    if train and sampler == "balanced":
        counts = df["Label"].value_counts()
        w = df["Label"].map(lambda c: 1.0 / counts[c]).to_numpy()
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(dataset),
                                    replacement=True, generator=g)
        return DataLoader(dataset, sampler=smp, **kw)
    if sampler not in (None, "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler!r}")
    return DataLoader(dataset, shuffle=bool(train), **kw)


# --------------------------------------------------------------------------- tập đánh giá trên GPU
class EvalSet:
    """Toàn bộ một tập (val hoặc test) dưới dạng tensor uint8 (N, 3, 256, 256) trên `device`.

    Giữ đúng thứ tự của df để ghép logit với tên file. ~700 MB cho 3.5k ảnh.
    """

    def __init__(self, df: pd.DataFrame, store: ImageStore, device, limit: int | None = None):
        if limit is not None:
            df = df.iloc[:limit]
        self.names = df["Filename"].tolist()
        self.y = df["Label"].to_numpy(dtype=np.int64)
        rows = store.rows(self.names)
        order = np.argsort(rows)                     # đọc memmap theo thứ tự tăng dần cho nhanh
        arr = np.empty((len(rows), IMG_SIDE, IMG_SIDE, 3), dtype=np.uint8)
        arr[order] = store.arr[rows[order]]
        self.X = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
        self.device = device

    def __len__(self) -> int:
        return len(self.names)


_EVALSETS: dict = {}


def get_evalset(name: str, df: pd.DataFrame, store: ImageStore, device, limit: int | None = None) -> EvalSet:
    """Nạp (và nhớ) một EvalSet; khoá gồm tên tập, số ảnh và thiết bị."""
    key = (name, limit, str(device), len(df))
    if key not in _EVALSETS:
        _EVALSETS[key] = EvalSet(df, store, device, limit)
    return _EVALSETS[key]
