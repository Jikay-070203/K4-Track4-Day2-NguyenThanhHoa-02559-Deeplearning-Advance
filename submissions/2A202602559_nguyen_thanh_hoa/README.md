# Lab Day 2 — DeepWeeds: backbone, công thức huấn luyện, suy luận

Sinh viên: **Nguyễn Thanh Hòa** · MSSV **2A202602559** · Track 4 · Ngày 2

## Link notebook chạy lại được

- Kaggle (notebook đã chạy xong, giữ nguyên output): [Lab Day 2 — Backbone, công thức huấn luyện và suy luận (Kaggle)](https://www.kaggle.com/code/nguyenthanhhoak18hcm/lab-day-2-backbone-c-ng-th-c-hu-n-luy-n-v-suy).

## Cách chạy lại

Notebook `code/lab_day2.ipynb` **tự chứa**: các ô `%%writefile` ghi toàn bộ module vào `code/`, ô đầu tự clone repo gốc của giảng viên (để lấy `eval.py`, không sửa), tải dữ liệu và chạy tới hết.

1. Mở `lab_day2.ipynb` trên Colab (hoặc Kaggle, bật Internet), chọn GPU (T4 trở lên).
2. (Tuỳ chọn) upload sẵn `images.zip` (MD5 `b7b30f96d466fba86016aa5a26606e0f`) rồi đặt `USER_IMAGES_ZIP` ở ô đầu; nếu không, notebook tải từ Zenodo (~490 MB) và kiểm tra MD5. Nhãn và fold lấy từ GitHub của tác giả (`train/val/test_subset0.csv`, không sửa).
3. *Runtime → Run all*. Mỗi lần chạy được lưu ở `runs/<exp_id>/seed<k>/`; chạy lại notebook sẽ nạp lại các run đã xong (an toàn khi bị ngắt kết nối). Đặt `USE_DRIVE = True` để lưu vào Google Drive.
4. Cuối notebook: `results.xlsx`, `curves/`, `predictions/`, `report.md` (bản tự sinh các bảng, phần nhận xét đã được viết nốt từ số liệu thật) và file `.zip` được tạo.

Thời gian: tổng thời gian huấn luyện cộng dồn của 31 lần chạy trong lần nộp bài ≈ 4,9 giờ trên Tesla T4 (cộng `train_time_s` trong `results/runs/*/seed*/summary.json`), chưa kể suy luận và đo độ trễ; `LITE = True` rút gọn số thí nghiệm (phải ghi rõ trong báo cáo).

## Phiên bản thư viện và seed

- Phiên bản của lần chạy nộp bài (Kaggle, ghi ở `results/env.json` và mục 2 của `report.md`): Python 3.13.15, PyTorch 2.11.0+cu128, timm 1.0.29, GPU Tesla T4. Cần: `torch`, `torchvision`, `timm`, `numpy`, `pandas`, `scikit-learn` (chỉ cho `eval`-tests của repo), `openpyxl`, `matplotlib`, `pillow`.
- Seed: sàng backbone và ablation dùng **seed 0**; `T00` (mốc) và `F01` (chung kết) chạy **seed 0, 1, 2**. Seed chỉ đổi khởi tạo head, thứ tự batch và augmentation, không đổi cách chia dữ liệu (S5). `cudnn.benchmark` bật nên không tái lập từng bit trên GPU.

## Cấu trúc thư mục nộp

```
2A202602559_nguyen_thanh_hoa/
├── README.md              # file này
├── results.xlsx           # 7 sheet: Backbones, Training, Inference, Final, PerClass, Latency, Summary
├── report.md              # báo cáo kết luận
├── curves/                # một ảnh training cho mỗi exp_id (B01.., T00.., T20, F01)
├── predictions/           # F01 / F01_uncal / R01 / T00: *_seed<k>_test.csv (+ *_val.csv), đúng định dạng eval.py
├── figures/               # EDA, đánh đổi backbone/suy luận, ablation, ma trận nhầm lẫn, ảnh bị đoán sai
├── results/               # log từng run (summary.json, history.csv, config.json), env.json, bảng CSV
└── code/
    ├── lab_day2.ipynb     # notebook (giữ output)
    └── dataset.py model.py losses.py train.py inference.py benchmark.py report_tools.py
```

`runs/` (checkpoint, logit) và dữ liệu không được commit (đã có trong `.gitignore` của repo).

## Kiểm tra số liệu bằng `eval.py` gốc

```bash
python eval.py score --pred "submissions/2A202602559_nguyen_thanh_hoa/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01
python eval.py grade --final "submissions/2A202602559_nguyen_thanh_hoa/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202602559_nguyen_thanh_hoa/predictions/T00_seed*_test.csv" \
    --uncal "submissions/2A202602559_nguyen_thanh_hoa/predictions/F01_uncal_seed*_test.csv" \
    --final-val "submissions/2A202602559_nguyen_thanh_hoa/predictions/F01_seed*_val.csv" \
    --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv --labels data/labels/labels.csv
```

(đường dẫn `data/labels/...` là nơi đặt các CSV của tác giả; notebook dùng thư mục tương ứng trên Colab).
