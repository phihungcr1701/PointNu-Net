# PointNu-Net: Train PanNuke nhanh nhất

Kết luận ngắn: repo này dễ chạy train hơn HoVer-Net nếu mục tiêu của bạn chỉ là huấn luyện trên PanNuke. Lý do là nó đã có script riêng `train_pannuke.py`, dataset class riêng `PannukeDataset`, và file cấu hình PanNuke sẵn trong `configs/pannuke.yaml`.

## Vì sao dễ hơn HoVer-Net

- Không cần tự sửa `config.py` để trỏ patch `.npy` như HoVer-Net.
- Không cần chạy bước `extract_patches.py` để chuyển dữ liệu về patch trung gian.
- PanNuke được đọc trực tiếp từ cấu trúc `.npy` có sẵn.
- Chỉ cần chuẩn bị đúng thư mục dữ liệu, rồi chạy train.

## Cấu trúc dữ liệu cần có

Repo này kỳ vọng PanNuke ở dạng:

```text
PointNu-Net/
└── datasets/
    └── PanNuKe/
        ├── images/
        │   ├── fold1/images.npy
        │   ├── fold2/images.npy
        │   └── fold3/images.npy
        └── masks/
            ├── fold1/masks.npy
            ├── fold2/masks.npy
            └── fold3/masks.npy
```

Ngoài ra, `train_pannuke.py` sẽ chọn `train_fold`, `val_fold`, `test_fold` theo kiểu 1-2-3 luân phiên.

## Cách train tối thiểu

1. Cài dependency:

```bash
pip install -r requirements.txt
```

2. Kiểm tra dữ liệu PanNuke đã đúng cấu trúc ở trên.

3. Chạy train:

```bash
python train_pannuke.py --name=pannuke_exp --seed=888 --train_fold=1 --val_fold=2 --test_fold=3
```

Hoặc đổi hoán vị fold theo nhu cầu:

- `train_fold=2 --val_fold=1 --test_fold=3`
- `train_fold=3 --val_fold=2 --test_fold=1`

## Cấu hình mặc định đáng chú ý

File `configs/pannuke.yaml` đã đặt sẵn:

- `dataroot: ./datasets/PanNuKe`
- `model.backbone: hrnet64`
- `model.num_classes: 6`
- `train.batch_size: 8`
- `train.max_epoch: 100`

Nếu máy yếu, giảm `batch_size` trong file YAML là cách chỉnh đơn giản nhất.

## Lưu ý thực tế

- Repo này phù hợp nhất nếu mục tiêu của bạn là train PanNuke bằng pipeline có sẵn.
- Script train không thấy bước validation riêng trong `train_pannuke.py`; nó chủ yếu train và lưu checkpoint định kỳ.
- Muốn đánh giá sau train, repo có `infer_pannuke.py` và `eval_pannuke.py`, nhưng nếu bạn chỉ ưu tiên train thì chưa cần đụng tới hai script này.

## Kết luận

Nếu so với HoVer-Net, PointNu-Net là lựa chọn dễ hơn cho nhu cầu của bạn vì có đường đi PanNuke riêng. Với mục tiêu "chỉ cần chạy train", đây là repo nên thử trước.
