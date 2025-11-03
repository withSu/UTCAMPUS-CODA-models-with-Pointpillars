# UT Campus Object Dataset (CODa) Object Detection Models

<b>Official model development kit for CODa.</b> We strongly recommend using this repository to run our pretrained
models and train on custom datasets. Thanks to the authors of ST3D++ and OpenPCDet from whom this repository
was adapted from.

![Sequence 0 Clip](./docs/codademo.gif)

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation.

## Getting Started

### Quicksetup with Docker

Please refer to [DOCKER.md](docs/DOCKER.md) to learn about how to pull our prebuilt docker images to **deploy our pretrained 3D object detection models.**

### Full Setup
Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more about how to use this project.

## License

Our code is released under the Apache 2.0 license.

## Paper Citation

If you find our work useful in your research, please consider citing our [paper](https://arxiv.org/abs/2309.13549) and [dataset](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV):

```
@inproceedings{zhang2023utcoda,
    title={Towards Robust 3D Robot Perception in Urban Environments: The UT Campus Object Dataset},
    author={Arthur Zhang and Chaitanya Eranki and Christina Zhang and Raymond Hong and Pranav Kalyani and Lochana Kalyanaraman and Arsh Gamare and Maria Esteva and Joydeep Biswas },
    booktitle={},
    year={2023}
}
```

## Dataset Citation
```
@data{T8/BBOQMV_2023,
author = {Zhang, Arthur and Eranki, Chaitanya and Zhang, Christina and Hong, Raymond and Kalyani, Pranav and Kalyanaraman, Lochana and Gamare, Arsh and Bagad, Arnav and Esteva, Maria and Biswas, Joydeep},
publisher = {Texas Data Repository},
title = {{UT Campus Object Dataset (CODa)}},
year = {2023},
version = {DRAFT VERSION},
doi = {10.18738/T8/BBOQMV},
url = {https://doi.org/10.18738/T8/BBOQMV}
}
```

## Acknowledgement

Our code is heavily based on [OpenPCDet v0.3](https://github.com/open-mmlab/OpenPCDet/commit/e3bec15f1052b4827d942398f20f2db1cb681c01). Thanks OpenPCDet Development Team for their awesome codebase.


Thank you to the authors of ST3D++ or OpenPCDet for an awesome codebase!
```
@article{yang2021st3d++,
  title={ST3D++: Denoised Self-training for Unsupervised Domain Adaptation on 3D Object Detection},
  author={Yang, Jihan and Shi, Shaoshuai and Wang, Zhe and Li, Hongsheng and Qi, Xiaojuan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
```
```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

---

# CODa Dataset 학습 가이드

## 📋 목차
1. [PKL 파일 개념](#pkl-파일-개념)
2. [발생한 문제들](#발생한-문제들)
3. [해결 과정](#해결-과정)
4. [논문/GitHub 설정 비교](#논문github-설정-비교)
5. [학습 명령어](#학습-명령어)
6. [평가 명령어](#평가-명령어)
7. [다른 데이터셋 적용 시 체크리스트](#다른-데이터셋-적용-시-체크리스트)

---

## 🔍 PKL 파일 개념

### PKL (Pickle) 파일이란?
- **역할**: 데이터셋의 메타데이터를 미리 처리하여 저장
- **목적**: 학습 시 매번 원본 데이터를 파싱하지 않고 빠르게 로드

### PKL 파일 구조
```
coda_infos_train.pkl
├── frame_id: "0501900"
├── image: {"image_path": "...", "image_shape": [...]}
├── point_cloud: {"lidar_path": "..."}
├── calib: {P0, P1, P2, P3, R0_rect, Tr_velo_to_cam}
└── annos:
    ├── name: ["Pedestrian", "Pole", ...]
    ├── bbox: [[x1,y1,x2,y2], ...]  # 2D bbox
    ├── location: [[x,y,z], ...]     # 3D center
    ├── dimensions: [[h,w,l], ...]   # 3D size
    ├── rotation_y: [...]            # Rotation
    ├── score: [...]
    ├── difficulty: [...]
    └── num_points_in_gt: [...]
```

### 왜 필요한가?
1. **속도**: 원본 JSON 파싱 대신 직렬화된 데이터 로드 (10-100배 빠름)
2. **전처리**: 좌표 변환, 필터링 등을 미리 수행
3. **일관성**: 모든 프레임이 동일한 포맷으로 저장
4. **데이터 증강**: GT database를 미리 생성하여 augmentation에 사용

---

## ❌ 발생한 문제들

### 1. 데이터 변환 문제
#### 문제 1-1: 이미지 확장자 불일치
- **증상**: `AssertionError: Image file does not exist: .../2d_rect_cam0_5_1900.jpg`
- **원인**: CODa는 `.png`인데 converter는 `.jpg` 예상
- **위치**: `tools/data_converter/coda_converter.py:288`

#### 문제 1-2: 2D bbox 파일 없음
- **증상**: `FileNotFoundError: .../2d_bbox/cam0/5/2d_bbox_cam0_5_1953.txt not found`
- **원인**: CODa는 LiDAR 전용 데이터셋, 2D bbox 미제공
- **위치**: `tools/data_converter/coda_converter.py:477-520`

#### 문제 1-3: 3D 데이터 경로 불일치
- **증상**: `AssertionError: Bin file does not exist: .../3d_raw/os1/5/...`
- **원인**: CODa는 `3d_comp` 사용, converter는 `3d_raw` 찾음
- **위치**: `tools/data_converter/coda_converter.py:291, 443-444`

### 2. PKL 생성 문제
#### 문제 2-1: Split 파라미터 버그 ⚠️ **Critical**
- **증상**: `Total samples for CODa dataset: 0`
- **원인**: `coda_dataset.py:136`에서 `self.split` 대신 `split` 파라미터 사용해야 함
```python
# WRONG:
split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')

# CORRECT:
split_dir = self.root_path / 'ImageSets' / (split + '.txt')
```

#### 문제 2-2: 잘못된 PKL 재사용
- **증상**: `AssertionError: Lidar files data/.../0102019.bin`
- **원인**: 다른 워크스페이스의 PKL 복사 (3,496 샘플 vs 140 샘플, 다른 sample ID)
- **해결**: 처음부터 재생성 필요

#### 문제 2-3: Python 모듈 캐싱
- **증상**: 코드 수정 후에도 버그 지속
- **원인**: Python이 이미 import된 모듈 캐시
- **해결**: `python -B -m pcdet.datasets.coda.coda_dataset` (bytecode 캐싱 비활성화)

### 3. 학습 설정 문제
#### 문제 3-1: WandB 초기화 오류
- **증상**: `wandb.errors.errors.UsageError: api_key not configured`
- **원인**: config에서 `WANDB: False`인데도 `wandb.init()` 호출
- **위치**: `tools/train_utils/train_utils.py:72, 42` / `tools/train.py:313`

#### 문제 3-2: Distributed training 파라미터
- **증상**: `RuntimeError: Default process group has not been initialized`
- **원인**: `torchrun`이 `LOCAL_RANK` 환경변수로 전달하는데 코드는 `--local-rank` 파라미터 예상
- **해결**: 환경변수 지원 추가 (`tools/train.py:53-57`)

### 4. 클래스 이름 문제
#### 문제 4-1: Car vs UtilityVehicle
- **증상**: Car AP 0%
- **원인**: Sequence 5에는 `UtilityVehicle`만 있고 `Car` 없음
- **해결**:
  - 옵션 1: 현재 모델 사용 (Pedestrian, Cyclist만 검출)
  - 옵션 2: config를 `UtilityVehicle`로 변경 후 재학습

---

## ✅ 해결 과정

### Step 0: 환경 준비
```bash
# 작업 디렉토리
cd /media/withsu/ROBOT_SSD_0/coda_clone2/coda-models

# 원본 데이터 위치 확인
ls /media/withsu/ROBOT_SSD_0/5/
# 출력: 2d_rect  3d_bbox  3d_comp  calibrations  metadata  poses  timestamps
```

### Step 1: 데이터 변환 (CODa → KITTI Format)

#### 1-1. Converter 수정
```bash
# 파일: tools/data_converter/coda_converter.py
```

**수정 1: 이미지 확장자** (Line 288)
```python
if "2d_rect"==modality:
    filetype = "png"  # 원래: "jpg"
```

**수정 2: 2D bbox 더미값** (Lines 477-520)
```python
# CODa는 2D bbox 없음 - 더미값 사용
twod_anno_dict = None
# ... 나중에 ...
bounding_box = [0.0, 0.0, 50.0, 50.0]  # 더미값 (LiDAR만 사용)
```

**수정 3: 3d_comp 경로** (Lines 291, 443-444)
```python
elif "3d_raw"==modality or "3d_comp"==modality:  # 원래: "3d_raw"만
    filetype = "bin"

# ...
bin_file = self.set_filename_by_prefix("3d_comp", "os1", traj, frame_idx)  # 원래: "3d_raw"
bin_path = join(self.load_dir, "3d_comp", "os1", traj, bin_file)
```

#### 1-2. 변환 실행
```bash
cd /media/withsu/ROBOT_SSD_0/coda_clone2/coda-models

export PYTHONPATH=$PWD:$PYTHONPATH

python tools/create_data.py coda \
  --root-path ./ \
  --out-dir ./data \
  --workers 8 \
  2>&1 | tee conversion.log
```

**예상 출력:**
```
Processing trajectory 5...
Converting frame 1800/2000...
...
Conversion complete: 200 frames
```

**결과 확인:**
```bash
ls data/coda_kitti_format/
# 출력: ImageSets  training  testing

ls data/coda_kitti_format/ImageSets/
# 출력: train.txt  val.txt  test.txt

wc -l data/coda_kitti_format/ImageSets/*.txt
# 출력:
#  140 train.txt
#   30 val.txt
#   30 test.txt
```

### Step 2: PKL 파일 생성

#### 2-1. Dataset 코드 수정
```bash
# 파일: pcdet/datasets/coda/coda_dataset.py
```

**수정 1: Split 파라미터** (Line 136) ⚠️ **가장 중요!**
```python
def set_sample_id_list(self, split):
    split_dir = self.root_path / 'ImageSets' / (split + '.txt')  # 원래: self.split
```

**수정 2: 이미지 확장자** (Line 162)
```python
img_file = root_split_path / 'image_0' / ('%s.png' % idx)  # 원래: .jpg
```

**수정 3: Data path** (Lines 671-672)
```python
data_path=ROOT_DIR / 'data' / 'coda_kitti_format',  # 원래: coda128_allclass_full
save_path=ROOT_DIR / 'data' / 'coda_kitti_format',
```

#### 2-2. Config 수정
```bash
# 파일: tools/cfgs/dataset_configs/da_coda_oracle_dataset_3class.yaml
```

```yaml
DATA_PATH: 'data/coda_kitti_format'  # 원래: '../data/coda_kitti_format'
```

#### 2-3. PKL 생성 실행
```bash
# -B 플래그로 Python bytecode 캐싱 비활성화 (중요!)
python -B -m pcdet.datasets.coda.coda_dataset \
  create_coda_infos \
  tools/cfgs/dataset_configs/da_coda_oracle_dataset_3class.yaml \
  2>&1 | tee pkl_generation.log
```

**예상 출력:**
```
---------------Start to generate data infos---------------
train sample_idx: 0501900
train sample_idx: 0501942
...
CODa info train file is saved to .../coda_infos_train.pkl
CODa info val file is saved to .../coda_infos_val.pkl
CODa info test file is saved to .../coda_infos_test.pkl
---------------Start create groundtruth database for data augmentation---------------
Database Pole: 1417
Database Pedestrian: 968
Database Tree: 971
Database Cyclist: 149
Database Railing: 1234
Database BikeRack: 316
Database UtilityVehicle: 25
---------------Data preparation Done---------------
```

**결과 확인:**
```bash
ls -lh data/coda_kitti_format/*.pkl
# 출력:
# -rw-r--r-- 1.2M coda_infos_train.pkl      (140 samples)
# -rw-r--r-- 243K coda_infos_val.pkl        (30 samples)
# -rw-r--r-- 241K coda_infos_test.pkl       (30 samples)
# -rw-r--r-- 1.5M coda_dbinfos_train.pkl    (GT database)
```

### Step 3: 학습 설정

#### 3-1. Config 수정
```bash
# 파일: tools/cfgs/coda_models/pointpillar_1x.yaml
```

**수정 1: Base config 경로** (Line 4)
```yaml
DATA_CONFIG:
    _BASE_CONFIG_: tools/cfgs/dataset_configs/da_coda_oracle_dataset_3class.yaml
```

**수정 2: WandB 비활성화** (Line 129)
```yaml
FINETUNE:
    WANDB: False
```

**수정 3: Balanced resampling** (Line 5)
```yaml
DATA_CONFIG:
    BALANCED_RESAMPLING: False  # 원래: True (에러 발생)
```

#### 3-2. Train 코드 수정
```bash
# 파일: tools/train.py
```

**수정 1: LOCAL_RANK 환경변수 지원** (Lines 53-57)
```python
args = parser.parse_args()

# torchrun이 환경변수로 전달
if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])

cfg_from_yaml_file(args.cfg_file, cfg)
```

**수정 2: WandB 체크** (Line 313)
```python
if ft_cfg is not None and ft_cfg.get('WANDB', False):  # 추가: WANDB 체크
    wandb.init(...)
```

#### 3-3. Train utils 수정
```bash
# 파일: tools/train_utils/train_utils.py
```

**수정 1: WandB 초기화 체크** (Line 72)
```python
if ft_cfg is not None and ft_cfg.get('WANDB', False):
    wandb.init(...)
```

**수정 2: WandB 로깅 체크** (Line 42)
```python
if ft_cfg is not None and ft_cfg.get('WANDB', False):
    wandb.log({"loss": loss, "lr": cur_lr, "iter": cur_it})
```

### Step 4: 학습 실행

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  tools/train.py \
  --cfg_file tools/cfgs/coda_models/pointpillar_1x.yaml \
  --launcher pytorch \
  --batch_size 2 \
  2>&1 | tee training_pointpillar_50epochs.log
```

**학습 파라미터:**
- Epochs: 50
- Batch size: 2 per GPU
- Learning rate: 0.003
- Optimizer: adam_onecycle
- 소요 시간: 약 40분 (RTX 3090 기준)

**결과:**
```
Epoch 50/50: loss=0.458
Checkpoints saved to: output/.../ckpt/checkpoint_epoch_50.pth
```

### Step 5: 평가 실행

#### 5-1. 평가 스크립트 생성
```bash
# 파일: eval_model.py (새로 생성)
```

```python
#!/usr/bin/env python3
"""Simple evaluation script for trained models"""
import sys, os, torch, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_yaml_file
from eval_utils import eval_utils

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    return parser.parse_args()

def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = args.split

    output_dir = Path('output') / cfg.EXP_GROUP_PATH / cfg.TAG / 'eval' / f'eval_{args.split}'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'eval_log.txt'
    logger = common_utils.create_logger(log_file, rank=0)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size, dist=False, workers=args.workers,
        logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    with torch.no_grad():
        eval_utils.eval_one_epoch(
            cfg, model, test_loader, 0, logger,
            dist_test=False, result_dir=output_dir, save_to_file=True
        )

if __name__ == '__main__':
    main()
```

#### 5-2. 평가 실행
```bash
python eval_model.py \
  --cfg_file tools/cfgs/coda_models/pointpillar_1x.yaml \
  --ckpt output/cfgs/coda_models/pointpillar_1x/defaultLR0.003000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth \
  --split val \
  --batch_size 1 \
  2>&1 | tee evaluation_val_epoch50.log
```

**결과 확인:**
```bash
# 로그 파일에서 AP 확인
grep "AP_R40" evaluation_val_epoch50.log

# 또는 결과 디렉토리 확인
ls output/cfgs/coda_models/pointpillar_1x/eval/eval_val/
```

**평가 결과 (Epoch 50):**
```
Pedestrian 3D AP_R40@0.50: 85.23% / 85.23% / 88.85% (easy/moderate/hard)
Cyclist 3D AP_R40@0.50:    12.69% / 12.69% / 63.15%
Car 3D AP_R40@0.70:         0.00% /  0.00% /  0.00% (데이터 없음)
```

---

## 📊 논문/GitHub 설정 비교

### Voxel Size 비교

| 구분 | X | Y | Z | MAX_VOXELS (train/test) | 비고 |
|------|---|---|---|------------------------|------|
| **GitHub 공식** | 0.1 | 0.1 | 6.0 | 80K / 90K | ✅ 권장 |
| **이전 설정** | 0.18 | 0.18 | 6.0 | 60K / 70K | ❌ 잘못됨 |
| **현재 설정** | 0.1 | 0.1 | 6.0 | 80K / 90K | ✅ 올바름 |

**Voxel Size가 중요한 이유:**
- 작은 voxel (0.1) → 높은 해상도, 더 정확한 검출
- 큰 voxel (0.18) → 낮은 해상도, 작은 객체 누락 가능
- 0.18 사용 시 성능 저하 예상: 17.94% mAP → 48.86% mAP (논문 기준 약 2.7배 차이)

### Point Cloud Range

| 구분 | Range (m) | 비고 |
|------|-----------|------|
| **Config** | [-35, -35, -2] ~ [35, 35, 4] | 70m × 70m |
| **실제 데이터** | 더 넓은 범위 가능 | Ouster OS1-128 |

### 학습 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| Epochs | 50 | GitHub 기본값 |
| Batch Size | 2 per GPU | GPU 메모리에 따라 조정 |
| Learning Rate | 0.003 | adam_onecycle |
| Optimizer | adam_onecycle | - |
| Weight Decay | 0.01 | - |
| Gradient Clip | 10 | - |

### 클래스 설정

#### GitHub 공식 설정
```yaml
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
```

#### 실제 데이터 (Sequence 5)
- **있는 클래스**: UtilityVehicle, Pedestrian, Bike (→Cyclist), Pole, Tree, Railing, Bike Rack
- **없는 클래스**: Car

#### 해결 방법
**옵션 1**: 현재 모델 사용 (Pedestrian, Cyclist만 검출)
**옵션 2**: CONFIG 수정
```yaml
CLASS_NAMES: ['UtilityVehicle', 'Pedestrian', 'Cyclist']

# Anchor config도 수정
ANCHOR_GENERATOR_CONFIG:
    - class_name: 'UtilityVehicle'  # 원래: 'Vehicle'
```

---

## 🚀 학습 명령어

### 전체 학습 파이프라인

```bash
# 0. 작업 디렉토리 이동
cd /media/withsu/ROBOT_SSD_0/coda_clone2/coda-models

# 1. 환경변수 설정
export PYTHONPATH=$PWD:$PYTHONPATH

# 2. 데이터 변환 (최초 1회만)
python tools/create_data.py coda \
  --root-path ./ \
  --out-dir ./data \
  --workers 8

# 3. PKL 생성 (최초 1회 또는 데이터 변경 시)
python -B -m pcdet.datasets.coda.coda_dataset \
  create_coda_infos \
  tools/cfgs/dataset_configs/da_coda_oracle_dataset_3class.yaml

# 4. 학습 실행
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  tools/train.py \
  --cfg_file tools/cfgs/coda_models/pointpillar_1x.yaml \
  --launcher pytorch \
  --batch_size 2
```

### Multi-GPU 학습

```bash
# 4개 GPU 사용
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  tools/train.py \
  --cfg_file tools/cfgs/coda_models/pointpillar_1x.yaml \
  --launcher pytorch \
  --batch_size 8  # 총 batch size (GPU당 2)
```

### 학습 재개 (Resume)

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  tools/train.py \
  --cfg_file tools/cfgs/coda_models/pointpillar_1x.yaml \
  --launcher pytorch \
  --batch_size 2 \
  --ckpt output/.../ckpt/checkpoint_epoch_30.pth  # 재개할 checkpoint
```

---

## 📈 평가 명령어

### Validation Set 평가

```bash
python eval_model.py \
  --cfg_file tools/cfgs/coda_models/pointpillar_1x.yaml \
  --ckpt output/cfgs/coda_models/pointpillar_1x/defaultLR0.003000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth \
  --split val \
  --batch_size 1
```

### Test Set 평가

```bash
python eval_model.py \
  --cfg_file tools/cfgs/coda_models/pointpillar_1x.yaml \
  --ckpt output/.../checkpoint_epoch_50.pth \
  --split test \
  --batch_size 1
```

### 모든 Checkpoint 평가

```bash
# output/.../ckpt/ 아래의 모든 checkpoint 평가
for ckpt in output/.../ckpt/checkpoint_epoch_*.pth; do
  epoch=$(basename $ckpt | grep -oP '\d+')
  echo "Evaluating epoch $epoch..."
  python eval_model.py \
    --cfg_file tools/cfgs/coda_models/pointpillar_1x.yaml \
    --ckpt $ckpt \
    --split val
done
```

### 결과 확인

```bash
# 로그 파일에서 AP 확인
grep "AP_R40" evaluation_val_epoch50.log

# 또는 간단히
grep -A3 "Pedestrian AP_R40" evaluation_val_epoch50.log
grep -A3 "Cyclist AP_R40" evaluation_val_epoch50.log
grep -A3 "Car AP_R40" evaluation_val_epoch50.log
```

---

## ✅ 다른 데이터셋 적용 시 체크리스트

### 1. 데이터 포맷 확인
- [ ] **이미지 확장자**: `.jpg` or `.png`?
- [ ] **LiDAR 포맷**: `.bin` (KITTI), `.pcd`, or `.ply`?
- [ ] **3D bbox 포맷**: JSON, TXT, or XML?
- [ ] **좌표계**: Camera or LiDAR 기준?

### 2. 클래스 확인
```bash
# 실제 데이터의 클래스 확인
find <data_path> -name "*.json" | xargs cat | grep '"classId"' | cut -d'"' -f4 | sort -u

# 또는 변환 후
cat data/kitti_format/training/label_all/*.txt | cut -d' ' -f1 | sort -u
```

- [ ] **클래스 이름**: Config의 `CLASS_NAMES`와 일치?
- [ ] **클래스 매핑**: Converter의 `class_map`에 모두 정의?

### 3. Converter 수정 체크리스트

**파일**: `tools/data_converter/<dataset>_converter.py`

- [ ] **이미지 확장자** 수정
```python
if "2d_rect"==modality:
    filetype = "png"  # 또는 "jpg"
```

- [ ] **LiDAR 경로** 확인
```python
bin_path = join(self.load_dir, "3d_comp", ...)  # 실제 경로로
```

- [ ] **클래스 매핑** 추가/수정
```python
self.dataset_to_kitti_class_map = {
    'YOUR_CLASS_1': 'Car',
    'YOUR_CLASS_2': 'Pedestrian',
    ...
}
```

- [ ] **2D bbox 처리** (없으면 더미값)
```python
bounding_box = [0.0, 0.0, 50.0, 50.0]  # LiDAR만 사용 시
```

### 4. Dataset 코드 수정 체크리스트

**파일**: `pcdet/datasets/<dataset>/<dataset>_dataset.py`

- [ ] **Split 파라미터 버그** 확인
```python
split_dir = self.root_path / 'ImageSets' / (split + '.txt')  # NOT self.split!
```

- [ ] **이미지 확장자**
```python
img_file = root_split_path / 'image_0' / ('%s.png' % idx)
```

- [ ] **Data path**
```python
data_path=ROOT_DIR / 'data' / '<your_dataset>',
```

### 5. Config 수정 체크리스트

**파일**: `tools/cfgs/<dataset>_models/<model>.yaml`

- [ ] **클래스 이름** 확인
```yaml
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']  # 실제 클래스로
```

- [ ] **Anchor 설정** (클래스별)
```yaml
ANCHOR_GENERATOR_CONFIG:
    - class_name: 'Car'  # CLASS_NAMES와 일치해야 함!
      anchor_sizes: [[4.7, 2.1, 1.7]]  # 실제 객체 크기 기반
```

- [ ] **Point Cloud Range**
```yaml
POINT_CLOUD_RANGE: [-35.0, -35.0, -2.0, 35.0, 35.0, 4.0]  # 데이터 범위에 맞게
```

- [ ] **Voxel Size** (중요!)
```yaml
VOXEL_SIZE: [0.1, 0.1, 6.0]  # 작을수록 정밀, 메모리 많이 사용
MAX_NUMBER_OF_VOXELS: {'train': 80000, 'test': 90000}
```

### 6. PKL 생성 전 체크

- [ ] **Python 캐시 비활성화**: `python -B` 사용
- [ ] **이전 PKL 삭제**: `rm data/<dataset>/coda_infos_*.pkl`
- [ ] **ImageSets 확인**: `train.txt`, `val.txt`, `test.txt` 존재?
- [ ] **샘플 ID 확인**: ImageSets의 ID와 실제 파일명 일치?

### 7. 학습 전 체크

- [ ] **GPU 메모리**: Batch size 조정 (2, 4, 8, ...)
- [ ] **Epochs**: 데이터셋 크기에 따라 (50-80 권장)
- [ ] **Learning Rate**: 0.001-0.003 (batch size에 비례)
- [ ] **WandB**: 사용 여부 설정

### 8. 디버깅 팁

#### PKL 생성 실패 시
```bash
# 1. 샘플 수 확인
python -c "
from pcdet.datasets.coda.coda_dataset import CODataset
from pcdet.config import cfg_from_yaml_file, cfg
cfg_from_yaml_file('tools/cfgs/dataset_configs/<config>.yaml', cfg)
dataset = CODataset(cfg.DATA_CONFIG, training=True, logger=None)
print(f'Total samples: {len(dataset)}')
"

# 2. 첫 샘플 로드 테스트
python -c "
from pcdet.datasets.coda.coda_dataset import CODataset
...
dataset = CODataset(...)
print(dataset[0])
"
```

#### 학습 시작 실패 시
```bash
# 1. Config 검증
python tools/train.py --cfg_file <config> --launcher none --epochs 1 --batch_size 1

# 2. 단일 배치 오버피팅 테스트
# Config에서 일시적으로:
# OPTIMIZATION.NUM_EPOCHS: 1
# DATA_PROCESSOR에서 샘플 1개만 사용
```

#### 평가 실패 시
```bash
# 1. Checkpoint 로드 확인
python -c "
import torch
ckpt = torch.load('<checkpoint>.pth')
print('Keys:', ckpt.keys())
print('Epoch:', ckpt['epoch'])
print('Model keys:', list(ckpt['model_state'].keys())[:5])
"

# 2. 단일 샘플 추론 테스트
python eval_model.py ... --batch_size 1 --split val
```

---

## 📝 주요 파일 경로 요약

```
coda-models/
├── data/
│   └── coda_kitti_format/
│       ├── ImageSets/
│       │   ├── train.txt (140 samples)
│       │   ├── val.txt (30 samples)
│       │   └── test.txt (30 samples)
│       ├── training/
│       │   ├── velodyne/         # LiDAR 데이터 (.bin)
│       │   ├── label_all/        # 3D bbox labels
│       │   ├── image_0/          # 이미지 (선택)
│       │   └── calib/            # Calibration
│       ├── coda_infos_train.pkl    (1.2MB)
│       ├── coda_infos_val.pkl      (243KB)
│       ├── coda_infos_test.pkl     (241KB)
│       └── coda_dbinfos_train.pkl  (1.5MB)
│
├── output/
│   └── cfgs/coda_models/pointpillar_1x/defaultLR0.003000OPTadam_onecycle/
│       ├── ckpt/
│       │   ├── checkpoint_epoch_1.pth
│       │   ├── ...
│       │   └── checkpoint_epoch_50.pth
│       ├── log_train_<timestamp>.txt
│       └── eval/
│           └── eval_val/
│               └── eval_log.txt
│
├── tools/
│   ├── cfgs/
│   │   ├── coda_models/
│   │   │   └── pointpillar_1x.yaml         # 모델 config
│   │   └── dataset_configs/
│   │       └── da_coda_oracle_dataset_3class.yaml  # 데이터 config
│   ├── data_converter/
│   │   └── coda_converter.py               # 데이터 변환
│   ├── train.py                            # 학습 스크립트
│   └── train_utils/
│       └── train_utils.py                  # 학습 유틸
│
├── pcdet/datasets/coda/
│   └── coda_dataset.py                     # Dataset 클래스
│
└── eval_model.py                           # 평가 스크립트 (새로 생성)
```

---

## 🔧 트러블슈팅

### Q1: "Total samples for CODa dataset: 0"
**A**: `coda_dataset.py:136`에서 `split` 파라미터 버그 확인
```python
split_dir = self.root_path / 'ImageSets' / (split + '.txt')  # NOT self.split!
```

### Q2: PKL 수정 후에도 변화 없음
**A**: Python 모듈 캐싱 때문. `-B` 플래그 사용
```bash
python -B -m pcdet.datasets.coda.coda_dataset create_coda_infos ...
```

### Q3: "RuntimeError: Default process group has not been initialized"
**A**: `LOCAL_RANK` 환경변수 지원 확인 (`train.py:53-57`)

### Q4: "wandb.errors.errors.UsageError: api_key not configured"
**A**: WandB 체크 누락. 다음 파일들 확인:
- `train_utils.py:72, 42`
- `train.py:313`

### Q5: Car AP가 0%
**A**: 데이터에 Car 클래스가 없음. UtilityVehicle 확인 필요

---

## 📚 참고 자료

- **CODa Dataset**: https://utexas.app.box.com/v/coda-paper
- **OpenPCDet**: https://github.com/open-mmlab/OpenPCDet
- **GitHub Repo**: https://github.com/ut-amrl/coda-models

---

**작성일**: 2025-11-02
**작성자**: Training Pipeline Documentation
**데이터셋**: CODa (UT Campus Object Dataset)
**모델**: PointPillar (Oracle, 3-class)

---

## 📊 최종 학습 결과 (PointPillar, Epoch 50)

### Validation Set 평가 결과 (30 samples)

**Pedestrian (보행자)**

| IoU | 3D AP | BEV AP |
|-----|-------|--------|
| 0.50 | 85.23% | 86.85% |
| 0.25 | 88.13% | 88.13% |

**Cyclist (자전거)**

| IoU | 3D AP | BEV AP |
|-----|-------|--------|
| 0.50 | 12.69% | 12.70% |
| 0.25 | 13.32% | 13.32% |

**Car**

| IoU | 3D AP | BEV AP |
|-----|-------|--------|
| 0.50 | 0.00% | 0.00% |
| 0.25 | N/A | N/A |

**참고사항:**
- Car 클래스는 Sequence 5에 데이터가 없어 0% AP
- Pedestrian 검출 성능 우수 (85-88%)
- Cyclist는 Easy/Moderate 난이도에서 낮은 성능 (데이터 불균형: Training 968 Pedestrian vs 149 Cyclist)
- 학습 시간: 약 40분 (RTX 3090 1 GPU 기준)
- 학습 파라미터: Voxel Size [0.1, 0.1, 6.0], Batch Size 2, 50 Epochs
