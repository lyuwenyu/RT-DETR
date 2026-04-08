# Training Dataset — TRACO6 (COCO format)

**Path:** `data/traco6_rt_detr/`  
**Format:** COCO JSON (same images and annotations used for all YOLO and RT-DETR training runs)

---

## Directory Layout

```
data/traco6_rt_detr/
├── dataset_meta.json
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
└── images/
    ├── train/   (34,864 images)
    ├── val/     ( 4,352 images)
    └── test/    ( 4,370 images)
```

---

## Split Summary

| Split  | Images | Annotations | Avg obj/img |
|--------|-------:|------------:|------------:|
| Train  | 34,864 |      35,940 |       1.031 |
| Val    |  4,352 |       4,474 |       1.028 |
| Test   |  4,370 |       4,520 |       1.034 |
| **Total** | **43,586** | **44,934** | |

**Total image data on disk:** ~23.4 GB  
**Image resolution:** 1280 × 960 px (JPEG)

---

## Classes (23)

COCO category IDs are **1-indexed** (id 1 = bigVan … id 23 = carCarrier).  
`remap_mscoco_category: False` is set in the training config — IDs are used as-is.

| ID | Class Name       | Train | Val | Test |
|----|------------------|------:|----:|-----:|
|  1 | bigVan           | 1,609 | 201 |  ~196 |
|  2 | bike_empty       | 1,619 | 203 |  ~203 |
|  3 | bus              | 1,270 | 158 |  ~158 |
|  4 | camperCaravan    | 1,628 | 204 |  ~204 |
|  5 | hatchback        | 1,622 | 202 |  ~202 |
|  6 | miniVan_type1    | 1,642 | 202 |  ~202 |
|  7 | miniVan_type2    | 1,617 | 207 |  ~207 |
|  8 | miniVan_type3    | 1,635 | 198 |  ~198 |
|  9 | motorcycle       | 1,779 | 214 |  ~214 |
| 10 | truckOpen        | 1,458 | 180 |  ~180 |
| 11 | sedan            | 1,631 | 203 |  ~203 |
| 12 | sportsClosed     | 1,617 | 199 |  ~199 |
| 13 | sportsOpen       | 1,715 | 216 |  ~216 |
| 14 | stationwagon     | 1,597 | 197 |  ~197 |
| 15 | trailerHighLong  | 1,274 | 158 |  ~158 |
| 16 | trailerHighShort | 1,633 | 205 |  ~205 |
| 17 | trailerLargeHeavy| 1,681 | 212 |  ~212 |
| 18 | trailerLowSmall  | 1,362 | 175 |  ~175 |
| 19 | truck            | 1,605 | 201 |  ~201 |
| 20 | pickupPassenger  | 1,611 | 202 |  ~202 |
| 21 | glassWagon       | 1,376 | 166 |  ~166 |
| 22 | bike_full        | 1,604 | 198 |  ~198 |
| 23 | carCarrier       | 1,355 | 173 |  ~173 |
| | **Total** | **35,940** | **4,474** | **4,520** |

**Class balance:** very well balanced — min 1,270 (bus), max 1,779 (motorcycle) in train, a 1.4× range.

---

## Key Characteristics

- **Single-object-per-image** dataset: average 1.03 annotations/image across all splits. Each image shows one vehicle captured from a fixed overhead or side camera.
- **Identical to the YOLO training set** — the `data/traco6_rt_detr/` COCO-format dataset is derived from the same source images and labels as `data/traco6_yolo_fixed/`, converted to COCO JSON format for the RT-DETR training pipeline.
- **No augmentation baked into the dataset** — all augmentation (RandomPhotometricDistort, RandomZoomOut, RandomIoUCrop, RandomHorizontalFlip, multi-scale resize) is applied on-the-fly during training via the dataloader config.
- **Augmentation policy:** heavy augmentation stops 3 epochs before the end of training (epoch 117/120) to allow the model to consolidate with clean samples.

---

## Model config reference

The training config that references this dataset:  
`third_party/RT-DETR/rtdetrv2_pytorch/configs/custom/rtdetrv2_r18vd_traco6.yml`

Training output directory:  
`runs/train/rtdetrv2_r18vd_traco6/`
