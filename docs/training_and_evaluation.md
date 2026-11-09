# Training and Evaluation Workflow

## Why This Stage Matters

The current detector uses pretrained YOLO weights. That is useful for a demo, but it is not enough for a sellable retail theft product because retail products vary by store, shelf, camera angle, lighting, and distance.

We need two kinds of evaluation:

1. Object detection accuracy: can the model detect people, products, and bags?
2. Event accuracy: can the full system correctly flag suspicious concealment events without too many false alerts?

This stage starts with object detection training and mAP evaluation. Event-level precision/recall comes after we collect human-reviewed event labels.

## Recommended Free Annotation Tools

Use CVAT, Label Studio, or Roboflow free/community options if available to you. CVAT is a strong choice because it can export Ultralytics YOLO detection datasets.

The recommended classes for the first custom model are:

- `person`
- `held_product`
- `bag`
- `basket_cart`

Do not try to create a separate class for every item a store can sell. For theft-review logic, we mainly need to know whether a sellable item is visible in the hand/body/bag interaction area. The product-zone change detector handles shelf/display disappearance separately.

For a store-specific SKU-recognition product, `held_product` can later become more detailed classes such as `phone`, `bottle`, `box`, `cosmetic`, `clothing_item`, etc., but that is a second product layer.

## Dataset Layout

Ultralytics YOLO detection training expects this structure:

```text
datasets/retail_theft_v1/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each label file uses YOLO normalized boxes:

```text
class_id x_center y_center width height
```

All box coordinates must be normalized from `0` to `1`.

## Commands

Create the dataset skeleton:

```bash
python -m retail_theft_detector.training init-dataset --name retail_theft_v1 --classes person held_product bag basket_cart
```

Extract frames from an uploaded local project for annotation:

```bash
python -m retail_theft_detector.training extract-frames --project-id PROJECT_ID --dataset retail_theft_v1 --every-seconds 0.5 --limit 300
```

Annotate the frames in CVAT or another tool, export as YOLO/Ultralytics YOLO, then place or split the labeled data into the dataset:

```bash
python -m retail_theft_detector.training split-labeled --images path/to/images --labels path/to/labels --dataset retail_theft_v1
```

Check whether the dataset is ready:

```bash
python -m retail_theft_detector.training check-dataset --dataset retail_theft_v1
```

Train:

```bash
python -m retail_theft_detector.training train --dataset retail_theft_v1 --model yolo11n.pt --epochs 80 --imgsz 640 --batch 8
```

Evaluate:

```bash
python -m retail_theft_detector.training evaluate --dataset retail_theft_v1 --weights runs\train\RUN_NAME\weights\best.pt --split val
```

Evaluation reports are saved under:

```text
reports/evaluation/
```

Training reports are saved under:

```text
reports/training/
```

## Colab GPU Training

If your laptop is not strong enough for training, use:

```text
notebooks/train_yolo_colab.ipynb
```

The notebook downloads a Kaggle YOLO dataset, trains on Colab GPU, evaluates, and saves `best.pt` plus an evaluation JSON report to Google Drive.

More notes:

```text
docs/colab_training.md
```

## What Accuracy Means Here

For the custom YOLO model, the main accuracy numbers are:

- `mAP50-95`: strict mean average precision across IoU thresholds
- `mAP50`: easier object-detection precision at IoU 0.50
- `mAP75`: stricter localization quality
- mean precision
- mean recall

For the complete theft-review system, we later need event-level metrics:

- alert precision
- alert recall
- false alerts per hour
- missed events
- timestamp error
- reviewer disagreement rate

## Minimum Dataset Recommendation

For a serious first evaluation:

- 300-500 labeled frames minimum for a smoke model
- 1,500-3,000 labeled frames for a meaningful first custom detector
- separate train/val/test videos so the model is tested on scenes it did not memorize
- examples of normal browsing, holding, returning, occlusion, bags, and actual/acted concealment

Do not train and test on frames from the same tiny clip and call it product accuracy. That will overstate performance.
