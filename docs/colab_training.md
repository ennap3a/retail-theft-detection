# Colab Training Plan

## What I Can and Cannot Do From Here

I can prepare the Colab notebook and training workflow. I cannot use a random third-party Kaggle/Google account, bypass login, or access your private Google Colab runtime from this local desktop session.

To get real metrics, you need to run:

```text
notebooks/train_yolo_colab.ipynb
```

inside your Google Colab account with a GPU runtime.

## Recommended Dataset Starting Points

Best first candidate:

- Kaggle: `simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm`

Why: it is described as a synthetic CCTV retail shoplifting dataset with YOLO annotations, which is closest to the object-detection training layer we need.

Other useful references:

- GitHub PoseLift / RetailS: better for pose/action anomaly modeling later, not a direct replacement for object-detection labels.
- Kaggle `thatbrownkid/thefttest`: based on surveillance anomaly detection video data, useful for event/action experiments.
- Kaggle `aravindm27/custom-theft-detection`: may be useful if the label quality and classes match our needs.

## No-Login Alternatives

The notebook supports three dataset modes:

```python
DATASET_SOURCE = "kaggle"
DATASET_SOURCE = "url_zip"
DATASET_SOURCE = "github_zip"
```

Use `url_zip` if you have a public direct YOLO dataset ZIP link:

```python
DATASET_SOURCE = "url_zip"
PUBLIC_DATASET_ZIP_URL = "https://example.com/dataset.zip"
```

Use `github_zip` if a public GitHub repository contains a YOLO dataset:

```python
DATASET_SOURCE = "github_zip"
GITHUB_REPO_ZIP_URL = "https://github.com/user/repo/archive/refs/heads/main.zip"
```

The ZIP still needs to contain a YOLO-style `data.yaml`, images, and labels. If it does not, the notebook will stop and ask you to inspect/restructure the dataset.

## How To Run

1. Open Google Colab.
2. Upload/open [notebooks/train_yolo_colab.ipynb](../notebooks/train_yolo_colab.ipynb).
3. Set runtime to GPU.
4. Run all cells.
5. If using Kaggle, upload `kaggle.json`. If using a public ZIP/GitHub ZIP, set the URL variables and no Kaggle login is needed.
6. Wait for training/evaluation.
7. Download or copy from Google Drive:
   - `best.pt`
   - `*_evaluation_report.json`
   - training plots

## After Training

Place `best.pt` in your local project, for example:

```text
models/retail_theft_v1_best.pt
```

Then the local detector can be configured to use:

```python
DetectorSettings(
    object_model="models/retail_theft_v1_best.pt",
    person_class_ids=(0,),
    product_class_ids=(1,),
    bag_class_ids=(2, 3),
)
```

The exact class IDs depend on the dataset's `data.yaml`. Always check the `names:` section before using the model.

## Metrics To Report

The notebook writes:

- `mAP50_95`
- `mAP50`
- `mAP75`
- `mean_precision`
- `mean_recall`

These are object-detection metrics, not full theft-event metrics. The next stage after this is event-level evaluation with human labels.
