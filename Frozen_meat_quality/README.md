# Frozen-Meat Packaging Quality Detection

This project was completed for MMA3001 Numerical Method and Machine Learning

The aim of this project is to use object detection to identify packaging and meat defects in pork rasher images. The model detects the location of the defect and classifies it into one of five classes. 


## Target Classes

1. `loose-meat`
2. `packaging-error`
3. `twisted-meat`
4. `unsealed`
5. `wrinkle`

## Dataset

The dataset used in this project is downloaded from Roboflow when the notebook is run

- Workspace: `hello-2aqe0`
- Project: `pork-rasher-error-packaging`
- Version: `4`
- Export format: `yolo26`
- Dataset licence recorded in the provided `data.yaml`: **CC BY 4.0**

The dataset itself is not uploaded to this Github repository. The Roboflow API key is entered separately in Google Colab and is not saved in the code or repository.

### Audited dataset

| Split | Images | Empty labels | loose-meat | packaging-error | twisted-meat | unsealed | wrinkle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 3349 | 684 | 66 | 1712 | 72 | 1055 | 18 |
| Validation | 120 | 14 | 3 | 34 | 3 | 81 | 0 |
| Test | 80 | 12 | 3 | 23 | 4 | 49 | 0 |

There are **3549 images** and **3123 labelled objects**. The class distribution is highly imbalanced. `wrinkle` has only 18 annotations and has no validation or test examples, so the model's performance on the `wrinkle` class cannot be properly evaluated.  

`dataset_errors.csv` records 710 empty-label files (684 train, 14 validation, 12 test). These are diagnostic entries, not automatically corrupted data; they may represent valid images containing no labelled defect.

## Computational Approach

Two pretrained YOLO26 object detectors are compared using the same main training settings:

- **Baseline:** YOLO26n
- **Alternative:** YOLO26s
- Epoch limit: 30
- Image size: 640
- Batch size: 16
- Seed: 42
- Model selection metric: validation mAP@50-95

The test set is not used for model selection.

## Measured Results

### Validation comparison

| Model | Precision | Recall | mAP@50 | mAP@50-95 | Inference (ms/image) | Training time (min) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | 0.8573 | 0.4685 | 0.5593 | 0.3638 | 12.47 | 28.50 |
| **YOLO26s** | 0.6912 | **0.6972** | **0.6587** | **0.3983** | **11.20** | 31.55 |

YOLO26s was chosen as the final model because it had the higher validation mAP50-95,MaAP50, and recall. 
Although YOLO26n had higher precision, YOLO26s was better at finding more of the actual defects. Its training time was also slightly longer. 

### Final test result - selected YOLO26s

| Metric | Result |
|---|---:|
| Precision | 0.4394 |
| Recall | 0.6513 |
| mAP@50 | 0.5902 |
| mAP@50-95 | 0.2776 |

The lower test mAP@50-95 than validation mAP@50-95 shows that generalisation remains limited. The severe class imbalance and very small support for rare classes are important limitations.

### Image-size sensitivity

| Image size | mAP@50-95 | Inference (ms/image) |
|---|---:|---:|
| 512 | 0.3855 | **6.98** |
| 640 | **0.3983** | 8.74 |

Using 640 pixels gave slightly higher validation accuracy, while 512 pixels reduced inference time. The project keeps 640 pixels because accuracy was the primary selection objective.

## Colab Notebook

The development notebook is available here:

https://colab.research.google.com/drive/1XHTmUWRXnp9ZdVKz3vzFldd1IE9jKkrd?usp=sharing

The repository also contains a standalone copy in `notebooks/`.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── Frozen_Meat_Packaging_Quality_FINAL.ipynb
├── src/
│   ├── __init__.py
│   └── utils.py
├── tests/
│   └── test_utils.py
├── scripts/
│   ├── generate_docs.py
│   └── run_tests.py
├── docs/
│   ├── index.html
│   └── test_report.html
├── outputs/
│   ├── best_frozen_meat.pt
│   ├── dataset_audit.csv
│   ├── dataset_errors.csv
│   ├── model_comparison.csv
│   ├── sensitivity.csv
│   ├── result_summary.txt
│   └── figures/
└── report/
```

## Installation

```bash
python -m pip install -r requirements.txt
```

For model training, a CUDA-capable GPU is recommended. The reported experiment used a Tesla T4 in Google Colab.

## Run the Notebook

1. Open `notebooks/Frozen_Meat_Packaging_Quality_FINAL.ipynb` in Google Colab.
2. Select a T4 GPU runtime.
3. Run the notebook from top to bottom.
4. Enter the Roboflow API key when requested.
5. Do not hard-code or commit the API key.

## Testing

Run the unit tests with:

```bash
python scripts/run_tests.py
```

The current repository test suite contains simple checks for YAML class loading, dataset auditing, dataset-error detection, model selection, and invalid/valid image handling. The generated report is stored at `docs/test_report.html`.

## HTML Source Documentation

Generate documentation with:

```bash
python scripts/generate_docs.py
```

The generated API documentation is stored at `docs/index.html`.

## Main Outputs

- `outputs/best_frozen_meat.pt` - selected YOLO26s weights
- `outputs/model_comparison.csv` - baseline vs alternative validation metrics
- `outputs/sensitivity.csv` - input-size sensitivity results
- `outputs/dataset_audit.csv` - split and class counts
- `outputs/dataset_errors.csv` - dataset diagnostic issues
- `outputs/result_summary.txt` - concise measured-result summary
- `outputs/figures/` - confusion matrices, training plots and example predictions

## Limitations

- Strong class imbalance: `packaging-error` and `unsealed` dominate the annotations.
- `wrinkle` has no validation/test annotations, so no validated performance claim can be made for this class.
- `loose-meat` and `twisted-meat` have only a few validation/test instances.
- Each candidate model was trained once using seed 42; small differences may be affected by random variation.
- The current project detects labelled visual defects. It does not prove industrial food safety, seal integrity, shelf life, or a complete pass/fail inspection policy.

## AI Use

This project, its computational workflow, model training, evaluation, data auditing, and documentation were independently designed, implemented, and authored by the student. AI tools (such as ChatGPT) were used strictly in an auxiliary and consultative capacity for debugging runtime errors, troubleshooting library syntax, and receiving organizational feedback on formatting. All technical decisions, experimental runs, and result interpretations remain entirely the student's work and responsibility.

## Licence

The code in this repository is released under the MIT License. Dataset licensing and ownership remain with the dataset provider; the provided Roboflow metadata records the dataset licence as CC BY 4.0.
