# Pixel-Play-2025

## Overview

This repository contains the code developed for the PIXEL Play 2025 competition, where I secured 1st place. The project focuses on animal image classification using two complementary approaches:

- A custom zero-shot learning (ZSL) pipeline built on a ResNet-50 backbone with predicate supervision.
- A CLIP-based zero-shot classification pipeline using prompt engineering for high accuracy.

The objective was to classify animals from image data while handling both seen and unseen classes through semantic reasoning.

## Project Structure

- `requirements.txt` — Python dependencies.
- `src/clip_classifier/` — CLIP-based zero-shot classifier.
  - `main.py` — entry point for running the CLIP classifier.
  - `classifier.py` — image classification logic.
  - `utils.py` — helper functions for prompts, embeddings, and CSV output.
- `src/zsl_resnet/` — ResNet-based ZSL pipeline.
  - `main.py` — CLI for training and prediction.
  - `dataset.py` — dataset loader, transforms, and preprocessing.
  - `model.py` — custom ResNet model architecture.
  - `losses.py` — loss functions used during training.
  - `train.py` — training loop, validation, and checkpoint saving.
  - `predict.py` — inference for seen and unseen classes.

## Task Description

The competition required building a solution capable of classifying animal images with a focus on zero-shot generalization. The dataset included:

- `classes.txt` — class names and indices.
- `predicate-matrix-continuous.txt` — predicate annotations for each class.
- `train/` — training images organized by class.
- `test/` — unlabeled test images for inference.

The challenge was to understand whether labels were binary or continuous, select appropriate loss functions, and design models that leverage class and predicate information effectively.

## Usage

### Environment Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Pretrained Models and Data

#### CLIP Model
The CLIP classifier automatically downloads pretrained weights on first run from HuggingFace.

- **Model**: `openai/clip-vit-large-patch14` (default)
- **Size**: ~1.4 GB
- **Auto-download**: Yes, HuggingFace transformers handles this
- **Cache location**: `~/.cache/huggingface/hub/`

To change the model, specify `--model_name` with any compatible CLIP variant from [HuggingFace](https://huggingface.co/models?search=clip).

#### ResNet-50 Backbone
The ZSL ResNet model uses a pretrained ResNet-50 from `torchvision.models`.

- **Pretrained**: Yes, `pretrained=True` in model initialization
- **Size**: ~100 MB
- **Auto-download**: Yes, torchvision handles this
- **Cache location**: `~/.cache/torch/hub/`

#### Dataset
Prepare your dataset directory with the following structure:

```
dataset_path/
├── classes.txt                       # Format: "index class_name" (one per line)
├── predicate-matrix-continuous.txt   # Shape: (num_classes, num_predicates)
├── train/                            # For ZSL ResNet only
│   ├── class_name_1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class_name_2/
│       └── img3.jpg
└── test/
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

### CLIP Classifier

This pipeline uses the OpenAI CLIP model for training-free zero-shot classification.

```bash
python src/clip_classifier/main.py --dataset_path <dataset_path> --output_csv clip_predictions.csv
```

Optional model selection:

```bash
python src/clip_classifier/main.py --dataset_path <dataset_path> --output_csv clip_predictions.csv --model_name openai/clip-vit-large-patch14
```

Expected dataset layout:

- `<dataset_path>/classes.txt`
- `<dataset_path>/test/*.jpg`

### ZSL ResNet Training

Train a custom ResNet-based model using predicate supervision:

```bash
python src/zsl_resnet/main.py train --dataset_path <dataset_path> --output_filename model.pth --num_epochs 10 --batch_size 32
```

### ZSL ResNet Prediction

Run inference on test images with a saved model:

```bash
python src/zsl_resnet/main.py predict --dataset_path <dataset_path> --model_path model.pth --output_csv zsl_predictions.csv
```

Expected dataset layout:

- `<dataset_path>/classes.txt`
- `<dataset_path>/predicate-matrix-continuous.txt`
- `<dataset_path>/train/<class_name>/*.jpg`
- `<dataset_path>/test/*.jpg`

## Technical Summary

### ResNet ZSL Approach

- Used a pretrained ResNet-50 backbone with a custom head.
- Added a predicate regression branch and class logits branch.
- Learned from continuous predicate labels and class indices simultaneously.
- Evaluated using training and validation splits to monitor overfitting.
- Tested several loss strategies, including:
  - `LabelSmoothingCrossEntropy`
  - `VSELoss`
  - Mean squared error and other variations
- Achieved a top score of approximately 60 with the ResNet-based method.

### CLIP Approach

- Adopted a powerful pretrained CLIP model for zero-shot classification.
- Generated text prompts for each animal class and averaged text embeddings.
- Tuned prompt templates to improve classification performance.
- The final prompt set included:

```python
def generate_text_prompts(animal_classes):
    templates = [
        "A photo of a {}.",
        "A realistic photo of a {}.",
        "An image of a {} in the wild.",
        "A {} in its natural habitat.",
        "A close-up of a {}."
    ]
    text_prompts = [template.format(animal) for animal in animal_classes for template in templates]
    return text_prompts
```

- This approach improved the score to above 90, with prompt quality being the most important factor.

## Key Learnings

- Understanding dataset structure and label format was critical.
- Distinguished between binary and continuous supervision and selected losses accordingly.
- Learnt class and predicate relationships for zero-shot learning.
- Applied transfer learning using ResNet-50 as the base architecture.
- Implemented normalization, padding, and preprocessing for image inputs.
- Built training and validation workflows to monitor model performance.
- Performed hyperparameter tuning to optimize validation behavior.
- Researched and applied CLIP for effective zero-shot classification.

## Result

- Placed 1st in PIXEL Play 2025.
- Demonstrated two methods: a custom ZSL ResNet pipeline and a CLIP-based zero-shot classifier.
- Established a practical workflow from data exploration through model selection and prompt optimization.

## Notes

This repository captures the complete pipeline developed during the hackathon, with a focus on reproducibility and professional ML engineering practice.
