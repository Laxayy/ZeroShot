"""
CLIP Classifier Entry Point

A command-line interface for running zero-shot animal classification using CLIP.
Supports custom CLIP model selection and handles batch image classification.
"""

import argparse

from classifier import run_clip_classifier


def main():
    parser = argparse.ArgumentParser(description='CLIP — Zero-Shot Animal Classification')
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_csv',   type=str, default='clip_predictions.csv')
    parser.add_argument('--model_name',   type=str, default='openai/clip-vit-large-patch14')

    args = parser.parse_args()

    run_clip_classifier(
        dataset_path = args.dataset_path,
        output_csv   = args.output_csv,
        model_name   = args.model_name
    )


if __name__ == '__main__':
    main()