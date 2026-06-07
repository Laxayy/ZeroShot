import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from utils import (
    load_animal_classes,
    generate_text_prompts,
    normalize_embeddings,
    save_results_to_csv,
    sort_csv
)


def classify_images(test_image_dir, animal_classes, text_embeddings, processor, model, device):
    test_images = [
        os.path.join(test_image_dir, f)
        for f in os.listdir(test_image_dir)
        if f.endswith(".jpg")
    ]
    print(f"Found {len(test_images)} images for classification.")
    results = []

    for img_path in test_images:
        image = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_embeddings = model.get_image_features(**image_inputs)
            image_embeddings = normalize_embeddings(image_embeddings)

            similarity_scores = image_embeddings @ text_embeddings.T
            best_match_idx = similarity_scores.argmax(dim=-1).item()
            predicted_class = animal_classes[best_match_idx]
            results.append({"Image Path": img_path, "Predicted Class": predicted_class})

            print(f"Image: {os.path.basename(img_path)}, Predicted Class: {predicted_class}")
    return results


def run_clip_classifier(dataset_path, output_csv, model_name='openai/clip-vit-large-patch14'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    animal_classes = load_animal_classes(f'{dataset_path}/classes.txt')
    print(f"Loaded {len(animal_classes)} animal classes.")

    text_prompts = generate_text_prompts(animal_classes)
    with torch.no_grad():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
        text_embeddings = model.get_text_features(**text_inputs)
        text_embeddings = normalize_embeddings(text_embeddings)

        num_templates = len(text_prompts) // len(animal_classes)
        aggregated_embeddings = text_embeddings.view(len(animal_classes), num_templates, -1).mean(dim=1)
        aggregated_embeddings = normalize_embeddings(aggregated_embeddings)

    results = classify_images(
        f'{dataset_path}/test',
        animal_classes,
        aggregated_embeddings,
        processor,
        model,
        device
    )

    raw_csv = 'predictions.csv'
    save_results_to_csv(results, raw_csv)
    sort_csv(raw_csv, output_csv)
    print(f"Classification completed. Results saved to {output_csv}.")