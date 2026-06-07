import csv
import torch
import pandas as pd


def load_animal_classes(file_path):
    animal_classes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            animal_name = ' '.join(parts[1:])
            animal_classes.append(animal_name)
    return animal_classes


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


def normalize_embeddings(embeddings):
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Path', 'Predicted Class'])
        for result in results:
            writer.writerow([result['Image Path'], result['Predicted Class']])
    print(f"Classification results saved to {output_file}")


def sort_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    df_sorted = df.sort_values(by='Image Path')
    df_sorted.to_csv(output_file, index=False)
    print(f"Sorted CSV file saved to {output_file}")