import os
import csv
import numpy as np
from glob import glob
from PIL import Image

import torch
import pandas as pd
from sklearn.preprocessing import normalize

from dataset import get_test_transforms
from model import CustomResNet


def predict(dataset_path, model_path, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = get_test_transforms()

    num_predicates = 85
    num_classes = 50
    model = CustomResNet(num_predicates=num_predicates, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_to_index = {}
    with open(f'{dataset_path}/classes.txt') as f:
        for line in f:
            index, class_name = line.split()
            class_to_index[int(index) - 1] = class_name.strip()

    predicate_continuous_mat = np.array(np.genfromtxt(f'{dataset_path}/predicate-matrix-continuous.txt', dtype='float32'))
    predicate_continuous_mat[predicate_continuous_mat > 0] /= 100.0

    modified_weights = model.fc_classes.weight.detach().cpu().numpy()
    seen_weights = normalize(modified_weights[:40], norm='l2')
    unseen_weights = normalize(predicate_continuous_mat[40:], norm='l2')

    def find_closest_class(image):
        with torch.no_grad():
            image = test_transform(image).unsqueeze(0).to(device)
            class_logits, predicates = model(image)
            predicates = predicates.cpu().numpy().flatten()
            predicates = normalize(predicates.reshape(1, -1), norm='l2')[0]

        seen_similarity = np.dot(seen_weights, predicates)
        unseen_similarity = np.dot(unseen_weights, predicates)

        if np.max(seen_similarity) > np.max(unseen_similarity):
            return np.argmax(seen_similarity)
        else:
            return 40 + np.argmax(unseen_similarity)

    predictions = []
    test_image_dir = f'{dataset_path}/test'

    for test_image_path in glob(os.path.join(test_image_dir, '*.jpg')):
        test_image = Image.open(test_image_path).convert('RGB')
        label_index = find_closest_class(test_image)
        animal_name = class_to_index.get(label_index, "Unknown")
        image_id = os.path.basename(test_image_path)
        predictions.append((image_id, animal_name))

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'class'])
        for image_id, animal_name in predictions:
            writer.writerow([image_id, animal_name])

    df = pd.read_csv(output_csv)
    df_sorted = df.sort_values(by='image_id')
    df_sorted.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")