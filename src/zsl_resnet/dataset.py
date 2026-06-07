import os
import numpy as np
from glob import glob
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms


class PadToSquare:
    def __call__(self, image):
        width, height = image.size
        max_side = max(width, height)
        padding = (
            (max_side - width) // 2,
            (max_side - height) // 2,
            (max_side - width + 1) // 2,
            (max_side - height + 1) // 2
        )
        return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')


class CustomDataset(data.Dataset):
    def __init__(self, image_dir, classes_file, predicate_file, transform=None):
        self.transform = transform
        self.predicate_continuous_mat = np.array(np.genfromtxt(predicate_file, dtype='float32'))
        self.predicate_continuous_mat[self.predicate_continuous_mat > 0] /= 100.0

        self.class_to_index = {}
        with open(classes_file) as f:
            for line in f:
                index, class_name = line.split()
                self.class_to_index[class_name.strip()] = int(index) - 1

        self.img_names = []
        self.img_index = []
        for class_name, class_index in self.class_to_index.items():
            folder_dir = os.path.join(image_dir, class_name)
            files = glob(os.path.join(folder_dir, '*.jpg'))
            for file_name in files:
                self.img_names.append(file_name)
                self.img_index.append(class_index)

        print(f"Loaded {len(self.img_names)} images from {image_dir}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = self.img_names[index]
        image = Image.open(img_path)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        img_index = self.img_index[index]
        predicate = self.predicate_continuous_mat[img_index, :]
        return image, predicate, img_path, img_index


def get_train_transforms():
    return transforms.Compose([
        PadToSquare(),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


def get_test_transforms():
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])