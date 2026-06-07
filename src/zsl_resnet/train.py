import numpy as np
import torch
import torch.optim as optim
from torch.utils import data

from dataset import CustomDataset, get_train_transforms
from model import CustomResNet
from losses import VSELoss, LabelSmoothingCrossEntropy


def train(dataset_path, num_epochs, eval_interval, learning_rate, output_filename, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_params = {'batch_size': batch_size, 'num_workers': 3}
    val_params = {'batch_size': batch_size, 'num_workers': 3}

    dataset = CustomDataset(
        f'{dataset_path}/train',
        f'{dataset_path}/classes.txt',
        f'{dataset_path}/predicate-matrix-continuous.txt',
        get_train_transforms()
    )

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.05 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = data.SubsetRandomSampler(train_indices)
    val_sampler = data.SubsetRandomSampler(val_indices)

    train_loader = data.DataLoader(dataset, sampler=train_sampler, **train_params)
    val_loader = data.DataLoader(dataset, sampler=val_sampler, **val_params)

    num_predicates = 85
    num_classes = 50
    model = CustomResNet(num_predicates=num_predicates, num_classes=num_classes).to(device)

    criterion_predicates = VSELoss(scale_factor=1.0)
    criterion_classes = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for i, (images, predicates, img_names, indexes) in enumerate(train_loader):
            if images.shape[0] < 2:
                break
            images = images.to(device)
            predicates = predicates.to(device)

            class_logits, pred_predicates = model(images)
            class_targets = indexes.to(device)

            loss_predicates = criterion_predicates(pred_predicates, predicates)
            loss_classes = criterion_classes(class_logits, class_targets)
            loss = 0.5 * loss_predicates + 0.5 * loss_classes

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % eval_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Predicate Loss: {loss_predicates.item():.4f}, Class Loss: {loss_classes.item():.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, predicates, img_names, indexes in val_loader:
                images = images.to(device)
                predicates = predicates.to(device)
                class_logits, pred_predicates = model(images)
                class_targets = indexes.to(device)

                loss_predicates = criterion_predicates(pred_predicates, predicates)
                loss_classes = criterion_classes(class_logits, class_targets)
                loss = 0.5 * loss_predicates + 0.5 * loss_classes
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), output_filename)
    print(f"Model saved to {output_filename}")