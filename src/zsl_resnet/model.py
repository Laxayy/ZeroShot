import torch.nn as nn
import torchvision


class CustomResNet(nn.Module):
    def __init__(self, num_predicates, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.fc_predicates = nn.Linear(2048, num_predicates)
        self.fc_classes = nn.Linear(num_predicates, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        predicates = self.fc_predicates(x)
        class_logits = self.fc_classes(predicates)
        return class_logits, predicates