import torch
import torch.nn as nn
import torchvision


class resnet(nn.Module):
    """
    Resnet model, pre-trained
    """
    def __init__(self, num_classes, freeze = False):
        """
        Constructor for resnet model
        """
        super(resnet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained = True)
        for layer in self.model.parameters():
            layer.requires_grad = freeze
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, X):
        """
        forward method for resnet model
        """
        return self.model(X)