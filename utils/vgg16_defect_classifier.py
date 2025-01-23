import torchvision
import torch.nn as nn


class VGGDefectClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, 2)

    def _freeze_params(self):
        for param in self.model.features[:23].parameters():
            param.requires_grad = False

    def forward(self, x):
        scores = self.model(x)

        if self.training:
            return scores

        else:
            probs = nn.functional.softmax(scores, dim=-1)
            return probs