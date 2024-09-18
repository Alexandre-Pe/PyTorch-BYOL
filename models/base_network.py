import torchvision.models as models
import torch
from models.mlp_head import MLPHead


class ResNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Model {kwargs['name']} not available.")

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)

class EfficientNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(EfficientNet, self).__init__()
        if kwargs['name'] == 'efficientnet_b0':
            efficientnet = models.efficientnet_b0(weights=None)
        elif kwargs['name'] == 'efficientnet_b1':
            efficientnet = models.efficientnet_b1(weights=None)
        elif kwargs['name'] == 'efficientnet_b2':
            efficientnet = models.efficientnet_b2(weights=None)
        elif kwargs['name'] == 'efficientnet_b3':
            efficientnet = models.efficientnet_b3(weights=None)
        elif kwargs['name'] == 'efficientnet_b4':
            efficientnet = models.efficientnet_b4(weights=None)
        elif kwargs['name'] == 'efficientnet_b5':
            efficientnet = models.efficientnet_b5(weights=None)
        elif kwargs['name'] == 'efficientnet_b6':
            efficientnet = models.efficientnet_b6(weights=None)
        elif kwargs['name'] == 'efficientnet_b7':
            efficientnet = models.efficientnet_b7(weights=None)
        else:
            raise ValueError(f"Model {kwargs['name']} not available.")

        self.encoder = torch.nn.Sequential(*list(efficientnet.children())[:-1])
        self.projetion = MLPHead(in_channels=efficientnet.classifier[1].in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
