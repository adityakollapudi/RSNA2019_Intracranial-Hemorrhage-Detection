import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import math

try:
    import pretrainedmodels
except ImportError:
    pretrainedmodels = None

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    EfficientNet = None


def _get_densenet_features(arch_name, pretrained=True):
    """Safely obtain DenseNet feature extractor compatible across torchvision versions."""
    fn = getattr(torchvision.models, arch_name)
    try:
        # torchvision >= 0.13
        if arch_name == 'densenet121':
            from torchvision.models import DenseNet121_Weights
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
        elif arch_name == 'densenet169':
            from torchvision.models import DenseNet169_Weights
            weights = DenseNet169_Weights.DEFAULT if pretrained else None
        else:
            weights = None
        return fn(weights=weights).features
    except (ImportError, AttributeError):
        # legacy torchvision
        return fn(pretrained=pretrained).features


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class se_resnext50_32x4d(nn.Module):
    def __init__(self, pretrained=True):
        super(se_resnext50_32x4d, self).__init__()
        if pretrainedmodels is None:
            raise RuntimeError(
                "pretrainedmodels is required for se_resnext50_32x4d.\n"
                "Install it via: pip install pretrainedmodels==0.7.4"
            )
        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](
            num_classes=1000,
            pretrained='imagenet' if pretrained else None
        )
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x


class se_resnext101_32x4d(nn.Module):
    def __init__(self, pretrained=True):
        super(se_resnext101_32x4d, self).__init__()
        if pretrainedmodels is None:
            raise RuntimeError(
                "pretrainedmodels is required for se_resnext101_32x4d.\n"
                "Install it via: pip install pretrainedmodels==0.7.4"
            )
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](
            num_classes=1000,
            pretrained='imagenet' if pretrained else None
        )
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x


class DenseNet169_change_avg(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet169_change_avg, self).__init__()
        self.densenet169 = _get_densenet_features('densenet169', pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet169(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class DenseNet121_change_avg(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet121_change_avg, self).__init__()
        self.densenet121 = _get_densenet_features('densenet121', pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        return x


