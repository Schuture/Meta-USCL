import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


class WNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(WNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


class ContrastiveModel(nn.Module):
    ''' feature extractor + projection head for contrastive learning '''

    def __init__(self, model_name='ResNet', depth=16,
                 feat_dim=128, pretrained=False):
        super(ContrastiveModel, self).__init__()
        if pretrained:
            print('ImageNet pretrained parameters loaded.')
        else:
            print('Random initialize model parameters.')
        
        backbone = self._get_basemodel(model_name, depth, pretrained)
        if model_name == 'ResNet':
            self.num_ftrs = backbone.fc.in_features
        elif model_name == 'VGG':
            self.num_ftrs = backbone.classifier._modules['0'].in_features
        elif model_name == 'DenseNet':
            self.num_ftrs = backbone.classifier.in_features
        elif model_name == 'ShuffleNet':
            self.num_ftrs = backbone.fc.in_features

        # This line of code may have bug for VGG, DenseNet.
        self.features = nn.Sequential(*list(backbone.children())[:-1]) # discard the last fc layer

        # projection MLP
        self.l1 = nn.Linear(self.num_ftrs, self.num_ftrs)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.num_ftrs, feat_dim)

        # representation dim is related to our network, but feature dim can be set.
        print('Representation dim: ', self.num_ftrs, '\n')

    def _get_basemodel(self, model_name, depth, pretrained):
        try:
            if model_name == 'ResNet':
                if depth == 18:
                    model = models.resnet18(pretrained=pretrained)
                if depth == 34:
                    model = models.resnet34(pretrained=pretrained)
                if depth == 50:
                    model = models.resnet50(pretrained=pretrained)
            if model_name == 'VGG':
                if depth == 11:
                    model = models.vgg11_bn(pretrained=pretrained)
                if depth == 13:
                    model = models.vgg13_bn(pretrained=pretrained)
                if depth == 16:
                    model = models.vgg16_bn(pretrained=pretrained)
                if depth == 19:
                    model = models.vgg19_bn(pretrained=pretrained)
            if model_name == 'DenseNet':
                if depth == 121:
                    model = models.densenet121(pretrained=pretrained)
                if depth == 161:
                    model = models.densenet161(pretrained=pretrained)
                if depth == 169:
                    model = models.densenet169(pretrained=pretrained)
                if depth == 201:
                    model = models.densenet201(pretrained=pretrained)
            if model_name == 'ShuffleNet':
                if depth == 1:
                    model = models.shufflenet_v2_x0_5(pretrained=pretrained)
                if depth == 2:
                    model = models.shufflenet_v2_x1_0(pretrained=pretrained)
                if depth == 3:
                    model = models.shufflenet_v2_x1_5(pretrained=pretrained)
                if depth == 4:
                    model = models.shufflenet_v2_x2_0(pretrained=pretrained)
            print('Feature extractor:', model_name, str(depth))
            return model
        except:
            raise ValueError('Invalid model name or depth.')

    def forward(self, x):
        h = self.features(x)

        h = F.adaptive_avg_pool2d(h, (1,1))
        h = h.squeeze()

        # print(h.shape)
        x = self.l1(h)
        x = self.relu(x)
        x = self.l2(x)
        
        return h, x # representations and the projections





