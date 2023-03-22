from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from resnet import ResNet18


def get_pretrained_resnet18(pretrained, state_dict_path):
    model = models.resnet18(pretrained=pretrained)
    if pretrained and state_dict_path == '':
        print('ImageNet pretrained parameters loaded.\n')
    
    num_ftrs = model.fc.in_features
    
    if state_dict_path:
        print('state_dict_path:', state_dict_path)
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        if 'weight' in state_dict.keys():
            state_dict = state_dict['weight']
        new_dict = {k:state_dict[k] for k in list(state_dict.keys()) if not k.startswith('l')} # 去掉2层MLP的参数
        new_dict = {k:new_dict[k] for k in list(new_dict.keys()) if not k.startswith('classifier')} # 去掉classifier的参数
        try:
            print('try loading state_dict of USCL resnet18')
            new_dict2 = OrderedDict()
            dic = {'features.0': 'conv1',
                   'features.1': 'bn1',
                   'features.4': 'layer1',
                   'features.5': 'layer2',
                   'features.6': 'layer3',
                   'features.7': 'layer4',
                   'linear': 'fc1'}
            for key in new_dict.keys():
                if key.startswith('features.'):
                    new_dict2[dic[key[:10]]+key[10:]] = new_dict[key]
                elif key.startswith('linear'):
                    new_dict2['fc1'+key[6:]] = new_dict[key]
            model_dict = model.state_dict()
            model_dict.update(new_dict2)
            model.load_state_dict(model_dict)
            print('state_dict of USCL resnet18 loaded')
        except:
            raise ValueError('state dict do not match')
    else:
        print('do not use semi/self-supervised pretrained state_dicts')
    
    return model