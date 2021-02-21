'''
Author: Ruizhi Liao

Model script to define and instantiate
residual network models
'''

import csv
import os
import numpy as np
from math import floor, ceil
import scipy.ndimage as ndimage

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                     stride=stride, padding=dilation, 
                     groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                     stride=stride, bias=False)


class BasicBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, 
                 norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2048_7_2_1(nn.Module):
    """ A residual network 7_2_1 
    with 7 "layers", 2x2 average pooling, and 1 fully connected layer.
    """

    def __init__(self, block, blocks_per_layers, output_channels=4, 
                 norm_layer=nn.BatchNorm2d, zero_init_residual=False):
        """ Input batch_sizex2048x2048x1          ->
            Conv1 batch_sizex512x512x8            ->
            "layer1" batch_sizex256x256x8         ->
            "layer2" batch_sizex128x128x16        ->
            "layer3" batch_sizex64x64x32          ->
            "layer4" batch_sizex32x32x64          ->
            "layer5" batch_sizex16x16x128         ->
            "layer6" batch_sizex8x8x192           ->
            "layer7" batch_sizex4x4x192           ->
            average pooling batch_sizex2x2x192    ->
            fc layer batch_sizexoutput_channelsx1
        """
        super(ResNet2048_7_2_1, self).__init__()

        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, 
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 8, blocks_per_layers[0], stride=2)
        self.layer2 = self._make_layer(block, 16, blocks_per_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, blocks_per_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, blocks_per_layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, blocks_per_layers[4], stride=2)
        self.layer6 = self._make_layer(block, 192, blocks_per_layers[5], stride=2)
        self.layer7 = self._make_layer(block, 192, blocks_per_layers[6], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(768, output_channels)
        self.softmax = nn.Softmax(dim=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_of_blocks, stride=1):
        norm_layer = self._norm_layer

        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride),
                                       norm_layer(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, num_of_blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        y_logits = self.fc1(z)
        y = self.softmax(y_logits)

        return y, z, y_logits

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory, epoch=-1): 
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """

        # Saving path should be a directory where the model and configuration can be saved
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        assert os.path.isdir(save_directory)

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # If we save using the predefined names, we can load using `from_pretrained`
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, 
                                             'pytorch_model_epoch'+str(epoch)+'.bin')

        torch.save(model_to_save.state_dict(), output_model_file)

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, block, blocks_per_layers, 
                        *inputs, **kwargs):
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Instantiate the model
        model = cls(block, blocks_per_layers, **kwargs)

        # if the user has not provided the ability to load in their own state dict, but our module
        # in this case it is easier to just use save_pretrained and from_pretrained to read that 
        # saved checkpoint fully
        if state_dict is None:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model


class ResNet1024_7_2_1(nn.Module):
    """ A residual network 7_2_1 
    with 7 "layers", 2x2 average pooling, and 1 fully connected layer.
    """

    def __init__(self, block, blocks_per_layers, output_channels=4, 
                 norm_layer=nn.BatchNorm2d, zero_init_residual=False):
        """ Input batch_sizex1024x1024x1          ->
            Conv1 batch_sizex512x512x8            ->
            "layer1" batch_sizex256x256x8         ->
            "layer2" batch_sizex128x128x16        ->
            "layer3" batch_sizex64x64x32          ->
            "layer4" batch_sizex32x32x64          ->
            "layer5" batch_sizex16x16x128         ->
            "layer6" batch_sizex8x8x192           ->
            "layer7" batch_sizex4x4x192           ->
            average pooling batch_sizex2x2x192    ->
            fc layer batch_sizexoutput_channelsx1
        """
        super(ResNet1024_7_2_1, self).__init__()

        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, 
                               padding=2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 8, blocks_per_layers[0], stride=2)
        self.layer2 = self._make_layer(block, 16, blocks_per_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, blocks_per_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, blocks_per_layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, blocks_per_layers[4], stride=2)
        self.layer6 = self._make_layer(block, 192, blocks_per_layers[5], stride=2)
        self.layer7 = self._make_layer(block, 192, blocks_per_layers[6], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(768, output_channels)
        self.softmax = nn.Softmax(dim=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_of_blocks, stride=1):
        norm_layer = self._norm_layer

        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride),
                                       norm_layer(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, num_of_blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        y_logits = self.fc1(z)
        y = self.softmax(y_logits)

        return y, z, y_logits

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory, epoch=-1): 
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """

        # Saving path should be a directory where the model and configuration can be saved
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        assert os.path.isdir(save_directory)

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # If we save using the predefined names, we can load using `from_pretrained`
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, 
                                             'pytorch_model_epoch'+str(epoch)+'.bin')

        torch.save(model_to_save.state_dict(), output_model_file)

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, block, blocks_per_layers, 
                        *inputs, **kwargs):
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Instantiate the model
        model = cls(block, blocks_per_layers, **kwargs)

        # if the user has not provided the ability to load in their own state dict, but our module
        # in this case it is easier to just use save_pretrained and from_pretrained to read that 
        # saved checkpoint fully
        if state_dict is None:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model


class ResNet512_6_2_1(nn.Module):
    """ A residual network 6_2_1 
    with 6 "layers", 2x2 average pooling, and 1 fully connected layer.
    """

    def __init__(self, block, blocks_per_layers, output_channels=4, 
                 norm_layer=nn.BatchNorm2d, zero_init_residual=False):
        """ Input batch_sizex512x512x1          ->
            Conv1 batch_sizex256x256x8            ->
            "layer1" batch_sizex128x128x16        ->
            "layer2" batch_sizex64x64x32          ->
            "layer3" batch_sizex32x32x64          ->
            "layer4" batch_sizex16x16x128         ->
            "layer5" batch_sizex8x8x192           ->
            "layer6" batch_sizex4x4x192           ->
            average pooling batch_sizex2x2x192    ->
            fc layer batch_sizexoutput_channelsx1
        """
        super(ResNet512_6_2_1, self).__init__()

        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=2, 
                               padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 8, blocks_per_layers[0], stride=2)
        self.layer2 = self._make_layer(block, 16, blocks_per_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, blocks_per_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, blocks_per_layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, blocks_per_layers[4], stride=2)
        self.layer6 = self._make_layer(block, 192, blocks_per_layers[5], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(768, output_channels)
        self.softmax = nn.Softmax(dim=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_of_blocks, stride=1):
        norm_layer = self._norm_layer

        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride),
                                       norm_layer(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, num_of_blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        y_logits = self.fc1(z)
        y = self.softmax(y_logits)

        return y, z, y_logits

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory, epoch=-1): 
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """

        # Saving path should be a directory where the model and configuration can be saved
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        assert os.path.isdir(save_directory)

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # If we save using the predefined names, we can load using `from_pretrained`
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, 
                                             'pytorch_model_epoch'+str(epoch)+'.bin')

        torch.save(model_to_save.state_dict(), output_model_file)

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, block, blocks_per_layers, 
                        *inputs, **kwargs):
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Instantiate the model
        model = cls(block, blocks_per_layers, **kwargs)

        # if the user has not provided the ability to load in their own state dict, but our module
        # in this case it is easier to just use save_pretrained and from_pretrained to read that 
        # saved checkpoint fully
        if state_dict is None:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model



class ResNet256_6_2_1(nn.Module):
    """ A residual network 6_2_1 
    with 6 "layers", 2x2 average pooling, and 1 fully connected layer.
    """

    def __init__(self, block, blocks_per_layers, output_channels=4, 
                 norm_layer=nn.BatchNorm2d, zero_init_residual=False):
        """ Input batch_sizex256x256x1          ->
            Conv1 batch_sizex256x256x8            ->
            "layer1" batch_sizex128x128x16        ->
            "layer2" batch_sizex64x64x32          ->
            "layer3" batch_sizex32x32x64          ->
            "layer4" batch_sizex16x16x128         ->
            "layer5" batch_sizex8x8x192           ->
            "layer6" batch_sizex4x4x192           ->
            average pooling batch_sizex2x2x192    ->
            fc layer batch_sizexoutput_channelsx1
        """
        super(ResNet256_6_2_1, self).__init__()

        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 8, blocks_per_layers[0], stride=2)
        self.layer2 = self._make_layer(block, 16, blocks_per_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, blocks_per_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, blocks_per_layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, blocks_per_layers[4], stride=2)
        self.layer6 = self._make_layer(block, 192, blocks_per_layers[5], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(768, output_channels)
        self.softmax = nn.Softmax(dim=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_of_blocks, stride=1):
        norm_layer = self._norm_layer

        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride),
                                       norm_layer(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, num_of_blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        y_logits = self.fc1(z)
        y = self.softmax(y_logits)

        return y, z, y_logits

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory, epoch=-1): 
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """

        # Saving path should be a directory where the model and configuration can be saved
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        assert os.path.isdir(save_directory)

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # If we save using the predefined names, we can load using `from_pretrained`
        if epoch == -1:
            output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        else:
            output_model_file = os.path.join(save_directory, 
                                             'pytorch_model_epoch'+str(epoch)+'.bin')

        torch.save(model_to_save.state_dict(), output_model_file)

    # based on 
    # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, block, blocks_per_layers, 
                        output_channels, *inputs, **kwargs):
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Instantiate the model
        model = cls(block, blocks_per_layers, output_channels=output_channels, **kwargs)

        # if the user has not provided the ability to load in their own state dict, but our module
        # in this case it is easier to just use save_pretrained and from_pretrained to read that 
        # saved checkpoint fully
        if state_dict is None:
            state_dict = torch.load(pretrained_model_path, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model


def build_resnet2048_7_2_1(block=BasicBlock, blocks_per_layers=[2, 2, 2, 2, 2, 2, 2], 
			    pretrained=False, pretrained_model_path=None, output_channels=4, **kwargs):
    model = ResNet2048_7_2_1(block, blocks_per_layers, output_channels=output_channels, **kwargs)
    if pretrained:
        model = model.from_pretrained(pretrained_model_path, block, blocks_per_layers,
                                      output_channels, **kwargs)
    return model


def build_resnet1024_7_2_1(block=BasicBlock, blocks_per_layers=[2, 2, 2, 2, 2, 2, 2], 
                pretrained=False, pretrained_model_path=None, output_channels=4, **kwargs):
    model = ResNet1024_7_2_1(block, blocks_per_layers, output_channels=output_channels, **kwargs)
    if pretrained:
        model = model.from_pretrained(pretrained_model_path, block, blocks_per_layers,
                                      output_channels, **kwargs)
    return model

def build_resnet512_6_2_1(block=BasicBlock, blocks_per_layers=[2, 2, 2, 2, 2, 2], 
                      pretrained=False, pretrained_model_path=None, output_channels=4, **kwargs):
    model = ResNet512_6_2_1(block, blocks_per_layers, output_channels=output_channels, **kwargs)
    if pretrained:
        model = model.from_pretrained(pretrained_model_path, block, blocks_per_layers,
                                      output_channels, **kwargs)
    return model

def build_resnet256_6_2_1(block=BasicBlock, blocks_per_layers=[2, 2, 2, 2, 2, 2], 
                      pretrained=False, pretrained_model_path=None, output_channels=4, **kwargs):
    model = ResNet256_6_2_1(block, blocks_per_layers, output_channels=output_channels, **kwargs)
    if pretrained:
        model = model.from_pretrained(pretrained_model_path, block, blocks_per_layers,
                                      output_channels, **kwargs)
    return model