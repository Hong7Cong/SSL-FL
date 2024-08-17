from torchvision import  transforms
import datetime
import json
import numpy as np
import time
from pathlib import Path
import torch
import torch.nn as nn
# from torch.optim import lr_scheduler
# import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
from datetime import date
import timm
# assert timm.__version__ == "0.3.2" # version check
from copy import deepcopy

import os
import sys
sys.path.insert(1, '/mnt/c/Users/PCM/Documents/GitHub/SSL-FL/code')
sys.path.insert(1, '/mnt/c/Users/PCM/Documents/GitHub/SSL-FL/segmenter')

import fed_mae.models_vit as models_vit
from fed_mae.engine_for_finetuning import train_one_epoch
import util.misc as misc
from util.FedAvg_utils import Partial_Client_Selection, valid, average_model
from util.data_utils import DatasetFLFinetune, create_dataset_and_evalmetrix
from util.start_config import print_options

mean = [0.6821, 0.4575, 0.2626]
std  = [0.1324, 0.1306, 0.1022]

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

mask_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

def imshow(inp, title=None, mean=np.array([ 0.7013, -0.1607, -0.7902]), std=np.array([0.5904, 0.5008, 0.3771])):
    """Input shound be tensor [3,224,224]"""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = mean
    # std = std
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def imshow2(inp, mask, alpha=0.5, title=None, mean=np.array([0.6821, 0.4575, 0.2626]), std=np.array([0.1324, 0.1306, 0.1022])):
    """Input shound be tensor [3,224,224]"""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = mean
    # std = std
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.imshow(mask, cmap='jet', alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def torch2numpy(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = mean
    # std = std
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)


class UNETR_2D(nn.Module):
    def __init__(self, cf, device, pretrained = '/mnt/c/Users/PCM/Documents/GitHub/SSL-FL/pretrained/SSFL/split2/checkpoint-6.pth'):
        super().__init__()
        self.cf = cf
        self.encoder = models_vit.__dict__['vit_base_patch16'](
        num_classes=5,
        drop_path_rate=0.1,
        global_pool=True,
        )

        model = models_vit.__dict__['vit_base_patch16'](
            num_classes=5,
            drop_path_rate=0.1,
            global_pool=True,
        )
        if(pretrained):
            state_dict = torch.load(pretrained)
            model.load_state_dict(state_dict['model'], strict=False)
        model.to(device)
        """ Patch + Position Embeddings """
        # self.patch_embed = nn.Linear(
        #     cf["patch_size"]*cf["patch_size"]*cf["num_channels"],
        #     cf["hidden_dim"]
        # )
        self.patch_embed = self.encoder.patch_embed
        
        self.positions = torch.arange(start=0, end=cf["num_patches"], step=1, dtype=torch.int32).to(device)
        self.pos_embed = nn.Embedding(cf["num_patches"], cf["hidden_dim"]).to(device)

        """ Transformer Encoder """
        # self.trans_encoder_layers = []

        # for i in range(cf["num_layers"]):
        #     layer = nn.TransformerEncoderLayer(
        #         d_model=cf["hidden_dim"],
        #         nhead=cf["num_heads"],
        #         dim_feedforward=cf["mlp_dim"],
        #         dropout=cf["dropout_rate"],
        #         activation=nn.GELU(),
        #         batch_first=True
        #     )
        #     self.trans_encoder_layers.append(layer)
        self.trans_encoder_layers = self.encoder.blocks
        
        """ CNN Decoder """
        ## Decoder 1
        self.d1 = DeconvBlock(cf["hidden_dim"], 512)
        self.s1 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 512),
            ConvBlock(512, 512)
        )
        self.c1 = nn.Sequential(
            ConvBlock(512+512, 512),
            ConvBlock(512, 512)
        )

        ## Decoder 2
        self.d2 = DeconvBlock(512, 256)
        self.s2 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 256),
            ConvBlock(256, 256)
        )
        self.c2 = nn.Sequential(
            ConvBlock(256+256, 256),
            ConvBlock(256, 256)
        )

        ## Decoder 3
        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128)
        )
        self.c3 = nn.Sequential(
            ConvBlock(128+128, 128),
            ConvBlock(128, 128)
        )

        ## Decoder 4
        self.d4 = DeconvBlock(128, 64)
        self.s4 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )
        self.c4 = nn.Sequential(
            ConvBlock(64+64, 64),
            ConvBlock(64, 64)
        )

        """ Output """
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Patch + Position Embeddings """
        patch_embed = self.patch_embed(inputs)   ## [8, 256, 768]

        positions = self.positions
        pos_embed = self.pos_embed(positions)   ## [256, 768]

        x = patch_embed + pos_embed ## [8, 256, 768]

        """ Transformer Encoder """
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        for i in range(self.cf["num_layers"]):
            layer = self.trans_encoder_layers[i]
            x = layer(x)

            if (i+1) in skip_connection_index:
                skip_connections.append(x)

        """ CNN Decoder """
        z3, z6, z9, z12 = skip_connections

        ## Reshaping
        batch = inputs.shape[0]
        z0 = inputs.view((batch, self.cf["num_channels"], self.cf["image_size"], self.cf["image_size"]))

        shape = (batch, self.cf["hidden_dim"], self.cf["width_patches"], self.cf["width_patches"])
        z3 = z3.reshape(shape)
        z6 = z6.reshape(shape)
        z9 = z9.reshape(shape)
        z12 = z12.reshape(shape)


        ## Decoder 1
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        ## Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        ## Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        ## Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        """ Output """
        output = self.output(x)

        return output