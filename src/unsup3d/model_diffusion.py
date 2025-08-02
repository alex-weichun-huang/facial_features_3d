import os
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from . import networks
from . import utils

import lpips
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from diffusers import UNet2DModel, DDIMScheduler

EPS = 1e-7


class Unsup3D_diffusion():
    def __init__(self, cfgs):
        super().__init__()
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7*self.max_depth + 0.3*self.min_depth))
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_flip = cfgs.get('lam_flip', 0.5)
        self.lam_flip_start_epoch = cfgs.get('lam_flip_start_epoch', 0)
        self.lr = cfgs.get('lr', 1e-4)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        self.lam_reg = cfgs.get('lam_reg', 0.1)
        self.weight_depth = cfgs.get('weight_depth', 1)
        self.weight_albedo = cfgs.get('weight_albedo', 0.1)
        self.run_test = cfgs.get('run_test', False)

        ## networks and optimizers
        repeat = int(math.log2(self.image_size / 64))
        self.netEA = UNet2DModel(
            sample_size=1,
            in_channels=512,
            out_channels=256,
            layers_per_block=2,
            block_out_channels=(128,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
        )
        self.netED = UNet2DModel(
            sample_size=1,
            in_channels=512,
            out_channels=256,
            layers_per_block=2,
            block_out_channels=(128,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
        )
        self.scheduler = DDIMScheduler(prediction_type='sample')
        self.scheduler.set_timesteps(5)
        self.netD = networks.EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None, repeat=repeat).requires_grad_(False)
        self.netA = networks.EDDeconv(cin=3, cout=3, nf=64, zdim=256, repeat=repeat).requires_grad_(False)
        self.netL = networks.Encoder(cin=3, cout=4, nf=32, repeat=repeat).requires_grad_(False)
        self.netV = networks.Encoder(cin=3, cout=6, nf=32, repeat=repeat).requires_grad_(False)
        self.netC = networks.ConfNet(cin=3, cout=2, nf=64, zdim=128, repeat=repeat).requires_grad_(False)
        self.network_names = [k for k in vars(self) if 'net' in k]
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        self.make_scheduler = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-5)

        ## other parameters
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)
        self.other_param_names = ['PerceptualLoss']

        ## depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth

    def init_optimizers(self):
        self.optimizer_names = []
        self.scheduler_names = []
        for net_name in ['netEA','netED']:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]
            scheduler = self.make_scheduler(optimizer)
            scheduler_name = net_name.replace('net','scheduler')
            setattr(self, scheduler_name, scheduler)
            self.scheduler_names += [scheduler_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()
