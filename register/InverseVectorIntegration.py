# -*- coding: utf-8 -*-
"""
Spatial transformer module for image registration.

__author__ = Xinzhe Luo
__version__ = 0.1

"""
import torch
import torch.nn as nn
# import torch.nn.functional as F
from register.SpatialTransformer import SpatialTransformer

class InverseVectorIntegration(nn.Module):

    def __init__(self, size, int_steps=0, **kwargs):
        super(InverseVectorIntegration, self).__init__()
        self.size = size
        self.int_steps = int_steps  #代表划分为多少个小步骤，对应K
        self.kwargs = kwargs
        self.transform = SpatialTransformer(self.size, padding_mode='zeros')

    def forward(self, flow):
        if flow is None:
            return None

        if self.int_steps > 0:
            vec = -torch.div(flow, 2 ** self.int_steps)  # 在正变换的基础上修改为负号即可
            for _ in range(self.int_steps):
                vec = vec + self.transform(vec, vec)  #不断对vec进行复合变换
            return vec
        else:
            return flow

