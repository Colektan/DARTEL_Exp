# -*- coding: utf-8 -*-
"""
Deformation regularization module for image registration.

__author__ = Xinzhe Luo
__version__ = 0.1

"""

import torch
import torch.nn as nn


class LocalDisplacementEnergy(nn.Module):

    def __init__(self, dimension, **kwargs):
        super(LocalDisplacementEnergy, self).__init__()
        self.dimension = dimension
        self.kwargs = kwargs

    def _gradient_dx(self, fv):
        if self.dimension == 3:
            return (fv[..., 2:, 1:-1, 1:-1] - fv[..., :-2, 1:-1, 1:-1]) / 2
        elif self.dimension == 2:
            return (fv[..., 2:, 1:-1] - fv[..., :-2, 1:-1]) / 2
        else:
            raise NotImplementedError

    def _gradient_dy(self, fv):
        if self.dimension == 3:
            return (fv[..., 1:-1, 2:, 1:-1] - fv[..., 1:-1, :-2, 1:-1]) / 2
        elif self.dimension == 2:
            return (fv[..., 1:-1, 2:] - fv[..., 1:-1, :-2]) / 2
        else:
            raise NotImplementedError

    def _gradient_dz(self, fv):
        if self.dimension == 3:  # 三维的情况下，计算第三维z的微分，错开两个像素相减
            return (fv[..., 1:-1, 1:-1, 2:] - fv[..., 1:-1, 1:-1, :-2]) / 2
        else:
            raise NotImplementedError

    def _gradient_txyz(self, Txyz, fn):
        if self.dimension == 3:  # 将微分计算应用到每个通道上
            return torch.stack([fn(Txyz[..., i, :, :, :]) for i in range(Txyz.size(-4))], dim=-4)
        elif self.dimension == 2:
            return torch.stack([fn(Txyz[..., i, :, :]) for i in range(Txyz.size(-3))], dim=-3)
        else:
            raise NotImplementedError


class BendingEnergy(LocalDisplacementEnergy):
    def __init__(self, alpha=1, **kwargs):
        super(BendingEnergy, self).__init__(**kwargs)
        self.alpha = alpha   #正则项系数

    def forward(self, flow):
        #  2D: [batch, 2, H, W]
        # 计算一阶梯度
        dfdx = self._gradient_txyz(flow, self._gradient_dx)  # [batch, 2, H-2, W-2]
        dfdy = self._gradient_txyz(flow, self._gradient_dy)  # [batch, 2, H-2, W-2]
        
        # 计算二阶梯度
        dfdxx = self._gradient_txyz(dfdx, self._gradient_dx) # [batch, 2, H-4, W-4]
        dfdyy = self._gradient_txyz(dfdy, self._gradient_dy) # [batch, 2, H-4, W-4]
        dfdxy = self._gradient_txyz(dfdx, self._gradient_dy) # [batch, 2, H-4, W-4]

        if self.dimension == 2:
            return self.alpha * torch.mean(dfdxx ** 2 + dfdyy ** 2 + 2 * dfdxy ** 2)

        elif self.dimension == 3:
            # 三维的情况下，补充关于第三维z的微分
            dfdz = self._gradient_txyz(flow, self._gradient_dz)
            dfdzz = self._gradient_txyz(dfdz, self._gradient_dz)
            dfdyz = self._gradient_txyz(dfdy, self._gradient_dz)
            dfdxz = self._gradient_txyz(dfdx, self._gradient_dz)

            return self.alpha * torch.mean(
                dfdxx ** 2 + dfdyy ** 2 + dfdzz ** 2 + 2 * dfdxy ** 2 + 2 * dfdxz ** 2 + 2 * dfdyz ** 2)

        else:
            raise NotImplementedError

class ElasticEnergy(LocalDisplacementEnergy):
    def __init__(self, alpha=1, beta=1, **kwargs):
        super(ElasticEnergy, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def forward(self, flow):
        dfdx = self._gradient_txyz(flow, self._gradient_dx)
        dfdy = self._gradient_txyz(flow, self._gradient_dy)

        
        if self.dimension == 2:
            return self.alpha * torch.mean(dfdx ** 2 + dfdy ** 2) + self.beta * torch.mean(dfdx ** 2 * dfdy ** 2)

        elif self.dimension == 3:
            dfdz = self._gradient_txyz(flow, self._gradient_dz)

            return self.alpha * torch.mean(dfdx ** 2 + dfdy ** 2 + dfdz ** 2) \
                   + self.beta * torch.mean(dfdx ** 2 * dfdy ** 2 + dfdx ** 2 * dfdz ** 2 + dfdy ** 2 * dfdz ** 2)

        else:
            raise NotImplementedError


class MembraneEnergy(LocalDisplacementEnergy):
    def __init__(self, beta=1, **kwargs):
        super(MembraneEnergy, self).__init__(**kwargs)
        self.beta = beta

    def forward(self, flow):
        dfdx = self._gradient_txyz(flow, self._gradient_dx)
        dfdy = self._gradient_txyz(flow, self._gradient_dy)

        if self.dimension == 2:
            return self.beta * torch.mean(dfdx ** 2 + dfdy ** 2)

        elif self.dimension == 3:
            dfdz = self._gradient_txyz(flow, self._gradient_dz)

            return self.beta * torch.mean(dfdx ** 2 + dfdy ** 2 + dfdz ** 2)

        else:
            raise NotImplementedError




class JacobianDeterminant(LocalDisplacementEnergy):
    def __init__(self, **kwargs):
        super(JacobianDeterminant, self).__init__(**kwargs)

    def forward(self, flow):
        dfdx = self._gradient_txyz(flow, self._gradient_dx)
        dfdy = self._gradient_txyz(flow, self._gradient_dy)

        if self.dimension == 2:
            return (dfdx[..., 0, :, :] + 1) * (dfdy[..., 1, :, :] + 1) - dfdx[..., 1, :, :] * dfdy[..., 0, :, :]

        elif self.dimension == 3:
            dfdz = self._gradient_txyz(flow, self._gradient_dz)

            return (dfdx[..., 0, :, :, :] + 1) * (dfdy[..., 1, :, :, :] + 1) * (dfdz[..., 2, :, :, :] + 1) \
                   + dfdx[..., 2, :, :, :] * dfdy[..., 0, :, :, :] * dfdz[..., 1, :, :, :] \
                   + dfdx[..., 1, :, :, :] * dfdy[..., 2, :, :, :] * dfdz[..., 0, :, :, :] \
                   - dfdx[..., 2, :, :, :] * (dfdy[..., 1, :, :, :] + 1) * dfdz[..., 0, :, :, :] \
                   + (dfdx[..., 0, :, :, :] + 1) * dfdy[..., 2, :, :, :] * dfdz[..., 1, :, :, :] \
                   + dfdx[..., 1, :, :, :] * dfdy[..., 0, :, :, :] * (dfdz[..., 2, :, :, :] + 1)

