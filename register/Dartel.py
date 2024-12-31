import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Dartel(nn.Module):
    def __init__(self, size, int_steps,interp_mode='bilinear', padding_mode='wrap'):
        super(Dartel, self).__init__()
        self.size = size  # 图像的大小
        self.dimension = len(self.size)  # 维度

        vectors = [torch.arange(0, s) for s in self.size]  
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  
        grid = torch.unsqueeze(grid, 0)
        grid = grid.to(torch.float32)
        self.register_buffer('grid', grid, persistent=False)  # 将网格坐标作为buffer
        self.interp_mode = interp_mode  # 插值模式
        self.padding_mode = padding_mode  # 填充模式

        
        self.basic_flows = nn.Parameter(torch.randn(1, 2, *self.size))  # 需要优化的向量场参数
        
        

    def getOverlapMask(self, src, flows=None, thetas=None, **kwargs):
        shape = src.shape[2:]
        assert len(shape) == self.dimension, "Expected volume dimension %s, got %s!" % (self.dimension, len(shape))

        if flows is None and thetas is None:
            return torch.ones(src.shape[0], 1, *shape, device=src.device, dtype=src.dtype)

        with torch.no_grad():
            new_locs = self._get_new_locs(thetas=thetas, flows=flows, **kwargs)
            mask = torch.zeros(src.shape[0], 1, *self.size, device=src.device, dtype=torch.uint8)
            for d in range(self.dimension):
                mask += new_locs[:, [d]].gt(shape[d]) + new_locs[:, [d]].le(0)

            mask = mask.eq(0).to(torch.float32)

        return mask
