from functools import partial
import numpy as np, torch as th
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(th.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn, self.norm = fn, th.nn.LayerNorm(dim)
    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor=4, dropout=0., layer=th.nn.Linear):
    return th.nn.Sequential(
    layer(dim, dim * expansion_factor),
    th.nn.GELU(),
    th.nn.Dropout(dropout),
    layer(dim * expansion_factor, dim),
    th.nn.Dropout(dropout))

def MLPMixer(image_shape, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0., *args, **kwargs):
    assert image_shape[-1] % patch_size == 0,\
    'ERROR <!> :: image dimensions must be divisible by patch size.'

    num_patches = (image_shape[-1] // patch_size) ** 2
    conv, mlp = partial(th.nn.Conv1d, kernel_size=1), th.nn.Linear

    return th.nn.Sequential(
    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
    th.nn.Linear((patch_size ** 2) * image_shape[0], dim),
    *[th.nn.Sequential(
    PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, conv)),
    PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, mlp))
    ) for _ in range(depth)],
    th.nn.LayerNorm(dim),
    Reduce('b n c -> b c', 'mean'),
    th.nn.Linear(dim, num_classes))
