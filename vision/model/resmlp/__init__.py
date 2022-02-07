import torch as th
from einops.layers.torch import Rearrange, Reduce


class Affine(th.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = th.nn.Parameter(th.ones( 1, 1, dim))
        self.b = th.nn.Parameter(th.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b


class PreAffinePostLayerScale(th.nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18: init_eps = 1e-1
        elif depth > 18 and depth <= 24: init_eps = 1e-5
        else: init_eps = 1e-6

        scale = th.zeros(1, 1, dim).fill_(init_eps)
        self.scale = th.nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x


def ResMLP(image_shape, patch_size, dim, depth, num_classes, expansion_factor=4, *args, **kwargs):
    image_height, image_width = image_shape[1:]
    assert (image_height % patch_size) == 0 and (image_width % patch_size) == 0,
    'image height and width must be divisible by patch size.'

    num_patches = (image_height // patch_size) * (image_width // patch_size)
    wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)

    return th.nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        th.nn.Linear((patch_size ** 2) * 3, dim),
        *[th.nn.Sequential(
        wrapper(i, th.nn.Conv1d(num_patches, num_patches, 1)),
        wrapper(i, th.nn.Sequential(
        th.nn.Linear(dim, dim * expansion_factor),
        th.nn.GELU(),
        th.nn.Linear(dim * expansion_factor, dim)))
        ) for i in range(depth)],
        Affine(dim),
        Reduce('b n c -> b c', 'mean'),
        th.nn.Linear(dim, num_classes))
