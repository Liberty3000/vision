import einops, torch as th

class Attention(th.nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        self.inner_dim = head_dim *  heads
        self.project = not (heads == 1 and head_dim == dim)

        self.heads, self.scale = heads, head_dim ** -0.5
        self.attend = th.nn.Softmax(dim=-1)
        self.to_qkv = th.nn.Linear(dim, self.inner_dim * 3, bias=False)

        self.head = th.nn.Sequential(
        th.nn.Linear(self.inner_dim, dim),
        th.nn.Dropout(dropout)
        if self.project else th.nn.Identity())

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = th.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        atn = self.attend(dots)

        latent = th.einsum('b h i j, b h j d -> b h i d', atn, v)

        latent = einops.rearrange(latent, 'b h n d -> b n (h d)')

        return self.head(latent), atn
