import einops, torch as th
from einops.layers.torch import Rearrange
from .transformer import Transformer

class ViT(th.nn.Module):
    def __init__(self, image_shape, patch_size, num_classes, patch_emb,
                 depth, heads, mlp_dim, pool='CLS', channels=3, head_dim=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()

        image_height, image_width = image_shape[1:]
        patch_height, patch_width = (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0,\
        'ERROR <!> :: image dimensions must be divisible by patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = image_shape[0] * patch_height * patch_width
        assert pool in {'CLS','mean'},\
        'ERROR <!> :: pooling must be either `CLS` ([CLS] token) or `mean` (mean pooling).'
        self.pool = pool

        self.embed_square_patches = th.nn.Sequential(
        Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_height, pw=patch_width),
        th.nn.Linear(patch_dim, patch_emb))

        self.pos_embedding = th.nn.Parameter(th.randn(1, num_patches + 1, patch_emb))

        self.dropout = th.nn.Dropout(emb_dropout)
        self.transformer = Transformer(patch_emb, depth, heads, head_dim, mlp_dim, dropout)

        self.mlp_head = th.nn.Sequential(
        th.nn.LayerNorm(patch_emb),
        th.nn.Linear(patch_emb, num_classes))

        self.CLS_token = th.nn.Parameter(th.randn(1, 1, patch_emb))


    def forward(self, img, attention=False):
        x = self.embed_square_patches(img)
        bsize, patches, _ = x.shape

        # append tokens to the beginning of each sequence in the batch
        cls_tokens = einops.repeat(self.CLS_token, '() n d -> b n d', b=bsize)
        x = th.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(patches + 1)]

        x = self.dropout(x)
        x,_= self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)

        if attention: return x, attention
        return x
