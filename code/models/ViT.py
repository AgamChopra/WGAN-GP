import torch.nn as nn
import torch

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_c = 3, embed_dim = 512):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size = patch_size, stride = patch_size)
        
    def forward(self, x):# x [batch, in_c, img_size, img_size] -> [batch, embed_size, n_patch/2, n_patch/2] -> [batch, embed_size, n_patch] -> [batch, n_patch, embed_size]
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
        
class Attention(nn.Module):
    def __init__(self, dim, n_heads = 8, qkv_bias = True, attn_p = 0., proj_p = 0.):
        super(Attention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x):
        n_samples, n_toks, dim = x.shape 
        if dim != self.dim:
            raise ValueError
        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_toks, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2,-1)
        dp = (q @ k_t) * self.scale
        attn = dp.softmax(dim = -1)
        attn = self.attn_drop(attn)
        w_av = attn @ v
        w_av = w_av.transpose(1,2)
        w_av = w_av.flatten(2)
        x = self.proj(w_av)
        x = self.proj_drop(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p = 0.):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features, hidden_features), nn.GELU(), 
                                    nn.Linear(hidden_features, out_features), nn.Dropout(p))
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio = 4.0, qkv_bias = True, p = 0., attn_p = 0.):
        super(Transformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps = 1E-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm2 = nn.LayerNorm(dim, eps = 1E-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features, dim, p)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size = 128, patch_size = 16, in_c = 3, n_classes = 1, embed_dim = 512, depth = 8, n_heads = 8, mlp_ratio = 4., qkv_bias = True, dropout=0.):
        super(VisionTransformer, self).__init__()
        self.dropout = dropout        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_c = in_c, embed_dim = embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,1+self.patch_embed.n_patches,embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.transformers = nn.ModuleList([Transformer(dim = embed_dim, n_heads = n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=dropout, attn_p=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps = 1E-6)
        self.head = nn.Linear(embed_dim, n_classes)
        self.prob_dist = nn.Softmax(dim = -1)
        
    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for t in self.transformers:
            x = t(x)
        x = self.norm(x)
        cls_token_final = x[:,0]
        x = self.head(cls_token_final)
        out = self.prob_dist(x)
        return out