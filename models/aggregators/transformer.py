import torch
import torch.nn as nn
from einops import repeat
from models.aggregators import BaseAggregator
from models.aggregators.shared_modules import Attention, FeedForward, PreNorm
import numpy as np

class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x, register_hook=False):
        for attn, ff in self.layers:
            x = attn(x, register_hook=register_hook) + x
            x = ff(x) + x
        return x


class Transformer(BaseAggregator):
    def __init__(
        self,
        *,
        num_classes,
        input_dim=2048,
        dim=512,
        depth=2,
        heads=8,
        mlp_dim=512,
        pool='cls',
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        subnetwork=True,
        register=0,
    ):
        super(BaseAggregator, self).__init__()
        assert pool in {
            'cls', 'mean', 'max', False
        }, 'pool type must be either cls (class token), mean (mean pooling), max max pooling) or False (no subnetwork pooling)'

        self.projection = nn.Sequential(nn.Linear(input_dim, 512, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.transformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        if register >0:
            self.register_token = nn.Parameter(torch.randn(1, register, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.subnetwork=subnetwork
        self.register=register

    def attention_rollout(self,a):
        return np.array(1),np.array(1)
    
    def forward(self, x, register_hook=False):
        if not self.subnetwork:
            x=[d for d in x if len(d.shape)==3]
            x=torch.cat(x,dim=1)
        b=1

        x = self.projection(x)

        if self.register>0:
            register_token = repeat(self.register_token, '1 s d -> b s d', b=b)
            x = torch.cat((register_token, x), dim=1)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, register_hook=register_hook)

        if self.pool == 'cls':
            x= x[:, 0]
        elif self.pool == 'mean': 
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x=x.max(dim=1).values
        # if no pooling, forward all.
        else:
            x=x[0]
        
        if not self.subnetwork:
            x=self.mlp_head(x)
            return x,x
        return x
