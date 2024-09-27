import torch
import torch.nn as nn
from models.aggregators.aggregator import BaseAggregator
from models.aggregators.shared_modules import MILAttention
from typing import Optional
from data_utils import generate_dropout_list
import numpy as np

class AttentionMIL(BaseAggregator):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        mlp_dim: int,
        clini_info,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        test_mode=False,
        **params
    ) -> None:
        """Create a new attention MIL model.
        Args:
            n_feats:  The nuber of features each bag instance has.
            encoder:  A network transforming bag instances into feature vectors.
        """
        super(BaseAggregator, self).__init__()
        self.encoder = encoder or nn.Sequential(
            nn.Linear(num_features, mlp_dim), nn.ReLU()
        )
        self.attention = attention or MILAttention(mlp_dim)
        self.head = head or nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Linear(mlp_dim, num_classes)
        )
        self.clini_info=clini_info
        self.test_mode=test_mode

    def forward(self, x, tiles=99999999, **kwargs):
        # if tiles is not None:
        #     assert bags.shape[0] == tiles.shape[0]
        # else:
        #     tiles = torch.tensor([bags.shape[1]],
        #                          device=bags.device).unsqueeze(0)
        if not self.test_mode:
            dropout_list= generate_dropout_list(x,self.stain_dropout)
        else:
            dropout_list= [True]*len(x)
        x_agg=[]
        for i, x_i in enumerate(x):
            if x_i is not None and dropout_list[i] and x_i.shape[1]>0:
                x_agg.append(x_i)
        x_agg = torch.cat(x_agg, dim=1)  # Add this line   

        embeddings = self.encoder(x_agg)

        # mask out entries if tiles < num_tiles
        masked_attention_scores = self._masked_attention_scores(
            embeddings, tiles
        )
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)

        return scores,weighted_embedding_sums
    
    def attention_rollout(self,x):
        return np.array([0]),np.array([0])
    

    def _masked_attention_scores(self, embeddings, tiles):
        """Calculates attention scores for all bags.
        Returns:
            A tensor containingtorch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = (torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device))

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < tiles).unsqueeze(-1)

        masked_attention = torch.where(
            attention_mask, attention_scores,
            torch.full_like(attention_scores, -1e3)
        )

        return torch.softmax(masked_attention, dim=1)
