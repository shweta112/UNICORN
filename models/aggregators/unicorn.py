from functools import partial
import torch
import torch.nn as nn
from einops import repeat
from models.aggregators.model_utils import safe_squeeze
import numpy as np

import torch
import torch.nn as nn
from einops import repeat
from data_utils import generate_dropout_list

from models.aggregators.shared_modules import Attention, FeedForward, PreNorm


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


class Transformer(nn.Module):
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
        super(Transformer, self).__init__()
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


    def forward(self, x, use_pos=False, register_hook=False):
        if not self.subnetwork:
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

        return x

class MultiTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim=512,
        mlp_dim=512,
        dim_head=64,
        dropout=0.,
        num_base_networks=2,
        stain_dropout=0.5,
        subnetworks,
        clini_info_dropout,
        heads=4,
        depth=2,
        register=False,
        modality_embedding=0,
        test_mode=False
    ):            
        super(MultiTransformer, self).__init__()
        self.test_mode=test_mode
        self.basis_transformers = nn.ModuleList(subnetworks)
        dim=dim+modality_embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.modality_embedding = nn.Parameter(torch.randn(1, num_base_networks, modality_embedding))
        self.transformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.register=register
        self.linear=nn.Linear(dim, num_classes)
        self.stain_dropout=stain_dropout
        self.clini_info_dropout=clini_info_dropout

    def forward(self, x):
        #batchsize
        b=1
        x_agg = [] 
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        if not self.test_mode:
             dropout_list= generate_dropout_list(x,self.stain_dropout)
        else:
            dropout_list= [True]*len(x)
        for i, x_i in enumerate(x):
            if x_i is not None and dropout_list[i] and x_i.shape[1]>0:
                x_i = self.basis_transformers[i](x_i).unsqueeze(0)
                x_agg.append(x_i)

        x_agg = torch.cat(x_agg, dim=1)
        x = torch.cat((cls_tokens, x_agg), dim=1)
        x = self.transformer(x)

        x_layer_norm=self.layer_norm(x[:,0])
        x=self.linear(x_layer_norm)
        return x,x_layer_norm
    
    def predict_single_patches(self, input_data):
    # Ensure the model is in evaluation mode
        self.eval()
        class_predictions=[]
        softmax=nn.Softmax(dim=1)
        
        with torch.no_grad():
            for i,stain_features in enumerate(input_data):
                class_predictions_staining=[]
                if stain_features is not None:
                    for j in range (stain_features.size(1)):
                        input_features=np.full(len(input_data), None, dtype=object)
                        single_patch_feature=stain_features[:,j,:]
                        input_features[i]=single_patch_feature.unsqueeze(0)
                        logits,_=self.forward(input_features)
                        class_predictions_staining.append(softmax(logits).cpu().numpy())
                    class_predictions.append(class_predictions_staining)
        return class_predictions
    
    def get_class_attention(self, attentions, patch_class_probabilities, predicted_class, clamp=True, clamp_value=0.05):
        '''
        attentions: a list of attentions, each entry corresponding to attentions from slides of one staining
        patch_class_probabilities: a list of lists of the output probability of the predicted class if feeding a single patch through the network
        predicted_class: which class was predicted for this slide
        clamp: True if top and bottom values should be clamped
        clamp_value: gives the percentiles we want to clamp to.
        '''
        patchwise_class_prob_of_prediction =[]
        class_attentions = []
        normalized_attentions=[]

        for stain_attentions, stain_class_probabilities in zip(attentions, patch_class_probabilities):

            if clamp:

                q05, q95 = np.quantile(stain_attentions, clamp_value), np.quantile(stain_attentions, 1 - clamp_value)
                stain_attentions= stain_attentions.clip(q05, q95)

            stain_attentions=(stain_attentions-np.min(stain_attentions))/(np.max(stain_attentions) - np.min(stain_attentions))
            patchwise_class_prob_stain = [safe_squeeze(class_pred)[predicted_class] for class_pred in stain_class_probabilities]
            normalized_attentions.append(stain_attentions)
            class_attention_stain = stain_attentions * np.array(patchwise_class_prob_stain)
            class_attentions.append(safe_squeeze(class_attention_stain))
            patchwise_class_prob_of_prediction.append(safe_squeeze(patchwise_class_prob_stain))
                
        class_attention_normalized = [(class_attention - np.min(np.concatenate(class_attentions))) / (np.max(np.concatenate(class_attentions)) - np.min(np.concatenate(class_attentions))) for class_attention in class_attentions]
        
        return class_attention_normalized, patchwise_class_prob_of_prediction,normalized_attentions
    
    def attention_rollout(self, input_data, class_token_idx=0, clamp=False, clamp_value=0.05):
        # Ensure the model is in evaluation mode
        self.eval()

        # Hook function to capture the attention weights
        def hook(attention_matrices, module, input_data, output):
            attention_weights = module.get_attention_map()  # Get attention map from the custom Attention class
            layer_idx = int(module.name.split('_')[-1])
            attention_matrices[layer_idx] = attention_weights.detach()

        # Function to register the hook for all the self-attention layers in a transformer
        def register_hooks(transformer, attention_matrices, prefix):
            handles = []
            for layer_idx, layer in enumerate(transformer.layers):
                handle = layer[0].fn.register_forward_hook(partial(hook, attention_matrices))
                layer[0].fn.name = f"{prefix}_layer_{layer_idx}"
                handles.append(handle)
            return handles

        # Function to compute the attention rollout for a transformer
        def compute_attention_rollout(model, input_data,seq_length,device):
            num_layers = len(model.transformer.layers)
            num_heads = model.transformer.layers[0][0].fn.heads

            attention_matrices = [
                torch.eye(seq_length, seq_length).unsqueeze(0).repeat(num_heads, 1, 1).to(device)
                for _ in range(num_layers)
            ]

            handles = register_hooks(model.transformer, attention_matrices, "basis")
            with torch.no_grad():
                out = model(input_data)

            for handle in handles:
                handle.remove()

            attention_matrices_combined = attention_matrices[0]
            for i in range(1, num_layers):
                attention_matrices_combined = torch.matmul(attention_matrices[i], attention_matrices_combined)

            return attention_matrices_combined

        # Compute the attention rollouts for each basis transformer
        basis_attention_rollouts = []

        device=None

        for i, (basis_transformer, x_i) in enumerate(zip(self.basis_transformers, input_data)):
            if input_data[i] is not None and input_data[i].shape[1]>0:
                device=input_data[i].device
                seq_length = input_data[i].size(1) + 1
                basis_attention_rollout = compute_attention_rollout(basis_transformer, x_i,seq_length,device)
                basis_attention_rollouts.append(basis_attention_rollout)


        # Compute the attention rollout for the top-level transformer
        top_attention_rollout = compute_attention_rollout(self, input_data,seq_length=len(input_data)+1,device=device)

        # Combine the attention rollouts for basis transformers and the top-level transformer
        class_token_top_attention=torch.mean(top_attention_rollout[:, :, class_token_idx, class_token_idx+1:],axis=1)
        attention_rollouts = []
        for i,basis_attention_rollout in enumerate(basis_attention_rollouts):
            attention_rollout_combined = class_token_top_attention[:,i]*basis_attention_rollout

            attention_rollouts.append(attention_rollout_combined)

        # Extract the attention weights for the class token, remove the self-attention from the class token
        class_token_attentions_per_head = [attention_rollout[:, :, class_token_idx, class_token_idx+1+self.register:] for attention_rollout in attention_rollouts]
        class_token_attentions=[safe_squeeze(torch.mean(staining_attentions,axis=1).cpu().numpy()) for staining_attentions in class_token_attentions_per_head]

        return class_token_attentions,safe_squeeze(class_token_top_attention.cpu().numpy())

