"""
Hacked together from https://github.com/lucidrains
"""

from functools import partial

import numpy as np
import torch
from models.aggregators.attentionmil import AttentionMIL
from models.aggregators.perceiver import Perceiver
from models.aggregators.transformer import Transformer


def get_networks(cfg,input_dim,subnetwork=True):
    subnetwork_name=cfg.subnetwork_name
    settings = cfg.subnetwork[subnetwork_name]
    networks=[]
    for dim in input_dim:
        if subnetwork_name.lower() == "perceiver":
            networks.append(Perceiver(**settings, num_classes=cfg.num_classes,input_channels=dim,subnetwork=subnetwork))
        if subnetwork_name.lower() == "transformer":
            networks.append(Transformer(**settings,num_classes=cfg.num_classes,input_dim=dim,subnetwork=subnetwork, register=cfg.register))
        if subnetwork_name.lower == "attentionmil":
            networks.append(AttentionMIL(subnetwork=subnetwork))

    return networks
    
def normalize_dict_values(d: dict) -> dict:
    return {k: v / sum(d.values()) for k, v in d.items()}

def get_staining_contribution(class_attention, batch,clamp_value=0.20):

    _, all_coords, _, _, _, feature_args, filenames = batch
    cut_off = np.quantile(class_attention, 1-clamp_value)
    class_attention[class_attention<cut_off]=0

    staining_contribution={}
    attention_shift=0
    for filename,staining_coords in zip(filenames,all_coords):
        staining=filename[0].split('_')[-1].replace(".h5","")
        staining_contribution[staining]=np.sum(class_attention[attention_shift:attention_shift+len(staining_coords[0])])
        attention_shift=attention_shift+len(staining_coords[0])

    return normalize_dict_values(staining_contribution)

# def multi_transformer_attention_rollout(model, input, class_token_idx=0,clamp=False,clamp_value=0.05):
#     sub_attentions=[]
#     for sub_transformer in model.basis_transformers:
#         sub_attention=attention_rollout(model, input, class_token_idx,clamp,clamp_value)
#         sub_attentions.append(sub_attention)
    
# --------------------
# Helpers
# --------------------
def attention_rollout(model, input, class_token_idx=0, clamp=False, clamp_value=0.05):
    # Ensure the model is in evaluation mode
    model.eval()

    # Hook function to capture the attention weights
    def hook(attention_matrices, module, input, output):
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
    def compute_attention_rollout(model, input,seq_length,device):
        num_layers = len(model.transformer.layers)
        num_heads = model.transformer.layers[0][0].fn.heads

        attention_matrices = [
            torch.eye(seq_length, seq_length).unsqueeze(0).repeat(num_heads, 1, 1).to(device)
            for _ in range(num_layers)
        ]

        handles = register_hooks(model.transformer, attention_matrices, "basis")
        with torch.no_grad():
            _ = model(input)

        for handle in handles:
            handle.remove()

        attention_matrices_combined = attention_matrices[0]
        for i in range(1, num_layers):
            attention_matrices_combined = torch.matmul(attention_matrices[i], attention_matrices_combined)

        return attention_matrices_combined

    # Compute the attention rollouts for each basis transformer
    basis_attention_rollouts = []

    device=None

    for i, (basis_transformer, x_i) in enumerate(zip(model.basis_transformers, input)):
        if input[i] is not None:
            device=input[i].device
            seq_length = input[i].size(1) + 1
            basis_attention_rollout = compute_attention_rollout(basis_transformer, x_i,seq_length,device)
            basis_attention_rollouts.append(basis_attention_rollout)


    # Compute the attention rollout for the top-level transformer
    top_attention_rollout = compute_attention_rollout(model, input,seq_length=len(input)+1,device=device)

    # Combine the attention rollouts for basis transformers and the top-level transformer
    class_token_top_attention=torch.mean(top_attention_rollout[:, :, class_token_idx, class_token_idx+1:],axis=1)
    attention_rollouts = []
    for i,basis_attention_rollout in enumerate(basis_attention_rollouts):
        attention_rollout_combined = class_token_top_attention[:,i]*basis_attention_rollout
        attention_rollouts.append(attention_rollout_combined)

    # Extract the attention weights for the class token, remove the self-attention from the class token
    class_token_attentions_per_head = [attention_rollout[:, :, class_token_idx, class_token_idx+1:] for attention_rollout in attention_rollouts]
    class_token_attentions=[safe_squeeze(torch.mean(staining_attentions,axis=1).cpu().numpy()) for staining_attentions in class_token_attentions_per_head]

    return class_token_attentions,safe_squeeze(class_token_top_attention.cpu().numpy())

def safe_squeeze(arr):
    # Squeeze the array
    squeezed = np.squeeze(arr)
    # If squeezing resulted in a scalar, convert it to a 1D array
    if squeezed.ndim == 0:
        squeezed = np.array([squeezed])
    return squeezed
