import logging
from typing import Final, TypedDict

import torch

logger = logging.getLogger(__name__)

ConversionResult = TypedDict('ConversionResult', {'new_state_dict': dict[str, torch.Tensor], 'missing_keys': set[str], 'unused_keys': set[str]})

key_mapping: Final[dict[str, str]] = {
    '{encoder_or_decoder}.layers.{layer_idx}.self_attn.out_proj.weight': '{encoder_or_decoder}.layers.{layer_idx}.self_attn.out_proj.weight',
    '{encoder_or_decoder}.layers.{layer_idx}.self_attn.out_proj.bias': '{encoder_or_decoder}.layers.{layer_idx}.self_attn.out_proj.bias',
    '{encoder_or_decoder}.layers.{layer_idx}.fc1.weight': '{encoder_or_decoder}.layers.{layer_idx}.linear1.weight',
    '{encoder_or_decoder}.layers.{layer_idx}.fc1.bias': '{encoder_or_decoder}.layers.{layer_idx}.linear1.bias',
    '{encoder_or_decoder}.layers.{layer_idx}.fc2.weight': '{encoder_or_decoder}.layers.{layer_idx}.linear2.weight',
    '{encoder_or_decoder}.layers.{layer_idx}.fc2.bias': '{encoder_or_decoder}.layers.{layer_idx}.linear2.bias',
    '{encoder_or_decoder}.layers.{layer_idx}.self_attn_layer_norm.weight': '{encoder_or_decoder}.layers.{layer_idx}.norm1.weight',
    '{encoder_or_decoder}.layers.{layer_idx}.self_attn_layer_norm.bias': '{encoder_or_decoder}.layers.{layer_idx}.norm1.bias',
    '{encoder_or_decoder}.embed_tokens.weight': '{encoder_or_decoder}_emb.weight',
    # Encoder layers
    'encoder.layers.{layer_idx}.final_layer_norm.weight': 'encoder.layers.{layer_idx}.norm2.weight',
    'encoder.layers.{layer_idx}.final_layer_norm.bias': 'encoder.layers.{layer_idx}.norm2.bias',
    # Decoder layers
    'decoder.layers.{layer_idx}.encoder_attn_layer_norm.weight': 'decoder.layers.{layer_idx}.norm2.weight',
    'decoder.layers.{layer_idx}.encoder_attn_layer_norm.bias': 'decoder.layers.{layer_idx}.norm2.bias',
    'decoder.layers.{layer_idx}.final_layer_norm.weight': 'decoder.layers.{layer_idx}.norm3.weight',
    'decoder.layers.{layer_idx}.final_layer_norm.bias': 'decoder.layers.{layer_idx}.norm3.bias',
    'decoder.layers.{layer_idx}.encoder_attn.out_proj.weight': 'decoder.layers.{layer_idx}.multihead_attn.out_proj.weight',
    'decoder.layers.{layer_idx}.encoder_attn.out_proj.bias': 'decoder.layers.{layer_idx}.multihead_attn.out_proj.bias',
    'decoder.output_projection.weight': 'output_projection.weight'
}


def get_proj_weights_and_bias(state_dict, layer_idx, encoder_or_decoder='encoder') -> dict[str, list[str] | torch.Tensor]:
    """
    Concatenates the k_proj, v_proj, and q_proj weights and biases into in_proj_weight and in_proj_bias.

    Args:
    - state_dict (dict): The state dictionary containing the weights and biases.
    - layer_idx (int): The index of the layer to process.

    Returns:
    - in_proj_weight (Tensor): Concatenated weight tensor for the in_proj layer.
    - in_proj_bias (Tensor): Concatenated bias tensor for the in_proj layer.
    """
    # Generate keys for k_proj, v_proj, q_proj for the given layer index
    k_weight_key = f"{encoder_or_decoder}.layers.{layer_idx}.self_attn.k_proj.weight"
    k_bias_key = f"{encoder_or_decoder}.layers.{layer_idx}.self_attn.k_proj.bias"
    v_weight_key = f"{encoder_or_decoder}.layers.{layer_idx}.self_attn.v_proj.weight"
    v_bias_key = f"{encoder_or_decoder}.layers.{layer_idx}.self_attn.v_proj.bias"
    q_weight_key = f"{encoder_or_decoder}.layers.{layer_idx}.self_attn.q_proj.weight"
    q_bias_key = f"{encoder_or_decoder}.layers.{layer_idx}.self_attn.q_proj.bias"
    
    # Ensure all keys exist in the state_dict
    for key in [k_weight_key, k_bias_key, v_weight_key, v_bias_key, q_weight_key, q_bias_key]:
        if key not in state_dict:
            raise KeyError(f"Key '{key}' not found in the state dict.")
    
    # Concatenate weights and biases
    q_weight = state_dict[q_weight_key]
    k_weight = state_dict[k_weight_key]
    v_weight = state_dict[v_weight_key]
    in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

    q_bias = state_dict[q_bias_key]
    k_bias = state_dict[k_bias_key]
    v_bias = state_dict[v_bias_key]
    in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

    projection_weights = {
        f"{encoder_or_decoder}.layers.{layer_idx}.self_attn.in_proj_weight": in_proj_weight,
        f"{encoder_or_decoder}.layers.{layer_idx}.self_attn.in_proj_bias": in_proj_bias,
        "removed_keys": [k_weight_key, k_bias_key, v_weight_key, v_bias_key, q_weight_key, q_bias_key]
    }

    # Cross-attention
    if encoder_or_decoder == 'decoder':
        encoder_attn_k_weight_key = f"{encoder_or_decoder}.layers.{layer_idx}.encoder_attn.k_proj.weight"
        encoder_attn_k_bias_key = f"{encoder_or_decoder}.layers.{layer_idx}.encoder_attn.k_proj.bias"
        encoder_attn_v_weight_key = f"{encoder_or_decoder}.layers.{layer_idx}.encoder_attn.v_proj.weight"
        encoder_attn_v_bias_key = f"{encoder_or_decoder}.layers.{layer_idx}.encoder_attn.v_proj.bias"
        encoder_attn_q_weight_key = f"{encoder_or_decoder}.layers.{layer_idx}.encoder_attn.q_proj.weight"
        encoder_attn_q_bias_key = f"{encoder_or_decoder}.layers.{layer_idx}.encoder_attn.q_proj.bias"

        encoder_attn_keys = [encoder_attn_k_weight_key, encoder_attn_k_bias_key, encoder_attn_v_weight_key, encoder_attn_v_bias_key, encoder_attn_q_weight_key, encoder_attn_q_bias_key]
        # Ensure all keys exist in the state_dict
        for key in encoder_attn_keys:
            if key not in state_dict:
                raise KeyError(f"Key '{key}' not found in the state dict.")
        
        # Concatenate weights and biases
        encoder_attn_q_weight = state_dict[encoder_attn_q_weight_key]
        encoder_attn_k_weight = state_dict[encoder_attn_k_weight_key]
        encoder_attn_v_weight = state_dict[encoder_attn_v_weight_key]
        encoder_in_proj_weight = torch.cat([encoder_attn_q_weight, encoder_attn_k_weight, encoder_attn_v_weight], dim=0)

        encoder_attn_q_bias = state_dict[encoder_attn_q_bias_key]
        encoder_attn_k_bias = state_dict[encoder_attn_k_bias_key]
        encoder_attn_v_bias = state_dict[encoder_attn_v_bias_key]
        encoder_in_proj_bias = torch.cat([encoder_attn_q_bias, encoder_attn_k_bias, encoder_attn_v_bias], dim=0)
        
        projection_weights[f"decoder.layers.{layer_idx}.multihead_attn.in_proj_weight"] = encoder_in_proj_weight
        projection_weights[f"decoder.layers.{layer_idx}.multihead_attn.in_proj_bias"] = encoder_in_proj_bias
        projection_weights['removed_keys'].extend(encoder_attn_keys)

    return projection_weights


def create_pytorch_state_dict(old_state_dict: dict[str, torch.Tensor]) -> ConversionResult:
    new_state_dict = {}

    all_keys = set(old_state_dict.keys())
    missing_keys = set()

    for layer_idx in range(12):
        for encoder_or_decoder in ['encoder', 'decoder']:
            for key_format, value_format in key_mapping.items():
                key = key_format.format(layer_idx=layer_idx, encoder_or_decoder=encoder_or_decoder)
                value = value_format.format(layer_idx=layer_idx, encoder_or_decoder=encoder_or_decoder)

                if key in all_keys:
                    new_state_dict[value] = old_state_dict[key]
                    all_keys.remove(key)
                elif value in new_state_dict:
                    continue
                else:
                    logger.warning(f"Key '{key}' not found in the state dict.")
                    missing_keys.add(key)
            
            attention_projections = get_proj_weights_and_bias(old_state_dict, layer_idx, encoder_or_decoder=encoder_or_decoder)
            all_keys -= set(attention_projections.pop('removed_keys'))
            new_state_dict.update(attention_projections)
    
    return {
        "new_state_dict": new_state_dict,
        "missing_keys": missing_keys,
        "unused_keys": all_keys
    }
