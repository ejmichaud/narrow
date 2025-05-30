from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

class VariableSizeLlamaRMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size, original_hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.variance_epsilon = eps
        self.D = hidden_size # dimension of pruned model
        self.Z = original_hidden_size - hidden_size # number of "hidden" zeros in residual stream
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # compute variance as if there were self.Z zeros appended to hidden_states
        variance = hidden_states.pow(2).mean(-1, keepdim=True) * self.D / (self.D + self.Z)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VariableSizeLlamaConfig(LlamaConfig):
    r"""
    This is a subclass of LlamaConfig that allows for variable intermediate sizes across layers.

    Args:
        intermediate_size (`int` or `List[int]`, *optional*, defaults to 11008):
            Dimension of the MLP representations. Can be a single integer (all layers have the same size) or a list of 
            integers (one per layer). If a list is provided, its length must match `num_hidden_layers`.
        original_hidden_size (`int`, *optional*, defaults to 2048):
        
        # ... all other args from LlamaConfig ...
    """

    model_type = "variable_size_llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        original_hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        self.original_hidden_size = original_hidden_size
        # Process the intermediate_size parameter
        if isinstance(intermediate_size, list):
            if len(intermediate_size) != num_hidden_layers:
                raise ValueError(
                    f"If intermediate_size is a list, it must have {num_hidden_layers} elements "
                    f"(one per layer), but got {len(intermediate_size)} elements."
                )
            self.intermediate_sizes = intermediate_size
            # Use the first value for compatibility with parent class
            intermediate_size_for_parent = intermediate_size[0]
        else:
            # If it's a single integer, create a list with the same value for all layers
            self.intermediate_sizes = [intermediate_size] * num_hidden_layers
            intermediate_size_for_parent = intermediate_size

        # Call the parent class constructor with the processed intermediate_size
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size_for_parent,  # Use the first value or original value
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
            **kwargs,
        )


class VariableSizeLlamaMLP(LlamaMLP):
    def __init__(self, config, layer_idx):
        # Don't call super().__init__(config) since we want to override the intermediate_size
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Use the layer-specific intermediate size
        self.intermediate_size = config.intermediate_sizes[layer_idx]
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]


class VariableSizeLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super(LlamaDecoderLayer, self).__init__()  # Call nn.Module.__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        
        # Use the variable size MLP
        self.mlp = VariableSizeLlamaMLP(config, layer_idx)
        
        self.input_layernorm = VariableSizeLlamaRMSNorm(config.hidden_size, config.original_hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = VariableSizeLlamaRMSNorm(config.hidden_size, config.original_hidden_size, eps=config.rms_norm_eps)


class VariableSizeLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(LlamaModel, self).__init__(config)  # Call PreTrainedModel.__init__
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Use variable-size decoder layers
        self.layers = nn.ModuleList(
            [VariableSizeLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        self.norm = VariableSizeLlamaRMSNorm(config.hidden_size, config.original_hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class VariableSizeLlamaForCausalLM(LlamaForCausalLM):
    config_class = VariableSizeLlamaConfig
    
    def __init__(self, config):
        # Don't call super().__init__(config) to avoid initializing the standard LlamaModel
        LlamaPreTrainedModel.__init__(self, config)
        
        self.model = VariableSizeLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

def create_variable_size_llama_config(config: LlamaConfig, intermediate_sizes: List[int]) -> VariableSizeLlamaConfig:
    """Create a VariableSizeLlamaConfig from a LlamaConfig and a list of intermediate sizes."""
    config_dict = config.to_dict()
    # Remove intermediate_size from the config dict since we'll pass it separately
    config_dict.pop('intermediate_size', None)
    return VariableSizeLlamaConfig(
        **config_dict,
        original_hidden_size=config.hidden_size,
        intermediate_size=intermediate_sizes,
    )

def convert_pruned_to_variable_size(model: LlamaForCausalLM) -> VariableSizeLlamaForCausalLM:
    """Convert a pruned LlamaForCausalLM model to a variable size LlamaForCausalLM model."""

    res_retained = (model.model.embed_tokens.weight.data != 0).any(dim=0)
    neurons_retained = [
        (layer.mlp.gate_proj.weight.data != 0).any(dim=1)
        for layer in model.model.layers
    ]

    intermediate_sizes = [sum(neurons_retained[i]) for i in range(len(neurons_retained))]
    config = create_variable_size_llama_config(model.config, intermediate_sizes)
    config.hidden_size = sum(res_retained)
    new_model = VariableSizeLlamaForCausalLM(config)

    new_model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data[:, res_retained]
    for layeri in range(len(model.model.layers)):
        # layernorms
        new_model.model.layers[layeri].input_layernorm.weight.data = model.model.layers[layeri].input_layernorm.weight.data[res_retained]
        new_model.model.layers[layeri].post_attention_layernorm.weight.data = model.model.layers[layeri].post_attention_layernorm.weight.data[res_retained]
        # mlp
        new_model.model.layers[layeri].mlp.gate_proj.weight.data = model.model.layers[layeri].mlp.gate_proj.weight.data[neurons_retained[layeri],:][:,res_retained]
        new_model.model.layers[layeri].mlp.up_proj.weight.data = model.model.layers[layeri].mlp.up_proj.weight.data[neurons_retained[layeri],:][:,res_retained]
        new_model.model.layers[layeri].mlp.down_proj.weight.data = model.model.layers[layeri].mlp.down_proj.weight.data[res_retained,:][:,neurons_retained[layeri]]
        # self-attention
        new_model.model.layers[layeri].self_attn.q_proj.weight.data = model.model.layers[layeri].self_attn.q_proj.weight.data[:, res_retained]
        new_model.model.layers[layeri].self_attn.k_proj.weight.data = model.model.layers[layeri].self_attn.k_proj.weight.data[:, res_retained]
        new_model.model.layers[layeri].self_attn.v_proj.weight.data = model.model.layers[layeri].self_attn.v_proj.weight.data[:, res_retained]
        new_model.model.layers[layeri].self_attn.o_proj.weight.data = model.model.layers[layeri].self_attn.o_proj.weight.data[res_retained, :]
    new_model.model.norm.weight.data = model.model.norm.weight.data[res_retained]
    
    return new_model
