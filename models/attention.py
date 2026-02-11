import torch.nn as nn
import torch
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, is_decoder: bool = False, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads."
        
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # [Batch, Seq, Head*Dim] -> [Batch, Seq, Head, Dim] -> [Batch, Head, Seq, Dim]
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,      # Query
            key_value_states=None,            # Key/Value
            past_key_value=None,
            attention_mask=None,
            output_attentions: bool = False,
            extra_attn=None,
            only_attn=False,
    ):
        """
        Input shape: [Batch, Seq_Len, Hidden]
        """
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # [Batch, Head, Tgt_Seq, Head_Dim]
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        if extra_attn is not None:
            attn_weights += extra_attn

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value

class Label2Text(nn.Module):
    
    def __init__(self, label_embedding_size, attn_hidden_size):
        super().__init__()
        self.proj_label = nn.Linear(label_embedding_size, 
                                    attn_hidden_size, bias=False)
    
    def forward(self, input_data, label_repr, padding_mask=None):
        
        label_repr = self.proj_label(label_repr)  
        # [batch_size, num_labels, attn_hidden_size]
        
        embedding_label = label_repr.transpose(1, 2)  
        # [batch_size, attn_hidden_size, num_labels]
        
        input_data = F.normalize(input_data, dim=-1, p=2)
        embedding_label = F.normalize(embedding_label, dim=1, p=2)
        
        # [batch_size, seq_len, num_labels]
        G = torch.bmm(input_data, embedding_label)
        
        if padding_mask is not None:
            padding_mask = padding_mask.eq(0)
            G = G.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        
        softmax_G = torch.softmax(G, dim=1)  
        # [batch_size, seq_len, num_labels]
        
        # [batch_size, num_labels, hidden_size]
        output = torch.bmm(softmax_G.transpose(1, 2), input_data)
        
        return output, softmax_G