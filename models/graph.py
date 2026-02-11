import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import os
from torch_geometric.nn import GCNConv, GATConv

# GraphAttention Class (For Graphormer)
class GraphAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, is_decoder=False, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, key_value_states=None, past_key_value=None, attention_mask=None, output_attentions=False, extra_attn=None, only_attn=False):
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None: attn_weights += extra_attn
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = (attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim))
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights, None

class GraphLayer(nn.Module):
    def __init__(self, config, graph_type):
        super(GraphLayer, self).__init__()
        self.config = config
        self.graph_type = graph_type
        # Initialize Graph Layer
        if self.graph_type == 'graphormer':
            self.graph = GraphAttention(config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob)
        elif self.graph_type == 'GCN':
            self.graph = GCNConv(config.hidden_size, config.hidden_size)
        elif self.graph_type == 'GAT':
            self.graph = GATConv(config.hidden_size, config.hidden_size, 1)

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = config.attention_probs_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, label_emb, extra_attn):
        residual = label_emb
        if self.graph_type == 'graphormer':
            label_emb, _, _ = self.graph(hidden_states=label_emb, extra_attn=extra_attn)
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.layer_norm(label_emb)

            residual = label_emb
            label_emb = self.fc2(self.activation_fn(self.fc1(label_emb)))
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.final_layer_norm(label_emb)
        elif self.graph_type == 'GCN' or self.graph_type == 'GAT':
            if label_emb.dim() == 3: label_emb = label_emb.squeeze(0)
            label_emb = self.graph(label_emb, edge_index=extra_attn)
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            if residual.dim() == 3 and label_emb.dim() == 2:
                label_emb = label_emb.unsqueeze(0)
            label_emb = residual + label_emb
            label_emb = self.layer_norm(label_emb)
        return label_emb

class GraphEncoder(nn.Module):
    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.hir_layers = nn.ModuleList([GraphLayer(config, graph_type) for _ in range(layer)])

        self.label_num = config.num_labels
        self.graph_type = graph_type

        self.label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)

        if self.graph_type == 'graphormer':
            parent_map = list(range(self.label_num))
            if path_list:
                for u, v in path_list:
                    if v < self.label_num:
                        parent_map[v] = u
            
            self.inverse_label_list = {}

            def get_root(parents, n):
                ret = []
                visited = set()
                # Use parents array instead of path_list tuple list
                while parents[n] != n:
                    if n in visited: # Cycle safety
                        break
                    visited.add(n)
                    ret.append(n)
                    n = parents[n]
                ret.append(n)
                ret.reverse()
                return ret

            for i in range(self.label_num):
                self.inverse_label_list.update({i: get_root(parent_map, i)})
            label_range = torch.arange(len(self.inverse_label_list))
            self.label_id = label_range
            node_list = {}

            def get_distance(node1, node2):
                p = 0
                q = 0
                node_list[(node1, node2)] = a = []
                node1 = self.inverse_label_list[node1]
                node2 = self.inverse_label_list[node2]
                while p < len(node1) and q < len(node2):
                    if node1[p] > node2[q]:
                        a.append(node1[p])
                        p += 1

                    elif node1[p] < node2[q]:
                        a.append(node2[q])
                        q += 1

                    else:
                        break
                return p + q

            self.distance_mat = self.label_id.reshape(1, -1).repeat(self.label_id.size(0), 1)
            hier_mat_t = self.label_id.reshape(-1, 1).repeat(1, self.label_id.size(0))
            self.distance_mat.map_(hier_mat_t, get_distance)
            self.distance_mat = self.distance_mat.view(1, -1)
            self.edge_mat = torch.zeros(self.label_num, self.label_num, 15,
                                        dtype=torch.long)
            for i in range(self.label_num):
                for j in range(self.label_num):
                    self.edge_mat[i, j, :len(node_list[(i, j)])] = torch.tensor(node_list[(i, j)])
            self.edge_mat = self.edge_mat.view(-1, self.edge_mat.size(-1))

            self.id_embedding = nn.Embedding(self.label_num, config.hidden_size, 0)
            self.distance_embedding = nn.Embedding(20, 1, 0)
            self.edge_embedding = nn.Embedding(self.label_num, 1, 0)
            self.label_id = nn.Parameter(self.label_id, requires_grad=False)
            self.edge_mat = nn.Parameter(self.edge_mat, requires_grad=False)
            self.distance_mat = nn.Parameter(self.distance_mat, requires_grad=False)
            self.label_name = []
            for i in range(len(self.label_dict)):
                self.label_name.append(self.label_dict[i])
            self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
            self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)
        else:
            # Force LongTensor for edge_index to avoid GAT errors
            self.path_list = nn.Parameter(torch.tensor(path_list, dtype=torch.long).transpose(0, 1), requires_grad=False)

    def forward(self, label_emb, embeddings):
        extra_attn = None
        
        if isinstance(label_emb, torch.Tensor):
            label_emb = label_emb.clone()
        
        # Handle nn.Embedding input - extract weights
        if isinstance(label_emb, nn.Embedding):
            label_emb = label_emb.weight  # [num_labels, hidden_size]
            label_emb = label_emb.clone()

        if self.graph_type == 'graphormer':
            label_mask = self.label_name != self.tokenizer.pad_token_id
            # full name
            label_name_emb = embeddings(self.label_name)
            label_emb = label_emb + (label_name_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1)

            label_emb = label_emb + self.id_embedding(self.label_id[:, None]).view(-1,
                                                                        self.config.hidden_size)
            extra_attn = self.distance_embedding(self.distance_mat) + self.edge_embedding(self.edge_mat).sum(
                dim=1) / (self.distance_mat.view(-1, 1) + 1e-8)
            extra_attn = extra_attn.view(self.label_num, self.label_num)
        if self.graph_type == 'GCN' or self.graph_type == 'GAT':
            extra_attn = self.path_list
        
        # Unsqueeze outside loop
        label_emb = label_emb.unsqueeze(0)
        
        for hir_layer in self.hir_layers:
            label_emb = hir_layer(label_emb, extra_attn)

        return label_emb.squeeze(0)
