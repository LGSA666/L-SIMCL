from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from .loss import multilabel_categorical_crossentropy, BCEloss, HBMloss, StructureAwareContrastiveLoss
from .attention import CrossAttention, Label2Text
from .graph import GraphEncoder 

class LSIMCLModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None, loss_type='zlpr', 
                 label_cpt=None, label_depths=None, **kwargs):
        super().__init__(config)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        
        self.num_labels = config.num_labels
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        self.layer = layer
        self.depth2label = depth2label
        self.path_list = path_list
        self.loss_type = loss_type.lower()
        
        # Initialize label_depths as buffer for label-level contrastive loss
        label_depths_data = kwargs.get('label_depths', None)
        if label_depths_data is not None:
            if not isinstance(label_depths_data, torch.Tensor):
                label_depths_data = torch.tensor(label_depths_data, dtype=torch.float32)
            self.register_buffer('label_depths', label_depths_data)
        else:
            self.label_depths = None
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax_entropy = kwargs.get('softmax_entropy', True) if loss_type == 'zlpr' else False
        self.tau = kwargs.get('temp', 0.1)
        self.graph_encoder = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)
        self.label_embeddings = nn.Parameter(torch.randn(self.num_labels, config.hidden_size) * 0.02)

        cross_attention_heads = kwargs.get('heads', None) 
        if cross_attention_heads is None:
            cross_attention_heads = config.num_attention_heads
        
        self.cross_attention = CrossAttention(
            embed_dim=config.hidden_size,
            num_heads=cross_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=False
        )
        
        self.label2text = Label2Text(
            label_embedding_size=config.hidden_size,
            attn_hidden_size=config.hidden_size
        )

        self.contrast_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True)
        self.linear1 = nn.Linear(config.hidden_size * self.num_labels, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, self.num_labels)
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        
        if self.loss_type == 'zlpr':
            self.criterion = None
        elif self.loss_type == 'bce':
            self.criterion = BCEloss(num_labels=config.num_labels)
        elif self.loss_type == 'hbm':
            self.criterion = HBMloss(
                num_labels=config.num_labels,
                alpha=kwargs.get('hbm_alpha', 1.0),
                margin=kwargs.get('hbm_margin', 0.1),
                use_hierarchy=True,
                depth2label=depth2label,
                path_list=path_list,
                mode=kwargs.get('hbm_loss_mode', 'unit')
            )
            self.hbm_bound_generator = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1, bias=False)
            )
            nn.init.xavier_normal_(self.hbm_bound_generator[0].weight)
            nn.init.zeros_(self.hbm_bound_generator[0].bias)
            nn.init.zeros_(self.hbm_bound_generator[2].weight)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Choose from ['zlpr', 'bce', 'hbm']")
        
        self.lambda_1 = kwargs.get('lambda_1', 0.1) # Sample-level CL weight
        self.lambda_2 = kwargs.get('lambda_2', 0.1) # Label-level CL weight
        
        # Sample-level Contrastive Loss
        self.structure_cl = StructureAwareContrastiveLoss(
            temperature=self.tau,
            num_labels=self.num_labels,
            device=self.device, 
            depth2label=depth2label,
            path_list=path_list,
            dropout_rate=kwargs.get('dropout_cl_rate', 0.1) 
        )
        self.init_weights()

    def depth_weighted_jaccard_similarity(self, labels):
        """
        Args:
            labels: [batch_size, num_labels] Multi-hot label matrix
        Returns:
            J: [batch_size, batch_size] Jaccard similarity matrix
            mu: scalar, sum(d_k + s_k)
        """
        if getattr(self, 'label_depths', None) is None:
            raise ValueError("label_depths is not initialized in the model.")
        
        depths = self.label_depths  # [num_labels]
        D = torch.max(depths)
        
        d_k = D - depths + 1
        
        children_count = torch.zeros(self.num_labels, device=depths.device)
        if self.path_list is not None:
            for parent, child in self.path_list:
                if 0 <= parent < self.num_labels:
                    children_count[parent] += 1
        s_k = torch.log(1 + children_count)
        
        # Combined Weight
        w_k = d_k + s_k  # [num_labels]
        mu = w_k.sum()
        
        labels = labels.float()
        weighted_disagree = (torch.matmul(labels * w_k, (1 - labels).T) +
                             torch.matmul((1 - labels) * w_k, labels.T))
        J = (mu - weighted_disagree) / mu
        
        return J, mu

    def label_contrastive_loss(self, label_embeddings, gold_labels, batch_idx, jaccard_sim, mu):
        """
        Label-level Contrastive Loss 
        Args:
            label_embeddings: [num_pos, hidden_size] positive label embeddings
            gold_labels: [num_pos] label indices
            batch_idx: [num_pos] batch indices
            jaccard_sim: [batch_size, batch_size] depth-weighted jaccard similarity
            mu: scalar, negative sample scaling factor
        """
        expanded_jaccard = jaccard_sim[batch_idx, :][:, batch_idx]
        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        def exp_cosine_sim(x1, x2, eps=1e-15):
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = x2.norm(p=2, dim=1, keepdim=True)
            sim = torch.matmul(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
            return torch.exp(torch.clamp(sim / self.tau, min=-20, max=20))

        for k in range(self.num_labels):
            pos_idx = (gold_labels == k).nonzero().squeeze(1)
            if pos_idx.numel() <= 1:
                continue
            neg_idx = (gold_labels != k).nonzero().squeeze(1)
            if neg_idx.numel() == 0:
                continue

            pos_samples = label_embeddings[pos_idx]
            neg_samples = label_embeddings[neg_idx]

            J_pp = expanded_jaccard[pos_idx, :][:, pos_idx]
            P = exp_cosine_sim(pos_samples, pos_samples) * J_pp

            J_pn = expanded_jaccard[pos_idx, :][:, neg_idx]
            N = mu * exp_cosine_sim(pos_samples, neg_samples) * (1 - J_pn)

            mask = ~torch.eye(pos_idx.numel(), device=self.device).bool()
            P_masked = P * mask.float()

            numerator = P_masked.sum(dim=1)
            denominator = numerator + N.sum(dim=1)

            valid = numerator > 0
            if valid.any():
                log_prob = torch.log(numerator[valid] / (denominator[valid] + 1e-8))
                loss += (-log_prob.mean())

        loss = loss / self.num_labels
        return loss


    def get_graph_label_embeddings(self):
        """
        Get Label Embeddings enhanced by Graph Encoder
        """
        word_embeddings = self.bert.embeddings.word_embeddings
        out = self.graph_encoder(self.label_embeddings, word_embeddings)
        return out


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            return_dict=None,
            label_depths=None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden_state = bert_output.last_hidden_state  # [Batch, Seq_Len, Hidden]
    
        sample_embedding = last_hidden_state[:, 0, :]   # [Batch, Hidden]
        encode_out = last_hidden_state[:, 1:, :]          # [Batch, Seq_Len-1, Hidden]

        graph_label_emb = self.get_graph_label_embeddings()  # [Num_Labels, Hidden]

        batch_size = input_ids.size(0)
        label_repr = graph_label_emb
        label_repr = label_repr.unsqueeze(0).expand(batch_size, -1, -1)

        
        if attention_mask is not None:
            sent_inputs_mask = attention_mask[:, 1:]  # [Batch, Seq_Len-1]
            extended_attention_mask = (1.0 - sent_inputs_mask[:, None, None, :].to(dtype=self.bert.dtype)) * -10000.0
        else:
            sent_inputs_mask = None
            extended_attention_mask = None

        # Cross Attention: Text (Query) attends to Labels (Key/Value)
        encode_out, _, _ = self.cross_attention(
            hidden_states=encode_out,          # Query: [Batch, Seq_Len-1, Hidden]
            key_value_states=label_repr,       # Key/Value: [Batch, Num_Labels, Hidden]
            attention_mask=None                # Unmasked Labels
        )
        # Output: [Batch, Seq_Len-1, Hidden]

        label_aware_embeddings, _ = self.label2text(
            input_data=encode_out,  # [Batch, Seq_Len-1, Hidden]
            label_repr=label_repr,  # [Batch, Num_Labels, Hidden]
            padding_mask=sent_inputs_mask  # [Batch, Seq_Len-1]
        )
        
        proj_label_embedding = self.dropout(label_aware_embeddings)
        graph_label_emb_expanded = graph_label_emb.unsqueeze(0).expand(batch_size, -1, -1)
        fusion_label_embedding = torch.cat([proj_label_embedding, graph_label_emb_expanded], dim=-1)

        fusion_attn_weights = self.contrast_proj(fusion_label_embedding)
        fusion_attn_weights = torch.softmax(fusion_attn_weights, dim=-1)
        fusion_attn_weights = torch.bmm(fusion_attn_weights, encode_out.transpose(1, 2))

        label_specific_embedding = torch.bmm(fusion_attn_weights, encode_out)
        features = self.dropout(label_specific_embedding)
        cls_embedding = features.view(batch_size, -1)

        intermediate_embedding = self.linear1(cls_embedding)
        intermediate_embedding = torch.relu(intermediate_embedding)

        logits = self.linear2(intermediate_embedding)
        
        total_loss = None
        cls_loss = torch.tensor(0.0).to(self.device)
        contrastive_loss = torch.tensor(0.0).to(self.device)
        sample_loss = torch.tensor(0.0).to(self.device)
        
        bound = None
        if self.loss_type == 'hbm':
            bound = self.hbm_bound_generator(sample_embedding)  # [batch_size, 1]
        
        if labels is not None:
            target_labels = labels.view(-1, self.num_labels)
            logits = torch.clamp(logits, min=-100, max=100)
            
            if self.loss_type == 'zlpr':
                cls_loss = multilabel_categorical_crossentropy(target_labels, logits)
            elif self.loss_type == 'bce':
                cls_loss = self.criterion(logits, target_labels)
            elif self.loss_type == 'hbm':
                cls_loss = self.criterion(logits, target_labels, bound=bound)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            total_loss = cls_loss
            contrastive_loss = torch.tensor(0.0).to(self.device)
            sample_loss = torch.tensor(0.0).to(self.device) # Initialize Sample-level Contrastive Loss
            
            # Construct List[List[int]] for labels (required for hard negative mining)
            batch_labels_list = []
            for i in range(target_labels.size(0)):
                 # Get active indices
                 indices = (target_labels[i] > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
                 # Ensure it's a list even if scalar
                 if isinstance(indices, int): indices = [indices]
                 batch_labels_list.append(indices)
            
            # Calculate Structure-Aware Contrastive Loss
            if getattr(self, 'structure_cl', None) is not None:
                # Input: Features (Pre-Dropout Label-Aware Representations)
                #        Labels List
                ls_loss = self.structure_cl(label_specific_embedding, batch_labels_list)
                
                if torch.isnan(ls_loss) or torch.isinf(ls_loss):
                    ls_loss = torch.tensor(0.0, device=self.device)
                
                # Weight by lambda_1
                sample_loss = ls_loss * self.lambda_1
                total_loss += sample_loss
            else:
                sample_loss = torch.tensor(0.0, device=self.device)

            if getattr(self, 'label_depths', None) is not None:
                jaccard_sim, mu = self.depth_weighted_jaccard_similarity(target_labels)
                
                jaccard_sim = jaccard_sim.to(self.device)
                mu = mu.to(self.device)

                flat_label_embeddings = features.view(-1, features.shape[-1])
                
                batch_idx = torch.arange(batch_size).to(self.device)
                batch_idx = batch_idx.unsqueeze(1).expand(-1, self.num_labels).flatten()

                mask = target_labels.to(torch.bool).flatten()

                label_ids = torch.arange(self.num_labels).to(self.device).expand(batch_size, self.num_labels).flatten()
                masked_gold_labels = torch.masked_select(label_ids, mask)
                masked_batch_idx = torch.masked_select(batch_idx, mask)
                masked_embeddings = flat_label_embeddings[mask]
                
                weighted_label_contrastive_loss = self.label_contrastive_loss(
                    masked_embeddings, 
                    masked_gold_labels, 
                    masked_batch_idx, 
                    jaccard_sim, 
                    mu
                )
                
                if torch.isnan(weighted_label_contrastive_loss) or torch.isinf(weighted_label_contrastive_loss):
                    weighted_label_contrastive_loss = torch.tensor(0.0, device=self.device)
                
                weighted_loss = weighted_label_contrastive_loss * self.lambda_2
                contrastive_loss = weighted_loss
                total_loss += weighted_loss
            else:
                contrastive_loss = torch.tensor(0.0).to(self.device)


        if not return_dict:
            output = (logits,) + bert_output[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return {
            'loss': total_loss,
            'logits': logits,
            'hidden_states': bert_output.hidden_states,
            'attentions': bert_output.attentions,
            'cls_loss': cls_loss,
            'contrastive_loss': contrastive_loss,
            'sample_loss': sample_loss,
            'bound': bound,
        }

    @torch.no_grad()
    def generate(self, input_ids, depth2label, threshold=0, **kwargs):
        """
        Args:
            input_ids: [batch_size, seq_len] input token IDs
            depth2label: {depth: [label_ids]} mapping from depth to label IDs
            threshold: prediction threshold (default 0 for logits)
            
        Returns:
            predict_labels: Predicted label lists [[label1, label2, ...], ...]
            prediction_scores: Original logits [batch_size, num_labels]
        """
        attention_mask = input_ids != self.config.pad_token_id
        outputs = self(input_ids, attention_mask=attention_mask)
        
        if isinstance(outputs, dict):
            prediction_scores = outputs['logits']
            batch_bounds = outputs.get('bound', None)
        else:
            prediction_scores = outputs.logits
            batch_bounds = getattr(outputs, 'bound', None)
        
        predict_labels = []
        
        # Iterate over each sample in batch
        for i, scores in enumerate(prediction_scores):
            single_pred = []
            
            # Determine threshold for current sample
            if batch_bounds is not None:
                current_threshold = batch_bounds[i].item()
            else:
                current_threshold = threshold
            
            # Hierarchical traversal using depth2label
            # Ensures consistency of label predictions across hierarchy
            for depth in sorted(depth2label.keys()):
                # Get all label IDs at this depth
                labels_at_depth = depth2label[depth]
                
                # Collect labels with score > threshold
                for label_id in labels_at_depth:
                    if label_id < len(scores):  # Ensure within bounds
                        if scores[label_id].item() > current_threshold:
                            single_pred.append(label_id)
            
            predict_labels.append(single_pred)
            
        return predict_labels, prediction_scores