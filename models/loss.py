import torch
import torch.nn as nn
import torch.nn.functional as F


import random
import math

def multilabel_categorical_crossentropy(y_true, y_pred):
    loss_mask = y_true != -100
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_pred.size(-1))
    y_true = y_true.masked_select(loss_mask).view(-1, y_true.size(-1))
    
    # Check if we have any valid elements
    if y_true.numel() == 0:
        return torch.tensor(0.0).to(y_pred.device)

    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    
    return (neg_loss + pos_loss).mean()

class BCEloss(nn.Module):
    def __init__(self, num_labels=None, **kwargs):
        super().__init__()
    
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='mean'
        )

class HBMloss(nn.Module):
    """
    Hierarchy-aware Biased Bound Margin Loss Function (HBM Loss)
    """
    
    def __init__(self, 
                 num_labels,
                 alpha=1.0,
                 margin=0.5,
                 use_hierarchy=True,
                 depth2label=None,
                 path_list=None,
                 mode='unit'):
        """
        Args:
            num_labels: Total number of labels
            alpha: HBM parameter (default: 1.0)
            margin: HBM margin (default: 0.5)
            use_hierarchy: Whether to use hierarchy weights (default: True)
            depth2label: Dict {depth: [label_ids]}
            path_list: List [(parent, child)]
            mode: 'unit' or 'hierarchy'
        """
        super().__init__()
        self.num_labels = num_labels
        self.alpha = alpha
        self.margin = margin
        self.use_hierarchy = use_hierarchy
        self.depth2label = depth2label
        self.mode = mode
        self.path_list = path_list
        
        if self.mode not in ['unit', 'hierarchy']:
            raise ValueError(f"Unknown HBM mode: {self.mode}. Choose 'unit' or 'hierarchy'.")
        
        # Build Units for 'unit' mode
        if self.mode == 'unit':
             self.units = self._build_units()
        
        # Compute hierarchy weights
        if use_hierarchy and depth2label is not None:
            self.register_buffer('label_weights', self._compute_hierarchy_weights())
        else:
            self.register_buffer('label_weights', torch.ones(num_labels))
            
    def _build_units(self):
        """
        Construct hierarchical units (Parent + Children group)
        """
        units = []
        if self.depth2label is None:
            return units
            
        # 1. Root Unit (Depth 1 labels)
        min_depth = min(self.depth2label.keys())
        root_children = [l for l in self.depth2label[min_depth] if l < self.num_labels]
        if root_children:
            units.append(root_children)
            
        # 2. Parent-Child Units
        if self.path_list:
            from collections import defaultdict
            parent_to_children = defaultdict(list)
            for u, v in self.path_list:
                if u < self.num_labels and v < self.num_labels:
                    parent_to_children[u].append(v)
            
            for p, children in parent_to_children.items():
                if children:
                    units.append(children)
        return units
    
    def _compute_hierarchy_weights(self):
        weights = torch.ones(self.num_labels)
        if self.depth2label is None:
            return weights
        
        for depth, label_ids in self.depth2label.items():
            weight = 1.0 + depth
            for label_id in label_ids:
                if label_id < self.num_labels:
                    weights[label_id] = weight
        
        weights = weights / weights.sum() * self.num_labels
        return weights
    
    def _compute_hbm_loss_exact(self, logits, targets, bound, label_weights=None):
        """
        Strict implementation of HBM logic from reference code (HBMChildCriterion).
        Includes Variance-based Bias Adjustment.
        
        logits: [Batch, N_Subset]
        targets: [Batch, N_Subset] (0 or 1)
        bound: [Batch, 1]
        """
        INF = torch.inf
        
        # Get pos, neg logits
        logits_transformed = (1 - 2 * targets) * logits          # l_neg, - l_pos
        logits_neg = logits_transformed.where(~targets.bool(), torch.tensor(-INF, device=logits.device))    # l_neg
        logits_pos = logits_transformed.where(targets.bool(), torch.tensor(-INF, device=logits.device))     # - l_pos
        
        # Compute standard deviation for biases
        logits_neg_padded = logits_neg # Shape [Batch, N]
        zero_like_padded = torch.zeros_like(logits_neg_padded)
        
        # Sigmoid check
        logits_neg_margin_mask = (2*(logits_neg_padded - bound)).sigmoid() > self.margin
        
        neg_mask = logits_neg_padded != -INF
        neg_mask_sum = neg_mask.sum(-1) # [Batch]
        
        # Compute Mean & Variance for active negatives
        logits_neg_zeropad = logits_neg_padded.where(neg_mask, zero_like_padded) # -INF -> 0 for sum
        
        # Avoid division by zero
        safe_neg_mask_sum = neg_mask_sum.clamp_min(1.0)
        logits_neg_mean = logits_neg_zeropad.sum(-1) / safe_neg_mask_sum
        
        # Subtract mean
        logits_neg_zeropad = logits_neg_zeropad - logits_neg_mean.unsqueeze(-1)
        logits_neg_zeropad = logits_neg_zeropad.where(neg_mask, zero_like_padded) # Restore 0s
        
        # Variance
        logits_neg_var = logits_neg_zeropad.square().sum(-1) / (neg_mask_sum - 1).clamp_min(1.0)
        logits_neg_std = logits_neg_var.unsqueeze(-1).nan_to_num(0.0).sqrt()
        
        # Adjust Neg Logits
        # "logits_neg = logits_neg_padded.where(logits_neg_margin_mask, -INF) - bound"
        # "logits_neg = logits_neg + logits_neg_std.detach()*self.alpha"
        
        logits_neg = logits_neg_padded.where(logits_neg_margin_mask, torch.tensor(-INF, device=logits.device)) - bound
        logits_neg = logits_neg + logits_neg_std.detach() * self.alpha
        
        
        # --- Pos logits with HBM ---
        logits_pos_padded = logits_pos
        
        logits_pos_margin_mask = (2*(logits_pos_padded + bound)).sigmoid() > self.margin
        
        pos_mask = logits_pos_padded != -INF
        pos_mask_sum = pos_mask.sum(-1)
        
        logits_pos_zeropad = logits_pos_padded.where(pos_mask, zero_like_padded)
        safe_pos_mask_sum = pos_mask_sum.clamp_min(1.0)
        logits_pos_mean = logits_pos_zeropad.sum(-1) / safe_pos_mask_sum
        
        logits_pos_zeropad = logits_pos_zeropad - logits_pos_mean.unsqueeze(-1)
        logits_pos_zeropad = logits_pos_zeropad.where(pos_mask, zero_like_padded)
        
        logits_pos_var = logits_pos_zeropad.square().sum(-1) / (pos_mask_sum - 1).clamp_min(1.0)
        logits_pos_std = logits_pos_var.unsqueeze(-1).nan_to_num(0.0).sqrt()
        
        # Adjust Pos Logits
        # "logits_pos = logits_pos_padded.where(logits_pos_margin_mask, -INF) + bound"
        # "logits_pos = logits_pos + logits_pos_std.detach()*self.alpha"
        
        logits_pos = logits_pos_padded.where(logits_pos_margin_mask, torch.tensor(-INF, device=logits.device)) + bound
        logits_pos = logits_pos + logits_pos_std.detach() * self.alpha

        zeros = torch.zeros(logits.shape[0], 1, device=logits.device, dtype=logits.dtype)
        
        # Note: Intentionally OMITTING label_weights addition inside logsumexp to strictly match Reference Code
        
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)
        
        neg_loss = torch.logsumexp(logits_neg, dim=-1)
        pos_loss = torch.logsumexp(logits_pos, dim=-1)
        
        loss = (neg_loss + pos_loss).mean()
        
        return loss

    def forward(self, logits, targets, bound=None):
        batch_size = logits.shape[0]
        
        if bound is None:
            bound = torch.full((batch_size, 1), 0.5, device=logits.device, dtype=logits.dtype)
        elif bound.dim() == 1:
            bound = bound.unsqueeze(-1)
        
        if self.mode == 'unit':
            if not self.units:
                return torch.tensor(0.0, device=logits.device)
            
            total_loss = torch.tensor(0.0, device=logits.device)
            num_valid_units = 0
            
            for unit_labels in self.units:
                # [Batch, Unit_Size]
                sub_logits = logits[:, unit_labels]
                sub_targets = targets[:, unit_labels]
                
                # Strictly follow "Level-wise" / "Unit-based" HBM.
                # Loss should only be calculated for samples where this unit is relevant (Active).
                # Relevance: The sample has at least one positive label in this unit (Teacher Forcing).
                
                sample_mask = sub_targets.sum(dim=1) > 0
                
                if sample_mask.sum() == 0:
                    continue
                
                # Filter rows
                active_logits = sub_logits[sample_mask]
                active_targets = sub_targets[sample_mask]
                active_bound = bound[sample_mask]
                
                sub_weights = None
                if self.use_hierarchy:
                    sub_weights = self.label_weights[unit_labels]
                
                # Call Strict Function
                unit_loss = self._compute_hbm_loss_exact(active_logits, active_targets, active_bound, sub_weights)
                total_loss += unit_loss
                num_valid_units += 1
            
            return total_loss / max(num_valid_units, 1)
            
        elif self.mode == 'hierarchy':
            # Layer-wise Sum using Strict HBM Logic on each layer
            if not self.depth2label:
                return torch.tensor(0.0, device=logits.device)
                
            total_loss = torch.tensor(0.0, device=logits.device)
            num_layers = 0
            
            for depth, label_ids in self.depth2label.items():
                if not label_ids:
                    continue
                    
                sub_logits = logits[:, label_ids]
                sub_targets = targets[:, label_ids]
                
                sub_weights = None
                if self.use_hierarchy:
                    sub_weights = self.label_weights[label_ids]
                
                layer_loss = self._compute_hbm_loss_exact(sub_logits, sub_targets, bound, sub_weights)
                total_loss += layer_loss
                num_layers += 1
                
            return total_loss / max(num_layers, 1)

class StructureAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, num_labels=None, device='cuda', depth2label=None, path_list=None, dropout_rate=0.1):
        super().__init__()
        self.temperature = temperature
        self.num_labels = num_labels
        self.device = device
        self.depth2label = depth2label # {depth: [label_ids]}
        self.path_list = path_list # [(parent, child)]
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Build Label Relations (Sibling, Parent, Child)
        self.label_relations = self._build_relations()
        # Calculate Max Depth D
        self.max_depth = max(depth2label.keys()) if depth2label else 1
        # Map Label to Depth
        self.label2depth = {}
        if depth2label:
            for d, lbs in depth2label.items():
                for l in lbs: self.label2depth[l] = d

    def _build_relations(self):
        """Build Label Relation Table: Parent, Children, Siblings"""
        relations = {i: {'parents': [], 'children': [], 'siblings': []} for i in range(self.num_labels)}
        
        # 1. Parent & Children
        if self.path_list:
            for u, v in self.path_list: # u -> v (Parent -> Child)
                if u < self.num_labels and v < self.num_labels:
                    relations[v]['parents'].append(u)
                    relations[u]['children'].append(v)
        
        # 2. Siblings (Sharing same Parent)
        for i in range(self.num_labels):
            parents = relations[i]['parents']
            for p in parents:
                for sibling in relations[p]['children']:
                    if sibling != i:
                        relations[i]['siblings'].append(sibling)
            relations[i]['siblings'] = list(set(relations[i]['siblings']))
            
        return relations

    def _get_ancestors(self, label_id):
        """Get all ancestors of a label"""
        ancestors = set()
        queue = [label_id]
        while queue:
            curr = queue.pop(0)
            for p in self.label_relations[curr]['parents']:
                if p not in ancestors:
                    ancestors.add(p)
                    queue.append(p)
        return ancestors

    def construct_hard_negative_labels(self, true_labels):
        """
        Construct Hard Negative Label Set based on Eq(10) & Eq(11)
        true_labels: List[int] (One Sample's positive label indices)
        """
        if not true_labels: 
            return [], -1
            
        true_labels_set = set(true_labels)
        y_d = max(true_labels, key=lambda l: self.label2depth.get(l, 0))
        
        candidates = []
        is_leaf = len(self.label_relations[y_d]['children']) == 0
        has_sibling = len(self.label_relations[y_d]['siblings']) > 0
        
        if is_leaf:
            if has_sibling:
                candidates = self.label_relations[y_d]['siblings']
            else:
                candidates = self.label_relations[y_d]['parents']
        else:
            candidates = self.label_relations[y_d]['children']
            
        if not candidates:
            all_indices = set(range(self.num_labels))
            neg_candidates = list(all_indices - true_labels_set)
            y_d_neg = random.choice(neg_candidates) if neg_candidates else y_d
        else:
            y_d_neg = random.choice(candidates)
            
        hard_neg_set = true_labels_set.copy()
        if y_d in hard_neg_set:
            hard_neg_set.remove(y_d) # Remove positive label itself
        
        hard_neg_set.add(y_d_neg) # Add hard negative label
        hard_neg_set.update(self._get_ancestors(y_d_neg)) # Add ancestors of hard negative
        
        return list(hard_neg_set), y_d_neg

    def forward(self, features, batch_labels_list):
        """
        Args:
            features: [Batch, Num_Labels, Hidden] - Label-Aware Representations (Pre-Dropout)
            batch_labels_list: List[List[int]] - True label indices for each sample
        """
        batch_size = features.size(0)
        device = features.device
        hidden_size = features.size(-1)
        
        Z_anchor_list = []
        Z_hard_neg_list = []
        weights_list = []
        valid_mask = []
        
        for i in range(batch_size):
            true_labels = batch_labels_list[i]
            if not true_labels:
                # Handle edge case
                Z_anchor_list.append(torch.zeros(hidden_size, device=device))
                Z_hard_neg_list.append(torch.zeros(hidden_size, device=device))
                weights_list.append(0.0)
                valid_mask.append(0.0)
                continue
                
            # Aggregate features corresponding to TRUE labels
            curr_features = features[i] # [Num_Labels, Hidden]
            anchor_feats = curr_features[true_labels]
            Z_anchor_i = anchor_feats.mean(dim=0)
            Z_anchor_list.append(Z_anchor_i)
            
            # Aggregate features corresponding to HARD NEGATIVE labels
            neg_labels, y_d_neg = self.construct_hard_negative_labels(true_labels)
            
            if not neg_labels:
                 # Fallback
                 Z_hat_i = Z_anchor_i
            else:
                 hard_neg_feats = curr_features[neg_labels]
                 Z_hat_i = hard_neg_feats.mean(dim=0)
            
            Z_hard_neg_list.append(Z_hat_i)
            
            d_neg = self.label2depth.get(y_d_neg, 0) if y_d_neg >= 0 else 0
            w_i = 1.0 + math.log(1.0 + d_neg / (self.max_depth + 1e-8))
            weights_list.append(w_i)
            valid_mask.append(1.0)

        # Stack Tensors
        Z_anchor = torch.stack(Z_anchor_list)       # [Batch, Hidden]
        Z_hard_neg = torch.stack(Z_hard_neg_list)   # [Batch, Hidden]
        
        Weights = torch.tensor(weights_list, device=device).view(-1, 1)
        Mask = torch.tensor(valid_mask, device=device).view(-1)
        
        Z_pos = self.dropout(Z_anchor)
        
        # Normalize
        Z_anchor = F.normalize(Z_anchor, dim=1)
        Z_pos = F.normalize(Z_pos, dim=1)
        Z_hard_neg = F.normalize(Z_hard_neg, dim=1)
        
        sim_pos = (Z_anchor * Z_pos).sum(dim=1, keepdim=True) / self.temperature
        exp_pos = torch.exp(sim_pos)
        
        sim_hard = (Z_anchor * Z_hard_neg).sum(dim=1, keepdim=True) / self.temperature
        exp_hard = Weights * torch.exp(sim_hard)
        
        sim_batch = torch.mm(Z_anchor, Z_anchor.t()) / self.temperature
        mask_diag = torch.eye(batch_size, device=device).bool()
        sim_batch.masked_fill_(mask_diag, float('-inf'))
        exp_batch = torch.sum(torch.exp(sim_batch), dim=1, keepdim=True)
        
        denominator = exp_pos + exp_hard + exp_batch
        log_prob = sim_pos - torch.log(denominator + 1e-8)
        
        loss = -log_prob.squeeze() * Mask
        return loss.sum() / (Mask.sum() + 1e-8)