"""Modeling components for GNN-AOP skin sensitization prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Dict, Tuple


class AssayFeatureEncoder(nn.Module):
    """
    Encode continuous assay values with learned embeddings for missing data.

    Rationale:
    - Currently only binary assay results are used; ~56% of continuous data ignored
    - Continuous values (DPRA%, EC1.5, CD86 RFI) are more informative than binary calls
    - Learn per-feature embeddings for missing values instead of simple imputation

    Features by Key Event:
    - MIE: dpra_mean, dpra_cys, dpra_lys, kdpra_log
    - KE1: keratinosens_ec15
    - KE2: hclat_mean, hclat_ec200, usens_cv70
    - Irritation: irritation_viability_mean
    - Corrosion: corrosion_ghs_score, corrosion_ter_mean
    """

    CONTINUOUS_FEATURES = {
        # MIE features (protein binding)
        'dpra_mean': {'ke': 'mie', 'range': (0, 100), 'higher_is_positive': True},
        'dpra_cys': {'ke': 'mie', 'range': (0, 100), 'higher_is_positive': True},
        'dpra_lys': {'ke': 'mie', 'range': (0, 100), 'higher_is_positive': True},
        'kdpra_log': {'ke': 'mie', 'range': (-5, 5), 'higher_is_positive': True},

        # KE1 features (keratinocyte activation)
        'keratinosens_ec15': {'ke': 'ke1', 'range': (0, 2000), 'higher_is_positive': False},

        # KE2 features (dendritic cell activation)
        'hclat_mean': {'ke': 'ke2', 'range': (0, 500), 'higher_is_positive': True},
        'hclat_ec200': {'ke': 'ke2', 'range': (0, 500), 'higher_is_positive': False},
        'usens_cv70': {'ke': 'ke2', 'range': (0, 500), 'higher_is_positive': False},

        # Irritation features
        'irritation_viability_mean': {'ke': 'irr', 'range': (0, 100), 'higher_is_positive': False},

        # Corrosion features
        'corrosion_ghs_score': {'ke': 'corr', 'range': (0, 4), 'higher_is_positive': True},
        'corrosion_ter_mean': {'ke': 'corr', 'range': (0, 500), 'higher_is_positive': False},

        # Molecular descriptors (v5 - for acute dermal toxicity)
        # These affect skin absorption and permeation
        'mol_weight': {'ke': 'mol', 'range': (0, 1000), 'higher_is_positive': True},
        'logp': {'ke': 'mol', 'range': (-5, 10), 'higher_is_positive': True},  # Lipophilicity
        'tpsa': {'ke': 'mol', 'range': (0, 200), 'higher_is_positive': False},  # Polar surface area
        'hbd': {'ke': 'mol', 'range': (0, 10), 'higher_is_positive': False},  # H-bond donors
        'hba': {'ke': 'mol', 'range': (0, 15), 'higher_is_positive': False},  # H-bond acceptors
        'num_rotatable_bonds': {'ke': 'mol', 'range': (0, 20), 'higher_is_positive': False},
        'fraction_csp3': {'ke': 'mol', 'range': (0, 1), 'higher_is_positive': False},
    }

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        n_features = len(self.CONTINUOUS_FEATURES)
        self.feature_names = list(self.CONTINUOUS_FEATURES.keys())

        # Learnable embedding for missing values (per feature)
        self.missing_embeddings = nn.ParameterDict({
            name.replace('.', '_'): nn.Parameter(torch.zeros(1))
            for name in self.CONTINUOUS_FEATURES
        })

        # Feature-wise normalization (computed from data during setup)
        self.register_buffer('means', torch.zeros(n_features))
        self.register_buffer('stds', torch.ones(n_features))

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def set_normalization(self, means: torch.Tensor, stds: torch.Tensor):
        """Set normalization parameters from training data"""
        self.means.copy_(means)
        self.stds.copy_(stds)

    def forward(self, assay_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode continuous assay features.

        Args:
            assay_values: Dict mapping feature names to [batch] tensors (NaN for missing)

        Returns:
            encoded: [batch, hidden_dim] assay feature embedding
        """
        batch_size = None
        device = None

        # Determine batch size and device from first available feature
        for name in self.feature_names:
            if name in assay_values and assay_values[name] is not None:
                batch_size = assay_values[name].size(0)
                device = assay_values[name].device
                break

        if batch_size is None:
            raise ValueError("No valid assay features provided")

        features = []
        for i, name in enumerate(self.feature_names):
            safe_name = name.replace('.', '_')
            if name in assay_values and assay_values[name] is not None:
                val = assay_values[name].view(batch_size)

                # Replace NaN with learned missing embedding
                mask = torch.isnan(val)
                missing_val = self.missing_embeddings[safe_name].expand(batch_size)
                val = torch.where(mask, missing_val, val)

                # Normalize
                val = (val - self.means[i]) / (self.stds[i] + 1e-8)
            else:
                # All missing - use learned embedding
                val = self.missing_embeddings[safe_name].expand(batch_size)

            features.append(val)

        x = torch.stack(features, dim=1)  # [batch, n_features]
        encoded = self.encoder(x)
        return encoded


class CausalPredictionHead(nn.Module):
    """
    Prediction head that outputs both prediction and epistemic uncertainty.

    Uses MC Dropout for uncertainty estimation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.var_head = nn.Linear(hidden_dim // 2, 1)  # Log variance

        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: Prediction logits
            uncertainty: Epistemic uncertainty (log variance)
        """
        h = self.layers(x)
        mean = self.mean_head(h)
        log_var = self.var_head(h)

        return mean.squeeze(-1), log_var.squeeze(-1)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout for uncertainty estimation"""
        self.train()  # Enable dropout

        predictions = []
        for _ in range(n_samples):
            mean, _ = self.forward(x)
            predictions.append(torch.sigmoid(mean))

        predictions = torch.stack(predictions, dim=0)

        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean_pred, uncertainty


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Reduces loss for well-classified examples, focuses on hard examples.
    Especially useful for sensitization where positive class is rare (~12%).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if y=1, 1-p if y=0
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * BCE_loss
        return focal_loss.mean()


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for ordinal regression.

    Uses cumulative probability approach:
    For each threshold k, we have binary classification: Y >= k vs Y < k

    Total loss = sum of BCE losses for all K-1 thresholds

    This preserves ordinal structure better than standard multi-class CE.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.class_weights = class_weights  # Optional [K-1] weights for thresholds

    def forward(
        self,
        cumulative_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cumulative_logits: [batch, K-1] logits for P(Y >= k)
            targets: [batch] class indices (0, 1, 2, ..., K-1)

        Returns:
            Ordinal cross-entropy loss
        """
        batch_size = cumulative_logits.size(0)
        num_thresholds = cumulative_logits.size(1)  # K-1

        # Create binary targets for each threshold
        # binary_targets[i, k] = 1 if targets[i] >= k+1, else 0
        binary_targets = torch.zeros_like(cumulative_logits)
        for k in range(num_thresholds):
            binary_targets[:, k] = (targets >= (k + 1)).float()

        # BCE loss for each threshold
        loss = self.bce(cumulative_logits, binary_targets)

        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights.to(loss.device)
            loss = loss * weights.unsqueeze(0)

        return loss.mean()


class UncertaintyWeightedLoss(nn.Module):
    """
    Learnable task-specific uncertainty weighting for multi-task learning.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall & Gal, 2018).

    The key insight is that each task has an inherent "homoscedastic" uncertainty
    that affects how much we should weight that task's loss. Tasks with higher
    uncertainty should contribute less to the total loss.

    Loss formula:
        L_total = sum_i [ L_i / (2 * sigma_i^2) + log(sigma_i) ]

    Where sigma_i is a learned parameter for each task. The log(sigma_i) term
    prevents the model from simply setting all sigmas to infinity.
    """

    def __init__(
        self,
        tasks: list,
        init_log_vars: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            tasks: List of task names to weight
            init_log_vars: Optional initial log-variance values (higher = lower initial weight)
        """
        super().__init__()

        # Default initialization based on expected difficulty
        # Higher init = lower initial weight for that task
        default_init = {
            'sensitization': 0.0,   # Baseline (most data)
            'irritation': 0.5,      # Less data
            'corrosion': 1.0,       # Least data
            'mie': 0.3,
            'ke1': 0.3,
            'ke2': 0.3,
            'potency': 0.0,         # Main sensitization potency task
        }
        init_log_vars = init_log_vars or default_init

        # Learnable log-variance parameters
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.tensor([init_log_vars.get(task, 0.0)]))
            for task in tasks
        })

        self.tasks = tasks

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, float]]]:
        """
        Weight losses by learned uncertainty.

        Args:
            losses: Dict mapping task_name -> loss tensor

        Returns:
            total_loss: Uncertainty-weighted sum
            weighted_info: Dict with raw loss, weighted loss, and log_var per task
        """
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        weighted_info = {}

        for task, loss in losses.items():
            if task in self.log_vars:
                log_var = self.log_vars[task]
                # precision = 1 / (2 * variance)
                precision = torch.exp(-log_var)
                # Weighted loss = loss / (2 * sigma^2) + log(sigma)
                # = loss * exp(-log_var) + 0.5 * log_var
                weighted_loss = precision * loss + 0.5 * log_var
                total = total + weighted_loss

                weighted_info[task] = {
                    'raw': loss.item(),
                    'weighted': weighted_loss.item(),
                    'log_var': log_var.item(),
                    'effective_weight': precision.item(),
                }
            else:
                # Task not in learnable parameters, add directly
                total = total + loss
                weighted_info[task] = {
                    'raw': loss.item(),
                    'weighted': loss.item(),
                    'log_var': 0.0,
                    'effective_weight': 1.0,
                }

        return total, weighted_info

    def get_task_weights(self) -> Dict[str, float]:
        """Get current effective weights (precision = exp(-log_var)) for logging."""
        return {
            task: torch.exp(-log_var).item()
            for task, log_var in self.log_vars.items()
        }

    def get_log_vars(self) -> Dict[str, float]:
        """Get current log-variance values for logging."""
        return {
            task: log_var.item()
            for task, log_var in self.log_vars.items()
        }


class CausalLoss(nn.Module):
    """
    Multi-objective loss for multi-task skin sensitization prediction.

    Components:
    1. Prediction loss (BCE for outcomes, or ordinal loss for potency)
    2. Key Event supervision loss (when NAM data available)
    3. Uncertainty calibration loss
    """

    def __init__(
        self,
        outcome_weight: float = 1.0,
        ke_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        use_focal_loss: bool = True,  # Enable focal loss for class imbalance
        focal_gamma: float = 2.0,  # Focal loss focusing parameter
        focal_alpha: float = 0.25,  # Focal loss balance parameter
        # Per-endpoint pos_weights based on class ratios
        sensitization_pos_weight: float = 8.0,  # Train: 7.7:1 neg:pos ratio
        irritation_pos_weight: float = 1.2,  # Train: ~1.1:1 ratio
        corrosion_pos_weight: float = 1.6,  # Train: ~1.6:1 ratio
        acute_dermal_pos_weight: float = 2.0,  # Toxic (GHS 1-3) vs non-toxic (GHS 4-5)
        use_ordinal_sensitization: bool = False,  # Use ordinal loss for potency
        ordinal_weight: float = 1.0,  # Weight for ordinal potency loss
        consistency_weight: float = 0.1,  # Mean Teacher consistency loss weight
        use_uncertainty_weighting: bool = False,  # Use learnable uncertainty weighting
        use_llna_auxiliary: bool = False,  # Use LLNA as pseudo-labels when human labels missing
        llna_auxiliary_weight: float = 0.3,  # Weight for LLNA pseudo-label loss (~61% agreement)
        llna_pos_weight: float = 0.5,  # Weight for LLNA+ pseudo-labels (79% PPV, reliable)
        llna_neg_weight: float = 0.1,  # Weight for LLNA- pseudo-labels (36% NPV, unreliable)
        use_asymmetric_llna: bool = False,  # Use asymmetric weighting for LLNA+/LLNA-
    ):
        super().__init__()

        self.outcome_weight = outcome_weight
        self.ke_weight = ke_weight
        self.uncertainty_weight = uncertainty_weight
        self.use_focal_loss = use_focal_loss
        self.sensitization_pos_weight = sensitization_pos_weight
        self.irritation_pos_weight = irritation_pos_weight
        self.corrosion_pos_weight = corrosion_pos_weight
        self.acute_dermal_pos_weight = acute_dermal_pos_weight
        self.use_ordinal_sensitization = use_ordinal_sensitization
        self.ordinal_weight = ordinal_weight
        self.consistency_weight = consistency_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_llna_auxiliary = use_llna_auxiliary
        self.llna_auxiliary_weight = llna_auxiliary_weight
        self.llna_pos_weight = llna_pos_weight
        self.llna_neg_weight = llna_neg_weight
        self.use_asymmetric_llna = use_asymmetric_llna

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ordinal_loss = OrdinalCrossEntropyLoss()

        # Per-endpoint pos_weights as buffers
        self.register_buffer('sens_pos_weight', torch.tensor([sensitization_pos_weight]))
        self.register_buffer('irr_pos_weight', torch.tensor([irritation_pos_weight]))
        self.register_buffer('corr_pos_weight', torch.tensor([corrosion_pos_weight]))
        self.register_buffer('dermal_pos_weight', torch.tensor([acute_dermal_pos_weight]))

        # Curriculum weights for progressive training
        # These are set externally by CurriculumScheduler during training
        self.curriculum_weights = {
            'sensitization': 1.0,
            'irritation': 1.0,
            'corrosion': 1.0,
            'acute_dermal': 1.0,
        }

        # Learnable uncertainty weighting (Kendall & Gal, 2018)
        if use_uncertainty_weighting:
            self.uncertainty_weighting = UncertaintyWeightedLoss(
                tasks=['sensitization', 'irritation', 'corrosion', 'acute_dermal', 'mie', 'ke1', 'ke2', 'potency']
            )
        else:
            self.uncertainty_weighting = None

    def set_curriculum_weights(self, weights: Dict[str, float]):
        """
        Update curriculum weights for progressive training.

        Called by CurriculumScheduler at the start of each epoch.

        Args:
            weights: Dict mapping endpoint names to weights (0 = skip, 1 = full weight)
        """
        self.curriculum_weights = weights.copy()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        ke_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss.

        Args:
            outputs: Model outputs
            targets: Ground truth outcomes {'sensitization', 'irritation', 'corrosion'}
            ke_targets: Optional Key Event targets {'mie', 'ke1', 'ke2'} from NAM data

        Returns:
            total_loss, loss_components
        """
        losses = {}
        loss_terms = []  # Collect all loss terms for proper gradient tracking

        # Ordinal sensitization potency loss (if enabled and potency targets available)
        if (self.use_ordinal_sensitization and
            'sensitization_cumlogits' in outputs and
            'sensitization_potency' in targets):

            cum_logits = outputs['sensitization_cumlogits']
            potency_target = targets['sensitization_potency']

            # Handle missing labels (-1 or NaN indicates missing)
            valid_mask = (potency_target >= 0) & (~torch.isnan(potency_target.float()))
            if valid_mask.sum() > 0:
                ordinal_l = self.ordinal_loss(
                    cum_logits[valid_mask],
                    potency_target[valid_mask].long()
                )
                losses['sensitization_potency_loss'] = ordinal_l
                loss_terms.append(self.ordinal_weight * ordinal_l)

        # Acute dermal ordinal loss (GHS categories 1-5)
        # GHS classes: 1=most toxic (LD50 <= 50), 5=least toxic (LD50 > 2000)
        # Target is 0-indexed: GHS 1 = class 0, GHS 5 = class 4
        if ('acute_dermal_cumlogits' in outputs and
            'acute_dermal_ghs' in targets):

            dermal_cumlogits = outputs['acute_dermal_cumlogits']
            ghs_target = targets['acute_dermal_ghs']

            # Handle missing labels (NaN indicates missing)
            # Convert GHS 1-5 to 0-indexed classes (0-4)
            valid_mask = ~torch.isnan(ghs_target)
            if valid_mask.sum() > 0:
                # GHS is 1-indexed (1-5), convert to 0-indexed (0-4)
                ghs_class = (ghs_target[valid_mask] - 1).long()
                ordinal_l = self.ordinal_loss(
                    dermal_cumlogits[valid_mask],
                    ghs_class
                )
                losses['acute_dermal_ghs_loss'] = ordinal_l
                # Apply curriculum weight for acute_dermal
                curriculum_w = self.curriculum_weights.get('acute_dermal', 1.0)
                loss_terms.append(self.ordinal_weight * curriculum_w * ordinal_l)

        # Outcome losses (binary)
        # For uncertainty weighting, we collect individual task losses separately
        individual_task_losses = {}

        # Get per-endpoint pos_weights
        endpoint_pos_weights = {
            'sensitization': self.sens_pos_weight,
            'irritation': self.irr_pos_weight,
            'corrosion': self.corr_pos_weight,
            'acute_dermal': self.dermal_pos_weight,
        }

        for endpoint in ['sensitization', 'irritation', 'corrosion', 'acute_dermal']:
            # Skip endpoints with zero curriculum weight
            curriculum_w = self.curriculum_weights.get(endpoint, 1.0)
            if curriculum_w == 0:
                continue

            if endpoint in outputs and endpoint in targets:
                pred = outputs[endpoint]
                target = targets[endpoint]
                pos_weight = endpoint_pos_weights[endpoint]

                # Handle missing labels
                valid_mask = ~torch.isnan(target)
                if valid_mask.sum() > 0:
                    if endpoint == 'sensitization' and self.use_ordinal_sensitization:
                        # For ordinal model, pred is already probability, convert target for BCE
                        with torch.cuda.amp.autocast(enabled=False):
                            loss = F.binary_cross_entropy(
                                pred[valid_mask].float(),
                                target[valid_mask].float(),
                                reduction='mean'
                            )
                    elif self.use_focal_loss:
                        # Focal loss + weighted BCE for all endpoints
                        focal_l = self.focal(pred[valid_mask], target[valid_mask])
                        weighted_bce = F.binary_cross_entropy_with_logits(
                            pred[valid_mask],
                            target[valid_mask],
                            pos_weight=pos_weight,
                            reduction='mean'
                        )
                        # Combine focal and weighted BCE
                        loss = 0.5 * focal_l + 0.5 * weighted_bce
                    else:
                        # Just weighted BCE for class imbalance
                        loss = F.binary_cross_entropy_with_logits(
                            pred[valid_mask],
                            target[valid_mask],
                            pos_weight=pos_weight,
                            reduction='mean'
                        )
                    losses[f'{endpoint}_loss'] = loss

                    # Store for uncertainty weighting, or add to loss_terms with curriculum weight
                    if self.uncertainty_weighting is not None:
                        individual_task_losses[endpoint] = loss
                    else:
                        loss_terms.append(self.outcome_weight * curriculum_w * loss)

        # LLNA Auxiliary Loss: Use LLNA as pseudo-labels when human labels are missing
        # This provides additional training signal from 352 LLNA-labeled samples without human labels
        if self.use_llna_auxiliary and 'sensitization_llna' in outputs and 'sensitization_llna' in targets:
            sens_pred = outputs['sensitization']  # Human sensitization prediction
            sens_target = targets['sensitization']
            llna_target = targets['sensitization_llna']

            # Find samples where human label is missing but LLNA is available
            human_missing = torch.isnan(sens_target)
            llna_available = ~torch.isnan(llna_target)
            use_llna_mask = human_missing & llna_available

            if use_llna_mask.sum() > 0:
                if self.use_asymmetric_llna:
                    # Asymmetric weighting: LLNA+ is reliable (79% PPV), LLNA- is unreliable (36% NPV)
                    # Apply different weights to LLNA+ and LLNA- pseudo-labels
                    llna_pos_mask = use_llna_mask & (llna_target == 1)
                    llna_neg_mask = use_llna_mask & (llna_target == 0)

                    total_llna_loss = torch.tensor(0.0, device=sens_pred.device)
                    n_pos = llna_pos_mask.sum()
                    n_neg = llna_neg_mask.sum()

                    # Loss for LLNA+ samples (higher weight - more reliable)
                    if n_pos > 0:
                        if self.use_focal_loss:
                            focal_pos = self.focal(sens_pred[llna_pos_mask], llna_target[llna_pos_mask])
                            bce_pos = F.binary_cross_entropy_with_logits(
                                sens_pred[llna_pos_mask], llna_target[llna_pos_mask], reduction='mean')
                            loss_pos = 0.5 * focal_pos + 0.5 * bce_pos
                        else:
                            loss_pos = F.binary_cross_entropy_with_logits(
                                sens_pred[llna_pos_mask], llna_target[llna_pos_mask], reduction='mean')
                        total_llna_loss = total_llna_loss + self.llna_pos_weight * loss_pos
                        losses['llna_pos_loss'] = loss_pos
                        losses['llna_pos_samples'] = n_pos.float()

                    # Loss for LLNA- samples (lower weight - less reliable)
                    if n_neg > 0:
                        if self.use_focal_loss:
                            focal_neg = self.focal(sens_pred[llna_neg_mask], llna_target[llna_neg_mask])
                            bce_neg = F.binary_cross_entropy_with_logits(
                                sens_pred[llna_neg_mask], llna_target[llna_neg_mask], reduction='mean')
                            loss_neg = 0.5 * focal_neg + 0.5 * bce_neg
                        else:
                            loss_neg = F.binary_cross_entropy_with_logits(
                                sens_pred[llna_neg_mask], llna_target[llna_neg_mask], reduction='mean')
                        total_llna_loss = total_llna_loss + self.llna_neg_weight * loss_neg
                        losses['llna_neg_loss'] = loss_neg
                        losses['llna_neg_samples'] = n_neg.float()

                    losses['llna_auxiliary_loss'] = total_llna_loss
                    losses['llna_auxiliary_samples'] = use_llna_mask.sum().float()
                    loss_terms.append(total_llna_loss)
                else:
                    # Symmetric weighting (original behavior)
                    # Use LLNA as pseudo-label for sensitization with reduced weight
                    # Apply same focal + weighted BCE combination as main loss
                    if self.use_focal_loss:
                        focal_l = self.focal(sens_pred[use_llna_mask], llna_target[use_llna_mask])
                        weighted_bce = F.binary_cross_entropy_with_logits(
                            sens_pred[use_llna_mask],
                            llna_target[use_llna_mask],
                            pos_weight=self.sens_pos_weight,
                            reduction='mean'
                        )
                        llna_aux_loss = 0.5 * focal_l + 0.5 * weighted_bce
                    else:
                        llna_aux_loss = F.binary_cross_entropy_with_logits(
                            sens_pred[use_llna_mask],
                            llna_target[use_llna_mask],
                            pos_weight=self.sens_pos_weight,
                            reduction='mean'
                        )

                    losses['llna_auxiliary_loss'] = llna_aux_loss
                    losses['llna_auxiliary_samples'] = use_llna_mask.sum().float()
                    loss_terms.append(self.llna_auxiliary_weight * llna_aux_loss)

        # Key Event supervision (when NAM data available)
        if ke_targets is not None:
            for ke in ['mie', 'ke1', 'ke2']:
                if ke in outputs and ke in ke_targets:
                    pred = outputs[ke]
                    target = ke_targets[ke]

                    valid_mask = ~torch.isnan(target)
                    if valid_mask.sum() > 0:
                        loss = self.bce(pred[valid_mask], target[valid_mask]).mean()
                        losses[f'{ke}_loss'] = loss

                        # Store for uncertainty weighting, or add to loss_terms
                        if self.uncertainty_weighting is not None:
                            individual_task_losses[ke] = loss
                        else:
                            loss_terms.append(self.ke_weight * loss)

        # Uncertainty calibration (if uncertainty outputs available)
        if 'sensitization_uncertainty' in outputs:
            # Penalize overconfident wrong predictions
            for endpoint in ['sensitization', 'irritation', 'corrosion', 'acute_dermal']:
                if f'{endpoint}_uncertainty' in outputs and endpoint in targets:
                    unc = outputs[f'{endpoint}_uncertainty']
                    # For ordinal sensitization, prediction is already probability
                    if endpoint == 'sensitization' and self.use_ordinal_sensitization:
                        pred = outputs[endpoint]
                    else:
                        pred = torch.sigmoid(outputs[endpoint])
                    target = targets[endpoint]

                    valid_mask = ~torch.isnan(target)
                    if valid_mask.sum() > 0:
                        # Error should correlate with uncertainty
                        error = (pred[valid_mask] - target[valid_mask]).abs()
                        unc_loss = F.mse_loss(torch.exp(unc[valid_mask]), error)
                        losses[f'{endpoint}_unc_loss'] = unc_loss
                        loss_terms.append(self.uncertainty_weight * unc_loss)

        # Mean Teacher consistency loss (when teacher outputs provided)
        if 'teacher_outputs' in outputs and self.consistency_weight > 0:
            teacher_outputs = outputs['teacher_outputs']
            for endpoint in ['sensitization', 'irritation', 'corrosion', 'acute_dermal']:
                if endpoint in teacher_outputs:
                    # Get student and teacher predictions
                    student_pred = outputs[endpoint]
                    teacher_pred = teacher_outputs[endpoint]

                    # For ordinal sensitization, student output is already probability
                    if endpoint == 'sensitization' and self.use_ordinal_sensitization:
                        student_prob = student_pred
                        teacher_prob = teacher_pred
                    else:
                        student_prob = torch.sigmoid(student_pred)
                        teacher_prob = torch.sigmoid(teacher_pred)

                    # MSE consistency loss (soft labels from teacher)
                    consistency = F.mse_loss(student_prob, teacher_prob.detach())
                    losses[f'{endpoint}_consistency'] = consistency
                    loss_terms.append(self.consistency_weight * consistency)

        # Sum all loss terms (ensures proper gradient flow)
        # Use uncertainty weighting if enabled
        if self.uncertainty_weighting is not None and individual_task_losses:
            # Apply learnable uncertainty weighting to all task losses
            uncertainty_loss, weighted_info = self.uncertainty_weighting(individual_task_losses)

            # Log uncertainty weights for monitoring
            for task, info in weighted_info.items():
                losses[f'{task}_effective_weight'] = info['effective_weight']

            # Add uncertainty-weighted losses plus any remaining loss terms
            if loss_terms:
                total_loss = uncertainty_loss + sum(loss_terms)
            else:
                total_loss = uncertainty_loss

        elif loss_terms:
            total_loss = sum(loss_terms)
        else:
            # Edge case: no valid labels in batch - create dummy loss with gradients
            for key in ['sensitization', 'irritation', 'corrosion']:
                if key in outputs:
                    total_loss = 0.0 * outputs[key].mean()
                    break
            else:
                # Fallback if no outputs available
                device = next(iter(outputs.values())).device if outputs else 'cpu'
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss, losses


from rdkit import Chem


class MoleculeFeaturizer:
    """Convert SMILES to molecular graph"""

    ATOM_FEATURES = {
        'atomic_num': list(range(1, 119)),
        'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW',
                      'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
        'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        'num_hs': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'],
        'is_aromatic': [False, True],
        'is_in_ring': [False, True],
    }

    BOND_FEATURES = {
        'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
        'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE'],
        'is_conjugated': [False, True],
    }

    @classmethod
    def get_atom_dim(cls):
        return sum(len(v) for v in cls.ATOM_FEATURES.values())

    @classmethod
    def get_bond_dim(cls):
        return sum(len(v) for v in cls.BOND_FEATURES.values())

    @staticmethod
    def _one_hot(value, choices):
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        return encoding

    @classmethod
    def smiles_to_graph(cls, smiles: str) -> Optional[Data]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = []
            features.extend(cls._one_hot(atom.GetAtomicNum(), cls.ATOM_FEATURES['atomic_num']))
            features.extend(cls._one_hot(str(atom.GetChiralTag()), cls.ATOM_FEATURES['chirality']))
            features.extend(cls._one_hot(atom.GetTotalDegree(), cls.ATOM_FEATURES['degree']))
            features.extend(cls._one_hot(atom.GetFormalCharge(), cls.ATOM_FEATURES['formal_charge']))
            features.extend(cls._one_hot(atom.GetTotalNumHs(), cls.ATOM_FEATURES['num_hs']))
            features.extend(cls._one_hot(str(atom.GetHybridization()), cls.ATOM_FEATURES['hybridization']))
            features.extend(cls._one_hot(atom.GetIsAromatic(), cls.ATOM_FEATURES['is_aromatic']))
            features.extend(cls._one_hot(atom.IsInRing(), cls.ATOM_FEATURES['is_in_ring']))
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge features
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])

            features = []
            features.extend(cls._one_hot(str(bond.GetBondType()), cls.BOND_FEATURES['bond_type']))
            features.extend(cls._one_hot(str(bond.GetStereo()), cls.BOND_FEATURES['stereo']))
            features.extend(cls._one_hot(bond.GetIsConjugated(), cls.BOND_FEATURES['is_conjugated']))
            edge_attr.extend([features, features])

        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, cls.get_bond_dim()), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
