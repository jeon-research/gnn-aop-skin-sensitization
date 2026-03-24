"""
Ensemble explanation combining multiple attribution methods.

Rank-based aggregation and consensus identification across
Integrated Gradients, GradCAM, Attention, and GNNExplainer.
"""

from typing import Dict, List

import torch
import numpy as np
from scipy import stats as scipy_stats


def normalize_importance(importance: torch.Tensor) -> torch.Tensor:
    """Normalize importance scores to [0, 1] per molecule."""
    if importance.numel() == 0:
        return importance
    min_val = importance.min()
    max_val = importance.max()
    if max_val > min_val:
        return (importance - min_val) / (max_val - min_val)
    return torch.zeros_like(importance)


class EnsembleExplanation:
    """Combine multiple explanation methods into consensus importance scores."""

    METHOD_NAMES = ['ig', 'gradcam', 'attention', 'gnnexplainer', 'pgexplainer', 'graphmask']

    def __init__(self, min_methods_for_consensus: int = 3):
        """
        Args:
            min_methods_for_consensus: Minimum number of methods that must
                agree on an atom being important for it to be in the consensus set.
        """
        self.min_methods_for_consensus = min_methods_for_consensus

    def rank_aggregate(self, method_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine methods via rank-based aggregation.

        For each method, rank atoms by importance. Average ranks across methods.
        Lower average rank = more important.

        Args:
            method_scores: Dict mapping method name -> [n_atoms] importance.

        Returns:
            [n_atoms] consensus importance scores (higher = more important).
        """
        if not method_scores:
            raise ValueError("No method scores provided")

        n_atoms = next(iter(method_scores.values())).numel()
        all_ranks = []

        for name, scores in method_scores.items():
            scores_np = scores.float().numpy()
            # scipy rankdata: 1 = lowest, n = highest
            ranks = scipy_stats.rankdata(scores_np, method='average')
            all_ranks.append(ranks)

        # Average rank across methods
        avg_ranks = np.mean(all_ranks, axis=0)

        # Convert to importance: higher rank = more important, normalize to [0, 1]
        importance = torch.tensor(avg_ranks, dtype=torch.float32)
        return normalize_importance(importance)

    def consensus_atoms(
        self,
        method_scores: Dict[str, torch.Tensor],
        top_k: int = None,
    ) -> torch.Tensor:
        """Identify atoms in top-K for >= min_methods_for_consensus methods.

        Args:
            method_scores: Dict mapping method name -> [n_atoms] importance.
            top_k: Number of top atoms per method. If None, uses n_atoms // 3.

        Returns:
            [n_atoms] binary mask (1 = consensus important atom).
        """
        n_atoms = next(iter(method_scores.values())).numel()
        if top_k is None:
            top_k = max(1, n_atoms // 3)

        vote_counts = torch.zeros(n_atoms)

        for name, scores in method_scores.items():
            top_indices = torch.topk(scores.float(), min(top_k, n_atoms)).indices
            vote_counts[top_indices] += 1

        consensus = (vote_counts >= self.min_methods_for_consensus).float()
        return consensus

    def combine(
        self,
        method_scores: Dict[str, torch.Tensor],
        top_k: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Full ensemble: rank aggregation + consensus.

        Returns:
            Dict with:
                'rank_importance': [n_atoms] rank-aggregated scores
                'consensus_mask': [n_atoms] binary consensus mask
                'method_agreement': scalar Spearman correlation (mean pairwise)
        """
        result = {
            'rank_importance': self.rank_aggregate(method_scores),
            'consensus_mask': self.consensus_atoms(method_scores, top_k=top_k),
        }

        # Compute mean pairwise Spearman correlation
        methods = list(method_scores.values())
        if len(methods) >= 2:
            correlations = []
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    rho, _ = scipy_stats.spearmanr(
                        methods[i].float().numpy(),
                        methods[j].float().numpy(),
                    )
                    if not np.isnan(rho):
                        correlations.append(rho)
            result['method_agreement'] = torch.tensor(
                np.mean(correlations) if correlations else 0.0
            )
        else:
            result['method_agreement'] = torch.tensor(0.0)

        return result

    @staticmethod
    def pairwise_spearman(method_scores: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute all pairwise Spearman correlations between methods.

        Returns:
            Dict mapping "method1_vs_method2" -> Spearman rho.
        """
        names = list(method_scores.keys())
        values = [method_scores[n].float().numpy() for n in names]
        result = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rho, _ = scipy_stats.spearmanr(values[i], values[j])
                key = f"{names[i]}_vs_{names[j]}"
                result[key] = float(rho) if not np.isnan(rho) else 0.0

        return result
