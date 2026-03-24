"""
Explanation extraction, AOP reference labels, and alignment metrics
for systematic comparison of GNN attributions to AOP-40 mechanisms.
"""

from src.explain.integrated_gradients import IntegratedGradients
from src.explain.gradcam import GradCAM
from src.explain.attention_extractor import AttentionExtractor
from src.explain.gnn_explainer import GNNExplainerWrapper
from src.explain.ensemble_explanation import EnsembleExplanation
from src.explain.aop_reference import AOPReference
from src.explain.alignment_metrics import compute_alignment_metrics
__all__ = [
    'IntegratedGradients',
    'GradCAM',
    'AttentionExtractor',
    'GNNExplainerWrapper',
    'EnsembleExplanation',
    'AOPReference',
    'compute_alignment_metrics',
]
