"""
Conformal prediction for molecular GNNs.

Split-conformal approach for calibrated uncertainty:
- Nonconformity scores from validation set
- Prediction sets at user-specified significance levels
- "Confident explanations" only when prediction set is singleton
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


class ConformalPredictor:
    """Split-conformal prediction for binary classification.

    Uses the validation set to calibrate nonconformity scores,
    then constructs prediction sets for test molecules.
    """

    def __init__(self, model: nn.Module, target_key: str = 'sensitization'):
        self.model = model
        self.target_key = target_key
        self.calibration_scores = None

    def _get_prediction(self, data: Data, device: torch.device) -> float:
        """Get predicted probability for a single molecule."""
        data = data.clone().to(device)
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            logit = outputs[self.target_key]
            if logit.dim() == 0:
                logit = logit.unsqueeze(0)
            prob = torch.sigmoid(logit).item()
        return prob

    def _nonconformity_score(self, prob: float, true_label: int) -> float:
        """Nonconformity score: 1 - predicted probability of true class."""
        if true_label == 1:
            return 1.0 - prob
        else:
            return prob  # 1 - (1 - prob) = prob

    def calibrate(
        self,
        cal_data: List[Data],
        cal_labels: np.ndarray,
        device: Optional[torch.device] = None,
    ):
        """Calibrate using a held-out calibration set.

        Args:
            cal_data: List of PyG Data objects (calibration molecules).
            cal_labels: Binary labels for calibration set.
            device: Device for model inference.
        """
        if device is None:
            device = next(self.model.parameters()).device

        scores = []
        for data, label in zip(cal_data, cal_labels):
            prob = self._get_prediction(data, device)
            score = self._nonconformity_score(prob, int(label))
            scores.append(score)

        self.calibration_scores = np.sort(scores)

    def _get_quantile(self, alpha: float) -> float:
        """Get the (1-alpha) quantile of calibration scores."""
        if self.calibration_scores is None:
            raise RuntimeError("Must call calibrate() first")
        n = len(self.calibration_scores)
        # Finite-sample valid quantile
        idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
        idx = min(idx, n - 1)
        return self.calibration_scores[idx]

    def predict_set(
        self,
        data: Data,
        alpha: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> Dict[str, any]:
        """Construct prediction set for a single molecule.

        Args:
            data: PyG Data object.
            alpha: Significance level (e.g., 0.1 for 90% coverage).
            device: Device.

        Returns:
            Dict with prediction set, predicted probability, and confidence info.
        """
        if device is None:
            device = next(self.model.parameters()).device

        prob = self._get_prediction(data, device)
        threshold = self._get_quantile(alpha)

        # Check which classes are in the prediction set
        # Class y is in set if nonconformity_score(prob, y) <= threshold
        score_positive = 1.0 - prob   # score for y=1
        score_negative = prob          # score for y=0

        prediction_set = []
        if score_negative <= threshold:
            prediction_set.append(0)
        if score_positive <= threshold:
            prediction_set.append(1)

        return {
            'prediction_set': prediction_set,
            'set_size': len(prediction_set),
            'is_singleton': len(prediction_set) == 1,
            'predicted_prob': prob,
            'predicted_class': int(prob >= 0.5),
            'score_positive': score_positive,
            'score_negative': score_negative,
            'threshold': threshold,
            'alpha': alpha,
        }

    def predict_batch(
        self,
        data_list: List[Data],
        alpha: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> List[Dict]:
        """Predict sets for a batch of molecules."""
        return [self.predict_set(d, alpha=alpha, device=device) for d in data_list]

    def evaluate_coverage(
        self,
        data_list: List[Data],
        true_labels: np.ndarray,
        alphas: List[float] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate conformal coverage and efficiency at multiple alpha levels.

        Args:
            data_list: List of PyG Data objects (test set).
            true_labels: Binary labels.
            alphas: List of significance levels to evaluate.

        Returns:
            Dict mapping alpha -> coverage metrics.
        """
        if alphas is None:
            alphas = [0.05, 0.10, 0.20]

        results = {}
        for alpha in alphas:
            predictions = self.predict_batch(data_list, alpha=alpha, device=device)

            # Coverage: fraction where true label is in prediction set
            covered = sum(
                1 for pred, label in zip(predictions, true_labels)
                if int(label) in pred['prediction_set']
            )
            coverage = covered / len(true_labels)

            # Efficiency: average set size
            avg_size = np.mean([p['set_size'] for p in predictions])

            # Singleton rate
            singleton_rate = np.mean([p['is_singleton'] for p in predictions])

            # Empty set rate
            empty_rate = np.mean([p['set_size'] == 0 for p in predictions])

            # Conditional coverage by class
            sens_mask = true_labels == 1
            nonsens_mask = true_labels == 0

            sens_covered = sum(
                1 for pred, label, is_sens in zip(predictions, true_labels, sens_mask)
                if is_sens and int(label) in pred['prediction_set']
            )
            nonsens_covered = sum(
                1 for pred, label, is_ns in zip(predictions, true_labels, nonsens_mask)
                if is_ns and int(label) in pred['prediction_set']
            )

            results[f'alpha_{alpha}'] = {
                'alpha': alpha,
                'target_coverage': 1 - alpha,
                'empirical_coverage': coverage,
                'avg_set_size': avg_size,
                'singleton_rate': singleton_rate,
                'empty_set_rate': empty_rate,
                'sensitizer_coverage': sens_covered / sens_mask.sum() if sens_mask.sum() > 0 else float('nan'),
                'nonsensitizer_coverage': nonsens_covered / nonsens_mask.sum() if nonsens_mask.sum() > 0 else float('nan'),
            }

        return results

    def get_confident_mask(
        self,
        data_list: List[Data],
        alpha: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Get boolean mask of molecules where the model is 'confident'.

        A molecule is confident when its conformal prediction set is a singleton.

        Returns:
            Boolean array of length len(data_list).
        """
        predictions = self.predict_batch(data_list, alpha=alpha, device=device)
        return np.array([p['is_singleton'] for p in predictions])
