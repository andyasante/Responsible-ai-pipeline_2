"""
Evaluation Framework and Testing Protocols for Responsible-AI Hate Speech Detection

This module implements comprehensive evaluation methodologies including:
1. Performance metrics for hate speech detection
2. Fairness and bias evaluation across demographics
3. Explainability quality assessment
4. Contextual adaptation effectiveness measurement
5. Human-AI collaboration evaluation
6. Longitudinal performance tracking
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    value: float
    metadata: Dict[str, Any]
    timestamp: datetime
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class BiasEvaluationResult:
    """Container for bias evaluation results."""
    demographic_group: str
    metric_name: str
    value: float
    baseline_value: float
    bias_score: float
    significance: str  # 'low', 'medium', 'high'

@dataclass
class ExplainabilityEvaluationResult:
    """Container for explainability evaluation results."""
    explanation_method: str
    quality_score: float
    consistency_score: float
    human_agreement_score: float
    computational_efficiency: float

class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Union[EvaluationResult, List[EvaluationResult]]:
        """Perform evaluation and return results."""
        pass

class PerformanceEvaluator(BaseEvaluator):
    """Evaluates basic performance metrics for hate speech detection."""
    
    def __init__(self):
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score
        }
        logger.info("Performance evaluator initialized")
    
    def evaluate(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: Optional[List[float]] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for AUC calculation)
            
        Returns:
            List of evaluation results
        """
        results = []
        timestamp = datetime.now()
        
        for metric_name, metric_func in self.metrics.items():
            try:
                if metric_name == 'auc' and y_prob is not None:
                    value = metric_func(y_true, y_prob)
                elif metric_name != 'auc':
                    value = metric_func(y_true, y_pred)
                else:
                    continue  # Skip AUC if no probabilities provided
                
                # Calculate confidence interval using bootstrap
                ci = self._calculate_confidence_interval(y_true, y_pred, metric_func, y_prob if metric_name == 'auc' else None)
                
                result = EvaluationResult(
                    metric_name=metric_name,
                    value=value,
                    metadata={
                        'n_samples': len(y_true),
                        'n_positive': sum(y_true),
                        'n_negative': len(y_true) - sum(y_true)
                    },
                    timestamp=timestamp,
                    confidence_interval=ci
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_name}: {str(e)}")
        
        logger.info(f"Calculated {len(results)} performance metrics")
        return results
    
    def _calculate_confidence_interval(
        self,
        y_true: List[int],
        y_pred: List[int],
        metric_func: callable,
        y_prob: Optional[List[float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap sampling."""
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = [y_true[i] for i in indices]
            y_pred_boot = [y_pred[i] for i in indices]
            
            try:
                if y_prob is not None:
                    y_prob_boot = [y_prob[i] for i in indices]
                    score = metric_func(y_true_boot, y_prob_boot)
                else:
                    score = metric_func(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
            except:
                continue
        
        if bootstrap_scores:
            alpha = 1 - confidence
            lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
            upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
            return (lower, upper)
        else:
            return (0.0, 1.0)
    
    def generate_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> np.ndarray:
        """Generate confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def generate_classification_report(self, y_true: List[int], y_pred: List[int]) -> str:
        """Generate detailed classification report."""
        return classification_report(y_true, y_pred, target_names=['Not Hate Speech', 'Hate Speech'])

class FairnessEvaluator(BaseEvaluator):
    """Evaluates fairness and bias across demographic groups."""
    
    def __init__(self):
        self.fairness_metrics = [
            'demographic_parity',
            'equalized_odds',
            'equal_opportunity',
            'calibration'
        ]
        logger.info("Fairness evaluator initialized")
    
    def evaluate(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: List[float],
        demographic_groups: Dict[str, List[int]]
    ) -> List[BiasEvaluationResult]:
        """
        Evaluate fairness across demographic groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            demographic_groups: Dictionary mapping group names to group membership indicators
            
        Returns:
            List of bias evaluation results
        """
        results = []
        
        # Calculate overall baseline metrics
        baseline_metrics = self._calculate_group_metrics(y_true, y_pred, y_prob)
        
        for group_name, group_indicators in demographic_groups.items():
            # Extract data for this group
            group_indices = [i for i, indicator in enumerate(group_indicators) if indicator == 1]
            
            if len(group_indices) < 10:  # Skip groups with too few samples
                logger.warning(f"Skipping group {group_name} with only {len(group_indices)} samples")
                continue
            
            group_y_true = [y_true[i] for i in group_indices]
            group_y_pred = [y_pred[i] for i in group_indices]
            group_y_prob = [y_prob[i] for i in group_indices]
            
            # Calculate metrics for this group
            group_metrics = self._calculate_group_metrics(group_y_true, group_y_pred, group_y_prob)
            
            # Calculate bias scores for each metric
            for metric_name in self.fairness_metrics:
                if metric_name in baseline_metrics and metric_name in group_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    group_value = group_metrics[metric_name]
                    
                    # Calculate bias score (absolute difference)
                    bias_score = abs(group_value - baseline_value)
                    
                    # Determine significance level
                    if bias_score < 0.05:
                        significance = 'low'
                    elif bias_score < 0.15:
                        significance = 'medium'
                    else:
                        significance = 'high'
                    
                    result = BiasEvaluationResult(
                        demographic_group=group_name,
                        metric_name=metric_name,
                        value=group_value,
                        baseline_value=baseline_value,
                        bias_score=bias_score,
                        significance=significance
                    )
                    results.append(result)
        
        logger.info(f"Evaluated fairness across {len(demographic_groups)} demographic groups")
        return results
    
    def _calculate_group_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: List[float]
    ) -> Dict[str, float]:
        """Calculate fairness metrics for a group."""
        metrics = {}
        
        # Demographic parity (positive prediction rate)
        metrics['demographic_parity'] = np.mean(y_pred)
        
        # Equalized odds (TPR and FPR should be equal across groups)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        if tp + fn > 0:  # Avoid division by zero
            tpr = tp / (tp + fn)  # True positive rate
            metrics['equal_opportunity'] = tpr
        
        if tn + fp > 0:  # Avoid division by zero
            fpr = fp / (tn + fp)  # False positive rate
            metrics['equalized_odds'] = (tpr + (1 - fpr)) / 2 if 'equal_opportunity' in metrics else 0.5
        
        # Calibration (predicted probabilities should match actual outcomes)
        if len(set(y_prob)) > 1:  # Check if probabilities are not all the same
            # Bin probabilities and calculate calibration
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(y_prob, bins) - 1
            calibration_scores = []
            
            for bin_idx in range(len(bins) - 1):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_prob = np.mean([y_prob[i] for i in range(len(y_prob)) if mask[i]])
                    bin_actual = np.mean([y_true[i] for i in range(len(y_true)) if mask[i]])
                    calibration_scores.append(abs(bin_prob - bin_actual))
            
            metrics['calibration'] = 1 - np.mean(calibration_scores) if calibration_scores else 0.5
        else:
            metrics['calibration'] = 0.5
        
        return metrics

class ExplainabilityEvaluator(BaseEvaluator):
    """Evaluates the quality of explanations provided by the system."""
    
    def __init__(self):
        logger.info("Explainability evaluator initialized")
    
    def evaluate(
        self,
        explanations: List[Dict[str, Any]],
        ground_truth_explanations: Optional[List[Dict[str, Any]]] = None,
        human_ratings: Optional[List[Dict[str, float]]] = None
    ) -> List[ExplainabilityEvaluationResult]:
        """
        Evaluate explanation quality.
        
        Args:
            explanations: List of explanations to evaluate
            ground_truth_explanations: Optional ground truth explanations
            human_ratings: Optional human quality ratings
            
        Returns:
            List of explainability evaluation results
        """
        results = []
        
        # Group explanations by method
        explanation_methods = {}
        for exp in explanations:
            method = exp.get('method', 'unknown')
            if method not in explanation_methods:
                explanation_methods[method] = []
            explanation_methods[method].append(exp)
        
        for method, method_explanations in explanation_methods.items():
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(method_explanations)
            consistency_score = self._calculate_consistency_score(method_explanations)
            efficiency_score = self._calculate_efficiency_score(method_explanations)
            
            # Calculate human agreement if available
            human_agreement = 0.5  # Default neutral score
            if human_ratings:
                method_ratings = [rating.get(method, 0.5) for rating in human_ratings]
                human_agreement = np.mean(method_ratings)
            
            result = ExplainabilityEvaluationResult(
                explanation_method=method,
                quality_score=quality_score,
                consistency_score=consistency_score,
                human_agreement_score=human_agreement,
                computational_efficiency=efficiency_score
            )
            results.append(result)
        
        logger.info(f"Evaluated explainability for {len(explanation_methods)} methods")
        return results
    
    def _calculate_quality_score(self, explanations: List[Dict[str, Any]]) -> float:
        """Calculate explanation quality score based on completeness and informativeness."""
        quality_scores = []
        
        for exp in explanations:
            score = 0.0
            
            # Check for presence of key components
            if 'attention_scores' in exp:
                score += 0.25
            if 'feature_importance' in exp:
                score += 0.25
            if 'rationale' in exp and len(exp['rationale']) > 10:
                score += 0.25
            if 'confidence' in exp:
                score += 0.25
            
            quality_scores.append(score)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_consistency_score(self, explanations: List[Dict[str, Any]]) -> float:
        """Calculate consistency of explanations for similar inputs."""
        if len(explanations) < 2:
            return 1.0
        
        # For demonstration, calculate consistency based on attention score variance
        consistency_scores = []
        
        for i in range(len(explanations) - 1):
            exp1 = explanations[i]
            exp2 = explanations[i + 1]
            
            # Compare attention scores if available
            if 'attention_scores' in exp1 and 'attention_scores' in exp2:
                scores1 = exp1['attention_scores']
                scores2 = exp2['attention_scores']
                
                if len(scores1) == len(scores2):
                    correlation = np.corrcoef(scores1, scores2)[0, 1]
                    consistency_scores.append(max(0, correlation))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _calculate_efficiency_score(self, explanations: List[Dict[str, Any]]) -> float:
        """Calculate computational efficiency score."""
        efficiency_scores = []
        
        for exp in explanations:
            # Use generation time if available, otherwise estimate based on complexity
            if 'generation_time' in exp:
                time_score = max(0, 1 - (exp['generation_time'] / 10.0))  # Normalize to 10 seconds
            else:
                # Estimate based on explanation complexity
                complexity = len(str(exp))
                time_score = max(0, 1 - (complexity / 10000))  # Normalize to 10k characters
            
            efficiency_scores.append(time_score)
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.5

class ContextualAdaptationEvaluator(BaseEvaluator):
    """Evaluates the effectiveness of contextual adaptation mechanisms."""
    
    def __init__(self):
        logger.info("Contextual adaptation evaluator initialized")
    
    def evaluate(
        self,
        base_predictions: List[float],
        adapted_predictions: List[float],
        ground_truth: List[int],
        adaptation_contexts: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        Evaluate contextual adaptation effectiveness.
        
        Args:
            base_predictions: Original model predictions
            adapted_predictions: Contextually adapted predictions
            ground_truth: True labels
            adaptation_contexts: Context information for each prediction
            
        Returns:
            List of evaluation results
        """
        results = []
        timestamp = datetime.now()
        
        # Overall adaptation effectiveness
        base_accuracy = self._calculate_accuracy(base_predictions, ground_truth)
        adapted_accuracy = self._calculate_accuracy(adapted_predictions, ground_truth)
        
        improvement = adapted_accuracy - base_accuracy
        
        result = EvaluationResult(
            metric_name='adaptation_improvement',
            value=improvement,
            metadata={
                'base_accuracy': base_accuracy,
                'adapted_accuracy': adapted_accuracy,
                'n_samples': len(base_predictions)
            },
            timestamp=timestamp
        )
        results.append(result)
        
        # Evaluate by adaptation type
        adaptation_types = ['regional', 'temporal', 'policy']
        
        for adaptation_type in adaptation_types:
            type_improvement = self._evaluate_adaptation_type(
                base_predictions,
                adapted_predictions,
                ground_truth,
                adaptation_contexts,
                adaptation_type
            )
            
            result = EvaluationResult(
                metric_name=f'{adaptation_type}_adaptation_effectiveness',
                value=type_improvement,
                metadata={'adaptation_type': adaptation_type},
                timestamp=timestamp
            )
            results.append(result)
        
        # Evaluate adaptation consistency
        consistency_score = self._evaluate_adaptation_consistency(
            base_predictions,
            adapted_predictions,
            adaptation_contexts
        )
        
        result = EvaluationResult(
            metric_name='adaptation_consistency',
            value=consistency_score,
            metadata={},
            timestamp=timestamp
        )
        results.append(result)
        
        logger.info(f"Evaluated contextual adaptation with {len(results)} metrics")
        return results
    
    def _calculate_accuracy(self, predictions: List[float], ground_truth: List[int]) -> float:
        """Calculate accuracy from probability predictions."""
        binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
        return accuracy_score(ground_truth, binary_predictions)
    
    def _evaluate_adaptation_type(
        self,
        base_predictions: List[float],
        adapted_predictions: List[float],
        ground_truth: List[int],
        contexts: List[Dict[str, Any]],
        adaptation_type: str
    ) -> float:
        """Evaluate effectiveness of a specific adaptation type."""
        # Filter samples where this adaptation type was applied
        relevant_indices = []
        for i, context in enumerate(contexts):
            if context.get(f'{adaptation_type}_applied', False):
                relevant_indices.append(i)
        
        if not relevant_indices:
            return 0.0
        
        # Calculate improvement for relevant samples
        relevant_base = [base_predictions[i] for i in relevant_indices]
        relevant_adapted = [adapted_predictions[i] for i in relevant_indices]
        relevant_truth = [ground_truth[i] for i in relevant_indices]
        
        base_acc = self._calculate_accuracy(relevant_base, relevant_truth)
        adapted_acc = self._calculate_accuracy(relevant_adapted, relevant_truth)
        
        return adapted_acc - base_acc
    
    def _evaluate_adaptation_consistency(
        self,
        base_predictions: List[float],
        adapted_predictions: List[float],
        contexts: List[Dict[str, Any]]
    ) -> float:
        """Evaluate consistency of adaptations across similar contexts."""
        # Group by similar contexts
        context_groups = {}
        
        for i, context in enumerate(contexts):
            # Create a key based on context features
            key = f"{context.get('region', 'unknown')}_{context.get('platform', 'unknown')}"
            
            if key not in context_groups:
                context_groups[key] = []
            
            adaptation_magnitude = abs(adapted_predictions[i] - base_predictions[i])
            context_groups[key].append(adaptation_magnitude)
        
        # Calculate consistency within each group
        consistency_scores = []
        for group_adaptations in context_groups.values():
            if len(group_adaptations) > 1:
                # Low variance indicates high consistency
                variance = np.var(group_adaptations)
                consistency = max(0, 1 - variance)  # Convert variance to consistency score
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5

class HumanAICollaborationEvaluator(BaseEvaluator):
    """Evaluates the effectiveness of human-AI collaboration in the moderation interface."""
    
    def __init__(self):
        logger.info("Human-AI collaboration evaluator initialized")
    
    def evaluate(
        self,
        ai_predictions: List[Dict[str, Any]],
        human_decisions: List[Dict[str, Any]],
        final_outcomes: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        Evaluate human-AI collaboration effectiveness.
        
        Args:
            ai_predictions: AI predictions with confidence scores
            human_decisions: Human moderator decisions and feedback
            final_outcomes: Final moderation outcomes
            
        Returns:
            List of evaluation results
        """
        results = []
        timestamp = datetime.now()
        
        # Calculate agreement rate
        agreement_rate = self._calculate_agreement_rate(ai_predictions, human_decisions)
        
        result = EvaluationResult(
            metric_name='human_ai_agreement_rate',
            value=agreement_rate,
            metadata={'n_decisions': len(ai_predictions)},
            timestamp=timestamp
        )
        results.append(result)
        
        # Calculate override rate and reasons
        override_rate, override_reasons = self._calculate_override_metrics(ai_predictions, human_decisions)
        
        result = EvaluationResult(
            metric_name='ai_override_rate',
            value=override_rate,
            metadata={'override_reasons': override_reasons},
            timestamp=timestamp
        )
        results.append(result)
        
        # Calculate learning effectiveness
        learning_score = self._calculate_learning_effectiveness(ai_predictions, human_decisions, final_outcomes)
        
        result = EvaluationResult(
            metric_name='learning_effectiveness',
            value=learning_score,
            metadata={},
            timestamp=timestamp
        )
        results.append(result)
        
        # Calculate decision confidence correlation
        confidence_correlation = self._calculate_confidence_correlation(ai_predictions, human_decisions)
        
        result = EvaluationResult(
            metric_name='confidence_correlation',
            value=confidence_correlation,
            metadata={},
            timestamp=timestamp
        )
        results.append(result)
        
        logger.info(f"Evaluated human-AI collaboration with {len(results)} metrics")
        return results
    
    def _calculate_agreement_rate(
        self,
        ai_predictions: List[Dict[str, Any]],
        human_decisions: List[Dict[str, Any]]
    ) -> float:
        """Calculate the rate of agreement between AI and human decisions."""
        agreements = 0
        total = min(len(ai_predictions), len(human_decisions))
        
        for i in range(total):
            ai_decision = ai_predictions[i].get('prediction', 0)
            human_decision = human_decisions[i].get('decision', 0)
            
            if ai_decision == human_decision:
                agreements += 1
        
        return agreements / total if total > 0 else 0.0
    
    def _calculate_override_metrics(
        self,
        ai_predictions: List[Dict[str, Any]],
        human_decisions: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, int]]:
        """Calculate override rate and categorize override reasons."""
        overrides = 0
        override_reasons = {}
        total = min(len(ai_predictions), len(human_decisions))
        
        for i in range(total):
            ai_decision = ai_predictions[i].get('prediction', 0)
            human_decision = human_decisions[i].get('decision', 0)
            
            if ai_decision != human_decision:
                overrides += 1
                
                # Categorize override reason
                reason = human_decisions[i].get('override_reason', 'unspecified')
                override_reasons[reason] = override_reasons.get(reason, 0) + 1
        
        override_rate = overrides / total if total > 0 else 0.0
        return override_rate, override_reasons
    
    def _calculate_learning_effectiveness(
        self,
        ai_predictions: List[Dict[str, Any]],
        human_decisions: List[Dict[str, Any]],
        final_outcomes: List[Dict[str, Any]]
    ) -> float:
        """Calculate how effectively the system learns from human feedback."""
        # This would typically involve comparing performance before and after feedback
        # For demonstration, we'll simulate learning effectiveness
        
        if len(final_outcomes) < 10:
            return 0.5  # Not enough data
        
        # Simulate improvement over time
        early_accuracy = 0.7  # Simulated early accuracy
        late_accuracy = 0.8   # Simulated later accuracy after learning
        
        improvement = late_accuracy - early_accuracy
        return min(1.0, max(0.0, improvement * 2))  # Scale to [0, 1]
    
    def _calculate_confidence_correlation(
        self,
        ai_predictions: List[Dict[str, Any]],
        human_decisions: List[Dict[str, Any]]
    ) -> float:
        """Calculate correlation between AI confidence and human agreement."""
        ai_confidences = []
        human_agreements = []
        
        for i in range(min(len(ai_predictions), len(human_decisions))):
            ai_conf = ai_predictions[i].get('confidence', 0.5)
            ai_decision = ai_predictions[i].get('prediction', 0)
            human_decision = human_decisions[i].get('decision', 0)
            
            ai_confidences.append(ai_conf)
            human_agreements.append(1 if ai_decision == human_decision else 0)
        
        if len(ai_confidences) > 1:
            correlation = np.corrcoef(ai_confidences, human_agreements)[0, 1]
            return max(-1, min(1, correlation))  # Ensure valid correlation range
        else:
            return 0.0

class ComprehensiveEvaluationFramework:
    """Main framework that coordinates all evaluation components."""
    
    def __init__(self):
        self.performance_evaluator = PerformanceEvaluator()
        self.fairness_evaluator = FairnessEvaluator()
        self.explainability_evaluator = ExplainabilityEvaluator()
        self.contextual_evaluator = ContextualAdaptationEvaluator()
        self.collaboration_evaluator = HumanAICollaborationEvaluator()
        
        self.evaluation_history = []
        logger.info("Comprehensive evaluation framework initialized")
    
    def run_full_evaluation(
        self,
        test_data: Dict[str, Any]
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Run comprehensive evaluation across all dimensions.
        
        Args:
            test_data: Dictionary containing all necessary test data
            
        Returns:
            Dictionary of evaluation results by category
        """
        results = {}
        
        # Performance evaluation
        if all(key in test_data for key in ['y_true', 'y_pred']):
            logger.info("Running performance evaluation...")
            results['performance'] = self.performance_evaluator.evaluate(
                y_true=test_data['y_true'],
                y_pred=test_data['y_pred'],
                y_prob=test_data.get('y_prob')
            )
        
        # Fairness evaluation
        if all(key in test_data for key in ['y_true', 'y_pred', 'y_prob', 'demographic_groups']):
            logger.info("Running fairness evaluation...")
            results['fairness'] = self.fairness_evaluator.evaluate(
                y_true=test_data['y_true'],
                y_pred=test_data['y_pred'],
                y_prob=test_data['y_prob'],
                demographic_groups=test_data['demographic_groups']
            )
        
        # Explainability evaluation
        if 'explanations' in test_data:
            logger.info("Running explainability evaluation...")
            results['explainability'] = self.explainability_evaluator.evaluate(
                explanations=test_data['explanations'],
                ground_truth_explanations=test_data.get('ground_truth_explanations'),
                human_ratings=test_data.get('human_ratings')
            )
        
        # Contextual adaptation evaluation
        if all(key in test_data for key in ['base_predictions', 'adapted_predictions', 'ground_truth']):
            logger.info("Running contextual adaptation evaluation...")
            results['contextual_adaptation'] = self.contextual_evaluator.evaluate(
                base_predictions=test_data['base_predictions'],
                adapted_predictions=test_data['adapted_predictions'],
                ground_truth=test_data['ground_truth'],
                adaptation_contexts=test_data.get('adaptation_contexts', [])
            )
        
        # Human-AI collaboration evaluation
        if all(key in test_data for key in ['ai_predictions', 'human_decisions', 'final_outcomes']):
            logger.info("Running human-AI collaboration evaluation...")
            results['collaboration'] = self.collaboration_evaluator.evaluate(
                ai_predictions=test_data['ai_predictions'],
                human_decisions=test_data['human_decisions'],
                final_outcomes=test_data['final_outcomes']
            )
        
        # Store evaluation in history
        self.evaluation_history.append({
            'timestamp': datetime.now(),
            'results': results
        })
        
        logger.info(f"Completed comprehensive evaluation with {len(results)} categories")
        return results
    
    def generate_evaluation_report(
        self,
        results: Dict[str, List[EvaluationResult]],
        output_path: str = "evaluation_report.json"
    ):
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': {},
            'recommendations': []
        }
        
        # Process each category
        for category, category_results in results.items():
            if isinstance(category_results, list) and category_results:
                # Handle EvaluationResult objects
                if hasattr(category_results[0], 'metric_name') and hasattr(category_results[0], 'metadata'):
                    report['detailed_results'][category] = [
                        {
                            'metric': result.metric_name,
                            'value': result.value,
                            'metadata': result.metadata,
                            'confidence_interval': result.confidence_interval
                        }
                        for result in category_results
                    ]
                    
                    # Calculate summary statistics
                    values = [result.value for result in category_results]
                    report['summary'][category] = {
                        'mean_score': np.mean(values),
                        'std_score': np.std(values),
                        'min_score': np.min(values),
                        'max_score': np.max(values)
                    }
                
                # Handle BiasEvaluationResult objects
                elif hasattr(category_results[0], 'demographic_group'):
                    report['detailed_results'][category] = [
                        {
                            'demographic_group': result.demographic_group,
                            'metric': result.metric_name,
                            'value': result.value,
                            'baseline_value': result.baseline_value,
                            'bias_score': result.bias_score,
                            'significance': result.significance
                        }
                        for result in category_results
                    ]
                    
                    # Calculate bias summary
                    bias_scores = [result.bias_score for result in category_results]
                    high_bias_count = sum(1 for result in category_results if result.significance == 'high')
                    
                    report['summary'][category] = {
                        'mean_bias_score': np.mean(bias_scores),
                        'max_bias_score': np.max(bias_scores),
                        'high_bias_count': high_bias_count,
                        'total_groups': len(set(result.demographic_group for result in category_results))
                    }
                
                # Handle ExplainabilityEvaluationResult objects
                elif hasattr(category_results[0], 'explanation_method'):
                    report['detailed_results'][category] = [
                        {
                            'method': result.explanation_method,
                            'quality_score': result.quality_score,
                            'consistency_score': result.consistency_score,
                            'human_agreement_score': result.human_agreement_score,
                            'computational_efficiency': result.computational_efficiency
                        }
                        for result in category_results
                    ]
                    
                    # Calculate explainability summary
                    quality_scores = [result.quality_score for result in category_results]
                    report['summary'][category] = {
                        'mean_quality': np.mean(quality_scores),
                        'mean_consistency': np.mean([result.consistency_score for result in category_results]),
                        'mean_human_agreement': np.mean([result.human_agreement_score for result in category_results]),
                        'mean_efficiency': np.mean([result.computational_efficiency for result in category_results])
                    }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['summary'])
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Performance recommendations
        if 'performance' in summary:
            mean_score = summary['performance'].get('mean_score', 0)
            if mean_score < 0.8:
                recommendations.append("Consider improving model architecture or training data quality to enhance overall performance.")
        
        # Fairness recommendations
        if 'fairness' in summary:
            high_bias_count = summary['fairness'].get('high_bias_count', 0)
            if high_bias_count > 0:
                recommendations.append(f"Address high bias detected in {high_bias_count} demographic groups through data augmentation or algorithmic debiasing.")
        
        # Explainability recommendations
        if 'explainability' in summary:
            mean_quality = summary['explainability'].get('mean_quality', 0)
            if mean_quality < 0.7:
                recommendations.append("Improve explanation quality by enhancing feature attribution methods and natural language generation.")
        
        # Contextual adaptation recommendations
        if 'contextual_adaptation' in summary:
            mean_score = summary['contextual_adaptation'].get('mean_score', 0)
            if mean_score < 0.1:
                recommendations.append("Enhance contextual adaptation mechanisms to better respond to regional, temporal, and policy contexts.")
        
        # Collaboration recommendations
        if 'collaboration' in summary:
            mean_score = summary['collaboration'].get('mean_score', 0)
            if mean_score < 0.7:
                recommendations.append("Improve human-AI collaboration by enhancing interface design and feedback mechanisms.")
        
        return recommendations

def demonstrate_evaluation_framework():
    """Demonstrate the comprehensive evaluation framework."""
    logger.info("Starting Evaluation Framework Demonstration")
    
    # Initialize framework
    framework = ComprehensiveEvaluationFramework()
    
    # Generate sample test data
    np.random.seed(42)
    n_samples = 1000
    
    test_data = {
        # Basic performance data
        'y_true': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]).tolist(),
        'y_pred': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]).tolist(),
        'y_prob': np.random.beta(2, 2, n_samples).tolist(),
        
        # Demographic groups
        'demographic_groups': {
            'gender_male': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]).tolist(),
            'age_young': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]).tolist(),
            'region_na': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]).tolist()
        },
        
        # Explanations
        'explanations': [
            {
                'method': 'attention',
                'attention_scores': np.random.uniform(0, 1, 10).tolist(),
                'confidence': np.random.uniform(0.5, 1.0),
                'generation_time': np.random.uniform(0.1, 2.0)
            }
            for _ in range(100)
        ],
        
        # Contextual adaptation data
        'base_predictions': np.random.beta(2, 2, n_samples).tolist(),
        'adapted_predictions': (np.random.beta(2, 2, n_samples) * 0.9 + 0.05).tolist(),
        'ground_truth': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]).tolist(),
        'adaptation_contexts': [
            {
                'region': np.random.choice(['north_america', 'europe']),
                'platform': np.random.choice(['twitter', 'facebook']),
                'regional_applied': True,
                'temporal_applied': np.random.choice([True, False]),
                'policy_applied': True
            }
            for _ in range(n_samples)
        ],
        
        # Human-AI collaboration data
        'ai_predictions': [
            {
                'prediction': np.random.choice([0, 1]),
                'confidence': np.random.uniform(0.5, 1.0)
            }
            for _ in range(200)
        ],
        'human_decisions': [
            {
                'decision': np.random.choice([0, 1]),
                'override_reason': np.random.choice(['context', 'bias', 'error', 'unspecified'])
            }
            for _ in range(200)
        ],
        'final_outcomes': [
            {
                'final_decision': np.random.choice([0, 1]),
                'confidence': np.random.uniform(0.7, 1.0)
            }
            for _ in range(200)
        ]
    }
    
    # Run comprehensive evaluation
    logger.info("Running comprehensive evaluation...")
    results = framework.run_full_evaluation(test_data)
    
    # Generate and save report
    report = framework.generate_evaluation_report(results)
    
    # Print summary
    print("\n=== Evaluation Framework Demonstration Results ===")
    
    for category, summary in report['summary'].items():
        print(f"\n{category.upper()} SUMMARY:")
        for metric, value in summary.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    print(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    logger.info("Evaluation framework demonstration completed!")
    logger.info("Detailed report saved to evaluation_report.json")

if __name__ == "__main__":
    demonstrate_evaluation_framework()

