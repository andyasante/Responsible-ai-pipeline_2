"""
Responsible-AI Hate Speech Detection Pipeline: Explainability and Interpretability Suite

This module implements comprehensive explainability components including:
- Attention heatmaps
- Integrated gradients
- SHAP summaries
- Natural language rationales
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Container for explanation results."""
    explanation_type: str
    data: Dict[str, Any]
    visualization: Optional[str] = None  # Base64 encoded image
    text_summary: Optional[str] = None

class BaseExplainer(ABC):
    """Abstract base class for all explainability methods."""
    
    @abstractmethod
    def explain(self, *args, **kwargs) -> ExplanationResult:
        """Generate explanation for the given input."""
        pass

class AttentionHeatmapExplainer(BaseExplainer):
    """
    Generates attention heatmaps showing which tokens the model focused on.
    """
    
    def __init__(self, colormap: str = "Reds"):
        self.colormap = colormap
        logger.info("Attention Heatmap Explainer initialized")
    
    def explain(
        self,
        tokens: List[str],
        attention_weights: np.ndarray,
        layer_idx: int = -1,
        head_idx: int = 0
    ) -> ExplanationResult:
        """
        Generate attention heatmap explanation.
        
        Args:
            tokens: List of input tokens
            attention_weights: Attention weights from transformer model
            layer_idx: Which layer to visualize (default: last layer)
            head_idx: Which attention head to visualize
            
        Returns:
            ExplanationResult containing heatmap data and visualization
        """
        # Extract attention for specified layer and head
        if len(attention_weights.shape) == 4:  # [batch, heads, seq, seq]
            attention = attention_weights[0, head_idx, :, :]
        elif len(attention_weights.shape) == 3:  # [heads, seq, seq]
            attention = attention_weights[head_idx, :, :]
        else:  # [seq, seq]
            attention = attention_weights
        
        # Focus on attention from [CLS] token to all other tokens
        cls_attention = attention[0, :len(tokens)]
        
        # Normalize attention scores
        cls_attention = cls_attention / np.sum(cls_attention)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # Create heatmap
        attention_matrix = cls_attention.reshape(1, -1)
        im = ax.imshow(attention_matrix, cmap=self.colormap, aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['Attention'])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        
        # Add title
        ax.set_title(f'Attention Heatmap (Layer {layer_idx}, Head {head_idx})')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Generate text summary
        top_indices = np.argsort(cls_attention)[-3:][::-1]
        top_tokens = [tokens[i] for i in top_indices]
        top_scores = [cls_attention[i] for i in top_indices]
        
        text_summary = f"Top attended tokens: {', '.join([f'{token} ({score:.3f})' for token, score in zip(top_tokens, top_scores)])}"
        
        return ExplanationResult(
            explanation_type="attention_heatmap",
            data={
                "tokens": tokens,
                "attention_scores": cls_attention.tolist(),
                "top_tokens": top_tokens,
                "top_scores": top_scores
            },
            visualization=image_base64,
            text_summary=text_summary
        )

class IntegratedGradientsExplainer(BaseExplainer):
    """
    Implements Integrated Gradients for feature attribution.
    """
    
    def __init__(self, steps: int = 50):
        self.steps = steps
        logger.info(f"Integrated Gradients Explainer initialized with {steps} steps")
    
    def explain(
        self,
        tokens: List[str],
        model_function: callable,
        baseline_function: callable,
        target_class: int
    ) -> ExplanationResult:
        """
        Generate integrated gradients explanation.
        
        Args:
            tokens: List of input tokens
            model_function: Function that returns model predictions
            baseline_function: Function that returns baseline (zero) input
            target_class: Target class for attribution
            
        Returns:
            ExplanationResult containing attribution scores
        """
        # For demonstration, simulate integrated gradients computation
        # In practice, this would involve actual gradient computation
        
        # Simulate attribution scores
        attributions = np.random.uniform(-1, 1, len(tokens))
        
        # Normalize attributions
        attributions = attributions / np.max(np.abs(attributions))
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        colors = ['red' if attr < 0 else 'green' for attr in attributions]
        bars = ax.bar(range(len(tokens)), attributions, color=colors, alpha=0.7)
        
        # Set labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Attribution Score')
        ax.set_title('Integrated Gradients Attribution')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, attr in zip(bars, attributions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                   f'{attr:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Generate text summary
        positive_indices = np.where(attributions > 0)[0]
        negative_indices = np.where(attributions < 0)[0]
        
        if len(positive_indices) > 0:
            top_positive_idx = positive_indices[np.argmax(attributions[positive_indices])]
            top_positive = (tokens[top_positive_idx], attributions[top_positive_idx])
        else:
            top_positive = None
            
        if len(negative_indices) > 0:
            top_negative_idx = negative_indices[np.argmin(attributions[negative_indices])]
            top_negative = (tokens[top_negative_idx], attributions[top_negative_idx])
        else:
            top_negative = None
        
        summary_parts = []
        if top_positive:
            summary_parts.append(f"Most positive: '{top_positive[0]}' ({top_positive[1]:.3f})")
        if top_negative:
            summary_parts.append(f"Most negative: '{top_negative[0]}' ({top_negative[1]:.3f})")
        
        text_summary = "; ".join(summary_parts)
        
        return ExplanationResult(
            explanation_type="integrated_gradients",
            data={
                "tokens": tokens,
                "attributions": attributions.tolist(),
                "positive_attributions": [(tokens[i], attributions[i]) for i in positive_indices],
                "negative_attributions": [(tokens[i], attributions[i]) for i in negative_indices]
            },
            visualization=image_base64,
            text_summary=text_summary
        )

class SHAPExplainer(BaseExplainer):
    """
    Implements SHAP (SHapley Additive exPlanations) for model explanations.
    """
    
    def __init__(self):
        logger.info("SHAP Explainer initialized")
    
    def explain(
        self,
        tokens: List[str],
        shap_values: np.ndarray,
        base_value: float,
        prediction: float
    ) -> ExplanationResult:
        """
        Generate SHAP explanation.
        
        Args:
            tokens: List of input tokens
            shap_values: SHAP values for each token
            base_value: Base value (expected model output)
            prediction: Actual model prediction
            
        Returns:
            ExplanationResult containing SHAP analysis
        """
        # Ensure SHAP values sum to prediction - base_value
        shap_sum = np.sum(shap_values)
        expected_sum = prediction - base_value
        
        if abs(shap_sum - expected_sum) > 1e-6:
            # Normalize SHAP values to satisfy additivity
            shap_values = shap_values * (expected_sum / shap_sum)
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate cumulative values for waterfall
        cumulative = [base_value]
        for shap_val in shap_values:
            cumulative.append(cumulative[-1] + shap_val)
        
        # Create waterfall bars
        x_pos = range(len(tokens) + 2)
        
        # Base value bar
        ax.bar(0, base_value, color='gray', alpha=0.7, label='Base Value')
        
        # SHAP value bars
        for i, (token, shap_val) in enumerate(zip(tokens, shap_values)):
            color = 'green' if shap_val > 0 else 'red'
            ax.bar(i + 1, shap_val, bottom=cumulative[i], color=color, alpha=0.7)
            
            # Add value labels
            ax.text(i + 1, cumulative[i] + shap_val/2, f'{shap_val:.3f}', 
                   ha='center', va='center', fontsize=8, rotation=90)
        
        # Final prediction bar
        ax.bar(len(tokens) + 1, prediction, color='blue', alpha=0.7, label='Prediction')
        
        # Set labels
        labels = ['Base'] + tokens + ['Prediction']
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Model Output')
        ax.set_title('SHAP Waterfall Plot')
        ax.legend()
        
        # Add horizontal lines for reference
        ax.axhline(y=base_value, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=prediction, color='blue', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Generate summary
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        top_features = [(tokens[i], shap_values[i]) for i in sorted_indices[:3]]
        
        text_summary = f"Base value: {base_value:.3f}, Prediction: {prediction:.3f}. "
        text_summary += f"Top contributing features: {', '.join([f'{token} ({val:.3f})' for token, val in top_features])}"
        
        return ExplanationResult(
            explanation_type="shap",
            data={
                "tokens": tokens,
                "shap_values": shap_values.tolist(),
                "base_value": base_value,
                "prediction": prediction,
                "top_features": top_features
            },
            visualization=image_base64,
            text_summary=text_summary
        )

class NaturalLanguageRationaleGenerator:
    """
    Generates human-readable explanations for model predictions.
    """
    
    def __init__(self):
        self.templates = {
            "hate_speech": [
                "This text was classified as hate speech because it contains {reasons}.",
                "The model identified hate speech due to {reasons}.",
                "This content was flagged as hate speech based on {reasons}."
            ],
            "not_hate_speech": [
                "This text was classified as non-hate speech because {reasons}.",
                "The model found no hate speech indicators: {reasons}.",
                "This content was deemed acceptable due to {reasons}."
            ]
        }
        logger.info("Natural Language Rationale Generator initialized")
    
    def generate_rationale(
        self,
        text: str,
        prediction: int,
        confidence: float,
        attention_result: ExplanationResult,
        shap_result: ExplanationResult,
        integrated_gradients_result: ExplanationResult
    ) -> str:
        """
        Generate a natural language rationale combining multiple explanation methods.
        
        Args:
            text: Original input text
            prediction: Model prediction (0: not hate speech, 1: hate speech)
            confidence: Prediction confidence
            attention_result: Attention explanation result
            shap_result: SHAP explanation result
            integrated_gradients_result: Integrated gradients result
            
        Returns:
            Natural language rationale string
        """
        # Extract key information from explanations
        top_attention_tokens = attention_result.data["top_tokens"][:2]
        top_shap_features = [feat[0] for feat in shap_result.data["top_features"][:2]]
        
        # Determine prediction type
        pred_type = "hate_speech" if prediction == 1 else "not_hate_speech"
        
        # Build reasons based on explanations
        reasons = []
        
        if prediction == 1:  # Hate speech
            if any(token.lower() in ["hate", "stupid", "idiot", "kill", "die"] for token in top_attention_tokens):
                reasons.append("explicit harmful language")
            if any(token.lower() in ["they", "those", "them", "people"] for token in top_attention_tokens):
                reasons.append("targeting language toward groups")
            if confidence > 0.7:
                reasons.append("strong linguistic patterns associated with hate speech")
            else:
                reasons.append("subtle indicators of potentially harmful content")
        else:  # Not hate speech
            if any(token.lower() in ["love", "great", "good", "nice", "happy"] for token in top_attention_tokens):
                reasons.append("positive sentiment and language")
            if any(token.lower() in ["weather", "food", "book", "family", "friends"] for token in top_attention_tokens):
                reasons.append("neutral, everyday topics")
            reasons.append("absence of harmful or targeting language")
        
        # Select template and format
        import random
        template = random.choice(self.templates[pred_type])
        
        if not reasons:
            reasons = ["the overall linguistic patterns in the text"]
        
        reason_text = ", ".join(reasons[:-1])
        if len(reasons) > 1:
            reason_text += f" and {reasons[-1]}"
        else:
            reason_text = reasons[0]
        
        rationale = template.format(reasons=reason_text)
        
        # Add confidence information
        confidence_text = f" The model's confidence in this classification is {confidence:.1%}."
        
        # Add specific token mentions
        if top_attention_tokens:
            token_text = f" Key words that influenced this decision include: {', '.join(top_attention_tokens)}."
        else:
            token_text = ""
        
        full_rationale = rationale + confidence_text + token_text
        
        logger.info("Generated natural language rationale")
        return full_rationale

class ExplainabilitySuite:
    """
    Comprehensive explainability suite combining all explanation methods.
    """
    
    def __init__(self):
        self.attention_explainer = AttentionHeatmapExplainer()
        self.integrated_gradients_explainer = IntegratedGradientsExplainer()
        self.shap_explainer = SHAPExplainer()
        self.rationale_generator = NaturalLanguageRationaleGenerator()
        logger.info("Explainability Suite initialized with all components")
    
    def explain_prediction(
        self,
        text: str,
        tokens: List[str],
        prediction: int,
        confidence: float,
        attention_weights: np.ndarray,
        model_function: Optional[callable] = None
    ) -> Dict[str, ExplanationResult]:
        """
        Generate comprehensive explanations for a model prediction.
        
        Args:
            text: Original input text
            tokens: Tokenized input
            prediction: Model prediction
            confidence: Prediction confidence
            attention_weights: Attention weights from model
            model_function: Model function for gradient computation
            
        Returns:
            Dictionary of explanation results
        """
        explanations = {}
        
        # Generate attention heatmap
        logger.info("Generating attention heatmap...")
        explanations["attention"] = self.attention_explainer.explain(
            tokens=tokens,
            attention_weights=attention_weights
        )
        
        # Generate integrated gradients (simulated)
        logger.info("Generating integrated gradients...")
        explanations["integrated_gradients"] = self.integrated_gradients_explainer.explain(
            tokens=tokens,
            model_function=model_function or (lambda x: np.random.random()),
            baseline_function=lambda: np.zeros_like(tokens),
            target_class=prediction
        )
        
        # Generate SHAP explanation (simulated)
        logger.info("Generating SHAP explanation...")
        # Simulate SHAP values
        shap_values = np.random.uniform(-0.5, 0.5, len(tokens))
        base_value = 0.0
        
        explanations["shap"] = self.shap_explainer.explain(
            tokens=tokens,
            shap_values=shap_values,
            base_value=base_value,
            prediction=confidence
        )
        
        # Generate natural language rationale
        logger.info("Generating natural language rationale...")
        rationale = self.rationale_generator.generate_rationale(
            text=text,
            prediction=prediction,
            confidence=confidence,
            attention_result=explanations["attention"],
            shap_result=explanations["shap"],
            integrated_gradients_result=explanations["integrated_gradients"]
        )
        
        explanations["rationale"] = ExplanationResult(
            explanation_type="natural_language_rationale",
            data={"rationale": rationale},
            text_summary=rationale
        )
        
        logger.info("Comprehensive explanation generation completed")
        return explanations
    
    def save_explanations(self, explanations: Dict[str, ExplanationResult], filepath: str):
        """Save explanations to file."""
        explanation_data = {}
        
        for key, result in explanations.items():
            explanation_data[key] = {
                "type": result.explanation_type,
                "data": result.data,
                "text_summary": result.text_summary,
                "has_visualization": result.visualization is not None
            }
        
        with open(filepath, 'w') as f:
            json.dump(explanation_data, f, indent=2)
        
        logger.info(f"Explanations saved to {filepath}")

def demonstrate_explainability_suite():
    """Demonstrate the explainability suite with sample data."""
    logger.info("Starting Explainability Suite Demonstration")
    
    # Initialize suite
    suite = ExplainabilitySuite()
    
    # Sample data
    sample_text = "This group of people is ruining our country!"
    sample_tokens = sample_text.split()
    sample_prediction = 1  # Hate speech
    sample_confidence = 0.85
    
    # Simulate attention weights (normally from transformer model)
    seq_len = len(sample_tokens)
    sample_attention = np.random.uniform(0, 1, (1, 8, seq_len, seq_len))  # [batch, heads, seq, seq]
    
    # Generate comprehensive explanations
    explanations = suite.explain_prediction(
        text=sample_text,
        tokens=sample_tokens,
        prediction=sample_prediction,
        confidence=sample_confidence,
        attention_weights=sample_attention
    )
    
    # Display results
    print(f"\n=== Explainability Results for: '{sample_text}' ===")
    print(f"Prediction: {'Hate Speech' if sample_prediction == 1 else 'Not Hate Speech'}")
    print(f"Confidence: {sample_confidence:.2%}")
    print()
    
    for explanation_type, result in explanations.items():
        print(f"--- {explanation_type.upper()} ---")
        print(f"Summary: {result.text_summary}")
        if result.visualization:
            print(f"Visualization: Available (base64 encoded)")
        print()
    
    # Save explanations
    suite.save_explanations(explanations, "sample_explanations.json")
    
    logger.info("Explainability suite demonstration completed!")

if __name__ == "__main__":
    demonstrate_explainability_suite()

