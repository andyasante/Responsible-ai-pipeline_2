"""
Simplified Demo Version of the Responsible-AI Hate Speech Detection Pipeline
Machine Learning Model Architecture

This is a demonstration version that shows the core concepts without requiring
heavy dependencies like PyTorch. It illustrates the architecture and design patterns.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the hate speech detection model."""
    model_name: str = "xlm-roberta-base"
    num_labels: int = 2
    num_languages: int = 10
    num_regions: int = 20
    num_events: int = 1000
    language_adapter_size: int = 64
    regional_adapter_size: int = 32
    event_embedding_dim: int = 64
    hidden_size: int = 768
    max_length: int = 512
    dropout: float = 0.1

class LanguageAdapterDemo:
    """
    Demonstration of Language Adapter concept.
    In the full implementation, this would be a PyTorch nn.Module.
    """
    
    def __init__(self, hidden_size: int, adapter_size: int = 64):
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        logger.info(f"Language Adapter initialized: {hidden_size} -> {adapter_size} -> {hidden_size}")
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Simulate forward pass through language adapter.
        In practice, this would involve learned linear transformations.
        """
        # Simulate down-projection, activation, and up-projection
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # For demo purposes, just add small random noise to simulate adaptation
        adaptation = np.random.normal(0, 0.01, hidden_states.shape)
        adapted_states = hidden_states + adaptation
        
        logger.debug(f"Language adaptation applied to shape: {hidden_states.shape}")
        return adapted_states

class RegionalAdapterDemo:
    """
    Demonstration of Regional Adapter concept.
    32-dimensional adapters for geographical/cultural adaptation.
    """
    
    def __init__(self, hidden_size: int, adapter_size: int = 32):
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        logger.info(f"Regional Adapter initialized: {hidden_size} -> {adapter_size} -> {hidden_size}")
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Simulate forward pass through regional adapter.
        """
        # Simulate regional adaptation
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # For demo purposes, apply region-specific transformation
        regional_factor = np.random.uniform(0.95, 1.05, hidden_states.shape)
        adapted_states = hidden_states * regional_factor
        
        logger.debug(f"Regional adaptation applied to shape: {hidden_states.shape}")
        return adapted_states

class EventEmbeddingDemo:
    """
    Demonstration of Event Embedding concept.
    64-dimensional embeddings for temporal events.
    """
    
    def __init__(self, num_events: int, embedding_dim: int = 64):
        self.num_events = num_events
        self.embedding_dim = embedding_dim
        
        # Simulate pre-computed event embeddings
        self.embeddings = np.random.normal(0, 0.1, (num_events, embedding_dim))
        logger.info(f"Event Embeddings initialized: {num_events} events, {embedding_dim} dimensions")
    
    def forward(self, event_ids: List[int]) -> np.ndarray:
        """
        Retrieve event embeddings for given event IDs.
        """
        embeddings = np.array([self.embeddings[event_id] for event_id in event_ids])
        logger.debug(f"Retrieved event embeddings for {len(event_ids)} events")
        return embeddings

class HateSpeechDetectionModelDemo:
    """
    Demonstration of the complete hate speech detection model architecture.
    This shows how all components integrate together.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Initialize components
        self.language_adapters = {
            f"lang_{i}": LanguageAdapterDemo(
                config.hidden_size, 
                config.language_adapter_size
            ) for i in range(config.num_languages)
        }
        
        self.regional_adapters = {
            f"region_{i}": RegionalAdapterDemo(
                config.hidden_size, 
                config.regional_adapter_size
            ) for i in range(config.num_regions)
        }
        
        self.event_embedding = EventEmbeddingDemo(
            config.num_events, 
            config.event_embedding_dim
        )
        
        logger.info("Hate Speech Detection Model initialized")
        logger.info(f"Configuration: {config}")
    
    def tokenize_text(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Simulate text tokenization (normally done by XLM-RoBERTa tokenizer).
        """
        # For demo purposes, create random token representations
        batch_size = len(texts)
        seq_length = self.config.max_length
        
        # Simulate tokenized inputs
        input_ids = np.random.randint(0, 50000, (batch_size, seq_length))
        attention_mask = np.ones((batch_size, seq_length))
        
        # Simulate XLM-RoBERTa hidden states
        hidden_states = np.random.normal(0, 0.1, (batch_size, seq_length, self.config.hidden_size))
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hidden_states": hidden_states
        }
    
    def forward(
        self,
        texts: List[str],
        language_ids: List[int],
        region_ids: List[int],
        event_ids: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Forward pass through the complete model.
        """
        batch_size = len(texts)
        
        # Step 1: Tokenize and get base representations
        tokenized = self.tokenize_text(texts)
        hidden_states = tokenized["hidden_states"]
        
        logger.info(f"Processing batch of {batch_size} texts")
        
        # Step 2: Apply language adapters
        for i in range(batch_size):
            lang_id = language_ids[i]
            if f"lang_{lang_id}" in self.language_adapters:
                hidden_states[i] = self.language_adapters[f"lang_{lang_id}"].forward(
                    hidden_states[i:i+1]
                )[0]
        
        # Step 3: Apply regional adapters
        for i in range(batch_size):
            region_id = region_ids[i]
            if f"region_{region_id}" in self.regional_adapters:
                hidden_states[i] = self.regional_adapters[f"region_{region_id}"].forward(
                    hidden_states[i:i+1]
                )[0]
        
        # Step 4: Pool sequence representations (simulate [CLS] token)
        pooled_output = hidden_states[:, 0, :]  # Use first token as sentence representation
        
        # Step 5: Get event embeddings
        event_embeds = self.event_embedding.forward(event_ids)
        
        # Step 6: Fuse text and event information
        fused_representation = np.concatenate([pooled_output, event_embeds], axis=-1)
        
        # Step 7: Classification (simulate final linear layer)
        # In practice, this would be learned weights
        classification_weights = np.random.normal(0, 0.1, (fused_representation.shape[1], self.config.num_labels))
        logits = np.dot(fused_representation, classification_weights)
        
        # Step 8: Get predictions
        predictions = np.argmax(logits, axis=-1)
        confidence_scores = np.max(logits, axis=-1)
        
        logger.info(f"Model forward pass completed for {batch_size} samples")
        
        return {
            "logits": logits,
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "hidden_states": fused_representation,
            "attention_weights": tokenized["attention_mask"]  # Simplified attention
        }

class ExplainabilityDemo:
    """
    Demonstration of the explainability suite components.
    """
    
    def __init__(self):
        logger.info("Explainability Suite initialized")
    
    def generate_attention_heatmap(self, attention_weights: np.ndarray, tokens: List[str]) -> Dict:
        """
        Simulate attention heatmap generation.
        """
        # For demo, create random attention scores
        attention_scores = np.random.uniform(0, 1, len(tokens))
        attention_scores = attention_scores / np.sum(attention_scores)  # Normalize
        
        heatmap_data = {
            "tokens": tokens,
            "attention_scores": attention_scores.tolist(),
            "explanation": "Attention heatmap shows which words the model focused on"
        }
        
        logger.info("Generated attention heatmap")
        return heatmap_data
    
    def generate_integrated_gradients(self, text: str, prediction: int) -> Dict:
        """
        Simulate integrated gradients explanation.
        """
        tokens = text.split()
        
        # Simulate importance scores
        importance_scores = np.random.uniform(-1, 1, len(tokens))
        
        explanation_data = {
            "tokens": tokens,
            "importance_scores": importance_scores.tolist(),
            "prediction": prediction,
            "explanation": "Integrated gradients show feature importance for the prediction"
        }
        
        logger.info("Generated integrated gradients explanation")
        return explanation_data
    
    def generate_shap_summary(self, features: List[str], prediction: int) -> Dict:
        """
        Simulate SHAP summary generation.
        """
        # Simulate SHAP values
        shap_values = np.random.uniform(-0.5, 0.5, len(features))
        
        shap_data = {
            "features": features,
            "shap_values": shap_values.tolist(),
            "prediction": prediction,
            "base_value": 0.0,
            "explanation": "SHAP values show how each feature contributes to the prediction"
        }
        
        logger.info("Generated SHAP summary")
        return shap_data
    
    def generate_natural_language_rationale(self, text: str, prediction: int, confidence: float) -> str:
        """
        Simulate natural language rationale generation.
        """
        if prediction == 1:  # Hate speech
            rationale = f"This text was classified as hate speech with {confidence:.2f} confidence. " \
                       f"The model identified potentially harmful language patterns and offensive content " \
                       f"that target specific groups or individuals."
        else:  # Not hate speech
            rationale = f"This text was classified as non-hate speech with {confidence:.2f} confidence. " \
                       f"The model found no significant indicators of harmful or offensive language " \
                       f"targeting specific groups or individuals."
        
        logger.info("Generated natural language rationale")
        return rationale

def demonstrate_pipeline():
    """
    Demonstrate the complete pipeline with sample data.
    """
    logger.info("Starting Responsible-AI Hate Speech Detection Pipeline Demonstration")
    
    # Initialize configuration
    config = ModelConfig()
    
    # Initialize model
    model = HateSpeechDetectionModelDemo(config)
    
    # Initialize explainability suite
    explainer = ExplainabilityDemo()
    
    # Sample data
    sample_texts = [
        "I love spending time with my friends and family.",
        "This group of people is ruining our country!",
        "Great weather today, perfect for a walk in the park.",
        "Those immigrants should go back where they came from."
    ]
    
    sample_language_ids = [0, 0, 0, 0]  # All English
    sample_region_ids = [0, 1, 0, 1]    # Mixed regions
    sample_event_ids = [0, 5, 0, 5]     # Some related to events
    
    # Run model inference
    logger.info("Running model inference...")
    results = model.forward(
        texts=sample_texts,
        language_ids=sample_language_ids,
        region_ids=sample_region_ids,
        event_ids=sample_event_ids
    )
    
    # Generate explanations for each sample
    logger.info("Generating explanations...")
    for i, text in enumerate(sample_texts):
        prediction = results["predictions"][i]
        confidence = results["confidence_scores"][i]
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {text}")
        print(f"Prediction: {'Hate Speech' if prediction == 1 else 'Not Hate Speech'}")
        print(f"Confidence: {confidence:.3f}")
        
        # Generate explanations
        tokens = text.split()
        
        # Attention heatmap
        attention_heatmap = explainer.generate_attention_heatmap(
            results["attention_weights"][i], tokens
        )
        print(f"Top attended words: {[tokens[j] for j in np.argsort(attention_heatmap['attention_scores'])[-3:]]}")
        
        # Integrated gradients
        integrated_gradients = explainer.generate_integrated_gradients(text, prediction)
        important_words = [tokens[j] for j in np.argsort(np.abs(integrated_gradients['importance_scores']))[-3:]]
        print(f"Most important words: {important_words}")
        
        # SHAP summary
        shap_summary = explainer.generate_shap_summary(tokens, prediction)
        contributing_words = [tokens[j] for j in np.argsort(np.abs(shap_summary['shap_values']))[-3:]]
        print(f"Top contributing words: {contributing_words}")
        
        # Natural language rationale
        rationale = explainer.generate_natural_language_rationale(text, prediction, confidence)
        print(f"Rationale: {rationale}")
    
    logger.info("Pipeline demonstration completed successfully!")

if __name__ == "__main__":
    demonstrate_pipeline()

