"""
Responsible-AI Hate Speech Detection Pipeline: Machine Learning Model Architecture and Training Pipeline

This module implements the core machine learning components for the hate speech detection pipeline,
including the XLM-RoBERTa model with language adapters, event embeddings, and regional adapters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    XLMRobertaModel, 
    XLMRobertaTokenizer, 
    XLMRobertaConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageAdapter(nn.Module):
    """
    Language-specific adapter module for XLM-RoBERTa.
    Allows efficient adaptation to specific languages without full model retraining.
    """
    
    def __init__(self, hidden_size: int, adapter_size: int = 64, dropout: float = 0.1):
        super(LanguageAdapter, self).__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        # Down-projection
        self.down_project = nn.Linear(hidden_size, adapter_size)
        # Up-projection
        self.up_project = nn.Linear(adapter_size, hidden_size)
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.normal_(self.down_project.weight, std=0.02)
        nn.init.normal_(self.up_project.weight, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the language adapter.
        
        Args:
            hidden_states: Input hidden states from XLM-RoBERTa
            
        Returns:
            Adapted hidden states
        """
        # Store original hidden states for residual connection
        residual = hidden_states
        
        # Adapter transformation
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.up_project(hidden_states)
        
        # Residual connection
        return residual + hidden_states

class RegionalAdapter(nn.Module):
    """
    Regional adapter module for capturing geographical and cultural nuances.
    32-dimensional adapters for different regions.
    """
    
    def __init__(self, hidden_size: int, adapter_size: int = 32, dropout: float = 0.1):
        super(RegionalAdapter, self).__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        # Down-projection
        self.down_project = nn.Linear(hidden_size, adapter_size)
        # Up-projection
        self.up_project = nn.Linear(adapter_size, hidden_size)
        # Activation and dropout
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.normal_(self.down_project.weight, std=0.02)
        nn.init.normal_(self.up_project.weight, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the regional adapter.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Regionally adapted hidden states
        """
        residual = hidden_states
        
        # Regional adaptation
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.up_project(hidden_states)
        
        # Residual connection
        return residual + hidden_states

class EventEmbedding(nn.Module):
    """
    Event embedding module for incorporating temporal event context.
    64-dimensional embeddings for significant events.
    """
    
    def __init__(self, num_events: int, embedding_dim: int = 64):
        super(EventEmbedding, self).__init__()
        self.num_events = num_events
        self.embedding_dim = embedding_dim
        
        # Event embedding layer
        self.event_embeddings = nn.Embedding(num_events, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.event_embeddings.weight, std=0.02)
    
    def forward(self, event_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through event embeddings.
        
        Args:
            event_ids: Tensor of event IDs
            
        Returns:
            Event embeddings
        """
        return self.event_embeddings(event_ids)

class HateSpeechDetectionModel(nn.Module):
    """
    Main hate speech detection model based on XLM-RoBERTa with adapters and event embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 2,
        num_languages: int = 10,
        num_regions: int = 20,
        num_events: int = 1000,
        language_adapter_size: int = 64,
        regional_adapter_size: int = 32,
        event_embedding_dim: int = 64,
        dropout: float = 0.1
    ):
        super(HateSpeechDetectionModel, self).__init__()
        
        # Load XLM-RoBERTa configuration and model
        self.config = XLMRobertaConfig.from_pretrained(model_name)
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_name)
        
        self.num_labels = num_labels
        self.hidden_size = self.config.hidden_size
        
        # Language adapters for each language
        self.language_adapters = nn.ModuleDict({
            f"lang_{i}": LanguageAdapter(
                self.hidden_size, 
                language_adapter_size, 
                dropout
            ) for i in range(num_languages)
        })
        
        # Regional adapters for each region
        self.regional_adapters = nn.ModuleDict({
            f"region_{i}": RegionalAdapter(
                self.hidden_size, 
                regional_adapter_size, 
                dropout
            ) for i in range(num_regions)
        })
        
        # Event embeddings
        self.event_embedding = EventEmbedding(num_events, event_embedding_dim)
        
        # Fusion layer for combining text, event, and regional information
        fusion_input_size = self.hidden_size + event_embedding_dim
        self.fusion_layer = nn.Linear(fusion_input_size, self.hidden_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_labels)
        )
        
        # Initialize fusion layer and classifier
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for custom layers."""
        for module in [self.fusion_layer, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor,
        region_ids: torch.Tensor,
        event_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hate speech detection model.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding tokens
            language_ids: Language IDs for language adapter selection
            region_ids: Region IDs for regional adapter selection
            event_ids: Event IDs for event embeddings
            labels: Ground truth labels (optional, for training)
            
        Returns:
            Dictionary containing logits, loss (if labels provided), and attention weights
        """
        # Get XLM-RoBERTa outputs
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        
        # Extract sequence output and attention weights
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        attention_weights = outputs.attentions  # For explainability
        
        # Apply language adapters
        batch_size = sequence_output.size(0)
        adapted_output = sequence_output.clone()
        
        for i in range(batch_size):
            lang_id = language_ids[i].item()
            if f"lang_{lang_id}" in self.language_adapters:
                adapted_output[i] = self.language_adapters[f"lang_{lang_id}"](sequence_output[i])
        
        # Apply regional adapters
        for i in range(batch_size):
            region_id = region_ids[i].item()
            if f"region_{region_id}" in self.regional_adapters:
                adapted_output[i] = self.regional_adapters[f"region_{region_id}"](adapted_output[i])
        
        # Pool the sequence output (use [CLS] token)
        pooled_output = adapted_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Get event embeddings
        event_embeds = self.event_embedding(event_ids)  # [batch_size, event_embedding_dim]
        
        # Fuse text and event information
        fused_input = torch.cat([pooled_output, event_embeds], dim=-1)
        fused_output = self.fusion_layer(fused_input)
        fused_output = F.relu(fused_output)
        
        # Classification
        logits = self.classifier(fused_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "logits": logits,
            "loss": loss,
            "attention_weights": attention_weights,
            "hidden_states": fused_output
        }

class HateSpeechDataset(Dataset):
    """
    Dataset class for hate speech detection with multilingual support and contextual information.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        language_ids: List[int],
        region_ids: List[int],
        event_ids: List[int],
        tokenizer: XLMRobertaTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.language_ids = language_ids
        self.region_ids = region_ids
        self.event_ids = event_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        language_id = self.language_ids[idx]
        region_id = self.region_ids[idx]
        event_id = self.event_ids[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'language_ids': torch.tensor(language_id, dtype=torch.long),
            'region_ids': torch.tensor(region_id, dtype=torch.long),
            'event_ids': torch.tensor(event_id, dtype=torch.long)
        }

class HateSpeechTrainer:
    """
    Training pipeline for the hate speech detection model.
    """
    
    def __init__(
        self,
        model: HateSpeechDetectionModel,
        tokenizer: XLMRobertaTokenizer,
        device: torch.device,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        
        # Initialize optimizer and scheduler (will be set during training)
        self.optimizer = None
        self.scheduler = None
    
    def prepare_optimizer_and_scheduler(self, num_training_steps: int):
        """Prepare optimizer and learning rate scheduler."""
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, train_dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_dataloader)
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Collect results
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            "loss": total_loss / len(eval_dataloader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        num_epochs: int = 3,
        save_path: str = "hate_speech_model"
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        # Prepare optimizer and scheduler
        num_training_steps = len(train_dataloader) * num_epochs
        self.prepare_optimizer_and_scheduler(num_training_steps)
        
        # Training history
        history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "eval_f1": []
        }
        
        best_f1 = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            
            # Evaluate
            eval_metrics = self.evaluate(eval_dataloader)
            
            # Update history
            history["train_loss"].append(train_loss)
            history["eval_loss"].append(eval_metrics["loss"])
            history["eval_accuracy"].append(eval_metrics["accuracy"])
            history["eval_f1"].append(eval_metrics["f1"])
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Eval Loss: {eval_metrics['loss']:.4f}")
            logger.info(f"Eval Accuracy: {eval_metrics['accuracy']:.4f}")
            logger.info(f"Eval F1: {eval_metrics['f1']:.4f}")
            
            # Save best model
            if eval_metrics["f1"] > best_f1:
                best_f1 = eval_metrics["f1"]
                self.save_model(save_path)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
        
        return history
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(save_path, "model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save model config
        config = {
            "num_labels": self.model.num_labels,
            "hidden_size": self.model.hidden_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay
        }
        
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")

def create_sample_data() -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
    """
    Create sample data for demonstration purposes.
    In a real implementation, this would load from the comprehensive dataset.
    """
    # Sample texts (in practice, these would come from Twitter, Facebook, Reddit)
    texts = [
        "I love spending time with my friends and family.",
        "This group of people is ruining our country!",
        "Great weather today, perfect for a walk in the park.",
        "Those immigrants should go back where they came from.",
        "Just finished reading an amazing book about history.",
        "Women are not capable of leadership roles.",
        "Looking forward to the weekend with my loved ones.",
        "All members of that religion are terrorists.",
        "The new restaurant in town has excellent food.",
        "People with disabilities are a burden on society."
    ]
    
    # Labels: 0 = not hate speech, 1 = hate speech
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    # Language IDs (0 = English, 1 = Spanish, 2 = French, etc.)
    language_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Region IDs (0 = North America, 1 = Europe, 2 = Asia, etc.)
    region_ids = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    
    # Event IDs (0 = no specific event, 1 = election, 2 = protest, etc.)
    event_ids = [0, 1, 0, 1, 0, 2, 0, 1, 0, 2]
    
    return texts, labels, language_ids, region_ids, event_ids

def main():
    """Main function to demonstrate the training pipeline."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    
    # Create sample data
    texts, labels, language_ids, region_ids, event_ids = create_sample_data()
    
    # Create dataset
    dataset = HateSpeechDataset(
        texts=texts,
        labels=labels,
        language_ids=language_ids,
        region_ids=region_ids,
        event_ids=event_ids,
        tokenizer=tokenizer,
        max_length=128
    )
    
    # Create data loaders (using same data for train/eval for demonstration)
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    eval_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Initialize model
    model = HateSpeechDetectionModel(
        model_name="xlm-roberta-base",
        num_labels=2,
        num_languages=10,
        num_regions=20,
        num_events=1000
    )
    
    # Initialize trainer
    trainer = HateSpeechTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=2e-5
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=2,
        save_path="./hate_speech_model"
    )
    
    logger.info("Training completed!")
    logger.info(f"Final metrics: {history}")

if __name__ == "__main__":
    main()

