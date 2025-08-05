"""
API endpoints for the Explainability Suite
Provides RESTful access to explanation generation capabilities.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import json
import logging
from typing import Dict, Any
import base64
from io import BytesIO
import os

# Import our explainability components
from explainability_suite import ExplainabilitySuite, ExplanationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize explainability suite
explainability_suite = ExplainabilitySuite()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "explainability-api"})

@app.route('/explain', methods=['POST'])
def explain_prediction():
    """
    Generate comprehensive explanations for a model prediction.
    
    Expected JSON payload:
    {
        "text": "input text",
        "tokens": ["list", "of", "tokens"],
        "prediction": 0 or 1,
        "confidence": 0.85,
        "attention_weights": [...] (optional, will simulate if not provided)
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['text', 'tokens', 'prediction', 'confidence']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        text = data['text']
        tokens = data['tokens']
        prediction = int(data['prediction'])
        confidence = float(data['confidence'])
        
        # Handle attention weights
        if 'attention_weights' in data:
            attention_weights = np.array(data['attention_weights'])
        else:
            # Simulate attention weights if not provided
            seq_len = len(tokens)
            attention_weights = np.random.uniform(0, 1, (1, 8, seq_len, seq_len))
        
        logger.info(f"Generating explanations for text: '{text[:50]}...'")
        
        # Generate explanations
        explanations = explainability_suite.explain_prediction(
            text=text,
            tokens=tokens,
            prediction=prediction,
            confidence=confidence,
            attention_weights=attention_weights
        )
        
        # Convert explanations to JSON-serializable format
        response_data = {}
        for key, result in explanations.items():
            response_data[key] = {
                "type": result.explanation_type,
                "data": result.data,
                "text_summary": result.text_summary,
                "visualization": result.visualization  # Base64 encoded image
            }
        
        logger.info("Explanations generated successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error generating explanations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/explain/attention', methods=['POST'])
def explain_attention_only():
    """
    Generate only attention heatmap explanation.
    """
    try:
        data = request.get_json()
        
        tokens = data['tokens']
        
        # Simulate or use provided attention weights
        if 'attention_weights' in data:
            attention_weights = np.array(data['attention_weights'])
        else:
            seq_len = len(tokens)
            attention_weights = np.random.uniform(0, 1, (seq_len, seq_len))
        
        result = explainability_suite.attention_explainer.explain(
            tokens=tokens,
            attention_weights=attention_weights
        )
        
        return jsonify({
            "type": result.explanation_type,
            "data": result.data,
            "text_summary": result.text_summary,
            "visualization": result.visualization
        })
        
    except Exception as e:
        logger.error(f"Error generating attention explanation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/explain/shap', methods=['POST'])
def explain_shap_only():
    """
    Generate only SHAP explanation.
    """
    try:
        data = request.get_json()
        
        tokens = data['tokens']
        prediction = float(data.get('prediction', 0.5))
        base_value = float(data.get('base_value', 0.0))
        
        # Use provided SHAP values or simulate them
        if 'shap_values' in data:
            shap_values = np.array(data['shap_values'])
        else:
            shap_values = np.random.uniform(-0.5, 0.5, len(tokens))
        
        result = explainability_suite.shap_explainer.explain(
            tokens=tokens,
            shap_values=shap_values,
            base_value=base_value,
            prediction=prediction
        )
        
        return jsonify({
            "type": result.explanation_type,
            "data": result.data,
            "text_summary": result.text_summary,
            "visualization": result.visualization
        })
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/explain/rationale', methods=['POST'])
def generate_rationale_only():
    """
    Generate only natural language rationale.
    """
    try:
        data = request.get_json()
        
        text = data['text']
        prediction = int(data['prediction'])
        confidence = float(data['confidence'])
        
        # Create mock explanation results for rationale generation
        mock_attention = ExplanationResult(
            explanation_type="attention_heatmap",
            data={"top_tokens": data.get('top_tokens', text.split()[:3])},
            text_summary=""
        )
        
        mock_shap = ExplanationResult(
            explanation_type="shap",
            data={"top_features": [(token, 0.1) for token in text.split()[:3]]},
            text_summary=""
        )
        
        mock_ig = ExplanationResult(
            explanation_type="integrated_gradients",
            data={},
            text_summary=""
        )
        
        rationale = explainability_suite.rationale_generator.generate_rationale(
            text=text,
            prediction=prediction,
            confidence=confidence,
            attention_result=mock_attention,
            shap_result=mock_shap,
            integrated_gradients_result=mock_ig
        )
        
        return jsonify({
            "type": "natural_language_rationale",
            "rationale": rationale
        })
        
    except Exception as e:
        logger.error(f"Error generating rationale: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/visualization/<explanation_type>/<image_id>', methods=['GET'])
def get_visualization(explanation_type, image_id):
    """
    Serve visualization images.
    This endpoint would typically serve cached visualizations.
    """
    try:
        # In a real implementation, you would retrieve the image from storage
        # For demo purposes, return a placeholder
        return jsonify({"message": f"Visualization for {explanation_type} with ID {image_id}"})
        
    except Exception as e:
        logger.error(f"Error serving visualization: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_explain', methods=['POST'])
def batch_explain():
    """
    Generate explanations for multiple predictions in batch.
    """
    try:
        data = request.get_json()
        
        batch_data = data['batch']
        results = []
        
        for item in batch_data:
            text = item['text']
            tokens = item['tokens']
            prediction = int(item['prediction'])
            confidence = float(item['confidence'])
            
            # Simulate attention weights
            seq_len = len(tokens)
            attention_weights = np.random.uniform(0, 1, (1, 8, seq_len, seq_len))
            
            # Generate explanations
            explanations = explainability_suite.explain_prediction(
                text=text,
                tokens=tokens,
                prediction=prediction,
                confidence=confidence,
                attention_weights=attention_weights
            )
            
            # Convert to JSON format
            item_result = {}
            for key, result in explanations.items():
                item_result[key] = {
                    "type": result.explanation_type,
                    "data": result.data,
                    "text_summary": result.text_summary,
                    "visualization": result.visualization
                }
            
            results.append({
                "input": item,
                "explanations": item_result
            })
        
        logger.info(f"Generated explanations for {len(batch_data)} items")
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error in batch explanation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """
    Get explainability suite configuration.
    """
    return jsonify({
        "available_methods": [
            "attention_heatmap",
            "integrated_gradients", 
            "shap",
            "natural_language_rationale"
        ],
        "supported_visualizations": [
            "heatmap",
            "bar_chart",
            "waterfall_plot"
        ],
        "version": "1.0.0"
    })

if __name__ == '__main__':
    logger.info("Starting Explainability API server...")
    app.run(host='0.0.0.0', port=5001, debug=True)

