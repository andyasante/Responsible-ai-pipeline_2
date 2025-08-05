"""
Test script for the Explainability API
"""

import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_explainability_api():
    """Test the explainability API functionality without network calls."""
    
    # Import the explainability suite directly
    from explainability_suite import ExplainabilitySuite
    import numpy as np
    
    logger.info("Testing Explainability API functionality...")
    
    # Initialize suite
    suite = ExplainabilitySuite()
    
    # Test data
    test_cases = [
        {
            "text": "I love spending time with my friends and family.",
            "tokens": ["I", "love", "spending", "time", "with", "my", "friends", "and", "family."],
            "prediction": 0,
            "confidence": 0.95
        },
        {
            "text": "This group of people is ruining our country!",
            "tokens": ["This", "group", "of", "people", "is", "ruining", "our", "country!"],
            "prediction": 1,
            "confidence": 0.85
        },
        {
            "text": "Great weather today, perfect for a walk in the park.",
            "tokens": ["Great", "weather", "today,", "perfect", "for", "a", "walk", "in", "the", "park."],
            "prediction": 0,
            "confidence": 0.92
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"Processing test case {i+1}: '{test_case['text'][:30]}...'")
        
        # Simulate attention weights
        seq_len = len(test_case['tokens'])
        attention_weights = np.random.uniform(0, 1, (1, 8, seq_len, seq_len))
        
        # Generate explanations
        explanations = suite.explain_prediction(
            text=test_case['text'],
            tokens=test_case['tokens'],
            prediction=test_case['prediction'],
            confidence=test_case['confidence'],
            attention_weights=attention_weights
        )
        
        # Convert to API response format
        api_response = {}
        for key, result in explanations.items():
            api_response[key] = {
                "type": result.explanation_type,
                "data": result.data,
                "text_summary": result.text_summary,
                "has_visualization": result.visualization is not None
            }
        
        results.append({
            "input": test_case,
            "explanations": api_response
        })
        
        # Print summary
        print(f"\n--- Test Case {i+1} ---")
        print(f"Text: {test_case['text']}")
        print(f"Prediction: {'Hate Speech' if test_case['prediction'] == 1 else 'Not Hate Speech'}")
        print(f"Confidence: {test_case['confidence']:.1%}")
        print()
        
        for explanation_type, explanation in api_response.items():
            print(f"{explanation_type.upper()}: {explanation['text_summary']}")
        print()
    
    # Save test results
    with open('api_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("API functionality test completed successfully!")
    logger.info("Results saved to api_test_results.json")
    
    # Test individual explanation methods
    logger.info("Testing individual explanation methods...")
    
    sample_tokens = ["This", "is", "a", "test", "sentence"]
    
    # Test attention explainer
    attention_weights = np.random.uniform(0, 1, (5, 5))
    attention_result = suite.attention_explainer.explain(
        tokens=sample_tokens,
        attention_weights=attention_weights
    )
    print(f"Attention Explanation: {attention_result.text_summary}")
    
    # Test SHAP explainer
    shap_values = np.random.uniform(-0.5, 0.5, 5)
    shap_result = suite.shap_explainer.explain(
        tokens=sample_tokens,
        shap_values=shap_values,
        base_value=0.0,
        prediction=0.7
    )
    print(f"SHAP Explanation: {shap_result.text_summary}")
    
    # Test rationale generator
    rationale = suite.rationale_generator.generate_rationale(
        text="This is a test sentence",
        prediction=0,
        confidence=0.8,
        attention_result=attention_result,
        shap_result=shap_result,
        integrated_gradients_result=attention_result  # Mock IG result
    )
    print(f"Natural Language Rationale: {rationale}")
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    test_explainability_api()

