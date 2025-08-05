"""
Test script for the Contextual Adaptation API
"""

import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_contextual_adaptation_api():
    """Test the contextual adaptation API functionality without network calls."""
    
    # Import the contextual adaptation pipeline directly
    from contextual_adaptation import ContextualAdaptationPipeline
    
    logger.info("Testing Contextual Adaptation API functionality...")
    
    # Initialize pipeline
    pipeline = ContextualAdaptationPipeline()
    
    # Test data
    test_cases = [
        {
            "text": "This group of politicians is ruining our democracy!",
            "base_prediction": 0.6,
            "region": "north_america",
            "organization": "Twitter",
            "platform": "twitter",
            "description": "Political criticism during election period"
        },
        {
            "text": "Women are not suited for leadership roles in technology.",
            "base_prediction": 0.8,
            "region": "europe",
            "organization": "Facebook",
            "platform": "facebook",
            "description": "Gender-based discrimination"
        },
        {
            "text": "I love the innovation happening in AI technology!",
            "base_prediction": 0.1,
            "region": "north_america",
            "organization": "Twitter",
            "platform": "twitter",
            "description": "Positive technology discussion"
        },
        {
            "text": "These immigrants are taking our jobs and destroying our culture!",
            "base_prediction": 0.9,
            "region": "europe",
            "organization": "Facebook",
            "platform": "facebook",
            "description": "Anti-immigration hate speech"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"Processing test case {i+1}: '{test_case['text'][:30]}...'")
        
        # Apply contextual adaptations
        adaptations = pipeline.adapt_prediction(
            text=test_case['text'],
            base_prediction=test_case['base_prediction'],
            region=test_case['region'],
            organization=test_case['organization'],
            platform=test_case['platform']
        )
        
        # Convert to API response format
        api_response = {
            "input": test_case,
            "adaptations": adaptations
        }
        
        results.append(api_response)
        
        # Print summary
        print(f"\n--- Test Case {i+1}: {test_case['description']} ---")
        print(f"Text: {test_case['text']}")
        print(f"Base Prediction: {test_case['base_prediction']:.3f}")
        print(f"Region: {test_case['region']}")
        print(f"Platform: {test_case['organization']} - {test_case['platform']}")
        print()
        
        print(f"Regional Adjustment: {adaptations['regional_adaptation']['adjustment']:+.3f}")
        print(f"Temporal Adjustment: {adaptations['temporal_adaptation']['adjustment']:+.3f}")
        print(f"Policy Adjustment: {adaptations['policy_adaptation']['adjustment']:+.3f}")
        print(f"Final Prediction: {adaptations['final_prediction']:.3f}")
        
        # Calculate total adjustment
        total_adjustment = adaptations['final_prediction'] - adaptations['base_prediction']
        print(f"Total Adjustment: {total_adjustment:+.3f}")
        print()
    
    # Save test results
    with open('contextual_api_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Contextual adaptation API functionality test completed successfully!")
    logger.info("Results saved to contextual_api_test_results.json")
    
    # Test individual components
    logger.info("Testing individual contextual adaptation components...")
    
    # Test regional adaptation
    print("\n=== Regional Adaptation Test ===")
    base_pred = 0.7
    na_pred = pipeline.regional_engine.get_regional_adaptation("north_america", base_pred)
    eu_pred = pipeline.regional_engine.get_regional_adaptation("europe", base_pred)
    
    print(f"Base prediction: {base_pred:.3f}")
    print(f"North America adaptation: {na_pred:.3f} (adjustment: {na_pred - base_pred:+.3f})")
    print(f"Europe adaptation: {eu_pred:.3f} (adjustment: {eu_pred - base_pred:+.3f})")
    
    # Test policy adaptation
    print("\n=== Policy Adaptation Test ===")
    twitter_pred = pipeline.policy_engine.get_policy_adaptation("Twitter", "twitter", base_pred)
    facebook_pred = pipeline.policy_engine.get_policy_adaptation("Facebook", "facebook", base_pred)
    
    print(f"Base prediction: {base_pred:.3f}")
    print(f"Twitter policy adaptation: {twitter_pred:.3f} (adjustment: {twitter_pred - base_pred:+.3f})")
    print(f"Facebook policy adaptation: {facebook_pred:.3f} (adjustment: {facebook_pred - base_pred:+.3f})")
    
    # Test temporal adaptation
    print("\n=== Temporal Adaptation Test ===")
    pipeline.temporal_engine.update_event_impacts(datetime.now())
    
    political_text = "The election results are being manipulated by corrupt politicians!"
    tech_text = "AI technology is revolutionizing our future!"
    
    political_pred = pipeline.temporal_engine.get_temporal_adaptation(political_text, base_pred, "north_america")
    tech_pred = pipeline.temporal_engine.get_temporal_adaptation(tech_text, base_pred, "north_america")
    
    print(f"Base prediction: {base_pred:.3f}")
    print(f"Political text adaptation: {political_pred:.3f} (adjustment: {political_pred - base_pred:+.3f})")
    print(f"Tech text adaptation: {tech_pred:.3f} (adjustment: {tech_pred - base_pred:+.3f})")
    
    logger.info("All contextual adaptation tests completed successfully!")

if __name__ == "__main__":
    test_contextual_adaptation_api()

