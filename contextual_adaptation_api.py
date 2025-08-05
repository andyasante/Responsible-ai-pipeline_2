"""
API endpoints for Contextual Adaptation System
Provides RESTful access to contextual adaptation capabilities.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Import our contextual adaptation components
from contextual_adaptation import ContextualAdaptationPipeline, ContextualEvent, RegionalContext, OrganizationalPolicy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize contextual adaptation pipeline
adaptation_pipeline = ContextualAdaptationPipeline()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "contextual-adaptation-api"})

@app.route('/adapt', methods=['POST'])
def adapt_prediction():
    """
    Apply contextual adaptations to a base prediction.
    
    Expected JSON payload:
    {
        "text": "input text",
        "base_prediction": 0.75,
        "region": "north_america",
        "organization": "Twitter",
        "platform": "twitter",
        "timestamp": "2024-01-15T10:30:00Z" (optional)
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['text', 'base_prediction', 'region', 'organization', 'platform']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        text = data['text']
        base_prediction = float(data['base_prediction'])
        region = data['region']
        organization = data['organization']
        platform = data['platform']
        
        # Parse timestamp if provided
        timestamp = None
        if 'timestamp' in data:
            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        
        logger.info(f"Adapting prediction for text: '{text[:50]}...'")
        
        # Apply contextual adaptations
        adaptations = adaptation_pipeline.adapt_prediction(
            text=text,
            base_prediction=base_prediction,
            region=region,
            organization=organization,
            platform=platform,
            timestamp=timestamp
        )
        
        logger.info("Contextual adaptations applied successfully")
        return jsonify(adaptations)
        
    except Exception as e:
        logger.error(f"Error applying contextual adaptations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/events', methods=['GET'])
def get_active_events():
    """Get currently active events."""
    try:
        current_time = datetime.now()
        active_events = adaptation_pipeline.db.get_active_events(current_time)
        
        events_data = []
        for event in active_events:
            events_data.append({
                "event_id": event.event_id,
                "name": event.name,
                "description": event.description,
                "event_type": event.event_type,
                "impact_level": event.impact_level,
                "affected_regions": event.affected_regions,
                "keywords": event.keywords
            })
        
        return jsonify({"active_events": events_data})
        
    except Exception as e:
        logger.error(f"Error retrieving active events: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/events', methods=['POST'])
def create_event():
    """
    Create a new contextual event.
    
    Expected JSON payload:
    {
        "event_id": "unique_id",
        "name": "Event Name",
        "description": "Event description",
        "start_date": "2024-01-15T00:00:00Z",
        "end_date": "2024-01-16T00:00:00Z",
        "event_type": "political",
        "impact_level": 0.8,
        "affected_regions": ["north_america"],
        "keywords": ["keyword1", "keyword2"]
    }
    """
    try:
        data = request.get_json()
        
        # Create event object
        event = ContextualEvent(
            event_id=data['event_id'],
            name=data['name'],
            description=data['description'],
            start_date=datetime.fromisoformat(data['start_date'].replace('Z', '+00:00')),
            end_date=datetime.fromisoformat(data['end_date'].replace('Z', '+00:00')) if data.get('end_date') else None,
            event_type=data['event_type'],
            impact_level=float(data['impact_level']),
            affected_regions=data['affected_regions'],
            keywords=data['keywords']
        )
        
        # Generate embedding
        event.embedding = adaptation_pipeline.event_generator.generate_embedding(event)
        
        # Store event
        adaptation_pipeline.db.store_event(event)
        
        logger.info(f"Created new event: {event.name}")
        return jsonify({"message": "Event created successfully", "event_id": event.event_id})
        
    except Exception as e:
        logger.error(f"Error creating event: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/regions', methods=['GET'])
def get_regions():
    """Get all registered regions."""
    try:
        regions_data = []
        for region_id, region in adaptation_pipeline.regional_engine.regional_contexts.items():
            regions_data.append({
                "region_id": region.region_id,
                "name": region.name,
                "cultural_factors": region.cultural_factors,
                "language_variants": region.language_variants,
                "sensitive_topics": region.sensitive_topics,
                "moderation_strictness": region.moderation_strictness
            })
        
        return jsonify({"regions": regions_data})
        
    except Exception as e:
        logger.error(f"Error retrieving regions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/regions', methods=['POST'])
def create_region():
    """
    Create a new regional context.
    
    Expected JSON payload:
    {
        "region_id": "unique_id",
        "name": "Region Name",
        "cultural_factors": {"conservatism": 0.6, "tolerance": 0.7},
        "language_variants": ["en-US"],
        "sensitive_topics": ["politics"],
        "moderation_strictness": 0.7
    }
    """
    try:
        data = request.get_json()
        
        # Create region object
        region = RegionalContext(
            region_id=data['region_id'],
            name=data['name'],
            cultural_factors=data['cultural_factors'],
            language_variants=data['language_variants'],
            sensitive_topics=data['sensitive_topics'],
            moderation_strictness=float(data['moderation_strictness'])
        )
        
        # Register region
        adaptation_pipeline.regional_engine.register_region(region)
        
        logger.info(f"Created new region: {region.name}")
        return jsonify({"message": "Region created successfully", "region_id": region.region_id})
        
    except Exception as e:
        logger.error(f"Error creating region: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/policies', methods=['GET'])
def get_policies():
    """Get all registered policies."""
    try:
        policies_data = []
        for key, policy in adaptation_pipeline.policy_engine.policies.items():
            policies_data.append({
                "policy_id": policy.policy_id,
                "organization": policy.organization,
                "platform": policy.platform,
                "policy_type": policy.policy_type,
                "strictness_level": policy.strictness_level,
                "specific_rules": policy.specific_rules,
                "last_updated": policy.last_updated.isoformat()
            })
        
        return jsonify({"policies": policies_data})
        
    except Exception as e:
        logger.error(f"Error retrieving policies: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/policies', methods=['POST'])
def create_policy():
    """
    Create a new organizational policy.
    
    Expected JSON payload:
    {
        "policy_id": "unique_id",
        "organization": "Organization Name",
        "platform": "platform_name",
        "policy_type": "content",
        "strictness_level": 0.8,
        "specific_rules": {"zero_tolerance": true}
    }
    """
    try:
        data = request.get_json()
        
        # Create policy object
        policy = OrganizationalPolicy(
            policy_id=data['policy_id'],
            organization=data['organization'],
            platform=data['platform'],
            policy_type=data['policy_type'],
            strictness_level=float(data['strictness_level']),
            specific_rules=data['specific_rules'],
            last_updated=datetime.now()
        )
        
        # Register policy
        adaptation_pipeline.policy_engine.register_policy(policy)
        
        logger.info(f"Created new policy: {policy.organization} - {policy.platform}")
        return jsonify({"message": "Policy created successfully", "policy_id": policy.policy_id})
        
    except Exception as e:
        logger.error(f"Error creating policy: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_adapt', methods=['POST'])
def batch_adapt():
    """
    Apply contextual adaptations to multiple predictions in batch.
    
    Expected JSON payload:
    {
        "batch": [
            {
                "text": "text1",
                "base_prediction": 0.6,
                "region": "north_america",
                "organization": "Twitter",
                "platform": "twitter"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        batch_data = data['batch']
        
        results = []
        for item in batch_data:
            adaptations = adaptation_pipeline.adapt_prediction(
                text=item['text'],
                base_prediction=float(item['base_prediction']),
                region=item['region'],
                organization=item['organization'],
                platform=item['platform'],
                timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')) if 'timestamp' in item else None
            )
            
            results.append({
                "input": item,
                "adaptations": adaptations
            })
        
        logger.info(f"Applied contextual adaptations to {len(batch_data)} items")
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error in batch adaptation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get contextual adaptation system configuration."""
    return jsonify({
        "adaptation_types": [
            "regional",
            "temporal", 
            "organizational_policy"
        ],
        "event_types": [
            "political",
            "social",
            "economic",
            "cultural",
            "crisis"
        ],
        "policy_types": [
            "content",
            "user",
            "community"
        ],
        "version": "1.0.0"
    })

if __name__ == '__main__':
    logger.info("Starting Contextual Adaptation API server...")
    app.run(host='0.0.0.0', port=5002, debug=True)

