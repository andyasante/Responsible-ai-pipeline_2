"""
Contextual Adaptation Mechanisms for Responsible-AI Hate Speech Detection

This module implements the contextual adaptation system that enables the model to:
1. Adapt to temporal events and their impact on hate speech patterns
2. Adjust for regional and cultural differences
3. Learn from organizational dependencies and policies
4. Continuously update based on emerging contexts
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContextualEvent:
    """Represents a temporal event that may influence hate speech patterns."""
    event_id: str
    name: str
    description: str
    start_date: datetime
    end_date: Optional[datetime]
    event_type: str  # 'political', 'social', 'economic', 'cultural', 'crisis'
    impact_level: float  # 0.0 to 1.0
    affected_regions: List[str]
    keywords: List[str]
    embedding: Optional[List[float]] = None

@dataclass
class RegionalContext:
    """Represents regional/cultural context for adaptation."""
    region_id: str
    name: str
    cultural_factors: Dict[str, float]
    language_variants: List[str]
    sensitive_topics: List[str]
    moderation_strictness: float  # 0.0 to 1.0
    embedding: Optional[List[float]] = None

@dataclass
class OrganizationalPolicy:
    """Represents organizational moderation policies."""
    policy_id: str
    organization: str
    platform: str
    policy_type: str  # 'content', 'user', 'community'
    strictness_level: float
    specific_rules: Dict[str, Any]
    last_updated: datetime

class ContextualDatabase:
    """Database for storing and retrieving contextual information."""
    
    def __init__(self, db_path: str = "contextual_data.db"):
        self.db_path = db_path
        self.init_database()
        logger.info(f"Contextual database initialized at {db_path}")
    
    def init_database(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                start_date TEXT NOT NULL,
                end_date TEXT,
                event_type TEXT NOT NULL,
                impact_level REAL NOT NULL,
                affected_regions TEXT,
                keywords TEXT,
                embedding TEXT
            )
        ''')
        
        # Regions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regions (
                region_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                cultural_factors TEXT,
                language_variants TEXT,
                sensitive_topics TEXT,
                moderation_strictness REAL NOT NULL,
                embedding TEXT
            )
        ''')
        
        # Policies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS policies (
                policy_id TEXT PRIMARY KEY,
                organization TEXT NOT NULL,
                platform TEXT NOT NULL,
                policy_type TEXT NOT NULL,
                strictness_level REAL NOT NULL,
                specific_rules TEXT,
                last_updated TEXT NOT NULL
            )
        ''')
        
        # Adaptation logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adaptation_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                adaptation_type TEXT NOT NULL,
                context_id TEXT NOT NULL,
                performance_before REAL,
                performance_after REAL,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_event(self, event: ContextualEvent):
        """Store a contextual event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO events 
            (event_id, name, description, start_date, end_date, event_type, 
             impact_level, affected_regions, keywords, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.name,
            event.description,
            event.start_date.isoformat(),
            event.end_date.isoformat() if event.end_date else None,
            event.event_type,
            event.impact_level,
            json.dumps(event.affected_regions),
            json.dumps(event.keywords),
            json.dumps(event.embedding) if event.embedding else None
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored event: {event.name}")
    
    def get_active_events(self, current_time: datetime) -> List[ContextualEvent]:
        """Get events that are currently active."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM events 
            WHERE start_date <= ? AND (end_date IS NULL OR end_date >= ?)
        ''', (current_time.isoformat(), current_time.isoformat()))
        
        events = []
        for row in cursor.fetchall():
            event = ContextualEvent(
                event_id=row[0],
                name=row[1],
                description=row[2],
                start_date=datetime.fromisoformat(row[3]),
                end_date=datetime.fromisoformat(row[4]) if row[4] else None,
                event_type=row[5],
                impact_level=row[6],
                affected_regions=json.loads(row[7]),
                keywords=json.loads(row[8]),
                embedding=json.loads(row[9]) if row[9] else None
            )
            events.append(event)
        
        conn.close()
        return events

class EventEmbeddingGenerator:
    """Generates embeddings for temporal events."""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        logger.info(f"Event embedding generator initialized with {embedding_dim} dimensions")
    
    def generate_embedding(self, event: ContextualEvent) -> List[float]:
        """Generate embedding for an event based on its characteristics."""
        # In practice, this would use a trained embedding model
        # For demonstration, we create embeddings based on event features
        
        # Base embedding from event type
        type_embeddings = {
            'political': np.random.normal(0.5, 0.1, self.embedding_dim),
            'social': np.random.normal(0.0, 0.1, self.embedding_dim),
            'economic': np.random.normal(-0.3, 0.1, self.embedding_dim),
            'cultural': np.random.normal(0.2, 0.1, self.embedding_dim),
            'crisis': np.random.normal(-0.8, 0.1, self.embedding_dim)
        }
        
        base_embedding = type_embeddings.get(event.event_type, np.zeros(self.embedding_dim))
        
        # Modify based on impact level
        impact_factor = event.impact_level * 2 - 1  # Scale to [-1, 1]
        base_embedding = base_embedding * (1 + impact_factor * 0.3)
        
        # Add noise for uniqueness
        noise = np.random.normal(0, 0.05, self.embedding_dim)
        final_embedding = base_embedding + noise
        
        # Normalize
        final_embedding = final_embedding / np.linalg.norm(final_embedding)
        
        return final_embedding.tolist()

class RegionalAdaptationEngine:
    """Handles regional and cultural adaptation."""
    
    def __init__(self):
        self.regional_contexts = {}
        self.adaptation_weights = {}
        logger.info("Regional adaptation engine initialized")
    
    def register_region(self, region: RegionalContext):
        """Register a regional context."""
        self.regional_contexts[region.region_id] = region
        
        # Generate adaptation weights based on cultural factors
        weights = {}
        for factor, value in region.cultural_factors.items():
            weights[factor] = value * region.moderation_strictness
        
        self.adaptation_weights[region.region_id] = weights
        logger.info(f"Registered region: {region.name}")
    
    def get_regional_adaptation(self, region_id: str, base_prediction: float) -> float:
        """Apply regional adaptation to a base prediction."""
        if region_id not in self.regional_contexts:
            return base_prediction
        
        region = self.regional_contexts[region_id]
        weights = self.adaptation_weights[region_id]
        
        # Apply cultural factor adjustments
        adjustment = 0.0
        for factor, weight in weights.items():
            if factor == 'conservatism':
                adjustment += weight * 0.1  # More conservative = stricter
            elif factor == 'tolerance':
                adjustment -= weight * 0.1  # More tolerant = less strict
            elif factor == 'formality':
                adjustment += weight * 0.05  # More formal = slightly stricter
        
        # Apply moderation strictness
        strictness_adjustment = (region.moderation_strictness - 0.5) * 0.2
        
        adapted_prediction = base_prediction + adjustment + strictness_adjustment
        adapted_prediction = max(0.0, min(1.0, adapted_prediction))  # Clamp to [0, 1]
        
        logger.debug(f"Regional adaptation for {region_id}: {base_prediction:.3f} -> {adapted_prediction:.3f}")
        return adapted_prediction

class TemporalAdaptationEngine:
    """Handles temporal event-based adaptation."""
    
    def __init__(self, db: ContextualDatabase):
        self.db = db
        self.event_impacts = {}
        logger.info("Temporal adaptation engine initialized")
    
    def update_event_impacts(self, current_time: datetime):
        """Update the impact of current events on model behavior."""
        active_events = self.db.get_active_events(current_time)
        
        self.event_impacts = {}
        for event in active_events:
            # Calculate time-based decay
            days_since_start = (current_time - event.start_date).days
            decay_factor = max(0.1, 1.0 - (days_since_start / 30.0))  # Decay over 30 days
            
            current_impact = event.impact_level * decay_factor
            self.event_impacts[event.event_id] = {
                'event': event,
                'current_impact': current_impact,
                'keywords': event.keywords
            }
        
        logger.info(f"Updated impacts for {len(active_events)} active events")
    
    def get_temporal_adaptation(self, text: str, base_prediction: float, region: str) -> float:
        """Apply temporal adaptation based on current events."""
        if not self.event_impacts:
            return base_prediction
        
        text_lower = text.lower()
        total_adjustment = 0.0
        
        for event_id, impact_data in self.event_impacts.items():
            event = impact_data['event']
            current_impact = impact_data['current_impact']
            
            # Check if this event affects the current region
            if region not in event.affected_regions:
                continue
            
            # Check for keyword matches
            keyword_matches = sum(1 for keyword in event.keywords if keyword.lower() in text_lower)
            if keyword_matches == 0:
                continue
            
            # Calculate adjustment based on event type and keyword matches
            keyword_factor = min(1.0, keyword_matches / len(event.keywords))
            
            if event.event_type in ['crisis', 'political']:
                # During crises or political events, be more sensitive
                adjustment = current_impact * keyword_factor * 0.15
            elif event.event_type in ['cultural', 'social']:
                # During cultural/social events, moderate adjustment
                adjustment = current_impact * keyword_factor * 0.1
            else:
                adjustment = current_impact * keyword_factor * 0.05
            
            total_adjustment += adjustment
        
        adapted_prediction = base_prediction + total_adjustment
        adapted_prediction = max(0.0, min(1.0, adapted_prediction))  # Clamp to [0, 1]
        
        if total_adjustment != 0:
            logger.debug(f"Temporal adaptation: {base_prediction:.3f} -> {adapted_prediction:.3f}")
        
        return adapted_prediction

class OrganizationalPolicyEngine:
    """Handles organizational policy adaptation."""
    
    def __init__(self, db: ContextualDatabase):
        self.db = db
        self.policies = {}
        logger.info("Organizational policy engine initialized")
    
    def register_policy(self, policy: OrganizationalPolicy):
        """Register an organizational policy."""
        key = f"{policy.organization}_{policy.platform}"
        self.policies[key] = policy
        
        # Store in database
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO policies 
            (policy_id, organization, platform, policy_type, strictness_level, specific_rules, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            policy.policy_id,
            policy.organization,
            policy.platform,
            policy.policy_type,
            policy.strictness_level,
            json.dumps(policy.specific_rules),
            policy.last_updated.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Registered policy: {policy.organization} - {policy.platform}")
    
    def get_policy_adaptation(self, organization: str, platform: str, base_prediction: float) -> float:
        """Apply organizational policy adaptation."""
        key = f"{organization}_{platform}"
        if key not in self.policies:
            return base_prediction
        
        policy = self.policies[key]
        
        # Apply strictness adjustment
        strictness_adjustment = (policy.strictness_level - 0.5) * 0.3
        
        # Apply specific rule adjustments
        rule_adjustment = 0.0
        for rule_type, rule_value in policy.specific_rules.items():
            if rule_type == 'zero_tolerance':
                rule_adjustment += 0.2 if rule_value else 0.0
            elif rule_type == 'context_sensitive':
                rule_adjustment -= 0.1 if rule_value else 0.0
            elif rule_type == 'user_reputation_factor':
                rule_adjustment += rule_value * 0.1
        
        adapted_prediction = base_prediction + strictness_adjustment + rule_adjustment
        adapted_prediction = max(0.0, min(1.0, adapted_prediction))  # Clamp to [0, 1]
        
        logger.debug(f"Policy adaptation for {key}: {base_prediction:.3f} -> {adapted_prediction:.3f}")
        return adapted_prediction

class ContextualAdaptationPipeline:
    """Main pipeline that coordinates all contextual adaptation mechanisms."""
    
    def __init__(self, db_path: str = "contextual_data.db"):
        self.db = ContextualDatabase(db_path)
        self.event_generator = EventEmbeddingGenerator()
        self.regional_engine = RegionalAdaptationEngine()
        self.temporal_engine = TemporalAdaptationEngine(self.db)
        self.policy_engine = OrganizationalPolicyEngine(self.db)
        
        # Initialize with sample data
        self._initialize_sample_data()
        
        logger.info("Contextual adaptation pipeline initialized")
    
    def _initialize_sample_data(self):
        """Initialize with sample contextual data."""
        # Sample events
        events = [
            ContextualEvent(
                event_id="election_2024",
                name="2024 Presidential Election",
                description="Major political election with high polarization",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                event_type="political",
                impact_level=0.9,
                affected_regions=["North America", "Global"],
                keywords=["election", "vote", "candidate", "democracy", "politics"]
            ),
            ContextualEvent(
                event_id="tech_conference_2024",
                name="Global Tech Conference",
                description="Major technology conference discussing AI and ethics",
                start_date=datetime(2024, 3, 15),
                end_date=datetime(2024, 3, 18),
                event_type="cultural",
                impact_level=0.4,
                affected_regions=["North America", "Europe", "Asia"],
                keywords=["technology", "AI", "innovation", "ethics", "future"]
            )
        ]
        
        for event in events:
            event.embedding = self.event_generator.generate_embedding(event)
            self.db.store_event(event)
        
        # Sample regions
        regions = [
            RegionalContext(
                region_id="north_america",
                name="North America",
                cultural_factors={
                    "conservatism": 0.6,
                    "tolerance": 0.7,
                    "formality": 0.5
                },
                language_variants=["en-US", "en-CA", "es-MX"],
                sensitive_topics=["politics", "religion", "race"],
                moderation_strictness=0.7
            ),
            RegionalContext(
                region_id="europe",
                name="Europe",
                cultural_factors={
                    "conservatism": 0.4,
                    "tolerance": 0.8,
                    "formality": 0.7
                },
                language_variants=["en-GB", "de-DE", "fr-FR", "es-ES"],
                sensitive_topics=["politics", "immigration", "history"],
                moderation_strictness=0.8
            )
        ]
        
        for region in regions:
            self.regional_engine.register_region(region)
        
        # Sample policies
        policies = [
            OrganizationalPolicy(
                policy_id="twitter_standard",
                organization="Twitter",
                platform="twitter",
                policy_type="content",
                strictness_level=0.7,
                specific_rules={
                    "zero_tolerance": False,
                    "context_sensitive": True,
                    "user_reputation_factor": 0.2
                },
                last_updated=datetime(2024, 1, 1)
            ),
            OrganizationalPolicy(
                policy_id="facebook_family",
                organization="Facebook",
                platform="facebook",
                policy_type="content",
                strictness_level=0.8,
                specific_rules={
                    "zero_tolerance": True,
                    "context_sensitive": False,
                    "user_reputation_factor": 0.1
                },
                last_updated=datetime(2024, 1, 1)
            )
        ]
        
        for policy in policies:
            self.policy_engine.register_policy(policy)
    
    def adapt_prediction(
        self,
        text: str,
        base_prediction: float,
        region: str,
        organization: str,
        platform: str,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Apply all contextual adaptations to a base prediction.
        
        Args:
            text: Input text to analyze
            base_prediction: Base model prediction (0.0 to 1.0)
            region: Regional identifier
            organization: Organization identifier
            platform: Platform identifier
            timestamp: Timestamp for temporal adaptation
            
        Returns:
            Dictionary containing adapted prediction and adaptation details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update temporal impacts
        self.temporal_engine.update_event_impacts(timestamp)
        
        # Apply adaptations in sequence
        prediction = base_prediction
        adaptations = {
            "base_prediction": base_prediction,
            "regional_adaptation": None,
            "temporal_adaptation": None,
            "policy_adaptation": None,
            "final_prediction": None
        }
        
        # Regional adaptation
        regional_prediction = self.regional_engine.get_regional_adaptation(region, prediction)
        adaptations["regional_adaptation"] = {
            "input": prediction,
            "output": regional_prediction,
            "adjustment": regional_prediction - prediction
        }
        prediction = regional_prediction
        
        # Temporal adaptation
        temporal_prediction = self.temporal_engine.get_temporal_adaptation(text, prediction, region)
        adaptations["temporal_adaptation"] = {
            "input": prediction,
            "output": temporal_prediction,
            "adjustment": temporal_prediction - prediction
        }
        prediction = temporal_prediction
        
        # Policy adaptation
        policy_prediction = self.policy_engine.get_policy_adaptation(organization, platform, prediction)
        adaptations["policy_adaptation"] = {
            "input": prediction,
            "output": policy_prediction,
            "adjustment": policy_prediction - prediction
        }
        prediction = policy_prediction
        
        adaptations["final_prediction"] = prediction
        
        logger.info(f"Contextual adaptation: {base_prediction:.3f} -> {prediction:.3f}")
        
        return adaptations

def demonstrate_contextual_adaptation():
    """Demonstrate the contextual adaptation pipeline."""
    logger.info("Starting Contextual Adaptation Pipeline Demonstration")
    
    # Initialize pipeline
    pipeline = ContextualAdaptationPipeline()
    
    # Test cases
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
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['description']} ---")
        print(f"Text: {test_case['text']}")
        print(f"Base Prediction: {test_case['base_prediction']:.3f}")
        
        adaptations = pipeline.adapt_prediction(
            text=test_case['text'],
            base_prediction=test_case['base_prediction'],
            region=test_case['region'],
            organization=test_case['organization'],
            platform=test_case['platform']
        )
        
        print(f"Regional Adjustment: {adaptations['regional_adaptation']['adjustment']:+.3f}")
        print(f"Temporal Adjustment: {adaptations['temporal_adaptation']['adjustment']:+.3f}")
        print(f"Policy Adjustment: {adaptations['policy_adaptation']['adjustment']:+.3f}")
        print(f"Final Prediction: {adaptations['final_prediction']:.3f}")
        
        results.append({
            "test_case": test_case,
            "adaptations": adaptations
        })
    
    # Save results
    with open('contextual_adaptation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Contextual adaptation demonstration completed!")
    logger.info("Results saved to contextual_adaptation_results.json")

if __name__ == "__main__":
    demonstrate_contextual_adaptation()

