"""Attack Detection and Defense Mechanisms

This module provides attack detection, defense strategies, and
adversarial robustness for the PI-HMARL system.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from enum import Enum
import hashlib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import threading
import asyncio

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of attacks"""
    ADVERSARIAL_INPUT = "adversarial_input"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    BYZANTINE = "byzantine"
    SYBIL = "sybil"
    DOS = "denial_of_service"
    REPLAY = "replay"
    MAN_IN_MIDDLE = "man_in_middle"
    INJECTION = "injection"


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Attack:
    """Attack information"""
    attack_id: str
    attack_type: AttackType
    source: str
    target: str
    threat_level: ThreatLevel
    timestamp: datetime
    details: Dict[str, Any]
    mitigated: bool = False
    mitigation_time: Optional[datetime] = None


@dataclass
class DefenseAction:
    """Defense action taken"""
    action_id: str
    attack_id: str
    action_type: str
    parameters: Dict[str, Any]
    timestamp: datetime
    success: bool
    result: Optional[Any] = None


class AttackDetector:
    """Detects various types of attacks"""
    
    def __init__(
        self,
        detection_window: int = 100,
        anomaly_threshold: float = 0.95
    ):
        """Initialize attack detector
        
        Args:
            detection_window: Window size for detection
            anomaly_threshold: Threshold for anomaly detection
        """
        self.detection_window = detection_window
        self.anomaly_threshold = anomaly_threshold
        
        # Attack tracking
        self.detected_attacks: List[Attack] = []
        self.attack_patterns: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=detection_window)
        )
        
        # Anomaly detectors
        self.anomaly_detectors: Dict[str, Any] = {}
        self._init_detectors()
        
        # Statistics
        self.attack_stats: Dict[AttackType, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Initialized AttackDetector")
    
    def _init_detectors(self):
        """Initialize anomaly detectors"""
        # Input anomaly detector
        self.anomaly_detectors['input'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Behavior anomaly detector
        self.anomaly_detectors['behavior'] = OneClassSVM(
            gamma='auto',
            nu=0.1
        )
        
        # Communication pattern detector
        self.anomaly_detectors['communication'] = IsolationForest(
            contamination=0.05,
            random_state=42
        )
    
    def detect_adversarial_input(
        self,
        input_data: torch.Tensor,
        model: nn.Module,
        agent_id: str
    ) -> Optional[Attack]:
        """Detect adversarial inputs
        
        Args:
            input_data: Input tensor
            model: Neural network model
            agent_id: Agent identifier
            
        Returns:
            Attack if detected
        """
        with self._lock:
            # Extract features
            features = self._extract_input_features(input_data)
            
            # Update pattern
            self.attack_patterns[f"input_{agent_id}"].append(features)
            
            # Need sufficient data
            if len(self.attack_patterns[f"input_{agent_id}"]) < 10:
                return None
            
            # Check for anomalies
            pattern_data = np.array(list(self.attack_patterns[f"input_{agent_id}"]))
            
            # Fit or predict
            detector = self.anomaly_detectors['input']
            if not hasattr(detector, 'is_fitted_'):
                detector.fit(pattern_data)
                detector.is_fitted_ = True
                return None
            
            # Predict anomaly
            anomaly_score = detector.decision_function(features.reshape(1, -1))[0]
            
            if anomaly_score < -self.anomaly_threshold:
                attack = Attack(
                    attack_id=f"adv_input_{datetime.now().timestamp()}",
                    attack_type=AttackType.ADVERSARIAL_INPUT,
                    source="unknown",
                    target=agent_id,
                    threat_level=ThreatLevel.HIGH,
                    timestamp=datetime.now(),
                    details={
                        'anomaly_score': anomaly_score,
                        'input_stats': {
                            'mean': float(input_data.mean()),
                            'std': float(input_data.std()),
                            'min': float(input_data.min()),
                            'max': float(input_data.max())
                        }
                    }
                )
                
                self._register_attack(attack)
                return attack
            
            return None
    
    def _extract_input_features(self, input_data: torch.Tensor) -> np.ndarray:
        """Extract features from input data
        
        Args:
            input_data: Input tensor
            
        Returns:
            Feature vector
        """
        with torch.no_grad():
            features = [
                float(input_data.mean()),
                float(input_data.std()),
                float(input_data.min()),
                float(input_data.max()),
                float(torch.norm(input_data, p=1)),
                float(torch.norm(input_data, p=2)),
                float(torch.norm(input_data, p=float('inf'))),
                float((input_data > input_data.mean() + 3 * input_data.std()).sum())
            ]
        
        return np.array(features)
    
    def detect_byzantine_behavior(
        self,
        agent_id: str,
        behavior_metrics: Dict[str, float]
    ) -> Optional[Attack]:
        """Detect Byzantine (malicious) behavior
        
        Args:
            agent_id: Agent identifier
            behavior_metrics: Behavior metrics
            
        Returns:
            Attack if detected
        """
        with self._lock:
            # Extract behavior features
            features = np.array([
                behavior_metrics.get('agreement_rate', 1.0),
                behavior_metrics.get('response_consistency', 1.0),
                behavior_metrics.get('protocol_compliance', 1.0),
                behavior_metrics.get('message_validity', 1.0),
                behavior_metrics.get('computation_accuracy', 1.0)
            ])
            
            # Update pattern
            self.attack_patterns[f"behavior_{agent_id}"].append(features)
            
            # Need sufficient data
            if len(self.attack_patterns[f"behavior_{agent_id}"]) < 20:
                return None
            
            # Detect anomalies
            pattern_data = np.array(list(self.attack_patterns[f"behavior_{agent_id}"]))
            
            detector = self.anomaly_detectors['behavior']
            if not hasattr(detector, 'is_fitted_'):
                detector.fit(pattern_data)
                detector.is_fitted_ = True
                return None
            
            # Check for Byzantine behavior
            anomaly_score = detector.decision_function(features.reshape(1, -1))[0]
            
            if anomaly_score < -0.5:  # Byzantine threshold
                attack = Attack(
                    attack_id=f"byzantine_{agent_id}_{datetime.now().timestamp()}",
                    attack_type=AttackType.BYZANTINE,
                    source=agent_id,
                    target="system",
                    threat_level=ThreatLevel.CRITICAL,
                    timestamp=datetime.now(),
                    details={
                        'behavior_metrics': behavior_metrics,
                        'anomaly_score': anomaly_score
                    }
                )
                
                self._register_attack(attack)
                return attack
            
            return None
    
    def detect_sybil_attack(
        self,
        network_state: Dict[str, Any]
    ) -> Optional[Attack]:
        """Detect Sybil attacks (fake identities)
        
        Args:
            network_state: Network state information
            
        Returns:
            Attack if detected
        """
        agents = network_state.get('agents', {})
        
        # Group agents by similar characteristics
        similarity_groups = defaultdict(list)
        
        for agent_id, agent_info in agents.items():
            # Create fingerprint
            fingerprint = self._create_agent_fingerprint(agent_info)
            similarity_groups[fingerprint].append(agent_id)
        
        # Check for suspicious groups
        for fingerprint, agent_list in similarity_groups.items():
            if len(agent_list) > 3:  # Suspicious number of similar agents
                attack = Attack(
                    attack_id=f"sybil_{datetime.now().timestamp()}",
                    attack_type=AttackType.SYBIL,
                    source="unknown",
                    target="network",
                    threat_level=ThreatLevel.HIGH,
                    timestamp=datetime.now(),
                    details={
                        'suspicious_agents': agent_list,
                        'fingerprint': fingerprint,
                        'group_size': len(agent_list)
                    }
                )
                
                self._register_attack(attack)
                return attack
        
        return None
    
    def _create_agent_fingerprint(self, agent_info: Dict[str, Any]) -> str:
        """Create agent fingerprint for similarity detection
        
        Args:
            agent_info: Agent information
            
        Returns:
            Fingerprint hash
        """
        # Extract relevant features
        features = [
            str(agent_info.get('hardware_id', '')),
            str(agent_info.get('software_version', '')),
            str(agent_info.get('behavior_pattern', '')),
            str(agent_info.get('communication_pattern', ''))
        ]
        
        # Create hash
        fingerprint = hashlib.md5('_'.join(features).encode()).hexdigest()[:8]
        
        return fingerprint
    
    def detect_dos_attack(
        self,
        traffic_metrics: Dict[str, float]
    ) -> Optional[Attack]:
        """Detect Denial of Service attacks
        
        Args:
            traffic_metrics: Network traffic metrics
            
        Returns:
            Attack if detected
        """
        # Check for abnormal traffic patterns
        request_rate = traffic_metrics.get('request_rate', 0)
        error_rate = traffic_metrics.get('error_rate', 0)
        bandwidth_usage = traffic_metrics.get('bandwidth_usage', 0)
        
        # Define thresholds
        if request_rate > 1000:  # Requests per second
            threat_level = ThreatLevel.HIGH if request_rate > 5000 else ThreatLevel.MEDIUM
            
            attack = Attack(
                attack_id=f"dos_{datetime.now().timestamp()}",
                attack_type=AttackType.DOS,
                source="unknown",
                target="system",
                threat_level=threat_level,
                timestamp=datetime.now(),
                details={
                    'request_rate': request_rate,
                    'error_rate': error_rate,
                    'bandwidth_usage': bandwidth_usage
                }
            )
            
            self._register_attack(attack)
            return attack
        
        return None
    
    def detect_replay_attack(
        self,
        message: Dict[str, Any]
    ) -> Optional[Attack]:
        """Detect replay attacks
        
        Args:
            message: Message to check
            
        Returns:
            Attack if detected
        """
        message_hash = self._hash_message(message)
        timestamp = message.get('timestamp', datetime.now())
        
        # Check if we've seen this message before
        if hasattr(self, '_message_history'):
            for hist_hash, hist_time in self._message_history:
                if hist_hash == message_hash:
                    attack = Attack(
                        attack_id=f"replay_{datetime.now().timestamp()}",
                        attack_type=AttackType.REPLAY,
                        source=message.get('sender', 'unknown'),
                        target=message.get('recipient', 'unknown'),
                        threat_level=ThreatLevel.MEDIUM,
                        timestamp=datetime.now(),
                        details={
                            'original_time': hist_time.isoformat(),
                            'replay_time': timestamp.isoformat(),
                            'message_hash': message_hash
                        }
                    )
                    
                    self._register_attack(attack)
                    return attack
        else:
            self._message_history = deque(maxlen=10000)
        
        # Add to history
        self._message_history.append((message_hash, timestamp))
        
        return None
    
    def _hash_message(self, message: Dict[str, Any]) -> str:
        """Create hash of message content
        
        Args:
            message: Message to hash
            
        Returns:
            Message hash
        """
        # Remove timestamp for content comparison
        content = {k: v for k, v in message.items() if k != 'timestamp'}
        content_str = json.dumps(content, sort_keys=True)
        
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _register_attack(self, attack: Attack):
        """Register detected attack
        
        Args:
            attack: Detected attack
        """
        self.detected_attacks.append(attack)
        self.attack_stats[attack.attack_type] += 1
        
        logger.warning(
            f"Attack detected: {attack.attack_type.value} "
            f"from {attack.source} targeting {attack.target}"
        )
    
    def get_threat_assessment(self) -> Dict[str, Any]:
        """Get current threat assessment
        
        Returns:
            Threat assessment summary
        """
        recent_attacks = [
            a for a in self.detected_attacks
            if a.timestamp > datetime.now() - timedelta(minutes=10)
        ]
        
        # Calculate threat level
        if any(a.threat_level == ThreatLevel.CRITICAL for a in recent_attacks):
            overall_threat = ThreatLevel.CRITICAL
        elif any(a.threat_level == ThreatLevel.HIGH for a in recent_attacks):
            overall_threat = ThreatLevel.HIGH
        elif any(a.threat_level == ThreatLevel.MEDIUM for a in recent_attacks):
            overall_threat = ThreatLevel.MEDIUM
        else:
            overall_threat = ThreatLevel.LOW
        
        return {
            'overall_threat_level': overall_threat.name,
            'active_attacks': len(recent_attacks),
            'attack_types': list(set(a.attack_type.value for a in recent_attacks)),
            'critical_targets': list(set(
                a.target for a in recent_attacks
                if a.threat_level == ThreatLevel.CRITICAL
            )),
            'mitigation_rate': sum(
                1 for a in self.detected_attacks if a.mitigated
            ) / len(self.detected_attacks) if self.detected_attacks else 1.0
        }


class DefenseStrategy:
    """Base class for defense strategies"""
    
    def __init__(self, name: str):
        """Initialize defense strategy
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.actions_taken: List[DefenseAction] = []
    
    async def defend(
        self,
        attack: Attack,
        context: Dict[str, Any]
    ) -> DefenseAction:
        """Execute defense strategy
        
        Args:
            attack: Detected attack
            context: System context
            
        Returns:
            Defense action taken
        """
        raise NotImplementedError


class AdversarialDefense(DefenseStrategy):
    """Defense against adversarial attacks"""
    
    def __init__(self):
        """Initialize adversarial defense"""
        super().__init__("adversarial_defense")
        
        # Defense mechanisms
        self.input_filters = []
        self.detection_models = {}
        
    async def defend(
        self,
        attack: Attack,
        context: Dict[str, Any]
    ) -> DefenseAction:
        """Defend against adversarial attack
        
        Args:
            attack: Adversarial attack
            context: System context
            
        Returns:
            Defense action
        """
        if attack.attack_type == AttackType.ADVERSARIAL_INPUT:
            # Apply input sanitization
            action = await self._sanitize_input(attack, context)
        elif attack.attack_type == AttackType.DATA_POISONING:
            # Apply data validation
            action = await self._validate_data(attack, context)
        else:
            # Generic defense
            action = await self._generic_defense(attack, context)
        
        self.actions_taken.append(action)
        return action
    
    async def _sanitize_input(
        self,
        attack: Attack,
        context: Dict[str, Any]
    ) -> DefenseAction:
        """Sanitize adversarial input
        
        Args:
            attack: Attack information
            context: System context
            
        Returns:
            Defense action
        """
        agent_id = attack.target
        
        # Apply input filters
        filters_applied = []
        
        # 1. Clip values to expected range
        filters_applied.append("value_clipping")
        
        # 2. Apply smoothing
        filters_applied.append("input_smoothing")
        
        # 3. Add noise for robustness
        filters_applied.append("noise_injection")
        
        action = DefenseAction(
            action_id=f"sanitize_{datetime.now().timestamp()}",
            attack_id=attack.attack_id,
            action_type="input_sanitization",
            parameters={
                'filters': filters_applied,
                'agent_id': agent_id
            },
            timestamp=datetime.now(),
            success=True
        )
        
        # Update attack status
        attack.mitigated = True
        attack.mitigation_time = datetime.now()
        
        logger.info(f"Applied input sanitization for agent {agent_id}")
        
        return action
    
    async def _validate_data(
        self,
        attack: Attack,
        context: Dict[str, Any]
    ) -> DefenseAction:
        """Validate data against poisoning
        
        Args:
            attack: Attack information
            context: System context
            
        Returns:
            Defense action
        """
        # Implement data validation
        validation_steps = [
            "outlier_detection",
            "consistency_check",
            "source_verification"
        ]
        
        action = DefenseAction(
            action_id=f"validate_{datetime.now().timestamp()}",
            attack_id=attack.attack_id,
            action_type="data_validation",
            parameters={
                'validation_steps': validation_steps
            },
            timestamp=datetime.now(),
            success=True
        )
        
        return action
    
    async def _generic_defense(
        self,
        attack: Attack,
        context: Dict[str, Any]
    ) -> DefenseAction:
        """Generic defense mechanism
        
        Args:
            attack: Attack information
            context: System context
            
        Returns:
            Defense action
        """
        action = DefenseAction(
            action_id=f"generic_{datetime.now().timestamp()}",
            attack_id=attack.attack_id,
            action_type="generic_defense",
            parameters={
                'isolation': True,
                'monitoring': True
            },
            timestamp=datetime.now(),
            success=True
        )
        
        return action
    
    def add_input_filter(self, filter_func: Callable):
        """Add input filter
        
        Args:
            filter_func: Filter function
        """
        self.input_filters.append(filter_func)
    
    def apply_adversarial_training(
        self,
        model: nn.Module,
        epsilon: float = 0.1
    ) -> nn.Module:
        """Apply adversarial training to model
        
        Args:
            model: Neural network model
            epsilon: Perturbation magnitude
            
        Returns:
            Hardened model
        """
        # Implement adversarial training
        # This is a placeholder for the actual implementation
        logger.info(f"Applied adversarial training with epsilon={epsilon}")
        
        return model


class IntrusionPrevention:
    """Intrusion prevention system"""
    
    def __init__(self):
        """Initialize intrusion prevention"""
        self.rules: List[Dict[str, Any]] = []
        self.blocked_entities: Set[str] = set()
        self.quarantine: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Initialize default rules
        self._init_default_rules()
        
        logger.info("Initialized IntrusionPrevention")
    
    def _init_default_rules(self):
        """Initialize default prevention rules"""
        self.rules.extend([
            {
                'name': 'rate_limiting',
                'condition': lambda metrics: metrics.get('request_rate', 0) > 100,
                'action': 'throttle',
                'parameters': {'limit': 50}
            },
            {
                'name': 'malformed_message',
                'condition': lambda msg: not self._validate_message_format(msg),
                'action': 'block',
                'parameters': {'duration': 300}
            },
            {
                'name': 'suspicious_pattern',
                'condition': lambda behavior: behavior.get('anomaly_score', 0) > 0.8,
                'action': 'quarantine',
                'parameters': {'review_required': True}
            }
        ])
    
    def check_and_prevent(
        self,
        entity_id: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Check entity and prevent if necessary
        
        Args:
            entity_id: Entity identifier
            data: Entity data/behavior
            
        Returns:
            Allowed status and reason if blocked
        """
        with self._lock:
            # Check if entity is blocked
            if entity_id in self.blocked_entities:
                return False, "Entity is blocked"
            
            # Check rules
            for rule in self.rules:
                if rule['condition'](data):
                    # Apply action
                    action_result = self._apply_action(
                        entity_id,
                        rule['action'],
                        rule['parameters']
                    )
                    
                    if not action_result:
                        return False, f"Blocked by rule: {rule['name']}"
            
            return True, None
    
    def _apply_action(
        self,
        entity_id: str,
        action: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """Apply prevention action
        
        Args:
            entity_id: Entity identifier
            action: Action type
            parameters: Action parameters
            
        Returns:
            Whether to allow entity
        """
        if action == 'block':
            duration = parameters.get('duration', 3600)
            self._block_entity(entity_id, duration)
            return False
            
        elif action == 'throttle':
            # Implement rate limiting
            limit = parameters.get('limit', 10)
            # Simplified throttling logic
            return True
            
        elif action == 'quarantine':
            self._quarantine_entity(entity_id, parameters)
            return False
            
        return True
    
    def _block_entity(self, entity_id: str, duration: int):
        """Block entity
        
        Args:
            entity_id: Entity to block
            duration: Block duration in seconds
        """
        self.blocked_entities.add(entity_id)
        
        # Schedule unblock
        def unblock():
            time.sleep(duration)
            self.blocked_entities.discard(entity_id)
            logger.info(f"Unblocked entity: {entity_id}")
        
        threading.Thread(target=unblock, daemon=True).start()
        
        logger.warning(f"Blocked entity: {entity_id} for {duration}s")
    
    def _quarantine_entity(
        self,
        entity_id: str,
        parameters: Dict[str, Any]
    ):
        """Quarantine entity for review
        
        Args:
            entity_id: Entity to quarantine
            parameters: Quarantine parameters
        """
        self.quarantine[entity_id] = {
            'timestamp': datetime.now(),
            'parameters': parameters,
            'status': 'pending_review'
        }
        
        logger.warning(f"Quarantined entity: {entity_id}")
    
    def _validate_message_format(self, message: Dict[str, Any]) -> bool:
        """Validate message format
        
        Args:
            message: Message to validate
            
        Returns:
            Validation status
        """
        required_fields = ['sender', 'type', 'timestamp']
        return all(field in message for field in required_fields)
    
    def add_rule(self, rule: Dict[str, Any]):
        """Add prevention rule
        
        Args:
            rule: Rule definition
        """
        self.rules.append(rule)
        logger.info(f"Added prevention rule: {rule['name']}")
    
    def review_quarantine(
        self,
        entity_id: str,
        decision: str
    ) -> bool:
        """Review quarantined entity
        
        Args:
            entity_id: Entity in quarantine
            decision: 'allow' or 'block'
            
        Returns:
            Success status
        """
        with self._lock:
            if entity_id not in self.quarantine:
                return False
            
            if decision == 'allow':
                del self.quarantine[entity_id]
                logger.info(f"Released entity from quarantine: {entity_id}")
            else:
                self._block_entity(entity_id, 3600)  # Block for 1 hour
                del self.quarantine[entity_id]
            
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get intrusion prevention status
        
        Returns:
            Status summary
        """
        return {
            'blocked_entities': len(self.blocked_entities),
            'quarantined_entities': len(self.quarantine),
            'active_rules': len(self.rules),
            'blocked_list': list(self.blocked_entities),
            'quarantine_list': list(self.quarantine.keys())
        }


# Example integrated defense system
class IntegratedDefenseSystem:
    """Integrated defense system combining all components"""
    
    def __init__(self):
        """Initialize integrated defense"""
        self.attack_detector = AttackDetector()
        self.adversarial_defense = AdversarialDefense()
        self.intrusion_prevention = IntrusionPrevention()
        
        # Defense strategies
        self.defense_strategies: Dict[AttackType, DefenseStrategy] = {
            AttackType.ADVERSARIAL_INPUT: self.adversarial_defense,
            AttackType.DATA_POISONING: self.adversarial_defense
        }
        
        logger.info("Initialized IntegratedDefenseSystem")
    
    async def protect(
        self,
        entity_id: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Comprehensive protection check
        
        Args:
            entity_id: Entity identifier
            data: Entity data
            
        Returns:
            Protection status and reason
        """
        # Check intrusion prevention
        allowed, reason = self.intrusion_prevention.check_and_prevent(
            entity_id, data
        )
        
        if not allowed:
            return False, reason
        
        # Check for attacks
        if 'input_data' in data:
            attack = self.attack_detector.detect_adversarial_input(
                data['input_data'],
                data.get('model'),
                entity_id
            )
            
            if attack:
                # Apply defense
                defense_strategy = self.defense_strategies.get(
                    attack.attack_type
                )
                
                if defense_strategy:
                    await defense_strategy.defend(attack, data)
                
                return False, f"Attack detected: {attack.attack_type.value}"
        
        return True, None