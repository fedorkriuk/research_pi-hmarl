"""Consensus Protocols for Multi-Agent Systems

This module implements various consensus protocols for distributed
decision-making and coordination.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class ConsensusType(Enum):
    """Types of consensus protocols"""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    BYZANTINE = "byzantine"
    RAFT = "raft"
    PBFT = "pbft"  # Practical Byzantine Fault Tolerance
    FEDERATED = "federated"


class VoteType(Enum):
    """Types of votes"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class Vote:
    """Vote in consensus protocol"""
    voter_id: int
    proposal_id: str
    vote_type: VoteType
    weight: float = 1.0
    timestamp: float = 0.0
    signature: Optional[str] = None
    justification: Optional[str] = None


@dataclass
class Proposal:
    """Proposal for consensus"""
    proposal_id: str
    proposer_id: int
    content: Dict[str, Any]
    timestamp: float
    deadline: float
    min_votes: int
    consensus_type: ConsensusType
    metadata: Dict[str, Any] = None


@dataclass
class ConsensusState:
    """Current consensus state"""
    proposal: Proposal
    votes: Dict[int, Vote]
    start_time: float
    status: str  # 'voting', 'accepted', 'rejected', 'timeout'
    result: Optional[Dict[str, Any]] = None
    round: int = 0


class ConsensusProtocol:
    """Main consensus protocol implementation"""
    
    def __init__(
        self,
        agent_id: int,
        total_agents: int,
        consensus_type: ConsensusType = ConsensusType.MAJORITY,
        byzantine_tolerance: float = 0.33
    ):
        """Initialize consensus protocol
        
        Args:
            agent_id: Agent identifier
            total_agents: Total number of agents
            consensus_type: Type of consensus
            byzantine_tolerance: Byzantine fault tolerance ratio
        """
        self.agent_id = agent_id
        self.total_agents = total_agents
        self.consensus_type = consensus_type
        self.byzantine_tolerance = byzantine_tolerance
        
        # Consensus tracking
        self.active_proposals: Dict[str, ConsensusState] = {}
        self.proposal_history = deque(maxlen=1000)
        self.vote_history = deque(maxlen=10000)
        
        # Protocol-specific components
        self.voting_mechanism = VotingMechanism(total_agents)
        self.byzantine_handler = ByzantineFaultTolerance(byzantine_tolerance)
        
        # Leadership (for Raft-like protocols)
        self.current_leader = None
        self.leader_election = LeaderElection(agent_id, total_agents)
        
        # Reputation tracking
        self.agent_reputation = defaultdict(lambda: 1.0)
        
        logger.info(f"Initialized {consensus_type.value} consensus for agent {agent_id}")
    
    def propose(
        self,
        content: Dict[str, Any],
        min_votes: Optional[int] = None,
        deadline: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new proposal
        
        Args:
            content: Proposal content
            min_votes: Minimum votes required
            deadline: Time limit for voting
            metadata: Additional metadata
            
        Returns:
            Proposal ID
        """
        # Generate proposal ID
        proposal_id = self._generate_proposal_id(content)
        
        # Default minimum votes
        if min_votes is None:
            if self.consensus_type == ConsensusType.BYZANTINE:
                min_votes = int(self.total_agents * (2/3)) + 1
            else:
                min_votes = int(self.total_agents / 2) + 1
        
        # Create proposal
        proposal = Proposal(
            proposal_id=proposal_id,
            proposer_id=self.agent_id,
            content=content,
            timestamp=time.time(),
            deadline=deadline,
            min_votes=min_votes,
            consensus_type=self.consensus_type,
            metadata=metadata
        )
        
        # Initialize consensus state
        consensus_state = ConsensusState(
            proposal=proposal,
            votes={},
            start_time=time.time(),
            status='voting'
        )
        
        self.active_proposals[proposal_id] = consensus_state
        
        logger.info(f"Created proposal {proposal_id}")
        return proposal_id
    
    def vote(
        self,
        proposal_id: str,
        vote_type: VoteType,
        justification: Optional[str] = None
    ) -> bool:
        """Cast vote on proposal
        
        Args:
            proposal_id: Proposal ID
            vote_type: Type of vote
            justification: Vote justification
            
        Returns:
            Success status
        """
        if proposal_id not in self.active_proposals:
            logger.warning(f"Proposal {proposal_id} not found")
            return False
        
        consensus_state = self.active_proposals[proposal_id]
        
        # Check if voting is still open
        if consensus_state.status != 'voting':
            logger.warning(f"Voting closed for proposal {proposal_id}")
            return False
        
        # Check deadline
        if time.time() > consensus_state.proposal.timestamp + consensus_state.proposal.deadline:
            consensus_state.status = 'timeout'
            return False
        
        # Create vote
        vote = Vote(
            voter_id=self.agent_id,
            proposal_id=proposal_id,
            vote_type=vote_type,
            weight=self._calculate_vote_weight(),
            timestamp=time.time(),
            signature=self._sign_vote(proposal_id, vote_type),
            justification=justification
        )
        
        # Record vote
        consensus_state.votes[self.agent_id] = vote
        self.vote_history.append(vote)
        
        # Check if consensus reached
        self._check_consensus(proposal_id)
        
        return True
    
    def receive_vote(
        self,
        vote: Vote
    ) -> bool:
        """Receive vote from another agent
        
        Args:
            vote: Vote to process
            
        Returns:
            Success status
        """
        proposal_id = vote.proposal_id
        
        if proposal_id not in self.active_proposals:
            logger.warning(f"Received vote for unknown proposal {proposal_id}")
            return False
        
        consensus_state = self.active_proposals[proposal_id]
        
        # Verify vote
        if not self._verify_vote(vote):
            logger.warning(f"Invalid vote from agent {vote.voter_id}")
            self._update_reputation(vote.voter_id, -0.1)
            return False
        
        # Check for Byzantine behavior
        if self.byzantine_handler.is_byzantine_vote(vote, consensus_state):
            logger.warning(f"Byzantine behavior detected from agent {vote.voter_id}")
            self._update_reputation(vote.voter_id, -0.5)
            return False
        
        # Record vote
        consensus_state.votes[vote.voter_id] = vote
        
        # Update reputation positively
        self._update_reputation(vote.voter_id, 0.01)
        
        # Check consensus
        self._check_consensus(proposal_id)
        
        return True
    
    def _check_consensus(self, proposal_id: str):
        """Check if consensus has been reached
        
        Args:
            proposal_id: Proposal ID
        """
        consensus_state = self.active_proposals[proposal_id]
        
        if consensus_state.status != 'voting':
            return
        
        # Apply appropriate consensus mechanism
        if self.consensus_type == ConsensusType.MAJORITY:
            result = self.voting_mechanism.check_majority(consensus_state)
        elif self.consensus_type == ConsensusType.WEIGHTED:
            result = self.voting_mechanism.check_weighted(consensus_state)
        elif self.consensus_type == ConsensusType.BYZANTINE:
            result = self.byzantine_handler.check_byzantine_consensus(consensus_state)
        elif self.consensus_type == ConsensusType.RAFT:
            result = self._check_raft_consensus(consensus_state)
        else:
            result = self.voting_mechanism.check_majority(consensus_state)
        
        if result['consensus_reached']:
            consensus_state.status = 'accepted' if result['approved'] else 'rejected'
            consensus_state.result = result
            
            # Move to history
            self.proposal_history.append(consensus_state)
            
            logger.info(f"Consensus reached for proposal {proposal_id}: {consensus_state.status}")
    
    def _check_raft_consensus(self, consensus_state: ConsensusState) -> Dict[str, Any]:
        """Check consensus using Raft protocol
        
        Args:
            consensus_state: Current consensus state
            
        Returns:
            Consensus result
        """
        # Simplified Raft consensus
        # Only leader's proposal can be accepted
        if consensus_state.proposal.proposer_id != self.current_leader:
            return {'consensus_reached': False, 'approved': False}
        
        # Check majority of votes
        return self.voting_mechanism.check_majority(consensus_state)
    
    def _generate_proposal_id(self, content: Dict[str, Any]) -> str:
        """Generate unique proposal ID
        
        Args:
            content: Proposal content
            
        Returns:
            Proposal ID
        """
        id_string = f"{self.agent_id}_{time.time()}_{str(content)}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]
    
    def _calculate_vote_weight(self) -> float:
        """Calculate vote weight based on reputation
        
        Returns:
            Vote weight
        """
        base_weight = 1.0
        reputation_bonus = self.agent_reputation[self.agent_id] - 1.0
        return base_weight + reputation_bonus * 0.5
    
    def _sign_vote(self, proposal_id: str, vote_type: VoteType) -> str:
        """Sign vote for authentication
        
        Args:
            proposal_id: Proposal ID
            vote_type: Vote type
            
        Returns:
            Vote signature
        """
        sign_string = f"{self.agent_id}_{proposal_id}_{vote_type.value}_{time.time()}"
        return hashlib.sha256(sign_string.encode()).hexdigest()[:8]
    
    def _verify_vote(self, vote: Vote) -> bool:
        """Verify vote authenticity
        
        Args:
            vote: Vote to verify
            
        Returns:
            Verification status
        """
        # Simplified verification
        # In practice would use cryptographic signatures
        return vote.signature is not None and len(vote.signature) == 8
    
    def _update_reputation(self, agent_id: int, delta: float):
        """Update agent reputation
        
        Args:
            agent_id: Agent ID
            delta: Reputation change
        """
        self.agent_reputation[agent_id] = max(
            0.0,
            min(2.0, self.agent_reputation[agent_id] + delta)
        )
    
    def get_active_proposals(self) -> List[Proposal]:
        """Get list of active proposals
        
        Returns:
            Active proposals
        """
        return [
            state.proposal for state in self.active_proposals.values()
            if state.status == 'voting'
        ]
    
    def get_consensus_status(self, proposal_id: str) -> Optional[ConsensusState]:
        """Get consensus status for proposal
        
        Args:
            proposal_id: Proposal ID
            
        Returns:
            Consensus state or None
        """
        return self.active_proposals.get(proposal_id)


class VotingMechanism:
    """Implements various voting mechanisms"""
    
    def __init__(self, total_agents: int):
        """Initialize voting mechanism
        
        Args:
            total_agents: Total number of agents
        """
        self.total_agents = total_agents
    
    def check_majority(self, consensus_state: ConsensusState) -> Dict[str, Any]:
        """Check simple majority consensus
        
        Args:
            consensus_state: Current consensus state
            
        Returns:
            Consensus result
        """
        votes = consensus_state.votes
        proposal = consensus_state.proposal
        
        # Count votes
        approve_count = sum(1 for v in votes.values() if v.vote_type == VoteType.APPROVE)
        reject_count = sum(1 for v in votes.values() if v.vote_type == VoteType.REJECT)
        
        total_votes = len(votes)
        
        # Check if enough votes
        if total_votes < proposal.min_votes:
            return {'consensus_reached': False, 'approved': False}
        
        # Check majority
        if approve_count > self.total_agents / 2:
            return {
                'consensus_reached': True,
                'approved': True,
                'approve_count': approve_count,
                'reject_count': reject_count,
                'total_votes': total_votes
            }
        elif reject_count > self.total_agents / 2:
            return {
                'consensus_reached': True,
                'approved': False,
                'approve_count': approve_count,
                'reject_count': reject_count,
                'total_votes': total_votes
            }
        
        return {'consensus_reached': False, 'approved': False}
    
    def check_weighted(self, consensus_state: ConsensusState) -> Dict[str, Any]:
        """Check weighted voting consensus
        
        Args:
            consensus_state: Current consensus state
            
        Returns:
            Consensus result
        """
        votes = consensus_state.votes
        proposal = consensus_state.proposal
        
        # Calculate weighted votes
        approve_weight = sum(
            v.weight for v in votes.values() 
            if v.vote_type == VoteType.APPROVE
        )
        reject_weight = sum(
            v.weight for v in votes.values() 
            if v.vote_type == VoteType.REJECT
        )
        
        total_weight = sum(v.weight for v in votes.values())
        
        # Check if enough participation
        if len(votes) < proposal.min_votes:
            return {'consensus_reached': False, 'approved': False}
        
        # Check weighted majority
        if approve_weight > total_weight * 0.5:
            return {
                'consensus_reached': True,
                'approved': True,
                'approve_weight': approve_weight,
                'reject_weight': reject_weight,
                'total_weight': total_weight
            }
        elif reject_weight > total_weight * 0.5:
            return {
                'consensus_reached': True,
                'approved': False,
                'approve_weight': approve_weight,
                'reject_weight': reject_weight,
                'total_weight': total_weight
            }
        
        return {'consensus_reached': False, 'approved': False}


class ByzantineFaultTolerance:
    """Handles Byzantine fault tolerance"""
    
    def __init__(self, tolerance_ratio: float = 0.33):
        """Initialize Byzantine fault tolerance
        
        Args:
            tolerance_ratio: Maximum ratio of Byzantine agents
        """
        self.tolerance_ratio = tolerance_ratio
        self.suspicious_agents = set()
        self.byzantine_behaviors = defaultdict(list)
    
    def is_byzantine_vote(
        self,
        vote: Vote,
        consensus_state: ConsensusState
    ) -> bool:
        """Check if vote exhibits Byzantine behavior
        
        Args:
            vote: Vote to check
            consensus_state: Current consensus state
            
        Returns:
            Whether vote is Byzantine
        """
        voter_id = vote.voter_id
        
        # Check for double voting
        if voter_id in consensus_state.votes:
            existing_vote = consensus_state.votes[voter_id]
            if existing_vote.vote_type != vote.vote_type:
                self.byzantine_behaviors[voter_id].append({
                    'type': 'double_vote',
                    'timestamp': time.time()
                })
                return True
        
        # Check for invalid timestamp
        if vote.timestamp < consensus_state.proposal.timestamp:
            self.byzantine_behaviors[voter_id].append({
                'type': 'invalid_timestamp',
                'timestamp': time.time()
            })
            return True
        
        # Check for suspicious patterns
        if self._is_suspicious_pattern(voter_id):
            self.suspicious_agents.add(voter_id)
            return True
        
        return False
    
    def check_byzantine_consensus(
        self,
        consensus_state: ConsensusState
    ) -> Dict[str, Any]:
        """Check consensus with Byzantine fault tolerance
        
        Args:
            consensus_state: Current consensus state
            
        Returns:
            Consensus result
        """
        votes = consensus_state.votes
        total_agents = len(votes)
        
        # Filter out suspicious votes
        valid_votes = {
            agent_id: vote for agent_id, vote in votes.items()
            if agent_id not in self.suspicious_agents
        }
        
        # Count valid votes
        approve_count = sum(
            1 for v in valid_votes.values() 
            if v.vote_type == VoteType.APPROVE
        )
        reject_count = sum(
            1 for v in valid_votes.values() 
            if v.vote_type == VoteType.REJECT
        )
        
        # Byzantine consensus requires 2/3 + 1
        required_votes = int(total_agents * (2/3)) + 1
        
        if approve_count >= required_votes:
            return {
                'consensus_reached': True,
                'approved': True,
                'approve_count': approve_count,
                'reject_count': reject_count,
                'byzantine_count': len(self.suspicious_agents)
            }
        elif reject_count >= required_votes:
            return {
                'consensus_reached': True,
                'approved': False,
                'approve_count': approve_count,
                'reject_count': reject_count,
                'byzantine_count': len(self.suspicious_agents)
            }
        
        return {'consensus_reached': False, 'approved': False}
    
    def _is_suspicious_pattern(self, agent_id: int) -> bool:
        """Check for suspicious behavior patterns
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Whether behavior is suspicious
        """
        behaviors = self.byzantine_behaviors[agent_id]
        
        # Too many Byzantine behaviors
        if len(behaviors) > 3:
            return True
        
        # Rapid Byzantine behaviors
        if len(behaviors) >= 2:
            recent_behaviors = [
                b for b in behaviors 
                if time.time() - b['timestamp'] < 60
            ]
            if len(recent_behaviors) >= 2:
                return True
        
        return False


class LeaderElection:
    """Implements leader election protocols"""
    
    def __init__(self, agent_id: int, total_agents: int):
        """Initialize leader election
        
        Args:
            agent_id: Agent identifier
            total_agents: Total number of agents
        """
        self.agent_id = agent_id
        self.total_agents = total_agents
        
        # Election state
        self.current_term = 0
        self.voted_for = None
        self.current_leader = None
        self.election_timeout = np.random.uniform(150, 300) / 1000  # seconds
        self.last_heartbeat = time.time()
        
        # Candidate state
        self.votes_received = set()
        self.state = 'follower'  # 'follower', 'candidate', 'leader'
    
    def start_election(self) -> Tuple[int, int]:
        """Start leader election
        
        Returns:
            (term, candidate_id)
        """
        self.current_term += 1
        self.state = 'candidate'
        self.voted_for = self.agent_id
        self.votes_received = {self.agent_id}
        
        logger.info(f"Agent {self.agent_id} starting election for term {self.current_term}")
        
        return self.current_term, self.agent_id
    
    def receive_vote_request(
        self,
        term: int,
        candidate_id: int
    ) -> Tuple[int, bool]:
        """Handle vote request
        
        Args:
            term: Election term
            candidate_id: Candidate requesting vote
            
        Returns:
            (current_term, vote_granted)
        """
        # Update term if necessary
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.state = 'follower'
        
        # Grant vote if haven't voted or voted for same candidate
        vote_granted = (
            term == self.current_term and
            (self.voted_for is None or self.voted_for == candidate_id)
        )
        
        if vote_granted:
            self.voted_for = candidate_id
            self.last_heartbeat = time.time()
        
        return self.current_term, vote_granted
    
    def receive_vote(
        self,
        term: int,
        voter_id: int,
        vote_granted: bool
    ) -> Optional[int]:
        """Receive vote response
        
        Args:
            term: Election term
            voter_id: Voter ID
            vote_granted: Whether vote was granted
            
        Returns:
            Agent ID if elected leader, None otherwise
        """
        if term != self.current_term or self.state != 'candidate':
            return None
        
        if vote_granted:
            self.votes_received.add(voter_id)
            
            # Check if won election
            if len(self.votes_received) > self.total_agents / 2:
                self.state = 'leader'
                self.current_leader = self.agent_id
                logger.info(f"Agent {self.agent_id} elected leader for term {term}")
                return self.agent_id
        
        return None
    
    def receive_heartbeat(self, term: int, leader_id: int):
        """Receive leader heartbeat
        
        Args:
            term: Current term
            leader_id: Leader ID
        """
        if term >= self.current_term:
            self.current_term = term
            self.current_leader = leader_id
            self.state = 'follower'
            self.last_heartbeat = time.time()
    
    def check_election_timeout(self) -> bool:
        """Check if election timeout occurred
        
        Returns:
            Whether to start election
        """
        if self.state == 'follower':
            if time.time() - self.last_heartbeat > self.election_timeout:
                return True
        return False
    
    def get_current_leader(self) -> Optional[int]:
        """Get current leader
        
        Returns:
            Leader ID or None
        """
        return self.current_leader