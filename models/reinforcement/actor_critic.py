"""
A2C Trading Agent - Advantage Actor-Critic for trading.

A2C is a synchronous version of A3C that's simpler to implement
and often performs just as well for single-environment training.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel, ModelPrediction


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network with shared feature extractor.
    
    Outputs both policy (action distribution) and value estimate.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128],
        continuous: bool = True
    ):
        super().__init__()
        
        self.continuous = continuous
        self.action_dim = action_dim
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Separate heads
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU()
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU()
        )
        
        if continuous:
            self.mean = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_probs = nn.Linear(hidden_dims[-1], action_dim)
        
        self.value = nn.Linear(hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(state)
        
        # Actor
        actor_features = self.actor_head(shared_features)
        if self.continuous:
            mean = torch.tanh(self.mean(actor_features))
            std = self.log_std.exp().expand_as(mean)
            policy = (mean, std)
        else:
            logits = self.action_probs(actor_features)
            policy = torch.softmax(logits, dim=-1)
        
        # Critic
        critic_features = self.critic_head(shared_features)
        value = self.value(critic_features)
        
        return policy, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        policy, value = self.forward(state)
        
        if self.continuous:
            mean, std = policy
            if deterministic:
                action = mean
                log_prob = None
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            if deterministic:
                action = policy.argmax(dim=-1)
                log_prob = None
            else:
                dist = Categorical(policy)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        policy, values = self.forward(states)
        
        if self.continuous:
            mean, std = policy
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            dist = Categorical(policy)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class A2CAgent(BaseModel):
    """
    Advantage Actor-Critic agent for trading.
    
    Key features:
    - Shared feature extractor for efficiency
    - Advantage-based updates for lower variance
    - Entropy regularization for exploration
    
    Usage:
        agent = A2CAgent(state_dim=50, action_dim=1)
        
        # Collect n_steps of experience
        for step in range(n_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, log_prob, value, done)
            state = next_state
        
        # Update after n_steps
        agent.update(next_state)
    """
    
    def __init__(
        self,
        name: str = "a2c_trader",
        state_dim: int = 50,
        action_dim: int = 1,
        hidden_dims: List[int] = [256, 128],
        continuous: bool = True,
        lr: float = 7e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 5
    ):
        super().__init__(name)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for A2CAgent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dims, continuous
        )
        
        # Optimizer
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=lr, alpha=0.99, eps=1e-5)
        
        # Buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Stats
        self.training_steps = 0
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Select action given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        
        action = action.squeeze().numpy()
        log_prob = log_prob.item() if log_prob is not None else 0.0
        value = value.item()
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """Store transition in buffer."""
        self.states.append(torch.FloatTensor(state))
        self.actions.append(torch.FloatTensor([action]) if np.isscalar(action) else torch.FloatTensor(action))
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self, next_state: np.ndarray = None) -> Dict[str, float]:
        """
        Update networks using collected experience.
        
        Args:
            next_state: State after last action (for bootstrapping)
        """
        if len(self.states) == 0:
            return {}
        
        # Get next value for bootstrapping
        if next_state is not None and not self.dones[-1]:
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                _, _, next_value = self.network.get_action(state_tensor, deterministic=True)
                next_value = next_value.item()
        else:
            next_value = 0
        
        # Compute returns and advantages using GAE
        returns = []
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Evaluate actions
        log_probs, values, entropy = self.network.evaluate_actions(states, actions)
        
        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = self.value_coef * nn.MSELoss()(values, returns)
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        total_loss = actor_loss + critic_loss + entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        self.training_steps += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item()
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'A2CAgent':
        """RL requires environment interaction."""
        logger.warning("A2C requires environment interaction.")
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Generate trading signal from state."""
        state = X.flatten() if len(X.shape) > 1 else X
        
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        action, _, value = self.select_action(state, deterministic=True)
        
        direction = float(action[0]) if hasattr(action, '__len__') else float(action)
        
        return ModelPrediction(
            model_name=self.name,
            direction=direction,
            magnitude=abs(direction) * 0.01,
            confidence=min(1.0, abs(direction)),
            metadata={
                'state_value': value,
                'raw_action': direction
            }
        )
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'continuous': self.continuous,
            'gamma': self.gamma,
            'network_state': self.network.state_dict(),
            'training_steps': self.training_steps
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.state_dim = state['state_dim']
        self.action_dim = state['action_dim']
        self.continuous = state['continuous']
        self.gamma = state['gamma']
        self.training_steps = state['training_steps']
        
        if 'network_state' in state:
            self.network.load_state_dict(state['network_state'])
