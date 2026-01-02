"""
PPO Trading Agent - Proximal Policy Optimization for trading.

PPO is a policy gradient method that's stable and sample-efficient,
making it well-suited for trading environments.
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
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel, ModelPrediction


class ActorNetwork(nn.Module):
    """Policy network that outputs action distribution."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Output mean and log_std for continuous action
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        mean = torch.tanh(self.mean_head(features))  # Bound to [-1, 1]
        log_std = torch.clamp(self.log_std_head(features), -20, 2)
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            return mean, None
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """Value network that estimates state value."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PPOMemory:
    """Experience buffer for PPO."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get_batches(self, batch_size: int):
        n = len(self.states)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                torch.stack([self.states[i] for i in batch_indices]),
                torch.stack([self.actions[i] for i in batch_indices]),
                torch.tensor([self.rewards[i] for i in batch_indices]),
                torch.stack([self.values[i] for i in batch_indices]),
                torch.tensor([self.log_probs[i] for i in batch_indices]),
                torch.tensor([self.dones[i] for i in batch_indices])
            )


class PPOAgent(BaseModel):
    """
    Proximal Policy Optimization agent for trading.
    
    The agent learns to output position sizes [-1, 1] based on
    market state features.
    
    Key features:
    - Clipped objective for stable training
    - GAE (Generalized Advantage Estimation)
    - Entropy bonus for exploration
    
    Usage:
        agent = PPOAgent(state_dim=50, action_dim=1)
        
        # Training loop
        for episode in range(1000):
            state = env.reset()
            done = False
            
            while not done:
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store_transition(state, action, reward, value, log_prob, done)
                state = next_state
            
            agent.update()
        
        # Inference
        action = agent.predict(current_state)
    """
    
    def __init__(
        self,
        name: str = "ppo_trader",
        state_dim: int = 50,
        action_dim: int = 1,
        hidden_dims: List[int] = [256, 128],
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64
    ):
        super().__init__(name)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PPOAgent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic = CriticNetwork(state_dim, hidden_dims)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Memory
        self.memory = PPOMemory()
        
        # Training stats
        self.training_steps = 0
        self.episode_rewards = []
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given state.
        
        Returns:
            action: Position size [-1, 1]
            log_prob: Log probability of action
            value: Estimated state value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)
        
        action = action.squeeze().numpy()
        log_prob = log_prob.item() if log_prob is not None else 0.0
        value = value.item()
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in memory."""
        self.memory.store(
            torch.FloatTensor(state),
            torch.FloatTensor([action]) if np.isscalar(action) else torch.FloatTensor(action),
            reward,
            torch.FloatTensor([value]),
            log_prob,
            done
        )
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """Update policy and value networks."""
        if len(self.memory.states) == 0:
            return {}
        
        # Get all data from memory
        states = torch.stack(self.memory.states)
        actions = torch.stack(self.memory.actions)
        old_log_probs = torch.tensor(self.memory.log_probs)
        
        # Compute advantages and returns
        values = [v.item() for v in self.memory.values]
        advantages, returns = self.compute_gae(
            self.memory.rewards,
            values,
            self.memory.dones
        )
        
        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Multiple epochs of updates
        for _ in range(self.n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Get current policy distribution
                mean, log_std = self.actor(batch_states)
                std = log_std.exp()
                dist = Normal(mean, std)
                
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Policy loss with clipping
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Value loss
                values = self.critic(batch_states).squeeze()
                critic_loss = self.value_coef * nn.MSELoss()(values, batch_returns.float())
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        self.memory.clear()
        self.training_steps += 1
        
        n_updates = self.n_epochs * (len(states) // self.batch_size + 1)
        
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'PPOAgent':
        """
        Fit is not directly applicable for RL.
        Use the training loop with select_action -> store_transition -> update.
        """
        logger.warning("PPO requires environment interaction. Use select_action/store_transition/update.")
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Generate trading signal from state.
        
        Args:
            X: State features
        """
        state = X.flatten() if len(X.shape) > 1 else X
        
        # Pad or truncate to expected state dimension
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
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_epsilon': self.clip_epsilon,
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'training_steps': self.training_steps
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.state_dim = state['state_dim']
        self.action_dim = state['action_dim']
        self.gamma = state['gamma']
        self.gae_lambda = state['gae_lambda']
        self.clip_epsilon = state['clip_epsilon']
        self.training_steps = state['training_steps']
        
        if 'actor_state' in state:
            self.actor.load_state_dict(state['actor_state'])
        if 'critic_state' in state:
            self.critic.load_state_dict(state['critic_state'])
