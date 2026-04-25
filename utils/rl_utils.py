"""
Reinforcement learning utilities for Q-Learning #20.

Pure numpy / PyTorch primitives for value-based RL: replay buffers,
prioritized sampling, epsilon-greedy exploration, target network updates,
evaluation rollouts, and learning-curve visualization.

No high-level RL libraries (stable-baselines3 / RLlib / tf-agents) - this
module exposes the from-scratch implementation that V1-V5 share.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np


# Reproducibility
def seed_everything(seed: int, env=None) -> None:
    """
    Seed all RNGs used by Q-Learning training.

    Seeds Python's `random`, numpy, and (if available) PyTorch's CPU + CUDA
    RNGs. Optionally seeds a gymnasium env's action_space sampler.

    Note: `env.reset(seed=...)` is the caller's responsibility per-episode;
    this helper only seeds the one-time action_space sampler.

    Args:
        seed: Random seed (project convention: 113).
        env: Optional gymnasium env. If provided, its action_space sampler
            is seeded.
    """
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch optional - V1 tabular variant doesn't need torch installed
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    if env is not None and hasattr(env, 'action_space'):
        env.action_space.seed(seed)


# Exploration primitives
def epsilon_greedy(
    q_values: np.ndarray,
    epsilon: float,
    n_actions: int,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Epsilon-greedy action selection.

    With probability (1 - epsilon), returns argmax(q_values). Otherwise returns
    a uniform random action over [0, n_actions). Pure numpy, framework-agnostic.

    Args:
        q_values: Q-value vector for current state, shape (n_actions,).
        epsilon: Exploration rate in [0, 1].
        n_actions: Number of discrete actions.
        rng: Optional numpy Generator. If None, uses numpy's global RNG
            (seeded by seed_everything).

    Returns:
        Selected action index in [0, n_actions).
    """
    rand_uniform = rng.random() if rng is not None else np.random.random()
    if rand_uniform < epsilon:
        if rng is not None:
            return int(rng.integers(0, n_actions))
        return int(np.random.randint(0, n_actions))
    return int(np.argmax(q_values))


@dataclass
class LinearEpsilonSchedule:
    """
    Linear epsilon decay from `start` to `end` over `decay_steps` steps.

    Standard DQN exploration schedule (Mnih 2015): linear decay over a
    fraction of total training steps, then holds at `end` forever.

    Example:
        schedule = LinearEpsilonSchedule(start=1.0, end=0.05, decay_steps=50_000)
        for t in range(total_steps):
            eps = schedule(t)
            action = epsilon_greedy(q_values, eps, n_actions)

    Args:
        start: Initial epsilon (typically 1.0 = pure exploration).
        end:   Final epsilon (typically 0.05 = 5% retained exploration).
        decay_steps: Steps over which to decay linearly.
    """
    start: float
    end: float
    decay_steps: int

    def __call__(self, step: int) -> float:
        """Return epsilon at the given step."""
        if step >= self.decay_steps:
            return self.end
        frac = step / self.decay_steps
        return self.start + (self.end - self.start) * frac
    

# Replay Buffer (V2-V4 use this; V5 uses PrioritizedReplayBuffer from chunk 3)
class ReplayBuffer:
    """
    Cyclic experience replay buffer.

    Stores (state, action, reward, next_state, done) transitions in a deque of
    fixed capacity. Sampling returns a batch of numpy arrays; caller is
    responsible for converting to framework tensors.

    Args:
        capacity: Max transitions stored. When full, oldest are evicted.
        rng: Optional numpy Generator for reproducible sampling. Defaults
            to numpy's global RNG (seeded by seed_everything).

    Example:
        buf = ReplayBuffer(capacity=10_000)
        buf.add(s, a, r, s_next, done)
        if len(buf) >= batch_size:
            batch = buf.sample(batch_size=64)
            states = torch.tensor(batch['states']).float().to(device)
    """
    def __init__(self, capacity: int, rng: Optional[np.random.Generator] = None):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = rng

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict:
        """
        Sample a batch of transitions uniformly at random.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict with keys 'states', 'actions', 'rewards', 'next_states',
            'dones', each a numpy array of shape (batch_size, ...).
        """
        if self.rng is not None:
            indices = self.rng.integers(0, len(self.buffer), size=batch_size)
        else:
            indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            'states':      np.array(states, dtype=np.float32),
            'actions':     np.array(actions, dtype=np.int64),
            'rewards':     np.array(rewards, dtype=np.float32),
            'next_states': np.array(next_states, dtype=np.float32),
            'dones':       np.array(dones, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)


# Target network update (DQN stability trick)
def soft_target_update(target_net, online_net, tau: float) -> None:
    """
    Polyak (soft) update of target network parameters in place.

        target_param = tau * online_param + (1 - tau) * target_param

    Use small tau (e.g., 0.005) for stable bootstrapping. For hard sync (used
    in Mnih 2015 DQN), use tau=1.0 every N steps - that's a one-liner:
    `target_net.load_state_dict(online_net.state_dict())`.

    Args:
        target_net: Target network (PyTorch nn.Module) to update in place.
        online_net: Online (training) network to copy from.
        tau: Mixing coefficient in [0, 1]. tau=1 is hard sync; tau=0 is no sync.
    """
    import torch  # lazy: V1 tabular doesn't need torch installed
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)


# Policy evaluation (greedy rollout)
def evaluate_policy(
    env,
    policy_fn: Callable[[np.ndarray], int],
    n_episodes: int = 100,
    seed: Optional[int] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate a policy via deterministic rollouts (no exploration).

    Runs `n_episodes` of `policy_fn(state) -> action` against `env`. Each
    episode runs until terminated or truncated.

    Args:
        env: gymnasium env (already constructed).
        policy_fn: Callable taking state and returning an integer action.
            Typically a closure over a trained Q-network or Q-table.
        n_episodes: Number of evaluation rollouts.
        seed: Optional base seed; episodes use seed, seed+1, ..., seed+n-1.

    Returns:
        (mean_return, std_return, returns_array of shape (n_episodes,))
    """
    returns = np.zeros(n_episodes, dtype=np.float64)
    for ep in range(n_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        ep_return = 0.0
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
        returns[ep] = ep_return
    return float(returns.mean()), float(returns.std()), returns


# Visualization
def plot_learning_curve(
    returns: np.ndarray,
    framework: str,
    window: int = 100,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot per-episode return + rolling mean.

    Two overlaid lines: raw episode returns (alpha=0.3) and rolling mean
    over `window` episodes (solid line). ASCII-only title for matplotlib safety.

    Args:
        returns: 1D array of per-episode returns.
        framework: 'PyTorch' or 'TensorFlow' (label on plot).
        window: Rolling-mean window size in episodes (default 100).
        title: Optional custom title; else auto-generated.
        save_path: Optional file path; if provided, savefig before show.
    """
    import matplotlib.pyplot as plt  # lazy: keep utils import light

    n = len(returns)
    if n >= window:
        rolling_mean = np.convolve(returns, np.ones(window) / window, mode='valid')
        x_rolling = np.arange(window - 1, n)
    else:
        rolling_mean = np.array([returns[:i+1].mean() for i in range(n)])
        x_rolling = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(returns, alpha=0.3, label='per-episode return')
    ax.plot(x_rolling, rolling_mean, linewidth=2,
            label=f'{window}-ep rolling mean' if n >= window else 'running mean')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title(title if title else f'Learning curve ({framework})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# Prioritized Experience Replay (V5 only - Schaul 2016)
class _SumTree:
    """
    Sum-tree data structure for O(log N) priority sampling.

    Internal nodes store the sum of their children's priorities; sampling a
    leaf with probability proportional to its priority is O(log N) via tree
    traversal from the root.

    Tree storage: numpy array of size 2*capacity - 1.
        - Leaves:         indices [capacity-1, 2*capacity-2]
        - Internal nodes: indices [0, capacity-2]
        - Root:           index 0
        - For node i:     left = 2i+1, right = 2i+2, parent = (i-1)//2

    Args:
        capacity: Number of leaves (= replay buffer capacity).
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.next_idx = 0   # cyclic write pointer (data-array index)
        self.size = 0       # number of leaves filled

    def total(self) -> float:
        """Sum of all priorities (root value)."""
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        """
        Write a new priority at the next leaf (cyclic).

        Returns:
            data-array index (0 to capacity-1) of the leaf just written.
        """
        leaf_data_idx = self.next_idx
        tree_idx = leaf_data_idx + self.capacity - 1
        self.update(tree_idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return leaf_data_idx

    def update(self, tree_idx: int, priority: float) -> None:
        """Update a leaf's priority and propagate change up to root."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get(self, s: float) -> Tuple[int, float, int]:
        """
        Find the leaf such that cumulative priority up to it is >= s.

        Args:
            s: Cumulative priority value in [0, total()].

        Returns:
            (tree_idx, priority, leaf_data_idx)
        """
        idx = 0  # start at root
        while idx < self.capacity - 1:  # while not at a leaf
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        leaf_data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), leaf_data_idx


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay (Schaul 2016).

    Samples transitions proportionally to TD-error magnitude raised to alpha.
    Importance-sampling weights correct for the bias from non-uniform sampling;
    beta is annealed from beta_start to 1.0 over training.

    Sum-tree gives O(log N) priority sampling vs O(N) cumulative-sum search.

    Args:
        capacity: Max transitions stored.
        alpha: Exponent on raw priorities (priority^alpha for sampling).
            alpha=0 reduces to uniform sampling; alpha=1 = full prioritization.
        epsilon: Small constant added to priorities to avoid 0-probability
            transitions (Schaul 2016 default 1e-6).
        rng: Optional numpy Generator for reproducible sampling.

    Example:
        per = PrioritizedReplayBuffer(capacity=50_000, alpha=0.6)
        per.add(s, a, r, s_next, done)  # uses current max priority
        batch, tree_indices, is_weights = per.sample(64, beta=0.4)
        # ... compute td_errors after gradient step ...
        per.update_priorities(tree_indices, np.abs(td_errors))
    """
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        rng: Optional[np.random.Generator] = None,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = rng
        self.tree = _SumTree(capacity)
        self.data: List = [None] * capacity
        self.max_priority = 1.0  # raw (pre-alpha) max for new transitions

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition with current max priority (so it's sampled at least once)."""
        priority_alpha = (self.max_priority + self.epsilon) ** self.alpha
        leaf_data_idx = self.tree.add(priority_alpha)
        self.data[leaf_data_idx] = (state, action, reward, next_state, done)

    def sample(
        self, batch_size: int, beta: float
    ) -> Tuple[dict, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions with priority + IS weights.

        Stratified sampling: divide [0, total] into batch_size segments,
        sample one transition uniformly within each segment, walk sum-tree
        to find the leaf.

        Args:
            batch_size: Number of transitions to sample.
            beta: Importance-sampling exponent in [0, 1]. beta=0 = no IS
                correction; beta=1 = full IS correction. Anneal 0.4 -> 1.0.

        Returns:
            (batch_dict, tree_indices, is_weights)
                batch_dict: same keys as ReplayBuffer.sample()
                tree_indices: shape (batch_size,), needed for update_priorities
                is_weights: shape (batch_size,), normalized so max=1
        """
        total_priority = self.tree.total()
        segment = total_priority / batch_size
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        transitions = []

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            if self.rng is not None:
                s = self.rng.uniform(lo, hi)
            else:
                s = np.random.uniform(lo, hi)
            tree_idx, priority, data_idx = self.tree.get(s)
            tree_indices[i] = tree_idx
            priorities[i] = priority
            transitions.append(self.data[data_idx])

        # IS weights: w_i = (N * P(i))^(-beta), normalized to max=1
        N = self.tree.size
        sampling_probs = priorities / total_priority
        is_weights = (N * sampling_probs) ** (-beta)
        is_weights = is_weights / is_weights.max()

        states, actions, rewards, next_states, dones = zip(*transitions)
        batch = {
            'states':      np.array(states, dtype=np.float32),
            'actions':     np.array(actions, dtype=np.int64),
            'rewards':     np.array(rewards, dtype=np.float32),
            'next_states': np.array(next_states, dtype=np.float32),
            'dones':       np.array(dones, dtype=np.float32),
        }
        return batch, tree_indices, is_weights.astype(np.float32)

    def update_priorities(
        self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update sampled transitions' priorities based on new TD-errors.

            stored_priority = (|td_error| + epsilon)^alpha

        Args:
            tree_indices: Sum-tree indices returned from sample().
            td_errors: Absolute TD-errors for each sampled transition.
        """
        for tree_idx, td_err in zip(tree_indices, td_errors):
            priority_alpha = (abs(td_err) + self.epsilon) ** self.alpha
            self.tree.update(tree_idx, priority_alpha)
            self.max_priority = max(self.max_priority, abs(td_err))

    def __len__(self) -> int:
        return self.tree.size