import dataclasses
import functools
from dataclasses import field
from functools import partial
from typing import Tuple, Optional, Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import tree_util

@tree_util.register_static
@dataclasses.dataclass
class Config:
    """Configuration for PPO algorithm."""
    # Environment parameters
    env_name: str = "CartPole-v1"
    seed: int = 42
    
    # Model parameters
    obs_dim: int = 4  # CartPole has 4D observation space
    act_dim: int = 2  # CartPole has 2 actions
    hidden_dim: int = 64
    dtype: Any = jnp.float32
    
    # PPO parameters
    num_iterations: int = 100
    batch_size: int = 2048
    num_epochs: int = 10
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    lmbda: float = 0.95
    clip_ratio: float = 0.2
    
    # Evaluation parameters
    eval_frequency: int = 10
    eval_episodes: int = 10

def jax_pytree_struct(cls, meta_fields: tuple = ()):
    """jax.tree_util.register_dataclass wrapper that automatically infers data_fields."""
    assert not dataclasses.is_dataclass(cls)
    cls = dataclasses.dataclass(cls)
    all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)
    return tree_util.register_dataclass(cls, data_fields=data_fields, meta_fields=meta_fields)

class _Init:
    """Base class for pytree data structures that will eventually contain jax.Arrays (e.g. layer definitions).
    Each subclass is responsible for defining abstract(), which returns an "abstract" version of the pytree containing
    ArrayInfos (i.e. metadata) instead of actual data. This class then helps generate the actual data.
    """
    @classmethod
    def abstract(cls, cfg: Config, *args, **kw):
        """Returns an instance of this class with ArrayInfos instead of jax.Arrays."""
        raise NotImplementedError

    @classmethod
    def init(cls, key: jax.random.PRNGKey, cfg: Config, *args, **kw):
        """Returns a pytree of randomly-initialized jax.Arrays corresponding to abstract()."""
        abstract = cls.abstract(cfg, *args, **kw)

        @jax.jit
        def _init():
            num_leaves = len(jax.tree.leaves(abstract, is_leaf=lambda x: isinstance(x, ArrayInfo)))
            key_iter = iter(jax.random.split(key, num_leaves))

            def init_leaf(x):
                if isinstance(x, ArrayInfo):
                    return x.initializer(next(key_iter), x.shape, x.dtype)
                return x

            return jax.tree.map(
                init_leaf,
                abstract,
                is_leaf=lambda x: isinstance(x, ArrayInfo),
            )
        return _init()

@jax_pytree_struct
class ArrayInfo:
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    initializer: Optional[callable] = None
    
    
@jax_pytree_struct
class Actor(_Init):
    layer_1: jax.Array | ArrayInfo
    layer_2: jax.Array | ArrayInfo
    layer_3: jax.Array | ArrayInfo
    bias_1: jax.Array | ArrayInfo
    bias_2: jax.Array | ArrayInfo
    bias_3: jax.Array | ArrayInfo
    
    @classmethod
    def abstract(cls, cfg: Config):
        _init = jax.nn.initializers.he_normal()
        dtype = cfg.dtype
        
        actor = Actor(
            layer_1 = ArrayInfo((cfg.obs_dim, cfg.hidden_dim), dtype, _init),
            layer_2 = ArrayInfo((cfg.hidden_dim, cfg.hidden_dim), dtype, _init),
            layer_3 = ArrayInfo((cfg.hidden_dim, cfg.act_dim), dtype, _init),
            bias_1 = ArrayInfo((cfg.hidden_dim,), dtype, jax.nn.initializers.zeros),
            bias_2 = ArrayInfo((cfg.hidden_dim,), dtype, jax.nn.initializers.zeros),
            bias_3 = ArrayInfo((cfg.act_dim,), dtype, jax.nn.initializers.zeros)
        )
        return actor

@jax_pytree_struct
class Critic(_Init):
    layer_1: jax.Array | ArrayInfo
    layer_2: jax.Array | ArrayInfo
    layer_3: jax.Array | ArrayInfo
    bias_1: jax.Array | ArrayInfo
    bias_2: jax.Array | ArrayInfo
    bias_3: jax.Array | ArrayInfo
    
    @classmethod
    def abstract(cls, cfg: Config):
        _init = jax.nn.initializers.he_normal()
        dtype = cfg.dtype
        
        critic = Critic(
            layer_1 = ArrayInfo((cfg.obs_dim, cfg.hidden_dim), dtype, _init),
            layer_2 = ArrayInfo((cfg.hidden_dim, cfg.hidden_dim), dtype, _init),
            layer_3 = ArrayInfo((cfg.hidden_dim, 1), dtype, _init),
            bias_1 = ArrayInfo((cfg.hidden_dim,), dtype, jax.nn.initializers.zeros),
            bias_2 = ArrayInfo((cfg.hidden_dim,), dtype, jax.nn.initializers.zeros),
            bias_3 = ArrayInfo((1,), dtype, jax.nn.initializers.zeros)
        )
        return critic

@jax.jit    
def mlp_block(x: jax.Array, model: Actor | Critic) -> jax.Array:
    """Simple MLP block."""
    x = jnp.einsum("i,ij->j", x, model.layer_1) + model.bias_1
    x = jnp.tanh(x)
    x = jnp.einsum("j,jk->k", x, model.layer_2) + model.bias_2
    x = jnp.tanh(x)
    x = jnp.einsum("k,kl->l", x, model.layer_3) + model.bias_3
    return x
  
@partial(jax_pytree_struct, meta_fields=("actor_opt", "critic_opt"))
class TrainingState:
    actor: Actor
    critic: Critic
    actor_opt: optax.GradientTransformation
    critic_opt: optax.GradientTransformation
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    key: jax.random.PRNGKey
    
def create_train_state(config: Config, rng_key: jax.Array) -> TrainingState:
    """Create training state."""
    actor = Actor.init(rng_key, config)
    critic = Critic.init(rng_key, config)
    
    actor_opt = optax.adam(config.actor_lr)
    critic_opt = optax.adam(config.critic_lr)
    
    actor_opt_state = actor_opt.init(actor)
    critic_opt_state = critic_opt.init(critic)
    
    return TrainingState(
        actor, 
        critic, 
        actor_opt, 
        critic_opt, 
        actor_opt_state, 
        critic_opt_state, 
        rng_key)

@jax_pytree_struct
class Transition:
    """Container for storing trajectory data"""
    states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_states: jax.Array
    dones: jax.Array
    log_probs: jax.Array
 
def get_advantage(carry, xs, cfg):
    last_gae = carry
    reward, value, next_value, dones = xs
    
    delta = reward + dones * cfg.gamma * next_value - value
    advantage = delta + dones * cfg.gamma * cfg.lmbda * last_gae
    return advantage, advantage   

@jax.jit
def compute_gae(rewards: jax.Array, 
                values: jax.Array, 
                next_values: jax.Array, 
                dones: jax.Array,
                last_gae: jax.Array, 
                cfg: Config) -> jax.Array:
    """Compute Generalized Advantage Estimation."""
    xs = (rewards, values, next_values, dones)
    
    _, advantages = jax.lax.scan(
        functools.partial(get_advantage, cfg=cfg),
        last_gae,  
        xs,        
        reverse=True
    )
    normalized_advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-7)
    returns = advantages + values
    
    return normalized_advantages, returns

@jax.jit
def actor_loss(actor: Actor, transitions: Transition, advantages: jax.Array, cfg: Config):
    """Actor loss."""
    logits = jax.vmap(lambda s: mlp_block(s, actor))(transitions.states)
    log_probs_all = jax.nn.log_softmax(logits)
    
    batch_indices = jnp.arange(transitions.actions.shape[0])
    new_log_probs = log_probs_all[batch_indices, transitions.actions]
    old_log_probs = transitions.log_probs
    ratio = jnp.exp(new_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
    actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
    
    return actor_loss

@jax.jit    
def critic_loss(critic: Critic, transitions: Transition, returns: jax.Array):
    """Critic loss."""
    values = jax.vmap(lambda s: mlp_block(s, critic))(transitions.states)
    critic_loss = jnp.mean((values - returns) ** 2)
    
    return critic_loss 


def update(state: TrainingState, transitions: Transition, advantages: jax.Array, returns: jax.Array, cfg: Config):
    """Update actor and critic parameters."""
    actor_loss_val, actor_grad = jax.value_and_grad(actor_loss)(state.actor, transitions, advantages, cfg)
    actor_updates, actor_opt_state = state.actor_opt.update(actor_grad, state.actor_opt_state)
    actor = optax.apply_updates(state.actor, actor_updates)
    
    critic_loss_val, critic_grad = jax.value_and_grad(critic_loss)(state.critic, transitions, returns)
    critic_updates, critic_opt_state = state.critic_opt.update(critic_grad, state.critic_opt_state)
    critic = optax.apply_updates(state.critic, critic_updates)
    
    return TrainingState(
        actor, 
        critic, 
        state.actor_opt, 
        state.critic_opt, 
        actor_opt_state, 
        critic_opt_state, 
        state.key), actor_loss_val, critic_loss_val
    
@jax.jit
def sample_action(actor: Actor, obs: jax.Array, key: jax.random.PRNGKey) -> Tuple:
    """Sample action from categorical distribution and compute log prob."""
    logits = mlp_block(obs, actor)
    action = jax.random.categorical(key, logits)
    
    log_probs = jax.nn.log_softmax(logits)
    log_prob = log_probs[action]
    
    return action, log_prob
    
def collect_data(state: TrainingState, env: gym.Env, cfg: Config):
    """Collect trajectory data."""
    batch_size = cfg.batch_size
    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []

    state_key, sample_key = jax.random.split(state.key)
    state = dataclasses.replace(state, key=state_key)
    
    obs, _ = env.reset()
    done = False
    steps = 0
    
    while steps < batch_size:
        sample_key, action_key = jax.random.split(sample_key)
        
        with jax.named_scope("sample_action"):
            action, log_prob = sample_action(state.actor, obs, action_key)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_obs)
        dones.append(float(done))
        log_probs.append(log_prob)
        
        obs = next_obs
        steps += 1
        
        if done:
            obs, _ = env.reset()
    
    trajectories = {
        'states': jnp.array(states),
        'actions': jnp.array(actions),
        'rewards': jnp.array(rewards),
        'next_states': jnp.array(next_states),
        'dones': jnp.array(dones),
        'log_probs': jnp.array(log_probs)
    }
    
    return trajectories, state

@jax.jit
def process_trajectories(critic: Critic, trajectories: dict, cfg: Config):
    """Process trajectory data."""
    states = trajectories['states']
    actions = trajectories['actions']
    rewards = trajectories['rewards']
    next_states = trajectories['next_states']
    dones = 1.0 - trajectories['dones']
    log_probs = trajectories['log_probs']
    
    values = jax.vmap(lambda s: mlp_block(s, critic))(states)
    next_values = jax.vmap(lambda s: mlp_block(s, critic))(next_states)
    
    with jax.named_scope("compute_gae"):
        advantages, returns = compute_gae(rewards, values, next_values, dones, jnp.zeros_like(values[0]), cfg)
    
    return Transition(states, actions, rewards, next_states, dones, log_probs), advantages, returns    

def evaluate(state: TrainingState, env: gym.Env, num_episodes: int = 10):
    """Evaluate policy by running multiple episodes and returning mean reward.
    
    This uses the deterministic (argmax) policy rather than sampling for evaluation.
    
    Args:
        state: Current training state containing actor and critic
        env: Gym environment to evaluate in
        num_episodes: Number of episodes to run for evaluation
        
    Returns:
        Mean episode return across all evaluation episodes
    """
    episode_returns = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            logits = mlp_block(obs, state.actor)
            action = jnp.argmax(logits).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        episode_returns.append(total_reward)
    
    mean_return = sum(episode_returns) / len(episode_returns)
    return mean_return

def train_ppo(env, cfg: Config):
    """Main PPO training loop."""
    key = jax.random.PRNGKey(cfg.seed)
    state = create_train_state(cfg, key)
    
    returns = []
    actor_losses = []
    critic_losses = []
    
    for iteration in range(cfg.num_iterations):
        with jax.named_scope("collect_data"):
            trajectories, state = collect_data(state, env, cfg)
        
        with jax.named_scope("process_trajectories"):
            transitions, advantages, returns_batch = process_trajectories(state.critic, trajectories, cfg)
        
        for epoch in range(cfg.num_epochs):
            state, actor_loss_val, critic_loss_val = update(
                state, transitions, advantages, returns_batch, cfg)
            
            actor_losses.append(float(actor_loss_val))
            critic_losses.append(float(critic_loss_val))
        
        if iteration % cfg.eval_frequency == 0:
            eval_return = evaluate(state, env, cfg.eval_episodes)
            returns.append(eval_return)
            print(f"Iteration {iteration}, Mean return: {eval_return:.2f}")
    
    return state, returns, actor_losses, critic_losses

def main():
    """Main entry point for PPO training."""
    cfg = Config()
    
    env = gym.make(cfg.env_name)
    
    cfg.obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        cfg.act_dim = env.action_space.n
    else:
        cfg.act_dim = env.action_space.shape[0]
    
    print(f"Training PPO on {cfg.env_name}")
    print(f"Observation space: {cfg.obs_dim}-dimensional")
    print(f"Action space: {cfg.act_dim} actions")
    
    with jax.named_scope("Training"):
        state, returns, actor_losses, critic_losses = train_ppo(env, cfg)
    
    final_return = evaluate(state, env, num_episodes=100)
    print(f"Training complete! Final performance: {final_return:.2f}")
    
    return state

if __name__ == "__main__":
    main()