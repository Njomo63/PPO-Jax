# Proximal Policy Optimization
Reinforcement learning involves an agent learning to make decisions by interacting with an environment. The agent follows a policy $\pi(a\mid s)$
 that maps states to action probabilities, and the goal is to find a policy that maximizes expected cumulative rewards. We can think of the policy as a model that takes in a state and outputs an action.
Proximal Policy Optimization (PPO) optimizes the policy by performing gradient ascent on the expected return while constraining policy updates to prevent large divergence from the previous policy. It use the clipped surrogate objective:

$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A^t, \text{clip} \left( r_t(\theta), 1 - \epsilon, 1 + \epsilon \right) A^t \right) \right]$

Where:

- $r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the probability ratio between new and old policies
- $\hat{A}_t$ is the estimated advantage
- $\epsilon$ is a hyperparameter that limits how much the policy can change

Advantage quantifies how much better an action is than others on average.
We calculate advantage using the Generalized Advantage Estimate (GAE):
$$
\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \cdots + (\gamma \lambda)^{T - t + 1} \delta_{T - 1},
$$

where  

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

The value function represents the expected return when starting from a specific state and following a particular policy thereafter.

