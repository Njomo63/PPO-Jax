# Proximal Policy Optimization
Reinforcement learning involves an agent learning to make decisions by interacting with an environment. The agent follows a policy $\pi(a\mid s)$
 that maps states to action probabilities, and the goal is to find a policy that maximizes expected cumulative rewards. We can think of the policy as a model that takes in a state and outputs an action.
Proximal Policy Optimization (PPO) optimizes the policy by performing gradient ascent on the expected return while constraining policy updates to prevent large divergence from the previous policy. It use the clipped surrogate objective:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip} \left( r_t(\theta), 1 - \epsilon, 1 + \epsilon \right) A_t \right) \right]
$$

Where:

- $r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the probability ratio between new and old policies
- $\hat{A}_t$ is the estimated advantage
- $\epsilon$ is a hyperparameter that limits how much the policy can change

Advantage quantifies how much better an action is than others on average.
We calculate advantage using the Generalized Advantage Estimate (GAE):

$$
\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \cdots + (\gamma \lambda)^{T - t + 1} \delta_{T - 1}
$$

where  

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

The value function represents the expected return when starting from a specific state and following a particular policy thereafter.

## Actor-Critic Architecture
PPO uses an actor-critic architecture that combines two neural networks working together. The actor network (policy) learns which actions to take in each state, while the critic network (value function) evaluates how good those states are.

The actor network outputs a probability distribution over actions, from which we sample during training to encourage exploration. The actor is trained to maximize the PPO objective, which encourages actions that lead to higher advantages while maintaining proximity to the previous policy.

The critic network estimates the value function $V(s)$ which represents the expected return from each state. It is trained to minimize the mean squared error between its predictions and the actual returns observed during training. When we calculate advantages (which guide the actor's learning), we're effectively asking "did this specific action perform better or worse than what the critic expected?" This is why it's called "critic" - it provides a baseline for judging the actor's individual action choices. This relationship creates a productive feedback loop: the actor tries to find better actions, the critic evaluates the resulting states, and the advantages derived from these evaluations help the actor improve further.