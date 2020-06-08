# REINFORCE Algorithm (Monte-Carlo policy gradient)

Monte-Carlo policy gradients proposed in [Policy Gradient Methods for
Reinforcement Learning with Function
Approximation,](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) uses several sample rollouts to collect and update policy parameters over time to obtain the optimal policy to maximize expected rewards.

![Pseudo Code for Reinforce algorithm](https://i.stack.imgur.com/D0K5F.png)

**Step 1 : Initialize** - Initialize a neural network (Policy network with some weigths and biases)</br>
**Step 2 : Rollouts** - Do episode rollouts for n number or episodes and collect the _observations, actions_ and _rewards_</br>
**Step 3 : Calculate loss and update the parameters** - Calculate the loss for the rollouts by using discounted rewards. Then update the policy parameters by using the custom loss calculated</br>
**Step 4 : Repeat** - Repeat step 1 and step 2 until a good enough (local optimum) policy is learnt by the agent.

## Loss function

Loss function can be identified as the main learning component of the RL agent as the better loss functions leads to more efficient and stable agents (Policies)
