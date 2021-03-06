# Reinforcement Learning
Reinforcement Learning is subfield of machine learning that is focused on behaviour of an agent in an environment when it is interacted with it. Being fairly new field, RL has shown some of very important breakthroughs in machine learning domain. Reinforcement Learning builds up on a very simple concept and expand across it. 

## Concept Behind Reinforcement Learning
Intuition behing RL is very simple. It can be explained as follow.
1. Initialise a software agent (Presumably a neural network)
2. Make agent interact with the environment (Make an action)
3. Collect the reward and the new state of the environment (scores of a game and frame of the game after we take the action)
4. Improve the agent to get better rewards over time
The whole subfield of reinforcement learning build up upone the afore mentioned concepts and the rest of the theory build up on that basis

![Basic concept of RL](https://www.researchgate.net/profile/Roohollah_Amiri/publication/323867253/figure/fig2/AS:606095550738432@1521515848671/Reinforcement-Learning-Agent-and-Environment.png)

## Types of Reinforcement Learning
1. Value Iteration Methods
2. Policy Iteration Methods
3. Model Based Methods

#### Before we dive deep into the rabbit hole
Though the concepts behind these approaches for reinforcement learning methods are quite simple it's quite easy to get lost(been there).
In order to have a good understanding about the concepts, getting familiar with the specific terminology is mandatory.

__State(_S<sub>t</sub>_)__: Information about the environment at time _t_</br>
__Action(_A<sub>t</sub>_)__: Action agent took at the time _t_</br>
__Action Probability of given state(_P(a|s)<sub>t</sub>_)__:

### 1. Value Based Methods (Q-value methods)
Value based methods focus on approximating the values of a states and taking the optimal actions in order to gain a good trajectory in order to maximize rewards. 

### 2. Policy Iteration Methods
Unlike value based methods, policy methods focus on approximating the trajectory in which the actions should be taken directly, in order to maximize the expected return. 

### 3. Model Based Methods
Model based methods try to approximate the dynamics of the environment in order for agent to chose optimal action trajectory.

## Policy iteration methods
Policy iteration methods have been getting so much attention recently. The reason behind the attention on policy iteration method is because policy gradient methods are relatively easier to implement and they are optimized to the required goal rather than predicting value of the current state to decide the optimal action (Value Iteration).
