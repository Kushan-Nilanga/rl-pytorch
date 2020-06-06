import gym

import torch as T 
import torch.nn as nn
import torch.nn.functional as f 
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np 


class Memory():
    def __init__(self):
        super(Memory, self).__init__()
        self.action_memory = []
        self.state_memory = []
        self.reward_memory = []
        self.done_memory = []
        self.logprob_memory = []

    def reset_memory(self):
        del self.action_memory[:]
        del self.state_memory[:]
        del self.reward_memory[:]
        del self.done_memory[:]
        del self.logprob_memory[:]

# reinforce agent
# action reward observation
# update the policy to maximize the reward
class PolicyNetwork(nn.Module):
    def __init__(self, gamma, input_size, output_size, hidden1_size, hidden2_size):
        super(PolicyNetwork, self).__init__()

        self.memory = Memory()

        self.gamma = gamma
        self.input_size = input_size
        
        self.input_layer = nn.Linear(input_size, hidden1_size)
        self.hidden1_layer = nn.Linear(hidden1_size, hidden2_size)
        self.hidden2_layer = nn.Linear(hidden2_size, output_size)
        self.optimiser = optim.Adam(self.parameters(), lr=0.0025)
        self.device = T.device('cuda:0' if T.cuda.is_available() == True else 'cpu:0')
        self.to(self.device)

    def forward(self, state):
        states = T.Tensor(state).to(self.device)
        x = f.relu(self.input_layer(states))
        x = f.relu(self.hidden1_layer(x)) 
        x = f.relu(self.hidden2_layer(x))
        return x

class Agent(object):
    def __init__(self, lr, input_size, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(gamma, input_size, n_actions, 256, 256)

    def select_action(self, state):
        probabilities = f.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        print("learning")
        G = np.zeros_like(self.reward_memory, dtype=np.float64)

        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean)/std

        G = T.Tensor(G, dtype=T.float32).to(self.policy.device)
        # calculating discounted rewards
        # rewards = []
        # discounted_reward = []
        # for reward, is_terminal in zip(reversed(self.memory.reward_memory), reversed(self.memory.done_memory)):
        #    if is_terminal:
        #        discounted_reward = 0
        #    discounted_reward = reward + (self.gamma * discounted_reward)
        #    rewards.insert(0, discounted_reward)

        # rewards = T.tensor(rewards).to(self.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        loss = 0 
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.policy.optimiser.step()

        self.action_memory=[]
        self.reward_memory=[]

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    reinforce_agent = Agent(0.001, 4, 0.99, 2)
    learning_done = False
    for i in range(1, 5000):
        state = env.reset()
        Done = False
        Reward = 0
        while Done==False:
            #env.render()
            action = reinforce_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            Done = done
            Reward+=reward

        print(i, Reward)

        if(i%10==0):
            reinforce_agent.learn()
            reinforce_agent.memory.reset_memory()