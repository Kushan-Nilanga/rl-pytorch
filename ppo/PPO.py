import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Memory():
    def __init__(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.done_memory = []
        self.log_probs_memory = []

    def clear_memory(self):
        del self.state_memory[:]
        del self.action_memory[:]
        del self.reward_memory[:]
        del self.done_memory[:]
        del self.log_probs_memory[:]


class A2CAgent(nn.Module):
    def __init__(self, gamma=0.9, input_size=4, hidden1_size=256, hidden2_size=256, output_size=2):
        super(A2CAgent, self).__init__()
        self.memory = Memory()
        self.input_size = input_size

        self.hidden_l1 = nn.Linear(input_size, hidden1_size)
        self.hidden_l2 = nn.Linear(hidden1_size, hidden2_size)

        self.action_head = nn.Linear(hidden2_size, output_size)
        self.value_head = nn.Linear(hidden2_size, 1)

        self.device = 'cuda:0' if T.cuda.is_available() == True else 'cpu:0'

    def forward(self, state):
        state = state.clone().detach().to(self.device)
        x = F.relu(self.hidden_l1(state))
        x = F.relu(self.hidden_l2(x))

        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value

    def evaluate(self, state, action):
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, T.squeeze(value), dist_entropy

    def select_action(self, state, memory):
        state = T.from_numpy(state).float().to(self.device)
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.state_memory.append(state)
        memory.action_memory.append(action)
        memory.log_probs_memory.append(dist.log_prob(action))
        return action.item()


class PPO:
    def __init__(self, gamma=0.99, clip=.2, lr=0.0025, input_size=4, hidden1_size=256, hidden2_size=256, output_size=2):
        super(PPO, self).__init__()
        self.lr = lr
        self.gamma = gamma
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.clip = clip
        self.K_epochs = 4
        self.lr = lr
        self.betas = (0.9, 0.999)
        self.device = 'cuda:0' if T.cuda.is_available() == True else 'cpu:0'

        self.policy = A2CAgent(self.gamma, self.input_size, self.hidden1_size,
                               self.hidden2_size, self.output_size).to(self.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=self.betas)

        self.policy_old = A2CAgent(self.gamma, self.input_size, self.hidden1_size,
                                   self.hidden2_size, self.output_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def learn(self, memory):
        rewards = []
        discounted_reward = []

        for reward, is_terminal in zip(reversed(memory.reward_memory), reversed(memory.done_memory)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma*discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = T.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = T.stack(memory.state_memory).to(self.device).detach()
        old_actions = T.stack(memory.action_memory).to(self.device).detach()
        old_logprobs = T.stack(memory.log_probs_memory).to(
            self.device).detach()

        # Evaluating old actions and values :
        logprobs, state_values, dist_entropy = self.policy.evaluate(
            old_states, old_actions)

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = T.exp(logprobs - old_logprobs.detach())

        # Finding Surrogate Loss:
        advantages = rewards - state_values.detach()
        surr1 = ratios * advantages
        surr2 = T.clamp(ratios, 1-self.clip, 1 + self.clip) * advantages
        loss = -T.min(surr1, surr2) + 0.5 * \
            (state_values - rewards)**2 - 0.01*dist_entropy

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss


if __name__ == '__main__':
    memory = Memory()
    ppo = PPO()
    env = gym.make("CartPole-v0")
    cu_rewards = []
    rolling_rewards = []
    episode_index = []

    for i in range(1, 1000):
        state = env.reset()
        episode_reward = 0
        Done = False
        while not Done:
            action = ppo.policy_old.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            memory.reward_memory.append(reward)
            memory.done_memory.append(done)
            Done = done

        if i % 10 == 0:
            print("learning loss", ppo.learn(memory).mean().item())
            memory.clear_memory()

        cu_rewards.append(episode_reward)
        rolling_rewards.append(np.mean(cu_rewards[10:]))
        episode_index.append(i)

        print(i, 'episode', episode_reward)

    plt.plot(episode_index, cu_rewards, label="score")
    plt.plot(episode_index, rolling_rewards, label="score")
    plt.legend()
    plt.show()
