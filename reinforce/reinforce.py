import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Categorical
import torch as t

import gym


class Agent(nn.Module):
    def __init__(self, gamma=0.99, input_dims=4, l1_dims=256, output_dims=2):
        super().__init__()  # nn.Module is initialised
        self.gamma = gamma
        self.device = "cuda:0" if t.cuda.is_available() else "cpu:1"
        self.l1 = nn.Linear(input_dims, l1_dims)
        self.l2 = nn.Linear(l1_dims, output_dims)
        self.optimizer = t.optim.Adam(self.parameters(), lr=1e-2)
        self.rews = []
        self.dones = []
        self.states = []
        self.actions = []
        self.logprobs = []
        self.to(self.device)

    def print(self):
        print("device -", self.device)
        print("gamma -", self.gamma)

    def select_action(self, observation):
        observation = t.from_numpy(observation).float().to(self.device)
        action_probs = self.forward(observation)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        self.logprobs.append(action_logprob)
        return action

    def forward(self, observation):
        x = f.relu(self.l1(observation))
        x = f.softmax(self.l2(x), dim=-1)
        return x

    def evaluate(self):
        states = t.tensor(self.states).float().to(self.device)
        action_probs = self.forward(states)
        dist = Categorical(action_probs)
        actions = dist.sample()
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs,  dist_entropy

    def learn(self):
        rewards = []
        discounted_reward = []
        for reward, is_terminal in zip(reversed(self.rews), reversed(self.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = discounted_reward + \
                (self.gamma*discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = t.tensor(self.rews).to(self.device).detach()
        rewards = (rewards - rewards.mean()) / \
            (rewards.std() + 1e-5)  # normalising the rewards

        log, entr = self.evaluate()
        loss = -(rewards * log * entr)

        self.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = Agent()
    agent.print()
    env = gym.make("CartPole-v0")
    observation = env.reset()

    episode_reward = 0
    for i in range(1000):
        done = False
        while(not done):
            # action = env.action_space.sample()  # agent.select_action
            action = agent.select_action(observation)
            observation, reward, done, info = env.step(action.item())
            agent.rews.append(reward)
            agent.dones.append(done)
            agent.actions.append(action)
            agent.states.append(observation)
            episode_reward += reward
            if done:
                observation = env.reset()
                print("episode", i, "reward", episode_reward)
                episode_reward = 0
                break
        if (i % 10 == 9):
            print("learning")
            agent.learn()
            agent.rews.clear()
            agent.dones.clear()
            agent.states.clear()
            agent.actions.clear()
            agent.logprobs.clear()
    env.close()
