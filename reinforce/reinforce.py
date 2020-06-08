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

    def print(self):
        print("device -", self.device)
        print("gamma -", self.gamma)

    def select_action(self, observation):
        action_probs = self.forward(observation)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action

    def forward(self, observation):
        observation = t.tensor(observation).to(self.device)
        x = f.relu(self.l1(observation))
        x = f.softmax(self.l2(x), dim=-1)
        return x

    def learn(self, rewards, dones, states):
        discounted_reward = []
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0

        self.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = Agent()
    agent.print()
    env = gym.make("CartPole-v0")
    observation = env.reset()

    rews = []
    dones = []
    states = []

    episode_reward = 0
    for i in range(20):
        done = False
        while(not done):
            env.render()
            # action = env.action_space.sample()  # agent.select_action
            action = agent.select_action(observation)
            observation, reward, done, info = env.step(action)
            rews.append(reward)
            dones.append(done)
            states.append(observation)
            episode_reward += reward
            if done:
                observation = env.reset()
                print("episode", i, "reward", episode_reward)
                episode_reward = 0
                break
        if (i % 10 == 0):
            print("learning")
            agent.learn(rews, dones, states)
            rews.clear()
            dones.clear()
            states.clear()
    env.close()
