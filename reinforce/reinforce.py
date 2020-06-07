import torch.nn as nn
import torch as t

import gym


class Agent(nn.Module):
    def __init__(self, gamma=0.99, input_dims=4, l1_dims=256):
        super().__init__()  # nn.Module is initialised
        self.gamma = gamma
        self.device = "cuda:0" if t.cuda.is_available() else "cpu:1"
        self.l1 = nn.Linear(input_dims, l1_dims)

    def print(self):
        print("device -", self.device)
        print("gamma -", self.gamma)

    def select_action(self, observation):
        observation = t.tensor(observation).to(self.device)
        return 1

    def forward(self):


if __name__ == "__main__":
    agent = Agent()
    agent.print()
    env = gym.make("CartPole-v0")
    observation = env.reset()

    episode_reward = 0
    for i in range(20):
        done = False
        while(not done):
            env.render()
            # action = env.action_space.sample()  # agent.select_action
            action = agent.select_action(observation)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                observation = env.reset()
                print("episode", i, "reward", episode_reward)
                episode_reward = 0
                break
        if (i % 10 == 0):
            print("learning")
    env.close()
