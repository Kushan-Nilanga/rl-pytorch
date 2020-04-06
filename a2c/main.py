import gym
from agent import Agent
import matplotlib.pyplot as plt 
import numpy as np
if __name__ == "__main__":
    agent = Agent(alpha=0.00005, beta=0.00025, input_dims=[4], gamma=0.99, n_actions=2, layer1_size=256, layer2_size=256)
    env = gym.make('CartPole-v1')
    score_history = []
    episode_history = []
    mean_history = []
    n_episodes = 3000
    for i in range(n_episodes):
        done = False
        score = 0 
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_

        print(i,". ", 'score: %.3f mean: %.3f' % (score, np.mean(score_history[5:])))
        score_history.append(score)
        mean_history.append(np.mean(score_history[5:]))
        episode_history.append(i)

    plt.plot(episode_history, score_history, label="score")
    plt.plot(episode_history, mean_history, label="mean")
    plt.legend()
    plt.show()