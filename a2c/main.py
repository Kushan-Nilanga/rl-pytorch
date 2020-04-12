import gym
from agent import Agent
import matplotlib.pyplot as plt 
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    agent = Agent(alpha=0.00005, beta=0.00025, input_dims=[4], gamma=0.99, n_actions=2, layer1_size=256, layer2_size=256)
    writer = SummaryWriter()
    env = gym.make('CartPole-v1')
    score_history = []
    episode_history = []
    mean_history = []
    n_episodes = 500
    render = False
    for i in range(n_episodes):
        done = False
        score = 0 
        observation = env.reset()
        al= []
        cl =[]
        a = None
        c = None
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if (render==True) : env.render()
            score += reward
            a, c = agent.learn(observation, reward, observation_, done)
            al.append(a.item())
            cl.append(c.item())
            observation = observation_
        print(i,'score: %.3f  mean: %.3f  actor loss: %.4f  critic loss: %.4f' % (score, np.mean(score_history[5:]), np.mean(al), np.mean(cl)))

        score_history.append(score)
        mean_history.append(np.mean(score_history[5:]))
        episode_history.append(i)
        writer.add_scalar("scores",score, i)

    writer.close()
    plt.plot(episode_history, score_history, label="score")
    plt.plot(episode_history, mean_history, label="mean")
    plt.legend()
    plt.show()