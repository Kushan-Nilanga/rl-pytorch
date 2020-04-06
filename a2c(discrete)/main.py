import gym
from agent import Agent
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    agent = Agent(alpha=0.0001, beta=0.0005, input_dims=[4], gamma=0.99, n_actions=2, l1_size=32, l2_size=32)
    env = gym.make('CartPole-v1')
    score_history = []
    episode_history = []
    n_episodes = 500
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

        print(i,". ", 'score: %.3f' % score)
        score_history.append(score)

    plt.plot(episode_history, score_history)