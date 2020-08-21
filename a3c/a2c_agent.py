import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class A2CAgent(nn.Module):
    def __init__(self, lr=0.001, gamma=.99, cpu="cpu:0"):
        super.__init__(A2CAgent, self)
        self._input_dims = 4
        self._action_dims = 2
        self.layer1_dims = 256
        self.layer2_dims = 256
        self.gamma = gamma
        self.lr = lr
        self.cpu = cpu
        self.layer1 = nn.Linear(self._input_dims, self.layer1_dims)
        self.layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.action_head = nn.Linear(self.layer2_dims, self._action_dims)
        self.critic_head = nn.Linear(self.layer2_dims, 1)
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.device = "cuda:0" if t.cuda.is_available() else self.cpu

    def __forward(self, observation):
        state = t.from_numpy(observation).to(self.device, dtype=t.float32).requires_grad(False)
        x = t.relu(self.layer1(state))
        x = t.relu(self.layer2(x))
        action_returns = self.action_head(x)
        value_returns = nn.Sigmoid(self.critic_head(x))
        return action_returns, value_returns

    def __select_action(self, input):
        action_probs, action_value = self.__forward(input)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def __evaluate(self, state, action):
        action_probs, action_value = self.__forward(state)
        dist = Categorical(action_probs)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprob, t.squeeze(action_value), dist_entropy

    def __learn(self):
        return self