import torch as T 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 

class Agent(nn.Module):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=2, layer1_size=1024, layer2_size=512, input_dims=4):
        super(Agent, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_actions = n_actions
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.input_dims = input_dims
        self.action_space = [i for i in range(self.n_actions)]  
        self.layer1 = nn.Linear(*self.input_dims, self.layer1_size)
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size)
        self.probs = nn.Linear(self.layer2_size, self.n_actions)
        self.values = nn.Linear(self.layer2_size, 1)
        """ 
            [
                {'params':self.layer1.weight, 'lr':self.alpha},
                {'params':self.layer1.bias, 'lr':self.alpha},
                {'params':self.layer2.weight, 'lr':self.alpha},
                {'params':self.layer2.bias, 'lr':self.alpha},
                {'params':self.probs.weight, 'lr':self.alpha},
                {'params':self.probs.bias, 'lr':self.alpha},
                {'params':self.values.weight, 'lr':self.beta},
                {'params':self.values.bias, 'lr':self.beta},
            ]
        ,  """ 
        self.optimizer = optim.Adam(self.parameters(),lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, observation):
        state_value = T.Tensor(observation).to(self.device)
        x = F.relu(self.layer1(state_value))
        x = F.relu(self.layer2(x))
        probs = F.softmax(self.probs(x))
        value = self.values(x)
        return probs, value

    def choose_action(self, observation):
        probabilities, _ = self.forward(observation)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        return action.item()

    def learn(self, state, reward, new_state, done):
        self.optimizer.zero_grad()
        _, critic_value = self.forward(state)
        _, critic_value_new = self.forward(new_state)
        delta = reward + self.gamma*critic_value_new*(1-int(done)) - critic_value
        
        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss+critic_loss).backward()
        self.optimizer.step()

