import torch as T
import torch.nn as nn 
import torch.functional as F 

class Memory:
    def __init__(self):
        action_memory = []
        state_memory = []
        reward_memory = []
        done_memory = []

    def clear_memory(self):
        del self.action_memory[:]
        del self.state_memory[:]
        del self.reward_memory[:]
        del self.done_memory[:]

class Agent(nn.Module):
    def __init__(self, input_size=4, hidden1_size=256, hidden2_size=256, output_size=2):
        super (Agent, self).__init__()
        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu:0'
        self.memory = Memory()

        policy = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size),
        )

    
    
