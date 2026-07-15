import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self,input_size):
        super(MLP,self).__init__()
        self.network=nn.Sequential(nn.Linear(input_size,64),nn.ReLU(),nn.Dropout(0.3),
                                   nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.3),
                                   nn.Linear(32,3))
    def forward(self,x):
        return self.network(x)
if __name__=="__main__":
    input_size=6
    model=MLP(input_size)
    print(model)