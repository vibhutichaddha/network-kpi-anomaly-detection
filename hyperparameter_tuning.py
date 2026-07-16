import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
y_val = torch.tensor(y_val, dtype=torch.long).squeeze()
input_size=X_train.shape[1]
class MLP(nn.Module):
    def __init__(self,input_size,hidden_units):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(input_size,hidden_units),nn.ReLU(),nn.Dropout(0.3),
                                   nn.Linear(hidden_units,hidden_units//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hidden_units//2,3))
    def forward(self,x):
        return self.network(x)
hidden_units_list=[32,64,128]
batch_sizes=[16,32,64]
epochs_list=[25,50,100]
results=[]
for hidden_units in hidden_units_list:
    for batch_size in batch_sizes:
        train_dataset=TensorDataset(X_train,y_train)
        val_dataset=TensorDataset(X_val,y_val)
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
        for epochs in epochs_list:
            print(f"Hidden Units={hidden_units}",f"Batch Size={batch_sizes}",f"Epochs={epochs}")
            model=MLP(input_size,hidden_units)
            criterion=nn.CrossEntropyLoss()
            optimizer=optim.Adam(model.parameters(),lr=0.001)
            for epoch in range(epochs):
                model.train()
                for features,labels in train_loader:
                    optimizer.zero_grad()
                    outputs=model(features)
                    loss=criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()
            model.eval()
            correct=0
            total=0
            with torch.no_grad():
                 outputs=model(features)
                 _,predicted=torch.max(outputs,1)
                 total+=labels.size(0)
                 correct+=(predicted==labels).sum().item()
            accuracy=100*correct/total
            print(f"Validation Accuracy:{accuracy:.2f}%")
            results.append([hidden_units,batch_size,epochs,round(accuracy,2)])
results_df=pd.DataFrame(results,columns=["Hidden Units","Batch Size","Epochs","Validation Accuracy"])
results_df.to_csv("hyperparameter_results.csv",index=False)
print("\nHyperparameter Tuning Complete")
print(results_df)