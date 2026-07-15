import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from simple_mlp import MLP
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
train_dataset=TensorDataset(X_train,y_train)
val_dataset=TensorDataset(X_val,y_val)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)
input_size=X_train.shape[1]
model=MLP(input_size)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=20
for epoch in range(epochs):
    model.train()
    running_loss=0.0
    for features,labels in train_loader:
        outputs=model(features)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    avg_loss=running_loss/len(train_loader)
    print(f"Epoch [{epoch+1/epochs}]Loss:{avg_loss:.4f}")
print("\nTraining Completed Successfully")