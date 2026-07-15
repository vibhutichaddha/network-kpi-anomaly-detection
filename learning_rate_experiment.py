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
learning_rates=[0.1,0.01,0.001]
epochs=20
for lr in learning_rates:
    print("="*50)
    print(f"Training of Learning Rate={lr}")
    print("="*50)
    model=MLP(input_size)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss=0
        correct=0
        total=0
        for features,labels in train_loader:
            optimizer.zero_grad()
            outputs=model(features)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            _,predicted=torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        train_accuracy=100*correct/total
        model.eval()
        val_correct=0
        val_total=0
        with torch.no_grad():
            for features,labels in val_loader:
                outputs=model(features)
                _,predicted=torch.max(outputs,1)
                val_total+=labels.size(0)
                val_correct+=(predicted==labels).sum().item()
        val_accuracy=100*val_correct/val_total
        print(f"Epoch{epoch+1:2d} |"
              f"Loss:{running_loss/len(train_loader):.4f} |"
              f"Train Acc:{train_accuracy:.2f}% |"
              f"Val Acc:{val_accuracy:.2f}%")
print("\nLearning Rate Experiment Successful")