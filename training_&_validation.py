import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
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
epochs=50
history=[]
for epoch in range(epochs):
    model.train()
    train_loss=0
    train_correct=0
    train_total=0
    for features,labels in train_loader:
        optimizer.zero_grad()
        outputs=model(features)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item() 
        _,predicted=torch.max(outputs,1)
        train_total+=labels.size(0)
        train_correct+=(predicted==labels).sum().item() 
    avg_train_loss=train_loss/len(train_loader)
    train_accuracy=100*train_correct/train_total
    model.eval()
    val_loss=0
    val_correct=0
    val_total=0
    with torch.no_grad():
        for features,labels in val_loader:
            outputs=model(features)
            loss=criterion(outputs,labels)
            optimizer.step()
            val_loss+=loss.item()
            _,predicted=torch.max(outputs,1)
            val_total+=labels.size(0)
            val_correct+=(predicted==labels).sum().item()
    avg_val_loss=val_loss/len(val_loader)
    val_accuracy=100*val_correct/val_total
    print(f"Epoch[{epoch+1}/{epochs}]"
          f"train loss:{avg_train_loss:.4f}"
          f"val loss:{avg_val_loss:.4f}"
          f"training accuracy:{train_accuracy:.2f}%"
          f"validation accuracy:{val_accuracy:.2f}%")
    torch.save(model.state_dict(),"mlp_model.pth")
    print("Model saved successfully")
    history.append([epoch+1, avg_train_loss, avg_val_loss, train_accuracy, val_accuracy])
    results=pd.DataFrame(history,columns=["Epoch","Training Loss","Validation Loss","Training Accuracy","Validation Accuracy"])
    results.to_csv("training_history.csv",index=False)
    print("\nTraining Completed Successfully")
    print("Data saved to training_history.csv")