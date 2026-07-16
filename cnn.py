import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
transform=transforms.ToTensor()
train_dataset=datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_dataset=datasets.MNIST(root="./data",train=False,download=True,transform=transform)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(32*14*14,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x
model=SimpleCNN()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=5
for epoch in range(epochs):
    running_loss=0
    correct=0
    total=0
    model.train()
    for images,labels in train_loader:
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    accuracy=100*correct/total
    print(f"Epoch {epoch+1}/{epochs} |"
          f"Loss: {running_loss/len(train_loader):.4f} |"
          f"Accuracy: {accuracy:.2f}")
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print("\nTest Accuracy:",100*correct/total)
torch.save(model.state_dict(),"cnn_mnist.pth")
print("Model Saved Successfully!")