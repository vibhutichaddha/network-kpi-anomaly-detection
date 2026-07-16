import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay)
from torch.utils.data import TensorDataset,DataLoader
from simple_mlp import MLP
X_test=np.load("X_test.npy")
y_test=np.load("y_test.npy")
X_test=torch.tensor(X_test,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.long).squeeze()
test_dataset=TensorDataset(X_test,y_test)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
input_size=X_test.shape[1]
model=MLP(input_size)
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()
true_labels=[]
predicted_labels=[]
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())
accuracy=accuracy_score(true_labels,predicted_labels)
precision=precision_score(true_labels,predicted_labels,average="weighted")
recall=recall_score(true_labels,predicted_labels,average="weighted")
f1=f1_score(true_labels,predicted_labels,average="weighted")
print("\nModel Evaluation")
print(f"Accuracy:{accuracy:.4f}")
print(f"Precision:{precision:.4f}")
print(f"Recall:{recall:.4f}")
print(f"F1 Score:{f1:.4f}")
cm=confusion_matrix(true_labels,predicted_labels)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Normal","Warning","Critical"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
print("\nConfusion matrix saved as confusion_matrix.png")