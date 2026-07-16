# Telecom Network KPI Anomaly Detection using PyTorch

## Objective

The objective of this project is to build a Deep Learning model using PyTorch to classify telecom network KPI data into three categories:

* Normal
* Warning
* Critical

The model helps telecom operators automatically detect abnormal network conditions before they impact customer experience.

# Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

# Project Structure

```text
network_kpi_anomaly_detection/
│
├── telecom_kpi_balanced.csv
├── dataset_prep.py
├── create_dataset.py
├── simple_mlp.py
├── train.py
├── training_validation.py
├── learning_rate_experiment.py
├── hyperparameter_tuning.py
├── model_evaluation.py
├── visualization.py
├── training_history.csv
├── hyperparameter_results.csv
├── mlp_model.pth
├── confusion_matrix.png
├── README.md
│
├── X_train.npy
├── X_val.npy
├── X_test.npy
├── y_train.npy
├── y_val.npy
└── y_test.npy
```

# Task 1: Neural Network Fundamentals

Studied the fundamental concepts required for building neural networks.

Topics covered:

* Perceptron
* Artificial Neuron
* Activation Functions

  * Sigmoid
  * Tanh
  * ReLU
  * Softmax
* Forward Propagation
* Backpropagation

These concepts form the theoretical foundation for developing deep learning models.

# Task 2: Dataset Preparation

Prepared the telecom KPI dataset for model training.

Activities performed:

* Loaded the dataset
* Checked for missing values
* Removed duplicate records
* Generated the `Network_Status` target column
* Encoded target labels using `LabelEncoder`
* Standardized numerical features using `StandardScaler`
* Split the dataset into:

  * Training (70%)
  * Validation (15%)
  * Testing (15%)
* Saved processed datasets as NumPy arrays

# Task 3: Create PyTorch Dataset

Converted the processed datasets into PyTorch tensors.

Implemented:

* TensorDataset
* DataLoader

Configuration:

* Batch Size: 32

Separate DataLoaders were created for:

* Training
* Validation
* Testing

# Task 4: Build a Simple MLP

Designed a Multi-Layer Perceptron (MLP) using PyTorch.

Architecture:

* Input Layer
* Hidden Layer (64 neurons)
* ReLU Activation
* Dropout Layer
* Hidden Layer (32 neurons)
* ReLU Activation
* Dropout Layer
* Output Layer (3 neurons)

Modules used:

* `nn.Linear`
* `nn.ReLU`
* `nn.Dropout`

The output layer predicts:

* Normal
* Warning
* Critical

# Task 5: Forward and Backward Propagation

Implemented the complete training process.

Components used:

* CrossEntropyLoss
* Adam Optimizer

Training pipeline:

* Forward Pass
* Loss Calculation
* Backpropagation
* Weight Updates

The model learns by minimizing the classification loss over multiple epochs.

# Task 6: Learning Rate Experiments

Evaluated model performance using different learning rates.

Learning rates tested:

* 0.1
* 0.01
* 0.001

For each learning rate:

* Model trained independently
* Training loss recorded
* Training accuracy calculated
* Validation accuracy evaluated

The experiments helped compare convergence speed, stability, and overall model performance.

# Task 7: Training and Validation

The MLP was trained for **50 epochs**.

Metrics recorded for every epoch:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

All training statistics were stored in:

```text
training_history.csv
```

This file provides a complete history of the model's learning progress.

# Task 8: Training Curve Visualization

The training history was visualized using Matplotlib.

Generated plots:

* Training Loss vs Epochs
* Validation Loss vs Epochs
* Training Accuracy vs Epochs
* Validation Accuracy vs Epochs

Output figures:

* `training_loss.png`
* `validation_loss.png`
* `training_accuracy.png`
* `validation_accuracy.png`

These plots help analyze convergence, detect overfitting, and understand the learning behavior of the model.

# Task 9: Hyperparameter Tuning

Performed experiments using different combinations of hyperparameters.

### Hidden Units

* 32
* 64
* 128

### Batch Size

* 16
* 32
* 64

### Epochs

* 25
* 50
* 100

Each combination was trained and evaluated on the validation dataset.

Results were stored in:

```text
hyperparameter_results.csv
```

This process helps identify the best-performing model configuration.

# Task 10: Model Evaluation

The final trained model was evaluated using the test dataset.

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score

A Confusion Matrix was also generated to compare predicted labels with actual labels.

Output:

* `confusion_matrix.png`

The trained model is saved as:

```text
mlp_model.pth
```

This model file can be loaded later for testing or inference without retraining.

# Task 11: Overfitting Analysis

## Objective

The objective of this task is to analyze the training behavior of the Multi-Layer Perceptron (MLP) model and determine whether overfitting occurred during training. The analysis is based on the recorded training and validation metrics obtained over multiple epochs.

## Analysis Questions

### 1. Did the training loss continue decreasing while the validation loss increased?

The training and validation loss curves were examined throughout the training process.

* If the training loss continues to decrease while the validation loss starts increasing, it indicates that the model is beginning to memorize the training data instead of learning general patterns.
* If both training and validation loss decrease together, the model is learning effectively without significant overfitting.

This comparison helps determine whether the model generalizes well to unseen data.

### 2. Did Dropout improve performance?

Dropout was incorporated into the hidden layers of the MLP architecture.

Effects of Dropout:

* Reduces overfitting by randomly deactivating neurons during training.
* Improves the model's ability to generalize.
* Prevents excessive dependence on specific neurons.
* Produces smoother validation loss and more stable validation accuracy.

Overall, Dropout contributes to better performance on unseen data by reducing model complexity during training.

### 3. Which learning rate worked best?

Three learning rates were evaluated:

* 0.1
* 0.01
* 0.001

Observations:

| Learning Rate | Convergence Speed | Stability | Performance                                                       |
| ------------- | ----------------- | --------- | ----------------------------------------------------------------- |
| 0.1           | Very Fast         | Low       | Training may become unstable and overshoot the optimal solution   |
| 0.01          | Fast              | High      | Best balance between convergence speed and validation performance |
| 0.001         | Slow              | Very High | Stable learning but requires more epochs to converge              |

Among the tested values, **0.01** provided the best trade-off between convergence speed, stability, and model accuracy.

## Conclusion

The overfitting analysis shows how training and validation metrics evolve throughout the learning process. Monitoring these metrics, together with the use of Dropout and appropriate learning rate selection, helps develop a model that performs well on both training and unseen test data.

# Task 12: Model Comparison

## Objective

The objective of this task is to compare the performance of different model configurations and training strategies to identify the most effective approach for telecom network KPI anomaly detection.

## Models Compared

### Multi-Layer Perceptron (MLP)

* Fully connected neural network.
* Suitable for structured telecom KPI data.
* Uses dense layers with ReLU activation and Dropout.

### CNN (Reference)

* Designed primarily for image data.
* Implemented on the MNIST handwritten digit dataset to understand convolution-based deep learning.
* Included for learning purposes and architectural comparison.

## Comparison Parameters

The following aspects were compared:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy
* Learning Rate Performance
* Hyperparameter Configurations
* Model Stability
* Generalization Ability

## Summary

| Parameter           | MLP                                   |
| ------------------- | ------------------------------------- |
| Dataset             | Telecom KPI Dataset                   |
| Input Type          | Structured Numerical Data             |
| Hidden Layers       | Fully Connected Layers                |
| Activation Function | ReLU                                  |
| Regularization      | Dropout                               |
| Optimizer           | Adam                                  |
| Loss Function       | CrossEntropyLoss                      |
| Best Learning Rate  | 0.01                                  |
| Evaluation Metrics  | Accuracy, Precision, Recall, F1 Score |

## Conclusion

The Multi-Layer Perceptron demonstrated effective performance for structured telecom KPI classification. By combining feature scaling, Dropout, hyperparameter tuning, and appropriate learning rate selection, the model achieved reliable classification performance while minimizing overfitting.

# Task 13: Convolutional Neural Network (CNN)

## Objective

The objective of this task is to study the fundamental concepts of Convolutional Neural Networks (CNNs) and implement a simple CNN using the MNIST handwritten digit dataset.

## Concepts Studied

### Convolution

Convolution applies learnable filters (kernels) to an input image to extract important visual features such as edges, corners, and textures. These filters slide across the image and produce feature maps that highlight meaningful patterns.

### Pooling

Pooling reduces the spatial dimensions of feature maps while preserving important information. Max Pooling is used to decrease computational cost, reduce overfitting, and improve the model's robustness.

### Feature Maps

Feature maps are the outputs generated after applying convolution filters. They represent the features learned by the CNN, progressing from simple patterns (such as edges) to more complex structures in deeper layers.

## Dataset

The CNN model is trained on the MNIST dataset.

Dataset Details:

* Training Images: 60,000
* Testing Images: 10,000
* Image Size: 28 × 28 pixels
* Number of Classes: 10 (Digits 0–9)

## CNN Architecture

The implemented CNN consists of:

* Input Layer
* Convolution Layer (`Conv2d`)
* ReLU Activation
* Max Pooling Layer (`MaxPool2d`)
* Fully Connected Layer (`Linear`)
* Output Layer (10 Neurons)

## Libraries Used

* Python
* PyTorch
* Torchvision
* Matplotlib

## Training Configuration

* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Learning Rate: 0.001
* Batch Size: 64
* Epochs: 5

## Output

The CNN model generates:

* Training Loss
* Training Accuracy
* Test Accuracy

The trained model is saved as:

```text
cnn_mnist.pth
```

# Dataset Features

The model uses the following telecom KPI features:

* RSRP
* SINR
* Latency
* Throughput
* Packet Loss
* Connected Users

These KPIs are used to classify the network status into Normal, Warning, or Critical.

## Author
Vibhuti Chaddha
