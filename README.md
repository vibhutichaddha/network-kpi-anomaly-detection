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
