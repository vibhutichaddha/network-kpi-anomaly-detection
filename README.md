# Telecom Network KPI Anomaly Detection using PyTorch

## Objective

The objective of this project is to build a Deep Learning model using PyTorch that classifies telecom network KPI data into three categories:

* Normal
* Warning
* Critical

The model helps network engineers automatically detect abnormal network conditions before they affect customer experience.

# Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn

# Project Structure

```text
network_kpi_anomaly_detection/
│
├── data/
│   └── telecom_kpi.csv
│
├── dataset_prep.py
├── create_dataset.py
├── model.py
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

Studied the basic concepts required for Deep Learning.

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

These concepts form the foundation of neural networks and explain how models learn from data.

# Task 2: Dataset Preparation

The telecom KPI dataset was cleaned and preprocessed before training.

Activities performed:

* Loaded the dataset
* Checked missing values
* Removed duplicate records
* Created the target label (`Network_Status`)
* Encoded target labels using `LabelEncoder`
* Standardized numerical features using `StandardScaler`
* Split the dataset into:

  * Training (70%)
  * Validation (15%)
  * Testing (15%)

The processed datasets were saved as NumPy arrays for later use.

# Task 3: Create PyTorch Dataset

The processed datasets were converted into PyTorch tensors.

Implemented:

* `TensorDataset`
* `DataLoader`

Configuration:

* Batch Size: **32**

Separate DataLoaders were created for:

* Training
* Validation
* Testing

This allows the model to efficiently process the data in mini-batches during training and evaluation.

# Task 4: Build a Simple MLP

A Multi-Layer Perceptron (MLP) was implemented using PyTorch.

The architecture consists of:

* Input Layer
* Hidden Layer (64 neurons)
* ReLU Activation
* Dropout Layer
* Hidden Layer (32 neurons)
* ReLU Activation
* Dropout Layer
* Output Layer (3 neurons)

PyTorch modules used:

* `nn.Linear`
* `nn.ReLU`
* `nn.Dropout`

The output layer predicts one of the following classes:

* Normal
* Warning
* Critical

# Dataset Features

The model uses the following KPI metrics:

* RSRP
* SINR
* Latency
* Throughput
* Packet Loss
* Connected Users

Depending on preprocessing, additional time-based features such as Hour, Day, and Month may also be included.

# Workflow

1. Load telecom KPI dataset.
2. Clean and preprocess the data.
3. Encode target labels.
4. Scale numerical features.
5. Split the dataset into training, validation, and testing sets.
6. Convert the data into PyTorch tensors.
7. Create `TensorDataset` and `DataLoader`.
8. Build the MLP architecture using PyTorch.

# Output

After completing these tasks, the project contains:

* Cleaned telecom KPI dataset
* Encoded target labels
* Standardized feature set
* Train, validation, and test datasets
* PyTorch DataLoaders
* Multi-Layer Perceptron (MLP) model ready for training

The next phase of the project focuses on training the model, evaluating its performance, and using it to predict telecom network anomalies.

## Author
Vibhuti Chaddha
