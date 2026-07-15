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

# Project Structure

```text
network_kpi_anomaly_detection/
│
├── telecom_kpi_balanced.csv
├── dataset_prep.py
├── create_dataset.py
├── simple_mlp.py
├── train.py
├── learning_rate_experiment.py
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

Studied the core concepts of Artificial Neural Networks.

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

These concepts provide the theoretical foundation for building and training deep learning models.

# Task 2: Dataset Preparation

The telecom KPI dataset was cleaned and preprocessed before training.

Activities performed:

* Loaded the dataset
* Checked missing values
* Removed duplicate records
* Generated the target column (`Network_Status`)
* Encoded target labels using `LabelEncoder`
* Standardized numerical features using `StandardScaler`
* Split the dataset into:

  * Training (70%)
  * Validation (15%)
  * Testing (15%)
* Saved the processed datasets as NumPy arrays

# Task 3: Create PyTorch Dataset

The processed data was converted into PyTorch tensors and organized for model training.

Implemented:

* TensorDataset
* DataLoader

Configuration:

* Batch Size: **32**

Separate DataLoaders were created for:

* Training
* Validation
* Testing

# Task 4: Build a Simple MLP

A Multi-Layer Perceptron (MLP) was implemented using PyTorch.

Model Architecture:

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

# Task 5: Forward and Backward Propagation

The MLP model was trained using the standard deep learning training pipeline.

Implemented:

* CrossEntropyLoss
* Adam Optimizer
* Forward Pass
* Loss Calculation
* Backpropagation
* Weight Updates

Training Process:

1. Load a batch of training data.
2. Perform a forward pass through the network.
3. Compute the classification loss.
4. Calculate gradients using backpropagation.
5. Update model weights using the Adam optimizer.
6. Repeat for multiple epochs until training is complete.

# Task 6: Learning Rate Experiments

The model was trained using three different learning rates to study their impact on performance.

Learning rates tested:

* 0.1
* 0.01
* 0.001

For each experiment:

* A new MLP model was initialized.
* The model was trained for multiple epochs.
* Training loss was recorded.
* Training accuracy was calculated.
* Validation accuracy was evaluated.

### Observations

| Learning Rate | Convergence Speed | Stability   | Expected Performance                        |
| ------------- | ----------------- | ----------- | ------------------------------------------- |
| 0.1           | Very Fast         | Less Stable | May fluctuate during training               |
| 0.01          | Fast              | Stable      | Good balance between speed and accuracy     |
| 0.001         | Slow              | Very Stable | Smooth convergence with gradual improvement |

# Dataset Features

The model uses the following telecom KPIs:

* RSRP
* SINR
* Latency
* Throughput
* Packet Loss
* Connected Users

Additional time-based features may also be included depending on the preprocessing strategy.

# Project Workflow

1. Load the telecom KPI dataset.
2. Clean and preprocess the data.
3. Generate and encode target labels.
4. Standardize numerical features.
5. Split the dataset into training, validation, and testing sets.
6. Convert the data into PyTorch tensors.
7. Create TensorDatasets and DataLoaders.
8. Build the MLP model.
9. Train the model using forward propagation and backpropagation.
10. Perform learning rate experiments and compare model performance.

# Output

After completing Tasks 1–6, the project includes:

* Cleaned and preprocessed telecom KPI dataset
* Encoded target labels
* Standardized feature set
* Training, validation, and testing datasets
* PyTorch TensorDatasets and DataLoaders
* Multi-Layer Perceptron (MLP) model
* Model training using CrossEntropyLoss and Adam optimizer
* Learning rate comparison experiments
* Training and validation accuracy for different learning rates

The project is now ready for the next phase, which focuses on model evaluation, performance visualization, and anomaly prediction on unseen telecom KPI data.

## Author
Vibhuti Chaddha
