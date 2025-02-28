# Predictive Maintenance Classification Model for Manufacturing Industries

## Overview
This project implements a machine learning pipeline to predict machine failure using sensor data. The dataset is sourced from Kaggle and consists of numerical features such as air temperature, process temperature, rotational speed, torque, and tool wear. A deep learning model using TensorFlow is trained to classify whether a machine will fail.

## Dataset
The dataset is downloaded from Kaggle using `kagglehub`:
- **Dataset Name**: Machine Predictive Maintenance Classification
- **File Used**: `predictive_maintenance.csv`
- **Target Variable**: `Target` (Binary classification: 1 for failure, 0 for no failure)
- **Features**:
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`
  - `Product Quality` (L, M, H)

## Dependencies
Ensure you have the following libraries installed before running the script:
```sh
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow kagglehub
```

## Project Workflow
### 1. **Data Loading**
- The dataset is downloaded and loaded into a Pandas DataFrame.
- Initial exploration using `.head()`, `.info()`, and `.describe()`.

### 2. **Data Cleaning**
- Checking for missing values and duplicates.
- Removing duplicate rows if found.

### 3. **Exploratory Data Analysis (EDA)**
- Distribution of `Target` (machine failure) using a countplot.
- Distribution of `Product Quality` (Low, Medium, High) derived from the `Product ID`.
- Correlation heatmap of numerical features.
- Pairplot visualization to analyze feature relationships with `Target`.

### 4. **Feature Engineering & Encoding**
- Mapping `Product Quality` into numeric values (`L:0, M:1, H:2`).
- Selecting relevant numerical features.

### 5. **Data Splitting**
- Splitting data into **training (80%)** and **testing (20%)** sets using `train_test_split` with stratification on `Target`.

### 6. **Deep Learning Model**
A feedforward neural network is implemented using TensorFlow/Keras:
- **Input Layer**: 6 features
- **Hidden Layers**:
  - Dense layer (16 neurons, ReLU activation)
  - Dense layer (8 neurons, ReLU activation)
- **Output Layer**:
  - Dense layer (1 neuron, Sigmoid activation for binary classification)

**Compilation:**
- Loss function: `binary_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`

### 7. **Training**
- Training for **15 epochs** with a batch size of **32**.
- Using **validation split of 20%**.

### 8. **Evaluation**
- Predictions are generated using the trained model.
- Model performance is evaluated using:
  - **Confusion Matrix**
  - **Classification Report**
  - **Accuracy & Loss Plots** (Training vs Validation)
- Final model performance (Loss & Accuracy) is printed.

## Results
- Model accuracy and loss are displayed after evaluation.
- Training & Validation performance trends are plotted, and can be seen after ruuning the model.

## How to Run
1. Download the dataset from Kaggle using the following command:
   ```python
   base_path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
   ```
2. Run the script:
   ```sh
   python your_script_name.py
   ```

## Future Improvements
- **Feature Selection**: Investigate feature importance to improve model efficiency.
- **Hyperparameter Tuning**: Optimize neural network architecture and training parameters.
- **Alternative Models**: Compare performance with tree-based models (Random Forest, XGBoost).
- **Deploy the Model**: Convert the model into an API using FastAPI or Flask.

## Author
- **Akash Raj Singh**
