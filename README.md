# winequality-datascience-end-to-end-mlops

## Template Script

The `template.py` script is designed to provide a basic structure for your data science project. It includes the following functionalities:

- **Data Loading**: Load your dataset from a specified path.
- **Data Preprocessing**: Perform basic preprocessing steps such as handling missing values and scaling features.
- **Model Training**: Train a machine learning model using the preprocessed data.
- **Model Evaluation**: Evaluate the trained model using appropriate metrics.
- **Model Saving**: Save the trained model to disk for future use.


### Workflows--ML Pipeline

1. Data Ingestion
2. Data Validation
3. Data Transformation-- Feature Engineering,Data Preprocessing
4. Model Trainer
5. Model Evaluation- MLFLOW,Dagshub

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity 
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py


### Usage

To use the `template.py` script, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/winequality-datascience-end-to-end-mlops.git
    cd winequality-datascience-end-to-end-mlops
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the script**:
    ```bash
    python template.py --data_path path/to/your/data.csv
    ```

### Example

Here is an example command to run the script:

```bash
python template.py --data_path data/winequality-red.csv
```

This command will load the dataset from `data/winequality-red.csv`, preprocess the data, train a model, evaluate it, and save the model to disk.

### Script Structure

The `template.py` script is structured as follows:

```python
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data(data_path):
    return pd.read_csv(data_path)

def preprocess_data(df):
    df = df.dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')
    return model

def save_model(model, model_path='model.joblib'):
    joblib.dump(model, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Template script for data science project')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()

    data = load_data(args.data_path)
    X, y = preprocess_data(data)
    model = train_model(X, y)
    save_model(model)
```

This template provides a starting point for your data science project, allowing you to focus on the specifics of your analysis and modeling.