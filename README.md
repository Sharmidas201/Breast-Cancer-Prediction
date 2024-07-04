# Breast-Cancer-Prediction
This project diagnoses breast cancer using a Random Forest classifier. The dataset contains features extracted from cell nuclei in breast biopsy samples, and the model predicts whether a tumor is benign or malignant.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview

The goal is to build a model that accurately classifies breast tumors as benign or malignant. Techniques like class weighting and SMOTE are used to handle class imbalance.

## Dataset

Features include:
- **radius_mean**: Mean radius
- **texture_mean**: Standard deviation of gray-scale values
- **perimeter_mean**: Mean perimeter
- **area_mean**: Mean area
- **smoothness_mean**: Mean local variation in radius lengths
- **compactness_mean**: Mean of perimeter^2 / area - 1.0
- **concavity_mean**: Mean of concave portions severity
- **concave points_mean**: Mean number of concave portions
- **radius_se**: Standard error of mean radius
- **radius_worst**: Largest mean radius value

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/breast-cancer-diagnosis.git
    cd breast-cancer-diagnosis
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Preprocess the data**:
    ```python
    # Load dataset and convert diagnosis column
    df = pd.read_csv('data.csv')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Split data into features and target variable
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    ```

2. **Balance the dataset using SMOTE**:
    ```python
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    ```

3. **Train the Random Forest model with GridSearchCV**:
    ```python
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                               param_grid, cv=5, scoring='recall')
    grid_search.fit(X_resampled, y_resampled)

    rf_best = grid_search.best_estimator_
    y_pred_best = rf_best.predict(X_test)

    print("Best Model Evaluation:")
    print(classification_report(y_test, y_pred_best))
    ```

## Results
```
Classification report for the best model:
          precision    recall  f1-score   support
       0       0.99      0.97      0.98        71
       1       0.95      0.98      0.97        43
accuracy                           0.97       114

