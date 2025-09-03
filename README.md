# Daytime Household Occupancy Prediction using Smart Grid Smart City (SGSC) Dataset

## Objective

This project focuses on building a classification model to predict whether a household is occupied during daytime hours using the Smart Grid Smart City (SGSC) dataset. The primary goal is to support energy efficiency strategies in Australia by leveraging machine learning to forecast occupancy patterns and optimize energy distribution accordingly.

## Problem Context

Accurately predicting daytime occupancy has practical implications for:
- Improving demand-side energy management
- Enabling adaptive pricing schemes
- Reducing energy waste
- Supporting policy decisions for grid operations and infrastructure

The project applies data mining and classification techniques to extract actionable insights from residential energy usage patterns and contextual metadata.

## Dataset Access

The dataset used for this project is publicly available from the Australian Government Open Data Portal:

**Smart Grid Smart City (SGSC) Customer Trial Data**  
Link: [https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data](https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data)

This dataset includes daily electricity consumption data, appliance ownership, occupancy schedules, and household characteristics across multiple regions in Australia.

## Phase 1: Data Preprocessing and Feature Engineering

### Dataset Overview

The SGSC dataset includes energy consumption readings along with household attributes such as appliance ownership, occupancy schedules, and regional identifiers.

### Cleaning and Transformation

- Irrelevant columns and constant values were removed.
- Missing values were handled appropriately depending on the data type.
- Categorical variables were encoded using a combination of label encoding and one-hot encoding.
- All features were standardized for use in distance-based algorithms.

### Feature Selection

To isolate the most influential features, ANOVA F-test-based feature selection was applied using `SelectKBest`, which ranked and selected the top 10 features most correlated with the target class. These included variables related to time usage, number of residents, and appliance presence.

## Phase 2: Model Development and Evaluation

### Models Trained

Multiple supervised learning models were implemented and compared, including:
- Logistic Regression (baseline)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Multi-Layer Perceptron (Neural Network)

### Tuning and Validation

Each model was optimized using `GridSearchCV` with 5-fold `StratifiedKFold` cross-validation to ensure balanced performance evaluation. Parameter grids were defined to explore hyperparameter spaces such as:
- Number of neighbors and distance metrics for KNN
- Tree depth and number of estimators for Random Forest
- Hidden layer sizes, learning rates, and activation functions for the neural network

### Evaluation Metrics

The models were assessed using:
- Accuracy
- Precision, Recall, and F1-Score
- Confusion Matrix
- ROC Curve and AUC Score

These metrics provided a comprehensive understanding of each model’s performance, particularly in handling class imbalances.

## Final Results

The best-performing model was a tuned Neural Network (MLPClassifier), which achieved:

- Accuracy: 83%
- High F1-score across both classes
- Strong AUC values demonstrating robust discrimination capability

This model successfully captured complex nonlinear relationships and generalized well to unseen data, outperforming tree-based and linear models in this context.

## Tools and Libraries

- Python 3.8+
- pandas, numpy for data manipulation
- scikit-learn for modeling, feature selection, and evaluation
- matplotlib, seaborn for visual analytics

## Key Achievements

- Designed and implemented a full classification pipeline tailored for occupancy prediction.
- Applied rigorous preprocessing and statistical feature selection to improve model interpretability.
- Validated and tuned multiple machine learning models to identify the most effective approach.
- Achieved a high-performing solution with 83% classification accuracy using a neural network model.

## Future Work

- Incorporate additional time-series features or temporal aggregation.
- Extend the model to predict occupancy at finer hourly intervals.
- Build a web-based interface for stakeholders to query occupancy predictions using live input data.

## Author

Varun Chandra Shekar  
Master of Data Science, RMIT University  
Melbourne, Australia
