# Daytime Household Occupancy Prediction using Smart Grid Smart City (SGSC) Dataset  

## Problem Statement  
Efficient energy management is central to modern smart grid initiatives. A critical driver of demand-side optimization is knowing **whether households are occupied during daytime hours**.  

Accurately predicting occupancy enables:  
- **Improved demand-side management**: aligning energy supply with actual household demand.  
- **Adaptive pricing schemes**: enabling flexible tariffs to encourage off-peak usage.  
- **Reduced energy waste**: minimizing unnecessary distribution to empty homes.  
- **Policy and infrastructure planning**: guiding government and utilities for sustainable grid operations.  

This project applies **data mining and machine learning classification techniques** to forecast household occupancy during daytime hours in Australia using the publicly available SGSC dataset.  

---

## Dataset Information  

The dataset used is the **Smart Grid Smart City (SGSC) Customer Trial Data**, provided by the **Australian Government Open Data Portal**:  
[Smart Grid Smart City Customer Trial Data](https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data)  

### Metadata Overview  
The SGSC dataset combines **demographics, tariffs, appliance ownership, and lifestyle/operational attributes**:  
- **Identifiers**: `CUSTOMER_KEY`, trial participation (`TRIAL_CUSTOMER_TYPE`), control group flags.  
- **Tariff and Technology Codes**: `TARIFF_PRODUCT_CD`, `FEEDBACK_TECH1_PRODUCT_CD`, `FEEDBACK_TECH2_PRODUCT_CD`.  
- **Household & Regional Info**: `UNITTYPE`, `CLIMATEZONE`, number of residents.  
- **Appliance Ownership**: `HAS_INTERNET_ACCESS`, `HAS_GAS_HEATING`, `HAS_POOLPUMP`, `HAS_AIRCON`, `HAS_GAS_COOKING`, `HAS_DRYER`.  
- **Usage Attributes**: `GasUse`, `ElectricityUse`, `DryerUse` (ordinal).  
- **Target Variable**: `IS_HOME_DURING_DAYTIME` (1 = occupied, 0 = unoccupied).  

This structured metadata allows machine learning models to connect household context and appliance usage to actual occupancy patterns.  

---

## Phase 1: Data Preprocessing & Feature Engineering  

### Cleaning & Transformation  
1. **Dropping irrelevant fields**: identifiers (`CUSTOMER_KEY`), administrative dates (`SMART_METER_INSTALLATION_DATE`, `OPERATION_FINISH_DATE`), and static codes.  
2. **Handling missing values**:  
   - Numerical attributes imputed with **group-based medians** (e.g., grouped by `UnitType`).  
   - Categorical attributes imputed with **mode** (most frequent).  
3. **Encoding categorical variables**:  
   - **Ordinal encoding** for ordered usage features (`GasUse`, `ElectricityUse`, `DryerUse`).  
   - **One-Hot encoding** for nominal features (`ClimateZone`, `UnitType`, appliance flags).  
4. **Standardization**: continuous variables standardized to improve distance-based models like KNN.  

### Feature Selection  
- Applied **ANOVA F-test (SelectKBest)** to rank features most correlated with occupancy.  
- Applied **Random Forest feature importance** to capture nonlinear influence.  
- Retained overlapping features from both methods (e.g., number of residents, appliance presence, climate zone).  

---

## Phase 2: Model Development & Evaluation  

### Models Trained  
- **Logistic Regression** (baseline)  
- **K-Nearest Neighbors (KNN)**  
- **Random Forest Classifier**  
- **Neural Network (MLPClassifier)**  

### Hyperparameter Tuning  
- Used **GridSearchCV** with **5-fold StratifiedKFold cross-validation**.  
- Tuned parameters included:  
  - KNN → neighbors, distance metric.  
  - Random Forest → number of trees, max depth.  
  - Neural Network → hidden layer sizes, activation, learning rate.  

### Evaluation Metrics  
- Accuracy  
- Precision, Recall, F1-score (per class)  
- Confusion Matrix  
- ROC Curve & AUC  

---

## Final Results  

### Random Forest (Best Model)  
- **Accuracy**: 71%  
- **ROC-AUC**: 0.78  
- **Macro-F1**: 0.63, **Weighted-F1**: 0.69  
- **Class 1 (Occupied)**: Precision 74%, Recall 88%, F1 = 0.81  
- **Class 0 (Unoccupied)**: Precision 59%, Recall 37%, F1 = 0.46  
- **Confusion Matrix**:  
  - TP = 2,733  
  - FN = 375  
  - TN = 548  
  - FP = 937  

**Insight**: The model emphasizes detecting occupied homes (high recall for class 1), which is preferable for grid optimization where missing occupancy has higher costs than false alarms.  

### Neural Network (MLP Classifier)  
- **Accuracy**: 70%  
- **ROC-AUC**: 0.77  
- **Class 1 (Occupied)**: Recall = 91% (strong)  
- **Class 0 (Unoccupied)**: Recall = 27% (weak)  

**Insight**: Neural Network captured nonlinear behavior but underperformed on class 0, limiting its balance.  

---

## Tools & Technologies  
- **Programming**: Python 3.8+  
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Techniques**: Feature selection (ANOVA F-test, Random Forest importance), cross-validation, hyperparameter tuning  
- **Visualization**: Confusion matrices, ROC curves, feature importance plots  

---

## Key Achievements  
Built a complete end-to-end **classification pipeline** for occupancy prediction.  
Applied rigorous preprocessing, encoding, and hybrid feature selection.  
Evaluated multiple models with **GridSearchCV + StratifiedKFold**.  
Achieved strong results with **Random Forest (ROC-AUC 0.78)**, aligning with the project’s energy efficiency goals.  

---

## Future Work  
- Add **time-series features** for hourly-level predictions.  
- Increase granularity to predict **hourly occupancy intervals**.  
- Develop a **web interface** for stakeholders to query predictions dynamically.  

---

## Author  
**Varun Chandra Shekar**  
Master of Data Science, RMIT University  
Melbourne, Australia  
