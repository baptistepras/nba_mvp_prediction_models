# 🏀 NBA MVP Prediction

This project aims to predict the **NBA Most Valuable Player (MVP)** of a given season using player and team statistics.  
The pipeline covers data collection, preprocessing, dataset building, model training, and evaluation.  
Multiple machine learning models are implemented and compared, with performance metrics reported to assess prediction quality.

---

## 📦 Project Structure

- `scripts_raw_data/` → scripts for data download and cleaning
- `scripts_data_process/` → scripts for dataset preparation  
- `train_models.py` → train and evaluate different machine learning models  
- `hyperparameters_tuning/` → scripts for hyperparameter tuning and feature selection  
- `models/` → trained models (ignored by Git)  
- `raw_data/`, `processed_data/` → datasets (ignored by Git)  

---

## ⚙️ Installation

Clone the repository and make sure you have Python 3.9+ with the required dependencies (I recommend using conda):

`conda env create -f environment.yml`

---

## 🚀 Usage

All commands should be run **from the project root**.

1. **Download and preprocess raw data**  
   ```
   python scripts_data_process/download_raw_data.py
   python scripts_data_process/build_team_mapping.py
   ```

2.	**Build datasets and split into train/test**
   ```
   python scripts_data_process/build_all_pipelines.py --pipelines all
   python scripts_data_process/build_splits.py --pipeline all
   ```

3.	**Train and evaluate models**
   ```
   python train_models.py --pipeline all --model <model_name>
   ```

Available <model_name> options:

	•	logreg → Logistic Regression
 
	•	rf → Random Forest
 
	•	xgb → XGBoost
 
	•	gb → Gradient Boosting
 
	•	histgb → Histogram-based Gradient Boosting
 
	•	lgbm → LightGBM

4.	**Hyperparameter tuning (optional)**
   ```
   python hyperparameters_tuning/hyperparameters_tuner.py --model <model_name> --param <param_name> --values <values>
   python hyperparameters_tuning/hyperparameters_tuner.py --model logreg --param C --values 0.01 0.1 1.0 5.0 10.0
   ```

⚠️ Logistic Regression (logreg) already comes with optimized hyperparameters.
For the other models, tuning is recommended; you must then change the default parameters in the code.

5.	**Feature selection (optional)**
   ```
   python hyperparameters_tuning/greedy_forward_feature_selection.py --pipeline all --model <model_name>
   python hyperparameters_tuning/greedy_backward_feature_selection.py --pipeline all --model <model_name>
   python hyperparameters_tuning/feature_selection.py --pipeline all --model <model_name> --min <min_features_to_keep> --max <max_features_to_keep>
   ```

⚠️ Feature selection has already been applied for Logistic Regression.
For the other models, these scripts can be used to explore dimensionality reduction and performance improvements.


📘 Note: Example commands are included at the bottom of each script for guidance.
