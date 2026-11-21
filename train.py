import os
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

root = tk.Tk()
root.withdraw()

print("Opening file picker... Please select your 'predictive_maintenance.csv' file for training.")
file_path = filedialog.askopenfilename(
    title="Select predictive_maintenance.csv",
    filetypes=[("CSV Files", "*.csv")]
)

if not file_path:
    print("[ERROR] No file selected!")
else:
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../outputs', exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(file_path)

    # 2. Domain Feature Engineering (Handbook Project 9)
    def engineer_features(data):
        d = data.copy()
        d['Temp_Diff'] = d['Process temperature [K]'] - d['Air temperature [K]']
        d['Power'] = d['Torque [Nm]'] * d['Rotational speed [rpm]'] * (2 * np.pi / 60)
        wear_limits = {'L': 200, 'M': 150, 'H': 120}
        d['Tool_Wear_Ratio'] = d['Tool wear [min]'] / d['Type'].map(wear_limits)
        return d

    df_feat = engineer_features(df)

    feature_cols = [
        'Type', 'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
        'Temp_Diff', 'Power', 'Tool_Wear_Ratio'
    ]

    X = df_feat[feature_cols]
    y = df_feat['Target']

    # 3. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Pipeline Construction
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), ['Type']),
            ('num', StandardScaler(), [col for col in feature_cols if col != 'Type'])
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=150, 
            random_state=42, 
            class_weight='balanced'
        ))
    ])

    # 5. Train & Evaluate
    model_pipeline.fit(X_train, y_train)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC Score:  {average_precision_score(y_test, y_prob):.4f}")

    # 6. Save Model & Test Results for Dashboard
    joblib.dump(model_pipeline, '../models/predictive_maintenance_rf.pkl')

    test_results = X_test.copy()
    test_results['Product ID'] = df.loc[X_test.index, 'Product ID']
    test_results['Actual_Target'] = y_test
    test_results['Failure_Type'] = df.loc[X_test.index, 'Failure Type']
    test_results['Failure_Prob'] = y_prob
    test_results.to_csv('../models/test_predictions.csv', index=False)

    print("\n[SUCCESS] Model training complete and artifacts saved in 'models/' folder!")