"""
experiments/generate_calvin_analysis.py
Extracts and summarizes model evaluation, threshold tuning, 
feature importance, and K-Means segmentation outputs for Calvin's section.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

def analyze_models():
    print("=" * 60)
    print("CALVIN ABRAHAM - MODEL EVALUATION & RESULTS SUMMARY")
    print("=" * 60)
    
    # 1. Classification Metrics Summary
    models = ["logistic_regression", "random_forest", "gradient_boosting", "decision_tree"]
    metrics_summary = []
    
    for m in models:
        test_path = os.path.join(BASE_DIR, m, "outputs", "test_metrics.json")
        cv_path = os.path.join(BASE_DIR, m, "outputs", "cv_metrics.json")
        thresh_path = os.path.join(BASE_DIR, m, "outputs", "threshold_analysis.json")
        
        test_data = load_json(test_path) or {}
        cv_data = load_json(cv_path) or {}
        thresh_data = load_json(thresh_path) or {}
        
        selected_cutoff = thresh_data.get("selected_threshold", "N/A")
        
        metrics_summary.append({
            "Model": m.replace("_", " ").title(),
            "CV F1 (0.50)": cv_data.get("f1_mean", cv_data.get("f1", "N/A")),
            "Selected Cutoff": selected_cutoff,
            "Test Precision": test_data.get("precision", "N/A"),
            "Test Recall": test_data.get("recall", "N/A"),
            "Test F1": test_data.get("f1", "N/A")
        })
        
    df_metrics = pd.DataFrame(metrics_summary)
    print("\n--- Model Comparison Summary ---")
    print(df_metrics.to_string(index=False))
    
    # 2. Logistic Regression Directional Feature Importance
    lr_pipeline_path = os.path.join(BASE_DIR, "logistic_regression", "outputs", "model_pipeline.joblib")
    if os.path.exists(lr_pipeline_path):
        try:
            pipeline = joblib.load(lr_pipeline_path)
            model = pipeline.named_steps.get("classifier", pipeline)
            preprocessor = pipeline.named_steps.get("preprocessor", None)
            
            if hasattr(model, "coef_"):
                coefs = model.coef_[0]
                feature_names = (
                    preprocessor.get_feature_names_out() 
                    if preprocessor and hasattr(preprocessor, "get_feature_names_out") 
                    else [f"Feature_{i}" for i in range(len(coefs))]
                )
                df_coefs = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
                df_coefs["Abs_Weight"] = df_coefs["Coefficient"].abs()
                df_coefs = df_coefs.sort_values(by="Abs_Weight", ascending=False).head(10)
                print("\n--- Logistic Regression Top 10 Coefficients (Directional) ---")
                print(df_coefs[["Feature", "Coefficient"]].to_string(index=False))
        except Exception as e:
            print(f"\n[Info] Could not inspect pipeline steps: {e}")

    # 3. K-Means Segment Profiles
    kmeans_summary_path = os.path.join(BASE_DIR, "kmeans", "outputs", "cluster_summary.json")
    km_data = load_json(kmeans_summary_path)
    if km_data:
        print("\n--- K-Means Cluster Metrics ---")
        print(json.dumps(km_data, indent=2))
        
    print("\nAnalysis ready. Copy the Markdown section into the project report.")

if __name__ == "__main__":
    analyze_models()
