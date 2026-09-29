import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from feature_extraction import (
    smiles_to_onehot, compute_fractional_entropy, formula_to_onehot, AHONet
)
from quantum_models import RCFQNN, compute_shap_analysis

def pad_and_flatten(matrix, max_flat_dim):
    batch_size = matrix.shape[0]
    flat = matrix.reshape(batch_size, -1)
    current_dim = flat.shape[1]
    if current_dim < max_flat_dim:
        padding = np.zeros((batch_size, max_flat_dim - current_dim), dtype=np.float32)
        return np.hstack([flat, padding])
    return flat[:, :max_flat_dim]

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2_Score": float(r2)
    }

def run_pipeline(k_folds=5):
    os.makedirs("analysis_results", exist_ok=True)
    
    # Synthetic dataset for execution
    num_samples = 40
    dummy_smiles = ["ClC1=CC=CC=C1", "CC(=O)O", "C1=CC=CN=C1", "CCO"] * 10
    dummy_formulas = ["C6H5Cl", "C2H4O2", "C5H5N", "C2H6O"] * 10
    dummy_mol_features = np.random.rand(num_samples, 20).astype(np.float32)
    y_pIC50 = np.random.rand(num_samples, 1).astype(np.float32)
    
    # 1. Feature Extraction Outputs
    out1 = smiles_to_onehot(dummy_smiles)
    out2 = compute_fractional_entropy(out1)
    out3 = formula_to_onehot(dummy_formulas)
    out4 = compute_fractional_entropy(out3)
    
    ahonet = AHONet(in_features=20, out_dim=32)
    out5 = ahonet(torch.tensor(dummy_mol_features)).detach().numpy()
    
    # 2. Maximum Feature Dimension Determination & Padding
    flat_dims = [
        np.prod(out1.shape[1:]), np.prod(out2.shape[1:]),
        np.prod(out3.shape[1:]), np.prod(out4.shape[1:]), np.prod(out5.shape[1:])
    ]
    max_dim = int(np.max(flat_dims))
    
    concatenated_features = np.hstack([
        pad_and_flatten(out1, max_dim), pad_and_flatten(out2, max_dim),
        pad_and_flatten(out3, max_dim), pad_and_flatten(out4, max_dim),
        pad_and_flatten(out5, max_dim)
    ])
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    overall_fold_summary = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(concatenated_features)):
        fold_num = fold + 1
        fold_dir = f"analysis_results/fold_{fold_num}"
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split Data
        X_train, X_val = concatenated_features[train_idx], concatenated_features[val_idx]
        y_train, y_val = y_pIC50[train_idx], y_pIC50[val_idx]
        
        # Separate Training and Validation Files
        np.save(f"{fold_dir}/X_train.npy", X_train)
        np.save(f"{fold_dir}/X_val.npy", X_val)
        np.save(f"{fold_dir}/y_train.npy", y_train)
        np.save(f"{fold_dir}/y_val.npy", y_val)
        
        # Train Model
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        
        model = RCFQNN(input_dim=concatenated_features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(2):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
        # Validation & Analysis Evaluation
        model.eval()
        val_tensor = torch.tensor(X_val)
        val_preds = model(val_tensor).detach().numpy()
        
        metrics = compute_metrics(y_val, val_preds)
        shap_importances = compute_shap_analysis(model, val_tensor[:5])
        
        # Save Metadata and Dimension Details File
        meta_info = {
            "fold_index": fold_num,
            "k_value": k_folds,
            "training_samples_count": len(X_train),
            "validation_samples_count": len(X_val),
            "max_feature_dimension": max_dim,
            "concatenated_feature_dim": int(concatenated_features.shape[1]),
            "feature_shapes": {
                "out1_smiles_oh": list(out1.shape),
                "out2_smiles_entropy": list(out2.shape),
                "out3_formula_oh": list(out3.shape),
                "out4_formula_entropy": list(out4.shape),
                "out5_fused_ahonet": list(out5.shape)
            },
            "validation_metrics": metrics,
            "shap_feature_importance_top10": shap_importances[:10]
        }
        
        with open(f"{fold_dir}/metadata_and_dimensions.json", "w") as f:
            json.dump(meta_info, f, indent=4)
            
        overall_fold_summary.append({"fold": fold_num, "metrics": metrics})
        print(f"Fold {fold_num}/{k_folds} Complete | Val RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2_Score']:.4f}")

    # Export Cross-Validation Summary Analysis File
    with open("analysis_results/overall_kfold_analysis.json", "w") as f:
        json.dump(overall_fold_summary, f, indent=4)

if __name__ == "__main__":
    run_pipeline(k_folds=5)