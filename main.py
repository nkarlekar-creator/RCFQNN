import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from feature_extraction import (
    smiles_to_onehot, apply_fractional_entropy, formula_to_onehot,
    hellinger_distance, AHONet
)
from quantum_models import RCFQNN, compute_exact_shapley_values

def pad_matrix_to_dim(flat_matrix, target_dim):
    """Pads features with zeros to bring all features to the highest dimension (Section 3.6)[cite: 1]"""
    batch_size, current_dim = flat_matrix.shape
    if current_dim < target_dim:
        padding = np.zeros((batch_size, target_dim - current_dim), dtype=np.float32)
        return np.hstack([flat_matrix, padding])
    return flat_matrix[:, :target_dim]

def run_pipeline(k_folds=5):
    os.makedirs("analysis_results", exist_ok=True)
    
    # 3.1 Data Acquisition & Input Processing[cite: 1]
    num_samples = 30
    dummy_smiles = ["ClC1=CC=CC=C1", "CC(=O)O", "C1=CC=CN=C1"] * 10
    dummy_formulas = ["C6H5Cl", "C2H4O2", "C5H5N"] * 10
    raw_mol_features = np.random.rand(num_samples, 1, 16, 16).astype(np.float32)
    y_pIC50 = np.random.rand(num_samples, 1).astype(np.float32)
    
    # Ground truth mean computation - Eq (6)[cite: 1]
    y_d_ground_truth = np.mean(y_pIC50)
    
    # Step 1: SMILES One-Hot Encoding (Output 1)[cite: 1]
    out1_smiles_oh, _ = smiles_to_onehot(dummy_smiles)
    
    # Step 2: Fractional Entropy for SMILES (Output 2)[cite: 1]
    out2_smiles_entropy = apply_fractional_entropy(out1_smiles_oh, q=1.5)
    
    # Step 3: Formula One-Hot Encoding (Output 3)[cite: 1]
    out3_formula_oh = formula_to_onehot(dummy_formulas)
    
    # Step 4: Fractional Entropy for Formula (Output 4)[cite: 1]
    out4_formula_entropy = apply_fractional_entropy(out3_formula_oh, q=1.5)
    
    # Step 5: Arrange features via Hellinger Distance & Fuse via AHONet (Output 5)[cite: 1]
    flat_out2 = out2_smiles_entropy.reshape(num_samples, -1)
    flat_out4 = out4_formula_entropy.reshape(num_samples, -1)
    
    # Quantify distance per sample - Eq (4)[cite: 1]
    h_distances = [hellinger_distance(flat_out2[i], flat_out4[i]) for i in range(num_samples)]
    
    # Extract fused features via AHONet backbone[cite: 1]
    ahonet_model = AHONet(in_channels=1, out_fused_dim=32)
    out5_fused = ahonet_model(torch.tensor(raw_mol_features)).detach().numpy()
    
    # Step 6: Feature Padding & Concatenation (Section 3.6)[cite: 1]
    flat_out1 = out1_smiles_oh.reshape(num_samples, -1)
    flat_out2 = out2_smiles_entropy.reshape(num_samples, -1)
    flat_out3 = out3_formula_oh.reshape(num_samples, -1)
    flat_out4 = out4_formula_entropy.reshape(num_samples, -1)
    flat_out5 = out5_fused.reshape(num_samples, -1)
    
    # Select highest dimension[cite: 1]
    dims = [flat_out1.shape[1], flat_out2.shape[1], flat_out3.shape[1], flat_out4.shape[1], flat_out5.shape[1]]
    max_dim = max(dims)
    
    # Zero pad each feature block[cite: 1]
    p1 = pad_matrix_to_dim(flat_out1, max_dim)
    p2 = pad_matrix_to_dim(flat_out2, max_dim)
    p3 = pad_matrix_to_dim(flat_out3, max_dim)
    p4 = pad_matrix_to_dim(flat_out4, max_dim)
    p5 = pad_matrix_to_dim(flat_out5, max_dim)
    
    # Concatenate into unified matrix (Dimension = 5 * max_dim)[cite: 1]
    X_concatenated = np.hstack([p1, p2, p3, p4, p5])
    
    # Step 7: K-Fold Training & Cross-Validation Split[cite: 1]
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    summary_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_concatenated)):
        fold_num = fold + 1
        fold_dir = f"analysis_results/fold_{fold_num}"
        os.makedirs(fold_dir, exist_ok=True)
        
        # Partition Training & Validation Data
        X_train, X_val = X_concatenated[train_idx], X_concatenated[val_idx]
        y_train, y_val = y_pIC50[train_idx], y_pIC50[val_idx]
        
        # Save exact numpy arrays for train/val split per fold
        np.save(f"{fold_dir}/X_train.npy", X_train)
        np.save(f"{fold_dir}/X_val.npy", X_val)
        np.save(f"{fold_dir}/y_train.npy", y_train)
        np.save(f"{fold_dir}/y_val.npy", y_val)
        
        # Initialize RCFQNN Model & Adam Optimizer[cite: 1]
        model = RCFQNN(concatenated_input_dim=X_concatenated.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Adam Optimizer[cite: 1]
        criterion = nn.MSELoss()
        
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        
        # Model Training[cite: 1]
        model.train()
        for epoch in range(2):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
        # Model Evaluation & Metrics[cite: 1]
        model.eval()
        val_x_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_preds = model(val_x_tensor).detach().numpy()
        
        mse = float(mean_squared_error(y_val, val_preds))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_val, val_preds))
        r2 = float(r2_score(y_val, val_preds))
        
        # Compute exact SHAP values for first validation sample[cite: 1]
        shap_vals = compute_exact_shapley_values(model, val_x_tensor[0])
        
        # Output Feature Dimensions Metadata[cite: 1]
        dim_file_data = {
            "fold_index": fold_num,
            "k_value": k_folds,
            "ground_truth_y_d_mean": float(y_d_ground_truth),
            "feature_dimensions": {
                "out1_smiles_onehot_dim": flat_out1.shape[1],
                "out2_smiles_entropy_dim": flat_out2.shape[1],
                "out3_formula_onehot_dim": flat_out3.shape[1],
                "out4_formula_entropy_dim": flat_out4.shape[1],
                "out5_fused_ahonet_dim": flat_out5.shape[1],
                "selected_max_dimension": max_dim,
                "final_concatenated_dimension": X_concatenated.shape[1]
            },
            "validation_metrics": {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2_Score": r2
            },
            "shap_explanation": {
                "shap_values_sample_0": shap_vals.tolist()
            }
        }
        
        # Save JSON metadata & dimensions[cite: 1]
        with open(f"{fold_dir}/feature_dimensions_and_metrics.json", "w") as f:
            json.dump(dim_file_data, f, indent=4)
            
        summary_results.append({"fold": fold_num, "metrics": dim_file_data["validation_metrics"]})
        print(f"Fold {fold_num} Complete | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # Save full K-Fold evaluation summary
    with open("analysis_results/kfold_summary.json", "w") as f:
        json.dump(summary_results, f, indent=4)

if __name__ == "__main__":
    run_pipeline(k_folds=5)