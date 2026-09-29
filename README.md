 RCFQNN: Hybrid Reflection-Coherent Feed-Forward Quantum Neural Network

This repository contains a research-oriented implementation of a **Hybrid Reflection-Coherent Feed-Forward Quantum Neural Network (RCFQNN)** for molecular feature processing and predictive modeling.

The implementation combines classical deep-learning feature extraction with quantum neural network components using **PyTorch** and **PennyLane**. The pipeline also includes feature engineering, cross-validation, regression metrics, and Shapley-value-based explainability.

## Repository

GitHub: https://github.com/nkarlekar-creator/RCFQNN

## Project Overview

The workflow implemented in this repository follows these major stages:

1. Molecular representation from SMILES strings.
2. One-hot feature encoding.
3. Fractional entropy-based feature transformation.
4. Molecular formula encoding.
5. Hellinger-distance-based feature comparison.
6. AHONet-style feature fusion using:
   - Efficient Channel Attention (ECA)
   - Second-order covariance pooling (MPN-COV)
7. Feature dimension alignment and concatenation.
8. Hybrid quantum modeling using:
   - Reflection Equivariant Quantum Neural Network (REQNN)
   - Coherent Feed-Forward Quantum Neural Network (CFFQNN)
9. Classical optimization using Adam.
10. K-fold cross-validation.
11. Regression evaluation using MSE, RMSE, MAE, and R².
12. Shapley-value-based model explainability.

## Architecture

```text
                 Molecular Input
                       |
             +---------+---------+
             |                   |
          SMILES             Molecular Formula
             |                   |
       One-Hot Encoding     One-Hot Encoding
             |                   |
      Fractional Entropy    Fractional Entropy
             |                   |
             +---------+---------+
                       |
                Feature Processing
                       |
                 AHONet Fusion
             (ECA + MPN-COV)
                       |
          Feature Padding / Alignment
                       |
              Feature Concatenation
                       |
              Classical Input Layer
                       |
              +--------+--------+
              |                 |
            REQNN             CFFQNN
              |                 |
              +--------+--------+
                       |
              Hybrid RCFQNN Output
                       |
              Regression Prediction
                       |
        +--------------+--------------+
        |                             |
   Performance Metrics          Explainability
 MSE / RMSE / MAE / R²          Shapley Values
```

## Files

### `feature_extraction.py`

Contains the classical feature-processing components, including:

- SMILES one-hot encoding
- Fractional entropy transformation
- Molecular formula encoding
- Hellinger distance calculation
- `ECANetModule`
- `MPN_COV_Pooling`
- `AHONet`

### `quantum_models.py`

Contains the quantum-learning components:

- PennyLane quantum device configuration
- Reflection Equivariant Quantum Neural Network (REQNN)
- Coherent Feed-Forward Quantum Neural Network (CFFQNN)
- Hybrid `RCFQNN` PyTorch model
- Exact Shapley-value calculation

The quantum circuits currently use a **4-qubit** PennyLane simulator based on `default.qubit`.

### `main.py`

Provides the primary end-to-end demonstration pipeline.

It:

- Creates a small synthetic molecular dataset.
- Extracts and fuses multiple feature representations.
- Performs 5-fold cross-validation.
- Trains the RCFQNN model.
- Calculates MSE, RMSE, MAE, and R².
- Generates Shapley-value explanations.
- Saves fold-level results under `analysis_results/`.

### `train_pipeline.py`

Provides an alternative training-pipeline implementation for experimentation with the same overall methodology.

> **Important:** `main.py` and `train_pipeline.py` currently use slightly different function/class names and interfaces. `main.py` is the recommended entry point for the ZIP version as provided. Before using `train_pipeline.py`, verify its function names and constructor arguments against the current implementations in `feature_extraction.py` and `quantum_models.py`.

## Requirements

Recommended Python version:

```text
Python 3.9+
```

Install the main dependencies:

```bash
pip install numpy torch scikit-learn pennylane
```

A typical `requirements.txt` can contain:

```text
numpy
torch
scikit-learn
pennylane
```

## Installation

Clone the repository:

```bash
git clone https://github.com/nkarlekar-creator/RCFQNN.git
cd RCFQNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the packages directly:

```bash
pip install numpy torch scikit-learn pennylane
```

## Running the Main Pipeline

Run:

```bash
python main.py
```

The program creates an `analysis_results/` directory containing fold-wise training/validation arrays and JSON analysis files.

Example structure:

```text
analysis_results/
├── fold_1/
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── feature_dimensions_and_metrics.json
├── fold_2/
├── fold_3/
├── fold_4/
├── fold_5/
└── kfold_summary.json
```

## Model Details

### REQNN

The Reflection Equivariant Quantum Neural Network uses:

- 4 qubits
- Y-axis angle embedding
- Symmetric parameterized rotations
- Reflection-symmetric CNOT connections
- Pauli-Z tensor-product expectation measurement

### CFFQNN

The Coherent Feed-Forward Quantum Neural Network uses:

- Four single-qubit parameterized rotations
- Controlled-Y rotations between neighboring qubits
- Pauli-Z expectation measurement

### RCFQNN

The hybrid model first maps the high-dimensional classical feature vector to four quantum inputs:

```text
High-dimensional features
        |
   Linear Adapter
        |
      Tanh
        |
    4 Quantum Inputs
       /       \
    REQNN     CFFQNN
       \       /
        Fusion
           |
     Regression Output
```

The final prediction combines the two quantum outputs using a nonlinear interaction/Taylor-series-inspired formulation.

## Explainability

The repository includes an exact Shapley-value implementation intended to quantify the contribution of individual input features to the model prediction.

The approach evaluates feature coalitions against a zero baseline and calculates the marginal contribution of each feature.

> **Computational note:** Exact Shapley-value computation has exponential complexity with respect to the number of input features. It can therefore become very expensive for high-dimensional feature vectors.

## Data

The current demonstration code uses **synthetic/dummy molecular inputs and randomly generated target values** for pipeline execution and software validation.

Examples include SMILES strings such as:

```text
ClC1=CC=CC=C1
CC(=O)O
C1=CC=CN=C1
```

and corresponding molecular formulas.

For scientific experiments, replace the synthetic inputs with the intended validated molecular dataset and experimentally obtained target values such as pIC50.

## Evaluation Metrics

The pipeline reports:

- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R² Score** — Coefficient of Determination

These metrics are generated for each validation fold and stored in JSON output files.

## Reproducibility

The K-fold splitter uses:

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

For publication-quality experiments, also consider explicitly setting random seeds for:

- NumPy
- PyTorch
- PennyLane
- Python's random module

and documenting the exact package versions and hardware/software environment.

## Important Research Use Note

This repository is a **research implementation/prototype**. The current code demonstrates the proposed computational workflow but should not be interpreted as a validated clinical, pharmaceutical, or production prediction system.

Before reporting scientific performance, use a validated dataset, appropriate train/test separation, independent testing, repeated experiments, and statistically justified comparisons with baseline methods.

## Citation

If you use this implementation in research, please cite the associated research work and acknowledge the repository:

```text
Karlekar, Nandkishor P. et al.
RCFQNN: Hybrid Reflection-Coherent Feed-Forward Quantum Neural Network.
GitHub Repository:
https://github.com/nkarlekar-creator/RCFQNN
```

Replace the citation above with the final bibliographic citation once the associated paper has been formally published.

## Author

**Dr. Nandkishor P. Karlekar**  
Associate Professor, School of Computing  
MIT-ADT University, Pune, India

GitHub: https://github.com/nkarlekar-creator

## License

No license file is currently included in this package. If this repository is intended for public reuse, add an appropriate open-source license such as MIT, Apache-2.0, or another license selected by the author.
