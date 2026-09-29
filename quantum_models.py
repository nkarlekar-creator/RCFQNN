import itertools
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# 4 Qubit Quantum Register
dev = qml.device("default.qubit", wires=4)

# 3.6.1(a) Reflection Equivariant Quantum Neural Network (REQNN)[cite: 1]
@qml.qnode(dev, interface="torch")
def reqnn_circuit(inputs, weights):
    # Reflection Equivariant Encoder (Amplitude / Angle with reflection invariance)[cite: 1]
    qml.AngleEmbedding(inputs[:4], wires=range(4), rotation='Y')
    
    # Symmetric operation gates for reflection equivariance[cite: 1]
    qml.RY(weights[0], wires=0)
    qml.RY(weights[0], wires=3)  # Symmetric pair
    qml.RY(weights[1], wires=1)
    qml.RY(weights[1], wires=2)  # Symmetric pair
    
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[3, 2])  # Reflection symmetric entanglement
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(3))

# 3.6.1(b) Coherent Feed-Forward Quantum Neural Network (CFFQNN)[cite: 1]
@qml.qnode(dev, interface="torch")
def cffqnn_circuit(inputs, weights):
    # Primary layer single-qubit rotations[cite: 1]
    for i in range(4):
        qml.RY(inputs[i] * weights[i], wires=i)
        
    # Parameterized controlled rotations (CRY) across hidden nodes - Eq (14) & Eq (15)[cite: 1]
    qml.CRY(weights[4], wires=[0, 1])
    qml.CRY(weights[5], wires=[1, 2])
    qml.CRY(weights[6], wires=[2, 3])
    
    return qml.expval(qml.PauliZ(3))

# 3.6.1 Hybrid RCFQNN Model[cite: 1]
class RCFQNN(nn.Module):
    def __init__(self, concatenated_input_dim):
        super(RCFQNN, self).__init__()
        # Input adaptation layer for quantum circuit mapping[cite: 1]
        self.input_adapter = nn.Linear(concatenated_input_dim, 4)
        
        # Quantum trainable parameters[cite: 1]
        self.reqnn_weights = nn.Parameter(torch.randn(2, dtype=torch.float32))
        self.cffqnn_weights = nn.Parameter(torch.randn(7, dtype=torch.float32))
        
        # Model network weights W_REQNN and W_CFFQNN[cite: 1]
        self.w_reqnn = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.w_cffqnn = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        x_adapted = torch.tanh(self.input_adapter(x)) * np.pi  # Rescale to [-pi, pi]
        batch_predictions = []
        
        for sample in x_adapted:
            # 1. Yield O_1 via REQNN and compute O_3 = O_1 * W_REQNN[cite: 1]
            o1 = reqnn_circuit(sample, self.reqnn_weights)
            o3 = o1 * self.w_reqnn
            
            # 2. Yield O_2 via CFFQNN and compute O_4 = O_2 * W_CFFQNN[cite: 1]
            o2 = cffqnn_circuit(sample, self.cffqnn_weights)
            o4 = o2 * self.w_cffqnn
            
            # 3. Section 3.6.1(c): Apply Taylor Series Concept - Eq (25)[cite: 1]
            # y = O_3 + O_4 + (O_3 * O_4) + 0.5 * (O_3^2 + O_4^2)[cite: 1]
            y_pred = o3 + o4 + (o3 * o4) + 0.5 * (o3**2 + o4**2)
            batch_predictions.append(y_pred)
            
        return torch.stack(batch_predictions).unsqueeze(1)

# 3.7 Explainability Analysis using Shapley Value Explanation (SHAP) - Eq (26)[cite: 1]
def compute_exact_shapley_values(model, x_sample):
    """
    Computes exact SHAP attributions using coalition vectors S in {0,1}^M 
    satisfying Symmetry, Dummy, Additivity, and Efficiency (Eq 26)[cite: 1].
    """
    model.eval()
    x_sample_np = x_sample.detach().numpy()
    M = x_sample_np.shape[0]  # Number of features
    
    baseline = np.zeros_like(x_sample_np)
    shap_values = np.zeros(M)
    
    # Iterate over all possible feature coalitions
    feature_indices = list(range(M))
    for i in range(M):
        other_features = [f for f in feature_indices if f != i]
        phi_i = 0.0
        
        for k in range(len(other_features) + 1):
            for subset in itertools.combinations(other_features, k):
                # Coalition S without feature i
                S_no_i = np.copy(baseline)
                S_no_i[list(subset)] = x_sample_np[list(subset)]
                
                # Coalition S with feature i
                S_with_i = np.copy(S_no_i)
                S_with_i[i] = x_sample_np[i]
                
                # Model predictions for both coalitions
                pred_no_i = model(torch.tensor(S_no_i, dtype=torch.float32).unsqueeze(0)).item()
                pred_with_i = model(torch.tensor(S_with_i, dtype=torch.float32).unsqueeze(0)).item()
                
                # Compute KernelSHAP / Shapley coalition weight
                weight = (np.math.factorial(k) * np.math.factorial(M - k - 1)) / np.math.factorial(M)
                phi_i += weight * (pred_with_i - pred_no_i)
                
        shap_values[i] = phi_i
        
    return shap_values