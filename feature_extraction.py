import re
import numpy as np
import torch
import torch.nn as nn

# 3.2 SMILES Notation & One-hot encoding matrix[cite: 1]
def smiles_to_onehot(smiles_list):
    unique_symbols = sorted(list(set("".join(smiles_list))))
    symbol_to_idx = {sym: i for i, sym in enumerate(unique_symbols)}
    max_len = max(len(s) for s in smiles_list)
    
    matrices = []
    for s in smiles_list:
        # Rows = unique symbols, Columns = position in SMILES sequence[cite: 1]
        mat = np.zeros((len(unique_symbols), max_len), dtype=np.float32)
        for col, sym in enumerate(s):
            row = symbol_to_idx[sym]
            mat[row, col] = 1.0
        matrices.append(mat)
    return np.array(matrices), symbol_to_idx

# 3.2.2 & 3.4 Entropy-based One-hot encoder using Fractional Entropy[cite: 1]
def apply_fractional_entropy(onehot_matrices, q=1.5):
    # Eq (2) & Eq (3): S_q = (1 - sum(p^q)) / (q - 1)[cite: 1]
    batch_size, rows, cols = onehot_matrices.shape
    entropy_matrices = np.zeros_like(onehot_matrices)
    
    for i in range(batch_size):
        mat = onehot_matrices[i]
        probs = np.mean(mat, axis=1, keepdims=True) + 1e-9
        probs = probs / np.sum(probs)
        
        # Compute fractional entropy value[cite: 1]
        s_q = (1.0 - np.sum(probs**q)) / (q - 1.0)
        entropy_matrices[i] = mat * s_q
        
    return entropy_matrices

# 3.3 Molecular Formula-based One-hot encoding[cite: 1]
def formula_to_onehot(formulas):
    all_elements = set()
    parsed_formulas = []
    
    for f in formulas:
        tokens = re.findall(r'([A-Z][a-z]*)(\d*)', f)
        counts = {elem: int(count) if count else 1 for elem, count in tokens}
        all_elements.update(counts.keys())
        parsed_formulas.append(counts)
        
    elem_list = sorted(list(all_elements))
    elem_to_idx = {elem: i for i, elem in enumerate(elem_list)}
    max_len = max(len(p) for p in parsed_formulas)
    
    matrices = []
    for counts in parsed_formulas:
        # Matrix stores count of atoms of elements in corresponding position[cite: 1]
        mat = np.zeros((len(elem_list), max_len), dtype=np.float32)
        for col, (elem, cnt) in enumerate(counts.items()):
            row = elem_to_idx[elem]
            mat[row, col] = float(cnt)
        matrices.append(mat)
    return np.array(matrices)

# 3.5.1(a) Hellinger Distance Calculation - Eq (4)[cite: 1]
def hellinger_distance(p, q):
    p_norm = p / (np.sum(p) + 1e-9)
    q_norm = q / (np.sum(q) + 1e-9)
    return (1.0 / np.sqrt(2.0)) * np.sqrt(np.sum((np.sqrt(p_norm + 1e-9) - np.sqrt(q_norm + 1e-9)) ** 2))

# 3.5.1(c) AHONet Structure (ECA-Net + MPN-COV)[cite: 1]
class ECANetModule(nn.Module):
    """Efficient Channel Attention Module without dimensionality reduction[cite: 1]"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECANetModule, self).__init__()
        k_size = int(abs((np.log2(channels) / gamma) + (b / gamma)))
        k_size = k_size if k_size % 2 != 0 else k_size + 1
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class MPN_COV_Pooling(nn.Module):
    """Second-order Covariance Pooling[cite: 1]"""
    def __init__(self):
        super(MPN_COV_Pooling, self).__init__()

    def forward(self, x):
        batch_size, C, H, W = x.shape
        N = H * W
        features = x.view(batch_size, C, N)
        
        # Compute sample covariance matrix[cite: 1]
        mean = torch.mean(features, dim=2, keepdim=True)
        features_centered = features - mean
        cov = torch.bmm(features_centered, features_centered.transpose(1, 2)) / (N - 1)
        return cov

class AHONet(nn.Module):
    """Attention High-Order Deep Network backbone[cite: 1]"""
    def __init__(self, in_channels, out_fused_dim):
        super(AHONet, self).__init__()
        # Custom Conv kernel replacing standard initial ResNet layer[cite: 1]
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.eca = ECANetModule(64)
        self.mpn_cov = MPN_COV_Pooling()
        self.fc = nn.Linear(64 * 64, out_fused_dim)

    def forward(self, x):
        x = torch.relu(self.initial_conv(x))
        x = self.eca(x)
        x_cov = self.mpn_cov(x)
        x_flat = x_cov.view(x_cov.size(0), -1)
        return self.fc(x_flat)