import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression

# Mutual Information Calculation Functions
def calculate_mutual_information_knn(state1, state2, k=3):
    """
    Calculate mutual information between two hidden states using k-NN method.
    This is more appropriate for continuous variables than discretization.
    
    Args:
        state1, state2: torch tensors representing hidden states
        k: number of nearest neighbors for estimation
    
    Returns:
        float: mutual information value
    """
    # Flatten the states to 1D
    flat1 = state1.flatten().detach().cpu().numpy()
    flat2 = state2.flatten().detach().cpu().numpy()
    
    # Ensure same length
    min_len = min(len(flat1), len(flat2))
    flat1 = flat1[:min_len]
    flat2 = flat2[:min_len]
    
    # Use sklearn's mutual_info_regression for continuous variables
    # Reshape for sklearn (samples, features)
    X = flat1.reshape(-1, 1)
    y = flat2
    
    # Calculate mutual information
    mi = mutual_info_regression(X, y, discrete_features=False, random_state=42)[0]
    
    return mi

def calculate_phi_knn(state1, state2, split_index=1, k=3):
    """
    Calculate phi (integrated information) between two hidden states using k-NN method.
    Phi measures the difference between total mutual information and the sum of 
    mutual information of split fragments.
    
    Args:
        state1, state2: torch tensors representing hidden states
        split_index: integer index to split the states (default 1)
        k: number of nearest neighbors for estimation
    
    Returns:
        float: phi value (total MI - sum of fragment MIs)
    """
    # Calculate total mutual information
    total_mi = calculate_mutual_information_knn(state1, state2, k)
    
    # Flatten the states to 1D
    flat1 = state1.flatten().detach().cpu().numpy()
    flat2 = state2.flatten().detach().cpu().numpy()
    
    # Ensure same length
    min_len = min(len(flat1), len(flat2))
    flat1 = flat1[:min_len]
    flat2 = flat2[:min_len]
    
    # Split the states at split_index
    if split_index >= min_len:
        split_index = min_len // 2  # Use middle if split_index is too large
    
    # Create fragments
    frag1_1 = flat1[:split_index]
    frag1_2 = flat1[split_index:]
    frag2_1 = flat2[:split_index]
    frag2_2 = flat2[split_index:]
    
    # Calculate mutual information for each fragment
    mi_frag1 = 0.0
    mi_frag2 = 0.0
    
    if len(frag1_1) > 0 and len(frag2_1) > 0:
        # Ensure same length for fragment 1
        min_frag1_len = min(len(frag1_1), len(frag2_1))
        if min_frag1_len > 0:
            X_frag1 = frag1_1[:min_frag1_len].reshape(-1, 1)
            y_frag1 = frag2_1[:min_frag1_len]
            mi_frag1 = mutual_info_regression(X_frag1, y_frag1, discrete_features=False, random_state=42)[0]
    
    if len(frag1_2) > 0 and len(frag2_2) > 0:
        # Ensure same length for fragment 2
        min_frag2_len = min(len(frag1_2), len(frag2_2))
        if min_frag2_len > 0:
            X_frag2 = frag1_2[:min_frag2_len].reshape(-1, 1)
            y_frag2 = frag2_2[:min_frag2_len]
            mi_frag2 = mutual_info_regression(X_frag2, y_frag2, discrete_features=False, random_state=42)[0]
    
    # Calculate phi as total MI minus sum of fragment MIs
    phi = total_mi - (mi_frag1 + mi_frag2)
    
    return phi

def calculate_mutual_information_discretized(state1, state2, bins=50):
    """
    Calculate mutual information using discretization (binning) method.
    This is the original method, kept for comparison.
    
    Args:
        state1, state2: torch tensors representing hidden states
        bins: number of bins for discretization
    
    Returns:
        float: mutual information value
    """
    # Flatten the states to 1D
    flat1 = state1.flatten().detach().cpu().numpy()
    flat2 = state2.flatten().detach().cpu().numpy()
    
    # Ensure same length
    min_len = min(len(flat1), len(flat2))
    flat1 = flat1[:min_len]
    flat2 = flat2[:min_len]
    
    # Discretize the continuous values into bins
    hist1, bin_edges1 = np.histogram(flat1, bins=bins)
    hist2, bin_edges2 = np.histogram(flat2, bins=bins)
    
    # Create joint histogram
    joint_hist, _, _ = np.histogram2d(flat1, flat2, bins=bins)
    
    # Normalize to get probabilities
    joint_prob = joint_hist / np.sum(joint_hist)
    prob1 = hist1 / np.sum(hist1)
    prob2 = hist2 / np.sum(hist2)
    
    # Calculate mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_prob[i, j] > 0 and prob1[i] > 0 and prob2[j] > 0:
                mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (prob1[i] * prob2[j]))
    
    return mi

def compute_mutual_information_matrix(hidden_states_list, method='knn', split_index=1):
    """
    Compute n×n mutual information matrix for n hidden states.
    
    Args:
        hidden_states_list: list of hidden states (tensors)
        method: 'knn' for k-nearest neighbors, 'discretized' for binning, or 'phi' for integrated information
        split_index: integer index to split states for phi calculation (default 1)
    
    Returns:
        numpy array: n×n mutual information matrix
    """
    n = len(hidden_states_list)
    mi_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # Self-mutual information (entropy)
                flat_state = hidden_states_list[i].flatten().detach().cpu().numpy()
                hist, _ = np.histogram(flat_state, bins=50)
                prob = hist / np.sum(hist)
                prob = prob[prob > 0]  # Remove zero probabilities
                mi_matrix[i, j] = entropy(prob, base=2)
            else:
                if method == 'knn':
                    mi_matrix[i, j] = calculate_mutual_information_knn(
                        hidden_states_list[i], 
                        hidden_states_list[j]
                    )
                elif method == 'phi':
                    mi_matrix[i, j] = calculate_phi_knn(
                        hidden_states_list[i], 
                        hidden_states_list[j],
                        split_index=split_index
                    )
                else:  # discretized
                    mi_matrix[i, j] = calculate_mutual_information_discretized(
                        hidden_states_list[i], 
                        hidden_states_list[j]
                    )
    
    return mi_matrix

def visualize_mi_matrix(mi_matrix, title="Mutual Information Matrix", export_dir=None, save_path=None):
    """Visualize the mutual information matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mi_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Mutual Information')
    plt.title(title)
    plt.xlabel('State Index')
    plt.ylabel('State Index')
    
    # Add text annotations
    for i in range(mi_matrix.shape[0]):
        for j in range(mi_matrix.shape[1]):
            plt.text(j, i, f'{mi_matrix[i, j]:.3f}', 
                    ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if export_dir and save_path:
        os.makedirs(export_dir, exist_ok=True)
        plt.savefig(os.path.join(export_dir, save_path), dpi=300, bbox_inches='tight')
        print(f"📊 Figure saved to: {os.path.join(export_dir, save_path)}")
    
    plt.show()
