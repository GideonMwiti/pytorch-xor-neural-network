"""
Project 12: Solving the Classic XOR Problem with Neural Networks in PyTorch
Author: Gideon Mwiti

Key Highlights:
- Historical & mathematical resolution of the Minsky & Papert (1969) Perceptron Dilemma.
- Direct comparative ablation:
  1) Single-Layer Perceptron (Linear Model): Bound to fail due to linear inseparability.
  2) Multi-Layer Perceptron (Non-linear MLP): Solves XOR with 100% accuracy.
- Geometric latent space analysis: Demonstrates how non-linear hidden representations 
  warp the input coordinates into a linearly separable manifold.
- High-resolution 4-panel diagnostic visual artifact:
  1) Linear Classifier Failure: Planar decision boundary unable to separate XOR.
  2) Non-Linear MLP Success: Complex contour boundary achieving 100% separation.
  3) Latent Representation Mapping: Geometric projection in hidden feature space.
  4) Training Convergence Trajectories: Binary Cross-Entropy Loss & Accuracy dynamics.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Canonical XOR Truth Table Dataset
XOR_INPUTS = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

XOR_TARGETS = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype=torch.float32)


def generate_synthetic_xor_distribution(points_per_cluster=100, noise=0.08, seed=42):
    """Generates continuous clusters around the 4 XOR vertices for rich visualization."""
    np.random.seed(seed)
    clusters = []
    labels = []
    
    centers = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    cluster_labels = [0.0, 1.0, 1.0, 0.0]
    
    for (cx, cy), lbl in zip(centers, cluster_labels):
        x = np.random.normal(cx, noise, points_per_cluster)
        y = np.random.normal(cy, noise, points_per_cluster)
        clusters.append(np.column_stack((x, y)))
        labels.append(np.full((points_per_cluster, 1), lbl))
        
    X_synth = torch.tensor(np.vstack(clusters), dtype=torch.float32)
    y_synth = torch.tensor(np.vstack(labels), dtype=torch.float32)
    return X_synth, y_synth


class LinearPerceptron(nn.Module):
    """
    Single-Layer Perceptron (Linear Model without hidden layers).
    y = sigmoid(w1*x1 + w2*x2 + b)
    Mathematically incapable of solving non-linearly separable XOR.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


class XORNeuralNetwork(nn.Module):
    """
    Multi-Layer Perceptron with non-linear hidden transformation.
    Architecture: 2 (Input) -> 4 (Hidden + ReLU) -> 2 (Latent Layer) -> 1 (Output)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(2, 1)

    def forward(self, x):
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        logits = self.out(h2)
        return logits

    def extract_latent_space(self, x):
        """Extracts the 2D hidden activation coordinates."""
        self.eval()
        with torch.no_grad():
            h1 = self.act1(self.fc1(x))
            h2 = self.act2(self.fc2(h1))
            return h2.cpu().numpy()


def train_model(model, X, y, epochs=1000, lr=0.05):
    """Trains a model on the continuous XOR distribution."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == y).float().mean().item() * 100.0
            
        history['loss'].append(loss.item())
        history['accuracy'].append(acc)
        
    return history


def generate_visual_artifact(linear_model, mlp_model, X_synth, y_synth, 
                             linear_hist, mlp_hist, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    
    # Generate 2D mesh grid for decision boundary visualization
    x_min, x_max = -0.4, 1.4
    y_min, y_max = -0.4, 1.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    X_np = X_synth.numpy()
    y_np = y_synth.numpy().ravel()
    
    # -------------------------------------------------------------
    # Panel 1: Single-Layer Perceptron (Linear Model Failure)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    linear_model.eval()
    with torch.no_grad():
        grid_probs = torch.sigmoid(linear_model(grid_tensor)).reshape(xx.shape).numpy()
        
    contour1 = ax1.contourf(xx, yy, grid_probs, levels=50, cmap='coolwarm', alpha=0.7)
    ax1.contour(xx, yy, grid_probs, levels=[0.5], colors='black', linewidths=2.5, linestyles='--')
    cbar1 = plt.colorbar(contour1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("P(Y = 1)", fontsize=10)
    
    # Scatter points
    ax1.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], c='blue', edgecolors='k', s=45, label='Class 0 (0,0 / 1,1)')
    ax1.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], c='red', edgecolors='k', s=45, label='Class 1 (0,1 / 1,0)')
    # Highlight exact truth table points
    ax1.scatter([0, 1], [0, 1], c='cyan', s=180, marker='X', edgecolors='black', linewidth=2, label='True (0,0 & 1,1) -> 0')
    ax1.scatter([0, 1], [1, 0], c='yellow', s=180, marker='*', edgecolors='black', linewidth=2, label='True (0,1 & 1,0) -> 1')
    
    ax1.set_title(f"Linear Perceptron (Minsky-Papert Limitation)\nSingle Linear Hyperplane Fails (Acc: {linear_hist['accuracy'][-1]:.1f}%)", 
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel("Input Feature $x_1$", fontsize=10)
    ax1.set_ylabel("Input Feature $x_2$", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, fontsize=8)
    
    # -------------------------------------------------------------
    # Panel 2: Multi-Layer Perceptron (Non-Linear Solution)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    mlp_model.eval()
    with torch.no_grad():
        mlp_probs = torch.sigmoid(mlp_model(grid_tensor)).reshape(xx.shape).numpy()
        
    contour2 = ax2.contourf(xx, yy, mlp_probs, levels=50, cmap='coolwarm', alpha=0.7)
    ax2.contour(xx, yy, mlp_probs, levels=[0.5], colors='black', linewidths=2.5, linestyles='-')
    cbar2 = plt.colorbar(contour2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("P(Y = 1)", fontsize=10)
    
    ax2.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], c='blue', edgecolors='k', s=45, label='Class 0')
    ax2.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], c='red', edgecolors='k', s=45, label='Class 1')
    ax2.scatter([0, 1], [0, 1], c='cyan', s=180, marker='X', edgecolors='black', linewidth=2)
    ax2.scatter([0, 1], [1, 0], c='yellow', s=180, marker='*', edgecolors='black', linewidth=2)
    
    ax2.set_title(f"Multi-Layer Perceptron (Non-Linear Architecture)\nNon-Linear Boundary Perfectly Solves XOR (Acc: {mlp_hist['accuracy'][-1]:.1f}%)", 
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel("Input Feature $x_1$", fontsize=10)
    ax2.set_ylabel("Input Feature $x_2$", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True, fontsize=8)

    # -------------------------------------------------------------
    # Panel 3: Latent Space Geometric Transformation
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    latent_coords = mlp_model.extract_latent_space(X_synth)
    canonical_latents = mlp_model.extract_latent_space(XOR_INPUTS)
    
    ax3.scatter(latent_coords[y_np == 0, 0], latent_coords[y_np == 0, 1], c='blue', alpha=0.6, s=40, label='Transformed Class 0')
    ax3.scatter(latent_coords[y_np == 1, 0], latent_coords[y_np == 1, 1], c='red', alpha=0.6, s=40, label='Transformed Class 1')
    
    # Canonical labels annotated in latent space
    markers = ['(0,0) [0]', '(0,1) [1]', '(1,0) [1]', '(1,1) [0]']
    colors = ['cyan', 'yellow', 'yellow', 'cyan']
    for idx, (hx, hy) in enumerate(canonical_latents):
        ax3.scatter(hx, hy, c=colors[idx], edgecolors='black', s=200, marker='D', zorder=5)
        ax3.annotate(f" {markers[idx]}", (hx, hy), fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8))
        
    ax3.set_title("Latent Space Representation ($h_1$ vs. $h_2$)\nHidden Layers Warp Coordinates into a Linearly Separable Manifold", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Hidden Dimension $h_1$ (Post-ReLU)", fontsize=10)
    ax3.set_ylabel("Hidden Dimension $h_2$ (Post-ReLU)", fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.legend(loc='upper right', frameon=True, fontsize=8)

    # -------------------------------------------------------------
    # Panel 4: Loss & Accuracy Convergence Trajectories
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    epochs = range(1, len(mlp_hist['loss']) + 1)
    
    ax4_acc = ax4.twinx()
    
    # Loss curves
    l1, = ax4.plot(epochs, linear_hist['loss'], 'r--', label='Linear Model Loss (Plateau)', linewidth=1.8)
    l2, = ax4.plot(epochs, mlp_hist['loss'], 'g-', label='MLP Loss (Rapid Minimization)', linewidth=2.2)
    ax4.set_xlabel("Training Iterations (Epochs)", fontsize=10)
    ax4.set_ylabel("BCE Loss", fontsize=10)
    
    # Accuracy curves
    l3, = ax4_acc.plot(epochs, linear_hist['accuracy'], 'orange', linestyle=':', label='Linear Acc (~50-75%)', linewidth=1.8)
    l4, = ax4_acc.plot(epochs, mlp_hist['accuracy'], 'darkgreen', linestyle='-', label='MLP Acc (Reaches 100%)', linewidth=2.0)
    ax4_acc.set_ylabel("Classification Accuracy (%)", fontsize=10)
    ax4_acc.set_ylim([40, 105])
    
    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    ax4.legend(lines, labels, loc='center right', frameon=True, fontsize=8.5)
    ax4.set_title("Convergence Dynamics: Linear Model vs. Multi-Layer Perceptron\nDemonstrating Non-Linear Optimization Trajectory", fontsize=11, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle("PyTorch XOR Problem: The Minsky-Papert Dilemma & Multi-Layer Neural Network Solution", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisual diagnostic report saved successfully at: {os.path.abspath(save_path)}")


def evaluate_truth_table(model, model_name):
    """Evaluates the discrete 4-point XOR truth table."""
    model.eval()
    with torch.no_grad():
        logits = model(XOR_INPUTS)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
    print(f"\n--- {model_name} Truth Table Predictions ---")
    print("  x1   x2  | True Y | Predicted P(Y=1) | Prediction | Status")
    print("------------------------------------------------------------")
    correct = 0
    for i in range(4):
        x1, x2 = XOR_INPUTS[i].numpy()
        true_y = int(XOR_TARGETS[i].item())
        prob = probs[i].item()
        pred = int(preds[i].item())
        status = "CORRECT" if pred == true_y else "FAILED"
        if pred == true_y:
            correct += 1
        print(f"  {x1:0.0f}    {x2:0.0f}  |   {true_y}    |     {prob:>8.4f}     |     {pred}      | {status}")
    print(f"Truth Table Accuracy: {correct}/4 ({correct * 25.0:.1f}%)")
    return correct * 25.0


def main():
    print("=" * 70)
    print("    PROJECT 12: SOLVING THE CLASSIC XOR PROBLEM WITH NEURAL NETWORKS")
    print("=" * 70)
    
    set_seed(42)
    device = torch.device("cpu")
    print(f"Execution Device    : {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "xor_neural_network_analysis.png")
    
    # 1. Generate Dataset
    print("\nGenerating XOR Distribution Dataset (400 Samples around Truth Table Vertices)...")
    X_synth, y_synth = generate_synthetic_xor_distribution(points_per_cluster=100, noise=0.08, seed=42)
    print(f"Dataset Size        : {len(X_synth)} points (200 Class 0, 200 Class 1)")
    print("-" * 70)
    
    # 2. Train Linear Perceptron (No Hidden Layers)
    print("\n[Model 1] Training Single-Layer Perceptron (Linear Model)...")
    set_seed(42)
    linear_model = LinearPerceptron().to(device)
    linear_hist = train_model(linear_model, X_synth, y_synth, epochs=600, lr=0.05)
    linear_tt_acc = evaluate_truth_table(linear_model, "Single-Layer Linear Perceptron")
    
    # 3. Train Multi-Layer Perceptron (With Non-Linear Hidden Layers)
    print("\n[Model 2] Training Multi-Layer Perceptron (2 -> 4 -> 2 -> 1 with ReLU)...")
    set_seed(42)
    mlp_model = XORNeuralNetwork().to(device)
    mlp_hist = train_model(mlp_model, X_synth, y_synth, epochs=600, lr=0.05)
    mlp_tt_acc = evaluate_truth_table(mlp_model, "Multi-Layer Perceptron (MLP)")
    
    # 4. Summary Benchmark Table
    print("\n" + "=" * 70)
    print("                    EXPERIMENTAL BENCHMARK SUMMARY")
    print("=" * 70)
    print(f" Architecture        | Parameters | Final BCE Loss | Truth Table Acc | Distribution Acc")
    print("-" * 70)
    print(f" Linear Perceptron   |     3      |     {linear_hist['loss'][-1]:.4f}     |      {linear_tt_acc:5.1f}%      |      {linear_hist['accuracy'][-1]:5.1f}%")
    print(f" Non-Linear MLP      |    25      |     {mlp_hist['loss'][-1]:.4f}     |     {mlp_tt_acc:5.1f}%      |     {mlp_hist['accuracy'][-1]:5.1f}%")
    print("=" * 70)
    
    # 5. Generate Visual Artifact
    generate_visual_artifact(linear_model, mlp_model, X_synth, y_synth,
                             linear_hist, mlp_hist, save_path)


if __name__ == "__main__":
    main()
