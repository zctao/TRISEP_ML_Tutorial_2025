import os
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_curve, auc, roc_auc_score

from plotting import plot_training_history, plot_train_vs_test, plot_roc_curve

def train_mlp(
    X_train, X_val,
    y_train, y_val,
    w_train, w_val,
    num_epochs=10,
    output_dir='mlp'
    ):
    # output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    w_train_tensor = torch.tensor(w_train.values, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    w_val_tensor = torch.tensor(w_val.values, dtype=torch.float32)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, w_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128)

    # Define a simple neural network model
    model = nn.Sequential(
        nn.Flatten(),  # Input layer
        nn.Linear(X_train.shape[1], 128),  # 1st hidden layer
        nn.ReLU(),  # Activation function
        nn.Linear(128, 128),  # 2nd hidden layer
        nn.ReLU(),  # Activation function
        nn.Linear(128, 1),  # Output layer
        nn.Sigmoid()  # Sigmoid activation for binary classification
    )

    # Binary Cross-Entropy Loss
    loss_fn = nn.BCELoss(reduction='none') # compute loss per sample

    # Using Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    # Train on GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        X_train_tensor = X_train_tensor.to('cuda')
        y_train_tensor = y_train_tensor.to('cuda')
        w_train_tensor = w_train_tensor.to('cuda')
        X_val_tensor = X_val_tensor.to('cuda')
        y_val_tensor = y_val_tensor.to('cuda')
        w_val_tensor = w_val_tensor.to('cuda')

    device = next(model.parameters()).device

    starting_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        weighted_train_loss = 0.0
        # Iterate over batches
        for X_batch, y_batch, w_batch in train_loader:
            # Move data to the appropriate device
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)

            # Forward pass
            outputs = model(X_batch).squeeze()
            loss = loss_fn(outputs, y_batch.float())
            weighted_loss = (loss * w_batch).mean()  # Apply weights to the loss

            # Backward pass and optimization
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            weighted_train_loss += weighted_loss

        weighted_train_loss = weighted_train_loss.detach().item() / len(train_loader)  # Average loss over batches

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = loss_fn(val_outputs, y_val_tensor.float())
            weighted_val_loss = (val_loss * w_val_tensor).mean()  # Apply weights to the loss
            weighted_val_loss = weighted_val_loss.detach().item()  # Convert to scalar

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {weighted_train_loss:.8f}, Validation Loss: {weighted_val_loss:.8f}")

    training_time = time.time() - starting_time
    print(f"Training time: {training_time} seconds")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

    # Plot training and validation loss
    plot_training_history(weighted_train_loss, weighted_val_loss, num_epochs)
    plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'))

    return model

# Evaluate model
def evaluate_mlp(
    model,
    X_train, X_test,
    y_train, y_test,
    w_train, w_test,
    output_dir='mlp'
    ):
    # output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    w_train_tensor = torch.tensor(w_train.values, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    w_test_tensor = torch.tensor(w_test.values, dtype=torch.float32)

    # Move model to GPU if available
    device = next(model.parameters()).device
    if device.type == 'cuda':
        X_train_tensor_gpu = X_train_tensor.to('cuda')
        X_test_tensor_gpu = X_test_tensor.to('cuda')
        y_pred_train = model(X_train_tensor_gpu).detach().cpu().numpy().ravel()
        y_pred_test = model(X_test_tensor_gpu).detach().cpu().numpy().ravel()
    else:
        y_pred_train = model(X_train_tensor).detach().numpy().ravel()
        y_pred_test = model(X_test_tensor).detach().numpy().ravel()

    # plot train vs test
    plot_train_vs_test(
        y_pred_train, y_train_tensor.numpy(), 
        y_pred_test, y_test_tensor.numpy(), 
        weights_train=w_train_tensor.numpy(), weights_test=w_test_tensor.numpy(),
        bins=25, out_range=(0, 1), density=False,
        xlabel="NN output", ylabel="Number of Events", title="Train vs Test"
    )
    plt.savefig(os.path.join(output_dir, 'train_vs_test.png'))

    plot_train_vs_test(
        y_pred_train, y_train_tensor.numpy(), 
        y_pred_test, y_test_tensor.numpy(), 
        weights_train=w_train_tensor.numpy(), weights_test=w_test_tensor.numpy(),
        bins=25, out_range=(0, 1), density=True,
        xlabel="NN output", ylabel="A.U.", title="Train vs Test"
    )
    plt.savefig(os.path.join(output_dir, 'train_vs_test_normalized.png'))

    # Compute and plot ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train_tensor.numpy(), y_pred_train, sample_weight=w_train_tensor.numpy())
    fpr_test, tpr_test, _ = roc_curve(y_test_tensor.numpy(), y_pred_test, sample_weight=w_test_tensor.numpy())
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    plot_roc_curve(
        [fpr_train, fpr_test], 
        [tpr_train, tpr_test], 
        [auc_train, auc_test], 
        labels=['Train', 'Test']
    )
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))

def feature_importance(model, feature_names, X, y, w, output_dir='mlp'):
    # evaluate feature importance using permutation importance

    model.eval()
    device = next(model.parameters()).device

    X_orig = X.copy()

    # Baseline score
    X_tensor = torch.tensor(X_orig, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).detach().cpu().numpy().ravel()
    baseline = roc_auc_score(y.values, y_pred, sample_weight=w.values)

    importances = []
    for i, vname in enumerate(feature_names):
        X_permuted = X_orig.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        X_tensor_perm = torch.tensor(X_permuted, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_perm = model(X_tensor_perm).detach().cpu().numpy().ravel()
        score = roc_auc_score(y.values, y_pred_perm, sample_weight=w.values)
        importances.append(baseline - score)

    importances = np.array(importances)
    sorted_idx = np.argsort(importances)

    # Plot feature importance
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure()
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel('Permutation Importance')
    plt.title('Permutation Importance from MLP')
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()