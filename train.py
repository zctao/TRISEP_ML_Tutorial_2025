import time
import uproot
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from plotting import plot_training_history, plot_train_vs_test, plot_roc_curve

######
## Load data
filename = "dataWW_d1.root"
treename = "tree_event"
with uproot.open(filename) as file:
    tree = file[treename]
    # load events as pandas DataFrame
    dataset = tree.arrays(library="pd")
    print(f"Loaded {len(dataset)} events from {filename}.")

# shuffle data
dataset = dataset.sample(frac=1).reset_index(drop=True)

######
# Examine dataset
# Print first few events
#print(dataset.head())

# Print dataset statistics
#print(dataset.describe())

label_weights = ( dataset[dataset.label==0].mcWeight.sum(), dataset[dataset.label==1].mcWeight.sum() )
print(f"Sum of label weights: Background, Signal = {label_weights}")

label_nevents = ( dataset[dataset.label==0].shape[0], dataset[dataset.label==1].shape[0] )
print(f"Total class number of events: Background, Signal = {label_nevents}")

######
# Event selection
# Select events with exactly 2 leptons
print(f"Dataset shape before selection: {dataset.shape}")
# And only events with positive weights
dataset_selected = dataset[ (dataset.lep_n == 2) & (dataset.mcWeight > 0) ]
print(f"Dataset shape after selection: {dataset_selected.shape}")

######
# Features to train on
features = [
    "met_et", "met_phi",
    "lep_pt_0", "lep_pt_1",
    "lep_eta_0", "lep_eta_1",
    "lep_phi_0", "lep_phi_1",
    "jet_n",
    "jet_pt_0", "jet_pt_1", 
    "jet_eta_0", "jet_eta_1",
    "jet_phi_0", "jet_phi_1"
]

target = dataset_selected["label"]
weights = dataset_selected["mcWeight"]

dataset_train = pd.DataFrame(dataset_selected, columns=features)
print(f"Training dataset shape: {dataset_train.shape}")
print(dataset_train.head())

# plot features
# TODO
# distributions, correlations
# also weights

######
# Preprocess data
# split dataset into training and test sets
test_size = 0.25  # 25% of the data will be used for testing
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    dataset_train, target, weights, test_size=test_size
)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
w_train = w_train.reset_index(drop=True)
w_test = w_test.reset_index(drop=True)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, w_train shape: {w_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, w_test shape: {w_test.shape}")

# further split test set into validation and test sets

X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
    X_test, y_test, w_test, test_size=0.5
)

# Standardize features
# scale to mean of 0 and standard deviation of 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Adjust event weights: train on equal amount of signal and background events; test with original weights
class_weights_train = (w_train[y_train == 0].sum(), w_train[y_train == 1].sum())
print(f"class_weights_train: Background, Signal = {class_weights_train}")

for i in range(len(class_weights_train)):
    w_train[y_train == i] *= max(class_weights_train) / class_weights_train[i] #equalize number of background and signal event
    w_test[y_test == i] *= 1/test_size/0.5  #increase test weight to compensate for sampling

print(f"Train weights: Background, Signal = {w_train[y_train == 0].sum()}, {w_train[y_train == 1].sum()}")
print(f"Test weights: Background, Signal = {w_test[y_test == 0].sum()}, {w_test[y_test == 1].sum()}")

######
# Train model
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
w_train_tensor = torch.tensor(w_train.values, dtype=torch.float32)

# Create DataLoader for training data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, w_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
w_val_tensor = torch.tensor(w_val.values, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
w_test_tensor = torch.tensor(w_test.values, dtype=torch.float32)

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

loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
starting_time = time.time()
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch, w_batch in train_loader:
        # Forward pass
        outputs = model(X_batch).squeeze()
        loss = loss_fn(outputs, y_batch.float())
        weighted_loss = (loss * w_batch).mean()  # Apply weights to the loss

        # Backward pass and optimization
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor).squeeze()
        val_loss = loss_fn(val_outputs, y_val_tensor.float())
        weighted_val_loss = (val_loss * w_val_tensor).mean()  # Apply weights to the loss

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {weighted_loss.item():.8f}, Validation Loss: {weighted_val_loss.item():.8f}")

training_time = time.time() - starting_time
print(f"Training time: {training_time} seconds")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

# Plot training and validation loss
plot_training_history(weighted_loss, weighted_val_loss, num_epochs)
plt.savefig('training_validation_loss.png')

######
# Evaluate model
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
plt.savefig('train_vs_test.png')

plot_train_vs_test(
    y_pred_train, y_train_tensor.numpy(), 
    y_pred_test, y_test_tensor.numpy(), 
    weights_train=w_train_tensor.numpy(), weights_test=w_test_tensor.numpy(),
    bins=25, out_range=(0, 1), density=True,
    xlabel="NN output", ylabel="A.U.", title="Train vs Test"
)
plt.savefig('train_vs_test_normalized.png')

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
plt.savefig('roc_curve.png')
