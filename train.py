import pandas as pd

import matplotlib.pyplot as plt
from plotting import plot_features, plot_correlations
from dataset import load_dataset, preprocess_dataset

######
## Load data
dataset = load_dataset("dataWW_d1.root", "tree_event")

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

# @@ Exercise: feafure engineering

# Target label
target = dataset["label"]
# Event weights
weights = dataset["mcWeight"]

# Make the datafram with selected features for training
dataset_train = pd.DataFrame(dataset, columns=features)
print(f"Training dataset shape: {dataset_train.shape}")
print(dataset_train.head())

# plot features
plot_features(dataset_train, target, weights)
plt.savefig('features.png')

plot_correlations(dataset_train, target)
plt.savefig('correlations.png')

######
# Preprocess data
# split dataset into training and test sets
test_size = 0.25  # 25% of the data will be used for testing
X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test = preprocess_dataset(dataset_train, target, weights, test_size=test_size)

######
# Train model
###
# Boosted decision trees
do_bdt = True
if do_bdt:
    from bdt import train_bdt, evaluate_bdt, feature_importance
    outdir_bdt = 'bdt'
    bdt = train_bdt(X_train, X_val, y_train, y_val, w_train, w_val, output_dir=outdir_bdt)
    evaluate_bdt(bdt, X_train, X_test, y_train, y_test, w_train, w_test, output_dir=outdir_bdt)
    feature_importance(bdt, dataset_train.columns, output_dir=outdir_bdt)

###
# Neural network
do_nn = True
if do_nn:
    from mlp import train_mlp, evaluate_mlp
    outdir_mlp = 'mlp'
    mlp = train_mlp(X_train, X_val, y_train, y_val, w_train, w_val, output_dir=outdir_mlp)
    evaluate_mlp(mlp, X_train, X_test, y_train, y_test, w_train, w_test, output_dir=outdir_mlp)