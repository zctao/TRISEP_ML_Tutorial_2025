import matplotlib.pyplot as plt
import numpy as np

def plot_features(features_df, target, weights):
    plt.figure()
    # background distributions
    ax = features_df[target==0].hist(
        weights=weights[target==0],
        figsize=(15,12),
        color='blue',
        alpha=0.5, 
        density=True,
        label='Background'
    )

    ax = ax.flatten()[:features_df.shape[1]]

    # signal distributions
    features_df[target==1].hist(
        weights=weights[target==1],
        figsize=(15,12),
        color='red',
        alpha=0.5,
        density=True,
        ax=ax,
        label='Signal'
    )

    # add legends
    for a in ax:
        a.legend()

def plot_training_history(training_loss, validation_loss, num_epochs):
    plt.figure()
    plt.plot(range(1, num_epochs + 1), [training_loss.item()] * num_epochs, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), [validation_loss.item()] * num_epochs, label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def plot_train_vs_test(
    y_pred_train, y_train, y_pred_test, y_test, 
    weights_train=None, weights_test=None,
    bins=25, out_range=(0,1), density=False,
    xlabel="", ylabel="a.u.", title=""
    ):
    plt.figure()

    if weights_train is not None:
        weights_train_0 = weights_train[y_train == 0]
        weights_train_1 = weights_train[y_train == 1]
    else:
        weights_train_0 = None
        weights_train_1 = None

    plt.hist(y_pred_train[y_train == 1], bins=bins, color='r', alpha=0.5, range=out_range, histtype='stepfilled', density=density, label='S (train)', weights=weights_train_1)

    plt.hist(y_pred_train[y_train == 0], bins=bins, color='b', alpha=0.5, range=out_range, histtype='stepfilled', density=density, label='B (train)', weights=weights_train_0)

    if weights_test is not None:
        weights_test_0 = weights_test[y_test == 0]
        weights_test_1 = weights_test[y_test == 1]
    else:
        weights_test_0 = None
        weights_test_1 = None

    hist, bins = np.histogram(y_pred_test[y_test==1], bins=bins, range=out_range, density=density, weights=weights_test_1)
    scale = len(y_pred_test[y_test==1]) / hist.sum()
    err = np.sqrt(hist*scale) / scale
    center = (bins[1:] + bins[:-1]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', color='r', label='S (test)')

    hist, bins = np.histogram(y_pred_test[y_test==0], bins=bins, range=out_range, density=density, weights=weights_test_0)
    scale = len(y_pred_test[y_test==0]) / hist.sum()
    err = np.sqrt(hist*scale) / scale
    center = (bins[1:] + bins[:-1]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', color='b', label='B (test)')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def plot_roc_curve(fpr_arr, tpr_arr, auc_arr, labels):
    plt.figure()
    for fpr, tpr, auc, label in zip(fpr_arr, tpr_arr, auc_arr, labels):
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()