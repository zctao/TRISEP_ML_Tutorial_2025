import os
import time
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc

from plotting import plot_train_vs_test, plot_roc_curve

def train_bdt(
    X_train, X_val,
    y_train, y_val,
    w_train, w_val,
    output_dir='bdt'
    ):
    # output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xgb = XGBClassifier(
        tree_method='hist', # 'gpu_hist' for GPU acceleration
        eval_metric='logloss'
    )
    # See https://xgboost.readthedocs.io/en/stable/parameter.html for more parameters

    starting_time = time.time()

    xgb.fit(X_train, y_train.values, sample_weight=w_train.values)

    training_time = time.time() - starting_time
    print(f"Training time: {training_time:.2f} seconds")

    # Save the trained model
    xgb.save_model(os.path.join(output_dir, 'model.json'))

    return xgb

def evaluate_bdt(
    model,
    X_train, X_test,
    y_train, y_test,
    w_train, w_test,
    output_dir='bdt'
    ):
    # output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_pred_train = model.predict_proba(X_train)[:, 1].ravel()
    y_pred_test = model.predict_proba(X_test)[:, 1].ravel()

    # plot train vs test
    plot_train_vs_test(
        y_pred_train, y_train, 
        y_pred_test, y_test, 
        weights_train=w_train.values, weights_test=w_test.values,
        bins=25, out_range=(0, 1), density=False,
        xlabel="BDT output", ylabel="Number of Events", title="Train vs Test"
    )
    plt.savefig(os.path.join(output_dir, 'train_vs_test.png'))

    plot_train_vs_test(
        y_pred_train, y_train, 
        y_pred_test, y_test, 
        weights_train=w_train.values, weights_test=w_test.values,
        bins=25, out_range=(0, 1), density=True,
        xlabel="BDT output", ylabel="A.U.", title="Train vs Test (Normalized)"
    )
    plt.savefig(os.path.join(output_dir, 'train_vs_test_normalized.png'))

    # Compute and plot ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train, sample_weight=w_train.values)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test, sample_weight=w_test.values)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    plot_roc_curve(
        [fpr_train, fpr_test],
        [tpr_train, tpr_test],
        [auc_train, auc_test],
        ['Train', 'Test']
    )
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))

def feature_importance(model, feature_names, output_dir='bdt'):
    """
    Plot feature importance from the trained BDT model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure()
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance from BDT')
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()