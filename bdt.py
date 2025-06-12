import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
import scipy.stats as stats

from plotting import plot_train_vs_test, plot_roc_curve, plot_confusion_matrix

def train_bdt(
    X_train,
    y_train,
    w_train,
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

    # plot confusion matrix
    cm_train = confusion_matrix(y_train, (y_pred_train > 0.5).astype(int), sample_weight=w_train.values)
    cm_test = confusion_matrix(y_test, (y_pred_test > 0.5).astype(int), sample_weight=w_test.values)
    plot_confusion_matrix(cm_train, cm_test)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

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

def feature_permutation(model, feature_names, X, y, w, output_dir='bdt'):
    """
    Compute and plot feature permutation importance.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result = permutation_importance(
        model, X, y, sample_weight=w.values,
        scoring='roc_auc', n_repeats=1, random_state=42, n_jobs=-1
    )

    sorted_idx = result.importances_mean.argsort()
    plt.figure()
    plt.barh(feature_names[sorted_idx], result.importances_mean[sorted_idx])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Permutation Importance from BDT')
    plt.savefig(os.path.join(output_dir, 'feature_permutation_importance.png'))
    plt.close()

def plot_learning_curve(
    X_train, y_train, w_train,
    output_dir='bdt'
    ):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xgb = XGBClassifier(
        tree_method='hist',
        eval_metric='logloss'
    )

    train_sizes = np.linspace(0.1, 1.0, 6)
    train_scores = []
    val_scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for frac in train_sizes:
        n_train = int(frac * len(X_train))
        X_sub = X_train[:n_train]
        y_sub = y_train[:n_train]
        w_sub = w_train[:n_train]

        train_score = []
        val_score = []

        for train_idx, val_idx in skf.split(X_sub, y_sub):
            X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_val = y_sub[train_idx], y_sub[val_idx]
            w_tr, w_val = w_sub[train_idx], w_sub[val_idx]

            xgb.fit(X_tr, y_tr, sample_weight=w_tr)
            train_score.append(xgb.score(X_tr, y_tr, sample_weight=w_tr))
            val_score.append(xgb.score(X_val, y_val, sample_weight=w_val))

        train_scores.append(np.mean(train_score))
        val_scores.append(np.mean(val_score))

    plt.figure()
    plt.plot(train_sizes, train_scores, label='Training Score')
    plt.plot(train_sizes, val_scores, label='Validation Score')
    plt.xlabel('Training Size (fraction)')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))

def hyperparameter_tuning(
    X_train, y_train, w_train,
    X_test, y_test, w_test,
    ):
    """
    Perform hyperparameter tuning for the BDT model.
    """
    print("Starting hyperparameter tuning...")
    # specify parameters, range, and distributions to sample from
    param_dist_XGB = {
        'max_depth': stats.randint(3, 12), # default 6
        'n_estimators': stats.randint(300, 800), #default 100
        'learning_rate': stats.uniform(0.1, 0.5), #def 0.3
    }

    gsearch = RandomizedSearchCV(
        estimator = XGBClassifier(tree_method="hist",eval_metric='logloss'),
        param_distributions = param_dist_XGB,
        scoring='roc_auc',
        n_iter=10,
        cv=2
    )

    gsearch.fit(X_train, y_train.values, sample_weight=w_train.values)

    print("Best parameters found: ", gsearch.best_params_)
    print("Best score: ", gsearch.best_score_)

    y_pred_test = gsearch.predict_proba(X_test)[:, 1].ravel()
    print("Score on test dataset: ",roc_auc_score(y_true=y_test.values, y_score=y_pred_test, sample_weight=w_test.values))
    dfsearch=pd.DataFrame.from_dict(gsearch.cv_results_)
    print(dfsearch)