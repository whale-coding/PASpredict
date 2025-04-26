from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV


# GridSearchCV
def grid_search_tuning(model, param_grid, X_train, y_train, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc'):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose,
                               scoring=scoring)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    print("Best parameters:", best_params)
    print('Best score:', grid_search.best_score_)

    return grid_search


# Calculation of evaluation indicators
def calculate_metrics(y_pred, y_proba, true_labels):
    # Calculate true negatives, false positives, false negatives, and true positives
    tn, fp, fn, tp = confusion_matrix(true_labels, y_pred).ravel()
    # Calculate metrics
    Recall = recall_score(true_labels, y_pred)
    Specificity = tn / (tn + fp)
    MCC = matthews_corrcoef(true_labels, y_pred)
    Precision = precision_score(true_labels, y_pred)
    False_positive_rate = fp / (fp + tn)
    False_negative_rate = fn / (tp + fn)
    F1 = f1_score(true_labels, y_pred)
    Acc = accuracy_score(true_labels, y_pred)
    AUC = roc_auc_score(true_labels, y_proba)
    AUPR = average_precision_score(true_labels, y_proba)

    return Recall, Specificity, Precision, F1, MCC, Acc, AUC, AUPR


# print metrics format
def format_metrics(recall, specificity, precision, f1, mcc, acc, auc, aupr):
    metrics_str = (
        f"Recall: {recall:.3f}\n"
        f"Specificity: {specificity:.3f}\n"
        f"Precision: {precision:.3f}\n"
        f"F1: {f1:.3f}\n"
        f"MCC: {mcc:.3f}\n"
        f"Accuracy: {acc:.3f}\n"
        f"AUC: {auc:.3f}\n"
        f"AUPR: {aupr:.3f}"
    )
    return metrics_str


# sequential_backward_selection
def sequential_backward_selection(X, y, model, cv=5, scoring='roc_auc'):
    sfs = SequentialFeatureSelector(model, direction='backward', cv=cv, scoring=scoring)
    X_train_selected = sfs.fit_transform(X, y)
    selected_features_indices = sfs.get_support(indices=True)

    return selected_features_indices, X_train_selected, y
