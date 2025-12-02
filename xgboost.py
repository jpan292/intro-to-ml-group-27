
import pandas as pd
import numpy as np
import data_module
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report




# Get data from the preprocessing file
df = data_module.load_data()
X, y, preprocessor = data_module.preprocess(df)
X_train, y_train, X_val, y_val, X_test, y_test = data_module.split(X, y, preprocessor)

pipeline = Pipeline([
    ('xgb', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    ))
])

# Setting up the hyperparameter grid for tuning
param_grid = {
    'xgb__n_estimators': [50, 100, 150],
    'xgb__max_depth': [3, 4, 5],
    'xgb__learning_rate': [0.01, 0.1, 0.2]
}

# We are using randomized search because it is faster than grid search
XGB_searched = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    scoring='roc_auc',
    refit=True,
    n_jobs=-1,
    random_state=42
)

XGB_searched.fit(X_train, y_train)
print(f"Best parameters: {XGB_searched.best_params_}")

# Train Final Model
best_xgb_model = XGBClassifier(
    max_depth=XGB_searched.best_params_.get('max_depth'),
    learning_rate=XGB_searched.best_params_.get('learning_rate'),
    n_estimators=XGB_searched.best_params_.get('n_estimators'),
    objective='binary:logistic',
    booster='gbtree',
    tree_method='hist',
    n_jobs=-1,
    subsample=1,
    reg_alpha=1,
    scale_pos_weight=1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

best_xgb_model.fit(X_train, y_train)


# Plot feature importances
feature_names = preprocessor.get_feature_names_out()

importances = best_xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20
top_indices = indices[:top_n]

f, ax = plt.subplots(figsize=(8, 8))
plt.title("Top 20 Most Important Variables for XGBoost")
sns.set_color_codes("bright")

sns.barplot(y=[feature_names[i] for i in top_indices],
            x=importances[top_indices],
            label="Total", color="b")

ax.set(ylabel="Variable", xlabel="Variable Importance")
sns.despine(left=True, bottom=True)
plt.show()



# Check for overfitting
train_pred = best_xgb_model.predict(X_train)
val_pred = best_xgb_model.predict(X_val)
test_pred = best_xgb_model.predict(X_test)
train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)
print("Check for overfitting")
print("Train Accuracy: ", train_acc)
print("Validation Accuracy: ", val_acc)
print("Test Accuracy: ", test_acc)



# Plot the ROC Curves
train_prob = best_xgb_model.predict_proba(X_train)[:, 1]
val_prob = best_xgb_model.predict_proba(X_val)[:, 1]
test_prob = best_xgb_model.predict_proba(X_test)[:, 1]
train_auc = roc_auc_score(y_train, train_prob)
val_auc = roc_auc_score(y_val, val_prob)
test_auc = roc_auc_score(y_test, test_prob)

plt.figure(figsize=(8, 6))
fpr_train, tpr_train, _ = roc_curve(y_train, train_prob)
plt.plot(fpr_train, tpr_train, linestyle='--', color='red', label=f'Train AUC = {train_auc:.3f}')
fpr_val, tpr_val, _ = roc_curve(y_val, val_prob)
plt.plot(fpr_val, tpr_val, linestyle='solid', color='blue', label=f'Validation AUC = {val_auc:.3f}')
fpr_test, tpr_test, _ = roc_curve(y_test, test_prob)
plt.plot(fpr_test, tpr_test, linestyle='dotted', color='green', label=f'Test AUC = {test_auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc= "lower right")
plt.show()

# Plot metrics
print(classification_report(y_test, test_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, test_pred))


# Plotting Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, test_prob)
avg_precision = average_precision_score(y_test, test_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='orange', lw=2, label=f'Average = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
