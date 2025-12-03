
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


best_model = XGB_searched.best_estimator_


# Plot feature importances
feature_names = preprocessor.get_feature_names_out()

importances = best_model.named_steps['xgb'].feature_importances_
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

val_pred = best_model.predict(X_val)
test_pred = best_model.predict(X_test)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)
print("Check for overfitting")

print("Validation Accuracy: ", val_acc)
print("Test Accuracy: ", test_acc)



# Plot the ROC Curves
val_prob = best_model.predict_proba(X_val)[:, 1]
test_prob = best_model.predict_proba(X_test)[:, 1]
val_auc = roc_auc_score(y_val, val_prob)
test_auc = roc_auc_score(y_test, test_prob)

plt.figure(figsize=(8, 6))
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
