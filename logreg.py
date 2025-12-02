# Users vs bots classification: Logistic Regression Model

#Import necessary libraries:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report)
import warnings
warnings.filterwarnings('ignore')
import data_module

df = data_module.load_data()
X, y, preprocessor = data_module.preprocess(df)
X_train, y_train, X_val, y_val, X_test, y_test = data_module.split(X, y, preprocessor)



logreg_classifier = LogisticRegression(max_iter=1000, random_state=42)
logreg_classifier.fit(X_train, y_train) # Train the model

#Acess feature's importance using coefficients
feature_names = preprocessor.get_feature_names_out()
coef = pd.Series(logreg_classifier.coef_[0], index=feature_names)
print(coef.sort_values(ascending=False))

"""## Evaluation"""

# Prediciton and evaluation for all models
models = {'Logistic Regression': logreg_classifier}

for model_name, model in models.items():
  ytrain_pred = model.predict(X_train)
  ytest_pred = model.predict(X_test)

  print(f"---{model_name}---")
  print("Test accuracy:", accuracy_score(y_test, ytest_pred))
  print("Precision:", precision_score(y_test, ytest_pred))
  print("Recall:", recall_score(y_test, ytest_pred))
  print("F1 score:", f1_score(y_test, ytest_pred))
  print("Confusion matrix:\n", confusion_matrix(y_test,ytest_pred))
  print("Classification Report: \n", classification_report(y_test,ytest_pred))

  #Calculate sensitivity and specificity
  # - True Positive: 1 predicted as 1
  # - True Negative: 0 predicted as 0
  # - False Positive: 1 predicted as 0
  # - False Negative: 0 predicted as 1
  tn, fp, fn, tp = confusion_matrix(y_test, ytest_pred).ravel()
  sensitivity = tp / (tp + fn) # recall
  specificity = tn / (tn + fp) # specificity
  print(f"Sensitivity (Recall) for {model_name}: {sensitivity:.4f}")
  print(f"Specificity for {model_name}: {specificity:.4f}")

  # Plot confusion matrix heatmap
  cm = confusion_matrix(y_test, ytest_pred)
  plt.figure()
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
  plt.title(f'Confusion Matrix for {model_name}')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.show()

  # ROC Curve and AUC
  ytest_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
  fpr, tpr, thresholds = roc_curve(y_test, ytest_prob)
  roc_auc = roc_auc_score(y_test, ytest_prob)

  plt.figure()
  plt.plot(fpr, tpr, label = f'{model_name} (AUROC = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], 'k--',color = 'gray')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'ROC Curve for {model_name}')
  plt.legend(loc = 'lower right')
  plt.show()
