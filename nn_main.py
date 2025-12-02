import pandas as pd
import numpy as np
import os
import random
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import data_module
"""
README:

2 model variants are trained and tested in this program. 

1. Simple NN with no hidden layers
2. Complex NN with 2 hidden layers
"""


def plot_confusion_matrix(cm, model_name):
    """Plot a 2x2 confusion matrix."""
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    classes = ["Human (0)", "Bot (1)"]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #Write counts on the matrix
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


"""
Run evaluation on test set:
- Confusion matrix
- Accuracy, Precision, Recall, F1, AUC
- ROC curve
"""
def evaluate_and_report(model, X_test, y_test, model_name):
    #Get predicted probabilities for the positive class
    y_proba = model.predict(X_test).ravel()

    #Convert probabilities to binary labels using 0.5 threshold
    y_pred = (y_proba >= 0.5).astype(int)

    #Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)

    #Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    #ROC + AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    print(f"\n===== Test metrics for {model_name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 score : {f1:.4f}")
    print(f"AUC      : {roc_auc:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc,
        "confusion_matrix": cm,
    }

"""
Basic NN (No Hidden Layers)
- Input directly to output

"""

def build_model_1(input_dim):

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model

"""
NN with 2 Hidden Layers
- Two hidden layers with ReLU
- Dropout to help prevent overfitting
- Hidden Layer 1 with 64 nodes
- Hidden Layer 2 with 32 nodes
"""

def build_model_2(input_dim):

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model



df = data_module.load_data()
X, y, preprocessor = data_module.preprocess(df)
X_train, y_train, X_val, y_val, X_test, y_test = data_module.split(X, y, preprocessor)


SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


n_samples, n_features = X.shape
print(f"Samples: {n_samples}, Features: {n_features}")



"""
#For Debug
print(f"Train size: {X_train.shape[0]}")
print(f"Val size:   {X_val.shape[0]}")
print(f"Test size:  {X_test.shape[0]}")

"""

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

n_features = X_train_scaled.shape[1]




early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

#Training Factors
EPOCHS = 50
BATCH_SIZE = 64

results_summary = {}

#Model 1
model1 = build_model_1(n_features)
model1.summary()

start_time = time.time()
history1 = model1.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)
time_model1 = time.time() - start_time
print(f"Training time (Model 1): {time_model1:.2f} seconds")

metrics1 = evaluate_and_report(model1, X_test_scaled, y_test, "Model 1: No Hidden Layers")
metrics1["training_time_sec"] = time_model1
results_summary["Model 1"] = metrics1

#Model 2
model2 = build_model_2(n_features)
model2.summary()

start_time = time.time()
history2 = model2.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)
time_model2 = time.time() - start_time
print(f"Training time (Model 2): {time_model2:.2f} seconds")

metrics2 = evaluate_and_report(model2, X_test_scaled, y_test, "Model 2: Hidden Layers")
metrics2["training_time_sec"] = time_model2
results_summary["Model 2"] = metrics2

plt.show()

