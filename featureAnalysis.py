import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('bots_vs_users.csv')
df_num = df.select_dtypes(include=[np.number])
corr = df_num.corr()

# Analysis of numeric features
#Check for leakage by using correlation matrix and eliminating abnormally high correlations (>=0.7)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    square=True,
    linewidths=.5,
    cbar=True
)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


X = df.drop(columns=['target'])
y = df['target']

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
plot_df = X_train.copy()
plot_df['target'] = y_train

cat_cols = plot_df.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=plot_df, x=col, hue='target')
    plt.title(f"Distribution of '{col}' by Target")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
