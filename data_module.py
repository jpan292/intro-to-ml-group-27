'''This module ensures all models train and evaluate on the same data '''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# constants
RANDOM_SEED = 42
TEST_SIZE = 0.3 #leave 70% for training
VAL_SIZE = 0.50 #validation and test set should each be 15% of og dataset

# function to load dataset, default location set
def load_data(path="./bots_vs_users.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    #all values in these columns are 1
    df = df.drop(columns=["has_short_name", "has_full_name", "city"], errors="ignore")
    df.head()
    # Preprocess the data by handling the empty cells and cells that contain "Unknown" as a value
    # Cannot drop empty cells because it is one of the signals indicating a bot account
    # Strategy: replace empty cells with NaN
    df.replace("",np.nan,inplace=True)

    # Identify numerical features
    num_cols = df.select_dtypes(include =['float64']).columns.tolist()

    # subscribers_count feature is supposed to be a numerical feature but has "Unknown" cells
    # strategy: convert "Uknown" to NaN and then append subscribers_count into numerical_cols
    df['subscribers_count'] = df['subscribers_count'].replace("Unknown",np.nan)
    df['subscribers_count'] = pd.to_numeric(df['subscribers_count'])
    num_cols.append('subscribers_count')

    # Categorical features are everything else except for the target feature
    cat_cols = [c for c in df.columns if c not in num_cols and c != "target"]

    # Add missingness indicators (in order to distinguish cells with actual values and empty cells)
    for col in num_cols:
        df[col+'_missing'] = df[col].isnull().astype(int)

    num_cols_with_indicator = num_cols + [col + '_missing' for col in num_cols]

    # Preprocessing Transformers
    # For numeric features, preprocess the data by median imputation (replace missing values with the median of the features)
    numeric_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'median'))
    ])

    # For categorical features use One Hot Encoding
    categorical_transformer = Pipeline(steps = [
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ])
    # Combining everything into a singular preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols_with_indicator),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )
   
    y = df["target"].astype("float32").values
    X_df = df.drop(columns=["target"])
    return X_df, y, preprocessor

def split(X, y, preprocessor, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED):
    #First split: 70% train, 30% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,#keep class balance consistent
    )
    #Second split: split temp into 50/50 into 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size= val_size,
        random_state= random_state,
        stratify=y_temp,
    )
    X_train = preprocessor.fit_transform(X_train)
    X_val   = preprocessor.transform(X_val)
    X_test  = preprocessor.transform(X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test
