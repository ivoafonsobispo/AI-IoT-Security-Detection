import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from result_metrics import calculate_metrics
from binary_model import train_binary_model

# Read CSV
df_train = pd.read_csv("../Datasets/flow-farm_train_smote.csv")
df_test = pd.read_csv("../Datasets/flow-farm_test.csv")

# Transform the labels into binary values
df_train['is_attack'] = df_train['type'].apply(lambda x: 0 if x == "normal" else 1)
df_train.groupby('is_attack')['is_attack'].count()

df_test['is_attack'] = df_test['type'].apply(lambda x: 0 if x == "normal" else 1)
df_test.groupby('is_attack')['is_attack'].count()

# Delete Multi-class Column
df_train = df_train.drop('type', axis=1)
df_test = df_test.drop('type', axis=1)

# Create Encoder Training
X_columns = df_train.columns.drop('is_attack')
X_columns_test = df_test.columns.drop('is_attack')

le = LabelEncoder()
le.fit(df_train["is_attack"].values)
le.fit(df_test["is_attack"].values)

X = df_train[X_columns].values
X_test = df_test[X_columns_test].values

y = df_train["is_attack"].values
y_test = df_test["is_attack"].values

y = le.transform(y)
y_test = le.transform(y_test)

# Split into training and testing sets
X_train, X_train_test, y_train, y_train_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

#train_binary_model(X_train, y_train, X_train_test, y_train_test)

# Validate Model
# Load the best saved model
best_model = load_model('best_model_binary_smote.h5')

pred = best_model.predict(X_test)
pred = np.round(pred).astype(int)
calculate_metrics("Binary SMOTE - DNN", y_test, pred)