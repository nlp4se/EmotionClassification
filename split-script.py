import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

df = pd.read_csv("GroundTruth2.csv", quotechar='"', sep=';')


# Define columns
label_cols = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation', 'Neutral']

# Separate features and labels
X = df[['sentence']].values
y = df[label_cols].values    # Binary multi-label array

# Perform 80-20 multi-label stratified split
X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)

# Convert back to DataFrames
train_df = pd.DataFrame(X_train, columns=['sentence'])
train_df[label_cols] = y_train

test_df = pd.DataFrame(X_test, columns=['sentence'])
test_df[label_cols] = y_test

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Optional: Check label balance
print("Train label distribution:\n", train_df[label_cols].sum())
print("Test label distribution:\n", test_df[label_cols].sum())
