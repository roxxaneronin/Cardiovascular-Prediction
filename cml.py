import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset from the provided link
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQBTwUSYhGmJo_-X_XTkLZZ0ZJn0GFaYhXPby8Ws0SbWnisLX4zTTJ4iXuxJ2P3QvYCRTEnWb6DFSK8/pub?gid=1836557400&single=true&output=csv"
dataset = pd.read_csv(url)

# Perform data pre-processing
# Drop any rows with missing values
dataset.dropna(inplace=True)

# Split the data into features (X) and labels (y)
X = dataset.drop('id', axis=1)
y = dataset['id']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data analysis and visualization
# Plot histograms of numeric features
dataset.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Plot bar charts for categorical features
categorical_features = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
for feature in categorical_features:
    sns.countplot(x=feature, data=dataset)
    plt.show()

# Calculate the correlation matrix
corr_matrix = dataset.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Feature scaling
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Machine learning techniques
#Support Vector Machines (SVM)
svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("Support Vector Machines (SVM) Accuracy:", svm_accuracy)

# K-Nearest Neighbor (KNN)
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)
print("K-Nearest Neighbor (KNN) Accuracy:", knn_accuracy)

# Decision Trees (DT)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Trees (DT) Accuracy:", dt_accuracy)

# Logistic Regression (LR)
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression (LR) Accuracy:", lr_accuracy)

# Random Forest (RF)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest (RF) Accuracy:", rf_accuracy)

# Model building with the best-performing technique
best_model = svm  # Choose the best-performing model here

# Train the model on the complete dataset
X_scaled = scaler.transform(X)
best_model.fit(X_scaled, y)

# Example prediction
example_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
example_data_scaled = scaler.transform(example_data)
prediction = best_model.predict(example_data_scaled)
print("Example prediction:", prediction)
