import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Load dataset
file_path = 'UNSW_NB15_training-set.csv'
data = pd.read_csv(file_path)

# Drop rows with missing values
data = data.dropna()

# Encode categorical columns
categorical_cols = ['proto', 'service', 'state', 'attack_cat']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Split dataset into features and target
X = data.iloc[:, :-2]  # Exclude last two columns: 'attack_cat' and 'label'
y = data.iloc[:, -1]   # Use 'label' column as target (0 = normal, 1 = attack)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("xtrain shape : ", X_train.shape)
print("xtest shape  : ", X_test.shape)
print("ytrain shape : ", y_train.shape)
print("ytest shape  : ", y_test.shape)

# Initialize and fit Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest.fit(X_train)

# Predict anomalies in the test set (-1 = anomaly, 1 = normal)
anomaly_pred = iso_forest.predict(X_test)

# Convert predictions to match label format (0 = normal, 1 = attack)
# Isolation Forest: -1 = anomaly (predict as 1), 1 = normal (predict as 0)
anomaly_pred_binary = [0 if x == 1 else 1 for x in anomaly_pred]

# 1. Visualize anomaly score distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=anomaly_pred, palette='viridis')
plt.title('Isolation Forest Anomaly Detection (Test Set)')
plt.xlabel('Anomaly Score (-1 = Anomaly, 1 = Normal)')
plt.ylabel('Count')
plt.show()

# 2. Cross-tabulation of predicted vs actual
anomaly_crosstab = pd.crosstab(pd.Series(anomaly_pred_binary, name='Predicted'), 
                               pd.Series(y_test.values, name='Actual'))
print("\nCross-tabulation of anomaly detection and actual labels:")
print(anomaly_crosstab)

# 3. Evaluation metrics
accuracy = accuracy_score(y_test, anomaly_pred_binary)
precision = precision_score(y_test, anomaly_pred_binary)
recall = recall_score(y_test, anomaly_pred_binary)
conf_matrix = confusion_matrix(y_test, anomaly_pred_binary)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

# 4. Full classification report
print("\nClassification Report:")
print(classification_report(y_test, anomaly_pred_binary, target_names=["Normal (0)", "Attack (1)"]))

import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
