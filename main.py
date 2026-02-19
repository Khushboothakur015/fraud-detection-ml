import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
print("Loading data...")
df = pd.read_csv('creditcard.csv')
pip install xgboost


# 2. Basic Exploration
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts(normalize=True) * 100}")

# 3. Data Preprocessing
# Standardize 'Amount' as it has a different scale compared to V1-V28
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
# Drop original Amount and Time (as Time is just seconds elapsed and often not directly useful without feature engineering)
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# 4. Feature and Target splitting
X = df.drop('Class', axis=1)
y = df['Class']

# 5. Train-Test Split (using a smaller test size to ensure enough fraud cases in both)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Model Training - Random Forest with Class Weighting
# Using class_weight='balanced' to handle the extreme imbalance
print("Training Random Forest model (this may take a minute)...")
model = RandomForestClassifier(n_estimators=100, 
                               random_state=42, 
                               n_jobs=-1, 
                               class_weight='balanced')
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# 8. Evaluation
print("\nEvaluation Results:")
print(classification_report(y_test, y_pred))

auprc = average_precision_score(y_test, y_probs)
print(f"Average Precision Score (PR-AUC): {auprc:.4f}")

# 9. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('/home/sandbox/confusion_matrix.png')
print("Confusion matrix saved to /home/sandbox/confusion_matrix.png")

# 10. Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
importances.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.savefig('/home/sandbox/feature_importance.png')
print("Feature importance plot saved to /home/sandbox/feature_importance.png")