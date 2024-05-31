# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv('/media/sf_predictive_maintenance/datasets/data_edited1.csv')
print(data.columns)
# drop the product id because ValueError and it isn't needed anyways
# drop type because ValueError and the author noted it was highly irrelevant anyways
data = data.drop(columns=['Product ID', 'Type'])

x = data.drop(columns=['failure'])
y = data['failure']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(x_test)

# Evaluate the model
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_log_reg))

# Initialize and train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

# Make predictions
y_pred_rf = rf.predict(x_test)

# Evaluate the model
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))

# Initialize and train the Gradient Boosting model
gb = GradientBoostingClassifier(random_state=42)
gb.fit(x_train, y_train)

# Make predictions
y_pred_gb = gb.predict(x_test)

# Evaluate the model
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_gb))