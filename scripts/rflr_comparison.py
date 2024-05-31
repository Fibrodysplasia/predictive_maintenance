# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load the edited dataset
data = pd.read_csv('/media/sf_predictive_maintenance/datasets/data_edited1.csv')
data = data.drop(columns=['Product ID',
                          'Type',
                          'TWF',
                          'HDF',
                          'PWF',
                          'OSF',
                          'RNF'])
x = data.drop(columns=['failure'])
y = data['failure']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# standardize
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# logistic regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train, y_train)
y_pred_lr = log_reg.predict(x_test)
y_prob_lr = log_reg.predict_proba(x_test)[:, 1]

# random forest
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

# NOTE TO SELF
# Here we have the probability something should be classified
# as a failure. This kind of info can be used to have a real-time
# prediction of whether something will fail and could be used to set
# a threshold (i.e. label something for failure when it's 85% likely to fail)
y_prob_rf = rf.predict_proba(x_test)[:, 1]

# evaluation
print("Logistic Regression ROC AUC: ", roc_auc_score(y_test, y_prob_lr))
print("Random Forest ROC AUC: ", roc_auc_score(y_test, y_prob_rf))

comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Logistic Regression Prediction': y_pred_lr,
    'Random Forest Prediction': y_pred_rf,
    'Logistic Regression Probability': y_prob_lr,
    'Random Forest Probability': y_prob_rf
})
# Check
print(comparison_df.head())

try:
    # feature importances for rf
    feature_importances_rf = rf.feature_importances_
    features = data.drop(columns=['failure']).columns
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances_rf
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importances - Random Forest')
    plt.savefig('../images/feature_importances_rf.png')
    plt.close()

    # coefficients for lr
    coefficients = log_reg.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.title('Coefficients - Logistic Regression')
    plt.savefig('../images/coefficients_lr.png')
    plt.close()
    
    print('Files saved to images folder.')
except Exception as e:
    print('Files could not be saved: {e}')
