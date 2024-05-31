# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# dataset
file_path = "/media/sf_predictive_maintenance/datasets/maintenance_data.csv"
data = pd.read_csv(file_path)

# structure
print("Shape:", data.shape)
print("First 5 Rows:\n", data.head())
print("Summary:\n", data.describe())
print("Info:\n", data.info())

# missing values
print("Missing Values:\n", data.isnull().sum())

# normalize
numeric_features = data.select_dtypes(include=[np.number]).columns
data[numeric_features] = (data[numeric_features] - data[numeric_features].mean()) / data[numeric_features].std()

# visualize
sns.pairplot(data)
plt.savefig("/media/sf_predictive_maintenance/images/pairplot.png")
plt.close()

# fix ValueError
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10,8))
sns.heatmap(numeric_data.corr(), annot=True, cmap=sns.diverging_palette(20, 220, as_cmap=True), fmt=".2f")
plt.savefig("/media/sf_predictive_maintenance/images/corr_heatmap.png")
plt.close()

print("Plots saved to images folder.")
