# imports
import pandas as pd
import numpy as np

file_path = '/media/sf_predictive_maintenance/datasets/maintenance_data.csv'
data = pd.read_csv(file_path)

# rename some things
column_renaming = {
    'Air temperature [K]': 'air_temp',
    'Process temperature [K]': 'proc_temp',
    'Rotational speed [rpm]': 'rpm',
    'Torque [Nm]': 'torque',
    'Tool wear [min]': 'wear',
    'Machine failure': 'failure'
}
data.rename(columns=column_renaming, inplace=True)

# aggregates
data['mean_air_temp'] = data['air_temp'].expanding().mean()
data['std_air_temp'] = data['air_temp'].expanding().std()

data['mean_proc_temp'] = data['proc_temp'].expanding().mean()
data['std_proc_temp'] = data['proc_temp'].expanding().std()

data['mean_rpm'] = data['rpm'].expanding().mean()
data['std_rpm'] = data['rpm'].expanding().std()

data['mean_torque'] = data['torque'].expanding().mean()
data['std_torque'] = data['torque'].expanding().std()

# rolling statistics
data['rolling_mean_air_temp'] = data['air_temp'].rolling(window=5).mean()
data['rolling_std_air_temp'] = data['air_temp'].rolling(window=5).std()

data['rolling_mean_proc_temp'] = data['proc_temp'].rolling(window=5).mean()
data['rolling_std_proc_temp'] = data['proc_temp'].rolling(window=5).std()

data['rolling_mean_rpm'] = data['rpm'].rolling(window=5).mean()
data['rolling_std_rpm'] = data['rpm'].rolling(window=5).std()

data['rolling_mean_torque'] = data['torque'].rolling(window=5).mean()
data['rolling_std_torque'] = data['torque'].rolling(window=5).std()

# Interaction features
data['torque_speed_product'] = data['torque'] * data['rpm']
data['temp_ratio'] = data['proc_temp'] / data['air_temp']

# Lag features
data['lag_1_torque'] = data['torque'].shift(1)
data['diff_lag_1_torque'] = data['torque'] - data['lag_1_torque']

# Cumulative features
data['cumulative_sum_torque'] = data['torque'].cumsum()

# Drop rows with NaN values generated by rolling and lag features
data.dropna(inplace=True)
print('Here is the data being saved:')
print()
print(data.head())

try:
    data.to_csv('/media/sf_predictive_maintenance/datasets/data_edited1.csv', index=False)
    print('Edited data saved to datasets folder.')
    # verify
    saved_data = pd.read_csv('/media/sf_predictive_maintenance/datasets/data_edited1.csv')
    print('Verified saved file exists, here are some rows:')
    print()
    print(saved_data.head())
except Exception as e:
    print('File not saved: {e}')