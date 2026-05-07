import pandas as pd

# Load the data
df = pd.read_csv('pulsar_data_train.csv')

# Filter for rows without missing values
df_complete = df.dropna()

# Filter for rows with at least one missing value
df_incomplete = df[df.isnull().any(axis=1)]

# Export to Excel
df_complete.to_excel('complete_data.xlsx', index=False)
df_incomplete.to_excel('incomplete_data.xlsx', index=False)