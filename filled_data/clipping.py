import pandas as pd

train = pd.read_excel('train_filled.xlsx')
test  = pd.read_excel('test_filled.xlsx')

# 1. Clip physically impossible imputed values
train['Standard deviation of the DM-SNR curve'] = train['Standard deviation of the DM-SNR curve'].clip(lower=7.37)
train['Skewness of the DM-SNR curve']           = train['Skewness of the DM-SNR curve'].clip(lower=-1.98)
test['Standard deviation of the DM-SNR curve']  = test['Standard deviation of the DM-SNR curve'].clip(lower=7.37)
test['Skewness of the DM-SNR curve']            = test['Skewness of the DM-SNR curve'].clip(lower=-1.98)

# 2. Split train into train/val for your CV loop
X = train.drop(columns='target_class').values
y = train['target_class'].values
X_unlabelled = test.drop(columns='target_class').values