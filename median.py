import pandas as pd

train_df = pd.read_csv('pulsar_data_train.csv')
test_df = pd.read_csv('pulsar_data_test.csv')

train_medians = train_df.median()
train_df = train_df.fillna(train_medians)
test_df = test_df.fillna(train_medians)

train_df.to_excel('train_filled_peter.xlsx', index=False)
test_df.to_excel('test_filled_peter.xlsx', index=False)