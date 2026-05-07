import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df_train = pd.read_csv('pulsar_data_train.csv')
df_test = pd.read_csv('pulsar_data_test.csv')

df_train.columns = [c.strip() for c in df_train.columns]
df_test.columns = [c.strip() for c in df_test.columns]

X_train = df_train.drop(columns=['target_class'])
y_train = df_train['target_class']
X_test = df_test.drop(columns=['target_class'])

X_combined = pd.concat([X_train, X_test], axis=0)

imputer = IterativeImputer(max_iter=10, random_state=42)
imputer.fit(X_combined)

X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

df_train_filled = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=df_train.index)
df_train_filled['target_class'] = y_train

df_test_filled = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=df_test.index)
df_test_filled['target_class'] = df_test['target_class'] 

df_train_filled.to_excel('train_filled.xlsx', index=False)
df_test_filled.to_excel('test_filled.xlsx', index=False)

print("Imputation finished. Files saved.")