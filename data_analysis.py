import pandas as pd

def export_analysis(input_file, output_path):
    df = pd.read_csv(input_file)
    
    missing = df.groupby('target_class').apply(lambda x: x.isnull().sum())
    cols = [c for c in df.columns if c != 'target_class']
    means = df.groupby('target_class')[cols].mean()
    medians = df.groupby('target_class')[cols].median()
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        missing.to_excel(writer, sheet_name='Missing_Data')
        means.to_excel(writer, sheet_name='Means')
        medians.to_excel(writer, sheet_name='Medians')
    
export_analysis(r"pulsar_data_train.csv", r"train_analysis.xlsx")
export_analysis(r"pulsar_data_test.csv", r"test_analysis.xlsx")