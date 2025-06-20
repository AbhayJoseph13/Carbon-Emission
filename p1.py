import pandas as p
import numpy as n

file_path = r"C:\Users\Abhay\Downloads\55e3c34c8c2c20110434.xls"
sheet_name = "Data"  

df = p.read_excel(file_path, sheet_name=sheet_name, engine='xlrd')

df_clean = df[(df['SCALE'] != 'Text') & (df['Decimals'] != 'Text')]

df_clean = df_clean.drop(columns=['Country name', 'Series code', 'SCALE', 'Decimals'], errors='ignore')

df_clean = df_clean.replace({'': n.nan, '..': n.nan, 'NA': n.nan, 'N/A': n.nan, '-': n.nan})

df_clean.iloc[:, 2:] = df_clean.iloc[:, 2:].apply(p.to_numeric, errors='coerce')

df_clean = df_clean.drop_duplicates()

threshold = 0.5 * df_clean.shape[0]
df_clean = df_clean.dropna(thresh=threshold, axis=1)

df_clean.iloc[:, 2:] = df_clean.iloc[:, 2:].apply(lambda col: col.fillna(col.mean()))

df_clean = df_clean.loc[~(df_clean.iloc[:, 2:].sum(axis=1) == 0)]

numeric_data = df_clean.iloc[:, 2:]
means = numeric_data.mean()
stds = numeric_data.std()

def is_not_outlier(row):
    return ((row >= (means - 3 * stds)) & (row <= (means + 3 * stds))).all()

mask = numeric_data.apply(is_not_outlier, axis=1)
df_clean = df_clean[mask]

df_clean['avg_value'] = df_clean.iloc[:, 2:].mean(axis=1)
df_clean['median_value'] = df_clean.iloc[:, 2:].median(axis=1)
df_clean['std_dev'] = df_clean.iloc[:, 2:].std(axis=1)

print("Basic Statistics:")
print(df_clean.describe())

print("\nCorrelation matrix (numeric columns):")
print(df_clean.iloc[:, 2:].corr())

df_clean.to_csv("cleaned_data_enhanced.csv", index=False)
print("\nCleaned and enhanced data saved as cleaned_data_enhanced.csv")
