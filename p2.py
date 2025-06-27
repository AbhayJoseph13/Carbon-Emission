import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load cleaned dataset from Week 1
data = pd.read_csv("cleaned_data_enhanced.csv")

print("Shape of the dataset:")
print(data.shape)

print("Available columns and their data types:")
print(data.dtypes)

print("Overview of the first 5 rows:")
print(data.head())

print("Descriptive statistics:")
print(data.describe().T)

# Feature engineering
data['en_ttl'] = data['en_per_gdp'] * data['gdp'] / 1000

# Feature selection
features_all = data[['country', 'year', 'cereal_yield', 'fdi_perc_gdp', 'gni_per_cap',
                    'en_per_gdp', 'en_per_cap', 'en_ttl', 'co2_ttl', 'co2_per_cap',
                    'co2_per_gdp', 'pop_urb_aggl_perc', 'prot_area_perc', 'gdp',
                    'pop_growth_perc', 'pop', 'urb_pop_growth_perc']]

# Correlation heatmap
sns.set_theme(font_scale=2)
plt.figure(figsize=(30, 20))
sns.heatmap(features_all.drop(['country'], axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f",
            center=0, vmin=-1, vmax=1)
plt.title('Correlation between features', fontsize=25, weight='bold')
plt.show()

sns.set_theme(font_scale=1)

# VIF calculation
features_for_vif = data[['cereal_yield', 'fdi_perc_gdp', 'gni_per_cap', 'en_per_cap', 'co2_per_cap',
                        'pop_urb_aggl_perc', 'prot_area_perc', 'gdp', 'pop_growth_perc', 'urb_pop_growth_perc']]

vif_data = pd.DataFrame()
vif_data["feature"] = features_for_vif.columns
vif_data["VIF"] = [variance_inflation_factor(features_for_vif.values, i)
                for i in range(features_for_vif.shape[1])]
print("\nVariance Inflation Factors:")
print(vif_data)

# Optional: additional plots and visualizations can be added below as needed
