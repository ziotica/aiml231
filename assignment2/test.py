import pandas as pd

# Sample numeric data (e.g., User IDs or Year codes)
df = pd.DataFrame({'year_code': [2022, 2021, 2023]})

# Convert to category
df['year_code'] = df['year_code'].astype('category')

# View the new dtype
print(df['year_code'])  # Output: category


for col in df:
    df[col] = df[col].astype('category').cat.codes.replace(-1, np.nan)