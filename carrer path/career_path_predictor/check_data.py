import pandas as pd
df = pd.read_csv('data/career_data.csv')
print(f'Dataset size: {len(df)}')
print(f'Unique careers: {df["career_path"].nunique()}')
print('Career distribution:')
print(df['career_path'].value_counts())