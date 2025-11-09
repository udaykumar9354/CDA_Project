import pandas as pd # type: ignore
import numpy as np # type: ignore

df = pd.read_csv('bank.csv', sep=',')
df.replace('unknown', np.nan, inplace=True)

categorical_cols = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'poutcome', 'deposit'
]

for col in categorical_cols:
    df[col] = df[col].astype('category')

# clean up missing values
df['job'] = df['job'].cat.add_categories("unknown_job").fillna("unknown_job")
df['education'] = df['education'].cat.add_categories("unknown_education").fillna("unknown_education")
df['contact'] = df['contact'].cat.add_categories("unknown_contact").fillna("unknown_contact")
df['poutcome'] = df['poutcome'].cat.add_categories("unknown_poutcome").fillna("unknown_poutcome")


df['deposit_output'] = df['deposit'].map({'yes': 1, 'no': 0})

print(df.info())
df.head()
