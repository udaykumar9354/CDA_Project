import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

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

df.info()

# Call Duration vs Deposit Subscription
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='duration', hue='deposit', fill=True, common_norm=False)
plt.title('Distribution of Call Duration for Deposit vs No Deposit')
plt.xlabel('Call Duration (seconds)')
plt.ylabel('Density')
plt.show()

# Deposit Rate vs Job Type
plt.figure(figsize=(10, 4))
sns.barplot(
    data=df, 
    x='job', 
    y='deposit_output', 
    order=df['job'].value_counts().index
)
plt.xticks(rotation=45)
plt.title('Deposit Probability by Job Type')
plt.ylabel('Probability of Subscription')
plt.gca().invert_yaxis()
plt.show()

# Deposit Rate vs Education Level
plt.figure(figsize=(8, 4))
sns.barplot(
    data=df,
    x='education',
    y='deposit_output',
    order=df['education'].value_counts().index
)
plt.xticks(rotation=45)
plt.gca().invert_yaxis()
plt.title('Deposit Probability by Education')
plt.show()

# Correlation Heatmap
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'deposit_output']
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
