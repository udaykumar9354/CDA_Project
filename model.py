import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from scipy.stats import chi2_contingency  # type: ignore
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

# Response Variable
df['deposit_output'] = df['deposit'].map({'yes': 1, 'no': 0})

df.info()

# Plots for understanding the data

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


# CHI-Square Tests for Categorical Variables

def chi_square_test(col):
    table = pd.crosstab(df[col], df['deposit'])
    chi2, p, dof, expected = chi2_contingency(table)

    print("\n" + "="*70)
    print(f"CHI-SQUARE TEST OF INDEPENDENCE: {col.upper()} vs DEPOSIT SUBSCRIPTION (YES/NO)")
    print("="*70)

    print(f" Variable Tested: {col}")
    print(f" Table Shape: {table.shape[0]} categories x 2 (deposit yes/no)\n")

    print("NULL HYPOTHESIS (H0):")
    print(f"  - {col} is independent of deposit subscription.")
    print("ALTERNATIVE HYPOTHESIS (H1):")
    print(f"  - {col} and deposit subscription are associated.\n")

    print(f"Chi-Square Statistic : {chi2:.4f}")
    print(f"Degrees of Freedom   : {dof}")
    print(f"P-value              : {p:.4f}\n")

    # INTERPRETATION
    if p < 0.05:
        print("INTERPRETATION:")
        print("   1. p-value < 0.05 -> Reject H0")
        print(f"  2. There is a statistically significant relationship between {col} and deposit (yes/no).")
        print("   3. This categorical variable is useful for prediction for deposit subscription.\n")
    else:
        print("INTERPRETATION:")
        print("   1. p-value ≥ 0.05 -> Fail to reject H0")
        print(f"  2. No evidence of an association between {col} and deposit (yes/no).")
        print("   3. This variable may have low predictive importance for deposit subscription.\n")

    print("- Expected Frequencies Matrix:")
    print(pd.DataFrame(expected, index=table.index, columns=table.columns))
    print("="*70)


# Variables to test
chi_square_test('job')
chi_square_test('marital')
chi_square_test('age')
chi_square_test('education')
chi_square_test('loan')

#multicollinearity
vif=pd.DataFrame()
vif["feature"]=df[num_cols].columns
vif["VIF"]= [variance_inflation_factor(df[num_cols].values, i) for i in range(len(df[num_cols].columns))]
print(vif)