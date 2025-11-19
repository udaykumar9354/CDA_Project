import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from scipy.stats import chi2_contingency  # type: ignore
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore
import statsmodels.formula.api as smf # type: ignore
from sklearn.metrics import roc_curve, auc 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score 

# LOAD & CLEAN DATA
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
df['deposit_output'] = df['deposit'].map({'yes': 1, 'no': 0}).astype(int)

df = df.drop(columns=['deposit'])

print(df.info())

# Plots for understanding the data 

# Call Duration vs Deposit Subscription 
plt.figure(figsize=(8, 5)) 
sns.kdeplot(data=df, x='duration', hue='deposit_output', fill=True, common_norm=False) 
plt.title('Distribution of Call Duration for Deposit vs No Deposit') 
plt.xlabel('Call Duration (seconds)') 
plt.ylabel('Density') 
plt.show() 

# Deposit Rate vs Job Type 
plt.figure(figsize=(10, 4)) 
sns.barplot( data=df, x='job', y='deposit_output', order=df['job'].value_counts().index ) 
plt.xticks(rotation=45) 
plt.title('Deposit Probability by Job Type') 
plt.ylabel('Probability of Subscription') 
plt.gca().invert_yaxis() 
plt.show() 

# Deposit Rate vs Education Level 
plt.figure(figsize=(8, 4)) 
sns.barplot( data=df, x='education', y='deposit_output', order=df['education'].value_counts().index ) 
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

# CHI-SQUARE TEST FUNCTION
def chi_square_test(col):
    table = pd.crosstab(df[col], df['deposit_output'])
    chi2, p, dof, expected = chi2_contingency(table)

    print("\n" + "="*70)
    print(f"CHI-SQUARE TEST OF INDEPENDENCE: {col.upper()} vs DEPOSIT SUBSCRIPTION")
    print("="*70)

    print(f"Variable Tested: {col}")
    print(f"Table Shape   : {table.shape[0]} categories x 2 (deposit yes/no)\n")

    print("NULL HYPOTHESIS (H0): Independent")
    print("ALT. HYPOTHESIS (H1): Associated\n")

    print(f"Chi-Square Statistic : {chi2:.4f}")
    print(f"Degrees of Freedom   : {dof}")
    print(f"P-value              : {p:.4f}\n")

    if p < 0.05:
        print("INTERPRETATION: Significant association (reject H0)\n")
    else:
        print("INTERPRETATION: No significant association (fail to reject H0)\n")

    print("Expected Frequencies:")
    print(pd.DataFrame(expected, index=table.index, columns=table.columns))
    print("="*70)


# Run chi-square tests
chi_square_test('job')
chi_square_test('marital')
chi_square_test('education')
chi_square_test('loan')
chi_square_test('housing')


# MULTICOLLINEARITY CHECK (VIF)
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
vif = pd.DataFrame()
vif["feature"] = num_cols
vif["VIF"] = [variance_inflation_factor(df[num_cols].values, i) for i in range(len(num_cols))]
print("\nVIF Values:\n", vif)


# LOGISTIC MODELS (WITH INTERACTIONS)
def run_logit(formula, name):
    print("\n" + "="*80)
    print(f"MODEL: {name}")
    print("="*80)
    print(f"Formula: {formula}\n")

    model = smf.logit(formula, data=df).fit(disp=False)
    print(model.summary())
    print(f"AIC: {model.aic:.2f}")

    odds = np.exp(model.params)
    print("\nOdds Ratios:")
    print(odds)

    return model


# 7 interaction models
models = {
    "Duration X Contact"         : "deposit_output ~ duration * contact",
    "Duration X Month"           : "deposit_output ~ duration * month",
    "Job X Education"            : "deposit_output ~ job * education",
    "Housing X Loan"             : "deposit_output ~ housing * loan",
    "Contact X Previous Outcome" : "deposit_output ~ contact * poutcome",
    "Age X Balance"              : "deposit_output ~ age * balance",
    "Campaign X Previous"        : "deposit_output ~ campaign * previous"
}

results = {}

for name, formula in models.items():
    results[name] = run_logit(formula, name)


# Full model with all predictors (main effects)
predictors = [c for c in df.columns if c != 'deposit_output']
full_formula = "deposit_output ~ " + " + ".join(predictors)

full_model = run_logit(full_formula, "Main Effects Model")
results["Main Effects Model"] = full_model
print()

#Wald test for significance of predictors in full model
print("WALD TEST FOR SIGNIFICANCE OF PREDICTORS IN FULL MODEL")
print(full_model.wald_test_terms(scalar=False))


# AIC COMPARISON TABLE
aic_table = pd.DataFrame({
    "Model": list(results.keys()),
    "AIC": [results[m].aic if results[m] is not None else np.nan for m in results]
})

print("\n\n==================== AIC COMPARISON TABLE ====================")
print(aic_table.sort_values("AIC"))
print("==============================================================")

#model after removing non significant predictors
formula="deposit_output ~ job+marital+education+housing+loan+contact+month+poutcome+balance+duration+campaign"
final_model = run_logit(formula, "Model after removing non-significant predictors")

#ROC curve
df["probs"]=final_model.predict(df)
fpr, tpr, _ = roc_curve(df['deposit_output'], df['probs'])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='red', label=f'Concordance index, c = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#Cross validation
varCat=['job','marital','education','housing','loan','contact','month','poutcome']
varNum=['balance','duration','campaign']
X=df[varCat + varNum]
y=df['deposit_output']

columns=ColumnTransformer([('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), varCat), ('num', StandardScaler(), varNum)])
pipeline = Pipeline([('preprocess', columns), ('classifier', LogisticRegression(max_iter=3000))])
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='roc_auc')
print(f"Cross-Validated Scores: {cv_scores}")
