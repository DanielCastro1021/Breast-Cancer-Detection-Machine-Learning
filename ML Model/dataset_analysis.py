# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns


# Read Dataset file
df = pd.read_csv('dataset.csv')

# Config Seaborn Graphs
sns.set_theme(style="ticks", color_codes=True)

# Check For Null Values and which atribuite has Null values.
print("\n")
print(df.isnull().any())
print("\n")

# Convert Missing values To -1
df['evaluation_bi_rads'] = np.where(
    df.evaluation_bi_rads == '?', -1, df.evaluation_bi_rads)
df['mass_shape'] = np.where(df.mass_shape == '?', -1, df.mass_shape)
df['mass_margin'] = np.where(df.mass_margin == '?', -1, df.mass_margin)
df['mass_density'] = np.where(df.mass_density == '?', -1, df.mass_density)
df['age'] = np.where(df.age == '?', -1, df.age)


# Convert Columns from object type to int type
print(df.dtypes)
print("\n")

for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name] = df[col_name].astype('int')

print(df.dtypes)
print("\n")


# Replace Nan With Mode for Categorical Variables and Mean for Age

# evaluation_bi_rads Range is 1 to 5
df.loc[df['evaluation_bi_rads'] < 1,
       'evaluation_bi_rads'] = df['evaluation_bi_rads'].mode()

df.loc[df['evaluation_bi_rads'] > 5,
       'evaluation_bi_rads'] = df['evaluation_bi_rads'].mode()

df['evaluation_bi_rads'].fillna(
    df['evaluation_bi_rads'].mode()[0], inplace=True)

# mass_shape Range is 1 to 4
df.loc[df['mass_shape'] < 1,
       'mass_shape'] = df['mass_shape'].mode()

df.loc[df['mass_shape'] > 4,
       'mass_shape'] = df['mass_shape'].mode()

df['mass_shape'].fillna(df['mass_shape'].mode()[0], inplace=True)

# mass_margin Range is 1 to 5
df.loc[df['mass_margin'] < 1,
       'mass_margin'] = df['mass_margin'].mode()

df.loc[df['mass_margin'] > 5,
       'mass_margin'] = df['mass_margin'].mode()

df['mass_margin'].fillna(df['mass_margin'].mode()[0], inplace=True)


# mass_density Range is 1 to 4
df.loc[df['mass_density'] < 1,
       'mass_density'] = df['mass_density'].mode()

df.loc[df['mass_density'] > 4,
       'mass_density'] = df['mass_density'].mode()

df['mass_density'].fillna(df['mass_density'].mode()[0], inplace=True)

# age
df['age'] = np.where(df.age == -1, df['age'].mean(), df.age)


# See Count of Values Submited per Atribuite
print("\n\n\n\nGravity Count Per Value Submited")
print(df['gravity'].value_counts())


print("\n\n\n\nAge Count Per Value Submited")
print(df['age'].value_counts())


print("\n\n\n\nEvaluation Bi Rads Count Per Value Submited")
print(df['evaluation_bi_rads'].value_counts())


print("\n\n\n\nMass Shape Count Per Value Submited")
print(df['mass_shape'] .value_counts())


print("\n\n\n\nMasse Margin Count Per Value Submited")
print(df['mass_margin'].value_counts())

print("\n\n\n\nMasse Desnsity Count Per Value Submited")
print(df['mass_density'].value_counts())


# See Data in Plots
for col_name in df.columns:
    if col_name != "age":
        sns.catplot(x=col_name, data=df, kind='count')
    if col_name != "gravity":
        sns.catplot(x=col_name, data=df, kind='box')


# See correlation matrix
corr = df.corr()
print(corr)
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()


independent_variables = ["evaluation_bi_rads", "age",
                         "mass_shape", "mass_margin", "mass_density"]
dependent_variables = ["gravity"]

X = df[independent_variables]
y = df[dependent_variables]

print("Done defining X and y")


thresh = 10  # threshold for VIF

for i in np.arange(0, len(independent_variables)):
    vif = [variance_inflation_factor(X[independent_variables].values, ix)
           for ix in range(X[independent_variables].shape[1])]
    maxloc = vif.index(max(vif))
    if max(vif) > thresh:
        print('vif : ', vif)
        print('dropping ',
              X[independent_variables]. columns[maxloc], ' at index ', maxloc)

        del independent_variables[maxloc]
    else:
        break

X = df[independent_variables]
print('Final variables : ', independent_variables)

# %%
