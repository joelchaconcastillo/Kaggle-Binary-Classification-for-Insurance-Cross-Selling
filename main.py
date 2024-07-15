import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_test = pd.read_csv('input/test.csv')
df = pd.read_csv('input/train.csv')
print("==========head========")
df.head()
print("==========info========")
df.info()
print("==========dtypes========")
df.dtypes
print("==========shape========")
df.shape

print("Checking duplicated entries")
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape

missing_values = df.isnull().sum()
missing_values


###################################################
##########Exploratory Data Analysis (EDA)##########
###################################################


sns.set(style="whitegrid")
sns.pair(df)

plt.figure(figsize=(7,4))
sns.boxplot(x=df['Annual_Premium'])
plt.savefig('boxplot_annual_premium.png', format='png') 
# Optional: Close the figure
plt.close()


plt.title('Distribution of Annual_Premium')
plt.xlabel('Annual_Premium')
plt.ylabel('Frequency')
#plt.show()
plt.savefig('distribution_annual_premium.png', format='png') 
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=False, bins=10)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
#plt.show()
plt.savefig('distribution_age_annual_premium.png', format='png') 
plt.close()


plt.figure(figsize=(10, 6))
sns.histplot(df['Region_Code'], kde=False, bins=12)
plt.title('Distribution of Region_Code')
plt.xlabel('Region_Code')
plt.ylabel('Frequency')
#plt.show()
plt.savefig('distribution_of_region_ode_annual_premium.png', format='png') 
plt.close()


