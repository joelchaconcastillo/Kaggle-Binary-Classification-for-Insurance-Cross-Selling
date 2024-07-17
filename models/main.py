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
sns.pairplot(df)

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
plt.savefig('distribution_of_region_code_annual_premium.png', format='png') 
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df['Vehicle_Age'], kde=False, bins=10)
plt.title('Distribution of Vehicle_Age')
plt.xlabel('Vehicle_Age')
plt.ylabel('Frequency')
plt.savefig('distribution_of_vehicle_age_annual_premium.png', format='png') 
plt.close()


##############Check for imbalanced data############

response_data = df['Response'].value_counts()
plt.figure(figsize=(6,6))
fig, ax = plt.subplots()
ax.pie(response_data, labels = [0, 1])
ax.set_title('Checking imbalance in training data or response')
plt.savefig('balance_labels.png', format='png') 
plt.close()
###########################################################
#######################Feature Engineering#################
###########################################################

df.info()
print("\n")
df.head()

def veh_a(Vehicle_Damage):
    if Vehicle_Damage == 'Yes':
        return 1
    else:
        return 0

df['Vehicle_Damages'] = df['Vehicle_Damage'].apply(veh_a)
df.drop(['Vehicle_Damage'],axis=1)

df['Vehicle_Age'] = df['Vehicle_Age'].astype('category')
df = pd.get_dummies(df, columns=['Vehicle_Age'])
df.head()

df_test.head()

df_test['Vehicle_Damages'] = df_test['Vehicle_Damage'].apply(veh_a)
df_test.drop(['Vehicle_Damage'],axis=1)

df_test['Vehicle_Age'] = df_test['Vehicle_Age'].astype('category')
df_test = pd.get_dummies(df_test, columns=['Vehicle_Age'])
df_test.head()

df_test['Gender'] = df_test['Gender'].astype('category')
df_test = pd.get_dummies(df_test, columns=['Gender'],drop_first=True)

df['Gender'] = df['Gender'].astype('category')
df = pd.get_dummies(df, columns=['Gender'],drop_first=True)

df = df.drop(['Vehicle_Damage'],axis=1)
df_test = df_test.drop(['Vehicle_Damage'],axis=1)

######################################################################
###############Splitting the data#####################################
######################################################################

X_train = df[['Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Vehicle_Damages', 'Vehicle_Age_1-2 Year','Vehicle_Age_< 1 Year', 'Vehicle_Age_> 2 Years', 'Gender_Male']]
y_train = df['Response']

X_test = df_test[['Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Vehicle_Damages', 'Vehicle_Age_1-2 Year','Vehicle_Age_< 1 Year', 'Vehicle_Age_> 2 Years', 'Gender_Male']]

y_train.head()

df_test.head()

###############################################################
#################Handling Imbalanced data######################
###############################################################

import imblearn
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(y_train.value_counts())
print()
print(y_train_smote.value_counts())


################################################################
##############Scaling Data######################################
################################################################
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)
data_scaled = scaler.fit_transform(df)

################################################################
#############Apply ML model#####################################
################################################################

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, y_train_smote, test_size=0.3, random_state=42)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

def model_prediction(model):
    model.fit(x_train,y_train)
    x_train_pred = model.predict(x_train)
    x_test_pred = model.predict(x_test)
    y_test_prob = model.predict_proba(x_test)[:, 1]

    a = accuracy_score(y_train,x_train_pred)*100
    b = accuracy_score(y_test,x_test_pred)*100
    c = precision_score(y_test,x_test_pred)
    d = recall_score(y_test,x_test_pred)
    e = roc_auc_score(y_test, y_test_prob)
    print(f"Accuracy_Score of {model} model on Training Data is:",a)
    print(f"Accuracy_Score of {model} model on Testing Data is:",b)
    print(f"Precision Score of {model} model is:",c)
    print(f"Recall Score of {model} model is:",d)
    print(f"AUC Score of {model} model is:", e)
    print("\n------------------------------------------------------------------------")
    print(f"Confusion Matrix of {model} model is:")
    cm = confusion_matrix(y_test,x_test_pred)
    plt.figure(figsize=(8,4))
    sns.heatmap(cm,annot=True,fmt="g",cmap="Greens")
    plt.show()

model_prediction(LogisticRegression())


