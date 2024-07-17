import numpy as np
import pandas as pd

df_train = pd.read_csv('/kaggle/input/lostinpircarhelsep/processed_train_data.csv')
df_test  = pd.read_csv('/kaggle/input/lostinpircarhelsep/processed_test_data.csv')

df_train.head()

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

categorical_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
binary_cols = ['Driving_License', 'Previously_Insured']
numeric_cols = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

preprocessor = ColumnTransformer(
        transformers = [
            ('cat', OneHotEncoder(), categorical_cols),
            ('num', MinMaxScaler(), numeric_cols)
            ])

df_train_processed = preprocessor.fit_transform(df_train)
imputer = SimpleImputer(strategy='mean')
df_train_processed = imputer.fit_transform(df_train_processed)
processed_cols = (preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist() + numeric_cols)
df_train_processed = pd.DataFrame(df_train_processed, columns=processed_cols)
df_train_processed['Response'] = df_train['Response']


df_test_processed = preprocessor.transform(df_test)
imputer = SimpleImputer(strategy='mean')
df_test_processed = imputer.fit_transform(df_test_processed)
processed_cols = (preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist() + numeric_cols)
df_test_processed = pd.DataFrame(df_test_processed, columns=processed_cols)

df_train_processed.head()

df_train_processed.shape

df_test_processed.head()

df_train.columns

df_train_processed.columns

df_train_processed['Driving_License'] = df_train['Driving_License']
df_test_processed['Driving_License'] = df_test['Driving_License']
df_train_processed['Previously_Insured'] = df_train['Previously_Insured']
df_test_processed['Previously_Insured'] = df_test['Previously_Insured']


df_train = df_train_processed
df_test =df_test_processed

cont_features = [
    'Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Damage']
cat_features = [ 
    'Vehicle_Age', 'Policy_Sales_Channel', 
    'Region_Code', 'Age', 'Vintage', 'Annual_Premium']

train_data = df_train
test_data  = df_test
features = cont_features + cat_features

df_train.head()

df_train['Response'].value_counts()

df_train.shape

df_test.head()

df_train.shape

import random
n= 9669866
df_train = df_train.sample(n=df_test.shape[0], random_state=42)

df_train.head()

df_test.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.activations import relu,elu

import tensorflow as tf
from tensorflow.keras.layers import Layer

categorical_cols = ['Gender_0', 'Gender_1', 'Vehicle_Age_0', 'Vehicle_Age_1', 'Vehicle_Age_2', 'Vehicle_Damage_0', 'Vehicle_Damage_1']
numeric_cols = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

scaler = MinMaxScaler()
df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
df_test[numeric_cols] = scaler.transform(df_test[numeric_cols])

X = df_train.drop('Response', axis=1)
y = df_train['Response']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

import keras

model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1]))
model.add(Dropout(0.1)),
model.add(Dense(128, activation='relu')),
model.add(Dense(64, activation='relu')),
model.add(Dropout(0.1)),
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])

from tensorflow.keras.callbacks import LearningRateScheduler
#lr_scheduler = LearningRateScheduler(scheduler)

model.fit(X_train, y_train, epochs=100, batch_size=1024, validation_data=(X_val, y_val))

y_pred = model.predict(df_test)

y_pred

df_sub = pd.read_csv('/kaggle/input/playground-series-s4e7/sample_submission.csv')

df_sub['Response'] = y_pred

df_sub.to_csv('submission.csv', index=False)

df_sub

df_sub['Response'].hist()


