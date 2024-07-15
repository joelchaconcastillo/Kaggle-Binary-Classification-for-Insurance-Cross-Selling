import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_test = pd.read_csv('input/health-insurance-cross-sell-prediction-data/test.csv')
df = pd.read_csv('input/health-insurance-cross-sell-prediction-data/train.csv')
print("==========head========")
df.head()
print("==========info========")
df.info()
print("==========dtypes========")
df.dtypes()
print("==========shape========")
df.shape()

df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape()

