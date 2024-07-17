import pandas as pd
import numpy as np
import seaborn as sns
import math
import copy
import pickle
import gc
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
path = './../../input'
tr_f = path+ '/train.csv' 
te_f = path+ '/test.csv'
sb_f = path+ '/sample_submission.csv'
TARGET = 'Response'
FOLDS = 5
SEED = 94
ES_ROUNDS = 200
LOG_STEPS = 500

###Data loading##
print('Loading data...')
train_data = pd.read_csv(tr_f, index_col='id')
test_data = pd.read_csv(te_f, index_col='id')
submission_data = pd.read_csv(sb_f,index_col='id')

train_data['is_train'] = 1
test_data['is_train'] = 0

# Combine train and test data for processing
combined_df = pd.concat([train_data, test_data])
del train_data, test_data
print('Data Load Successfully!')


def transform_categorical_features(df):
    print('==> Transforming categorical features')
    gender_map = {'Male': 0, 'Female': 1}
    vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    vehicle_damage_map = {'No': 0, 'Yes': 1}
    
    df['Gender'] = df['Gender'].map(gender_map)
    df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age_map)
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map(vehicle_damage_map)
    
    return df

def adjust_data_types(df):
    print('==> Adjusting data types')
    df['Region_Code'] = df['Region_Code'].astype(int)
    df['Annual_Premium'] = df['Annual_Premium'].astype(int)
    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype(int)
    
    return df

def create_additional_features(df):
    print('==> Creating additional features')
    ##This concatenate two columns and from it creates an ID for each tuple
    df['Prev_Insured_Annual_Premium'] = pd.factorize(df['Previously_Insured'].astype(str) + df['Annual_Premium'].astype(str))[0]
    df['Prev_Insured_Vehicle_Age'] = pd.factorize(df['Previously_Insured'].astype(str) + df['Vehicle_Age'].astype(str))[0]
    df['Prev_Insured_Vehicle_Damage'] = pd.factorize(df['Previously_Insured'].astype(str) + df['Vehicle_Damage'].astype(str))[0]
    df['Prev_Insured_Vintage'] = pd.factorize(df['Previously_Insured'].astype(str) + df['Vintage'].astype(str))[0]
    
    return df

def optimize_memory_usage(df):
    print('==> Optimizing memory usage')
    start_mem_usage = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type.name in ['category', 'object']:
            raise ValueError(f"Column '{col}' is of type '{col_type.name}'")

        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem_usage = df.memory_usage().sum() / 1024**2
    print(f'------ Memory usage before: {start_mem_usage:.2f} MB')
    print(f'------ Memory usage after: {end_mem_usage:.2f} MB')
    print(f'------ Reduced memory usage by {(100 * (start_mem_usage - end_mem_usage) / start_mem_usage):.1f}%')

    return df




combined_df = transform_categorical_features(combined_df) #TODO: we don't need one-hot encoding
combined_df = adjust_data_types(combined_df)
combined_df = create_additional_features(combined_df)
combined_df = optimize_memory_usage(combined_df)


# Split the data back into train and test sets
train_data = combined_df[combined_df['is_train'] == 1].drop(columns=['is_train'])
test_data = combined_df[combined_df['is_train'] == 0].drop(columns=['is_train'])

# Ensure the target variable is of integer type
train_data[TARGET] = train_data[TARGET].astype(int)

###########Split the data into features and target

# X_train = train_data.drop(TARGET, axis=1)
# y_train = train_data[TARGET]
X_test = test_data
X = train_data.drop(columns=[TARGET])
y = train_data[TARGET]


def train_and_evaluate(model, X, y, X_test, folds=5, random_state=None):
    """
       basic modeling
    """
    print(f'Training {model.__class__.__name__}\n')

    scores = []
    feature_importances = np.zeros(X.shape[1])
    evaluation_history = []

    oof_pred_probs = np.zeros(X.shape[0])
    test_pred_probs = np.zeros(X_test.shape[0])

    skf = StratifiedKFold(n_splits=FOLDS, random_state=94, shuffle=True)

    for fold_index, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model_clone = copy.deepcopy(model)
        model_clone.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=LOG_STEPS)

        feature_importances += model_clone.feature_importances_ / folds
        evaluation_history.append(model_clone.evals_result())

        y_pred_probs = model_clone.predict_proba(X_val)[:, 1]
        oof_pred_probs[val_index] = y_pred_probs

        temp_test_pred_probs = model_clone.predict_proba(X_test)[:, 1]
        test_pred_probs += temp_test_pred_probs / folds

        auc_score = roc_auc_score(y_val, y_pred_probs)
        scores.append(auc_score)

        print(f'\n--- Fold {fold_index + 1} - AUC: {auc_score:.5f}\n\n')

        del model_clone
        gc.collect()

    print(f'------ Average AUC: {np.mean(scores):.5f} ± {np.std(scores):.5f}\n\n')

    return oof_pred_probs, test_pred_probs

X_test = X_test.drop(columns=['Response'])

########################################################################################
##########Model Training and evaluation#################################################
########################################################################################
xgb_params = {
    'colsample_bylevel': 0.21416299403295808,
    'colsample_bynode': 0.6208356380953189,
    'colsample_bytree': 0.19219680400212635,
    'gamma': 0.6051664515971382,
    'max_bin': 682,
    'max_delta_step': 5.264818337431145,
    'max_depth': 68,
    'min_child_weight': 5.23408291542125,
    'n_estimators': 10000,
    'n_jobs': -1,
    'objective': "binary:logistic",
    'random_state': SEED,
    'reg_alpha': 0.46516016901463414,
    'reg_lambda': 0.8410553418334902,
    'subsample': 0.802533192662325,
    'verbosity': 0,
    'eval_metric': "auc",
    'early_stopping_rounds': ES_ROUNDS,
    'tree_method': 'gpu_hist'  # Use GPU acceleration
}

xgb_model = XGBClassifier(**xgb_params)
# Call train_and_evaluate function with XGBClassifier model
oof_pred_probs, test_pred_probs = train_and_evaluate(xgb_model, X, y, X_test, folds=FOLDS, random_state=SEED)


####################Submission#################
submission = pd.DataFrame({
    'id': submission_data.index,
    'Response': test_pred_probs
})
submission.to_csv('submission.csv', index=False)
submission.head()

