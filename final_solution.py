## Imports
import os
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import optuna

# Using Only All Applicable Features for Modeling
## Loading in the Training Data
train_df = pd.read_csv('train.csv')

## Formatting Pitch Type Column
one_hot_encoded_df = pd.get_dummies(train_df['pitch_type'], prefix='pitch_type')
train_df = pd.concat([train_df, one_hot_encoded_df], axis=1)
train_df.drop('pitch_type', axis=1, inplace=True)

## Feature Engineering
train_df['opposite_hand'] = train_df['is_lhp'] != train_df['is_lhb']
train_df['on_base'] = train_df['on_1b'] + train_df['on_2b'] + train_df['on_3b']
train_df['late_game'] = train_df['inning'] >= 7
train_df['pressure'] = (train_df['outs_when_up'] == 2) & ((train_df['on_2b'] == 1) | (train_df['on_3b'] == 1))
train_df['clutch'] = train_df['pressure'] & (train_df['late_game'] == 1)
train_df['fatigue'] = train_df['pitch_number'] / 110
train_df['momentum_x'] = train_df['release_speed'] * train_df['release_pos_x']
train_df['momentum_z'] = train_df['release_speed'] * train_df['release_pos_z']
train_df['total_acceleration'] = np.sqrt(train_df['ax']**2 + train_df['ay']**2 + train_df['az']**2)
train_df['pitch_break_horizontal'] = train_df['vx0'] - train_df['pfx_x']
train_df['pitch_break_vertical'] = train_df['vz0'] - train_df['pfx_z']

## Rolling statistics
train_df = train_df.sort_values(by=['uid', 'pitch_number'])
train_df['avg_speed_last_5'] = train_df.groupby('uid')['release_speed'].rolling(window=5, min_periods=1).mean().reset_index(drop=True)
train_df['avg_speed_last_10'] = train_df.groupby('uid')['release_speed'].rolling(window=10, min_periods=1).mean().reset_index(drop=True)
train_df['avg_spin_last_5'] = train_df.groupby('uid')['release_spin_rate'].rolling(window=5, min_periods=1).mean().reset_index(drop=True)
train_df['avg_spin_last_10'] = train_df.groupby('uid')['release_spin_rate'].rolling(window=10, min_periods=1).mean().reset_index(drop=True)

## Set Features and Target
feats = ['release_speed', 'release_pos_x', 'release_pos_z', 'is_lhp', 'is_lhb', 'balls', 'strikes', 
         'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 
         'is_top', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'effective_speed', 
         'release_spin_rate', 'release_extension', 'release_pos_y', 'pitch_number', 'spin_axis', 'spray_angle', 
         'bat_speed', 'swing_length', 'pitch_type_CH', 'pitch_type_CU', 'pitch_type_EP', 'pitch_type_FA',
         'pitch_type_FC', 'pitch_type_FF', 'pitch_type_FO', 'pitch_type_FS', 'pitch_type_KC', 'pitch_type_KN', 
         'pitch_type_SC', 'pitch_type_SI', 'pitch_type_SL', 'pitch_type_ST', 'pitch_type_SV', 'opposite_hand', 
         'on_base', 'late_game', 'pressure', 'clutch', 'fatigue', 'momentum_x', 'momentum_z', 
         'total_acceleration', 'pitch_break_horizontal', 'pitch_break_vertical', 'avg_speed_last_5', 
         'avg_speed_last_10', 'avg_spin_last_5', 'avg_spin_last_10']
target = 'outcome_code'

## Model Training with K-Fold Cross Validation
folds = 3
kf = KFold(n_splits=folds, shuffle=True)
outputs = pd.DataFrame()

for train_idx, test_idx in kf.split(train_df):
    train = train_df.iloc[train_idx]
    test = train_df.iloc[test_idx]
    model = cb.CatBoostClassifier(iterations=1000, verbose=False, loss_function='MultiClassOneVsAll', eval_metric='AUC')
    model.fit(train[feats], train[target])
    _df = pd.DataFrame(model.predict_proba(test[feats]), index=test.index)
    outputs = pd.concat([outputs, _df])

# Feature Importance Extraction
feature_importances = model.get_feature_importance()
feature_names = model.feature_names_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

print('---------------------------------------------------')
print(importance_df.head(10))

# Selecting Important Features
threshold = 2.20
selected_features = importance_df[importance_df['Importance'] > threshold]['Feature'].tolist()
print(f"Selected Features: {selected_features}")

folds = 3
kf = KFold(folds, shuffle=True)
outputs = pd.DataFrame()
for train_idx, test_idx in kf.split(train_df):
    train = train_df.iloc[train_idx]
    test = train_df.iloc[test_idx]
    model = cb.CatBoostClassifier(iterations=1000, verbose=False, loss_function='MultiClassOneVsAll', eval_metric='AUC')
    model.fit(train.loc[:, selected_features], train[target])
    _df = pd.DataFrame(model.predict_proba(test.loc[:, selected_features]), index=test.index)
    outputs = pd.concat([outputs, _df])
## Post-Model Processing
train_df = pd.concat([train_df, outputs], axis=1)
train_df['e'] = train_df.loc[:, range(5)].sum(axis=1).sub(1)
train_df[0] = train_df[0].sub(train_df['e'])
## Output Summary
train_df = train_df.rename(columns={0:'out',1:'single',2:'double',3:'triple',4:'home_run'})
print(train_df.loc[:, ['uid','out','single','double','triple','home_run']].sample(10))
print(roc_auc_score(train_df[target], train_df.loc[:, ['out','single','double','triple','home_run']], multi_class='ovr'))