# I'll First Alter the Features and Try to Optimize the ROC AUC Score Through This
# This Produced a ROC AUC Score of 0.7216
## Imports
import os
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
## Loading in the Training Data
train_df = pd.read_csv('train.csv')
## Reformatting Pitch Type Column
one_hot_encoded_df = pd.get_dummies(train_df['pitch_type'], prefix='pitch_type')
train_df = pd.concat([train_df, one_hot_encoded_df], axis=1)
train_df = train_df.drop('pitch_type', axis=1)
## Setting the Features
print(train_df.columns)
feats = ['release_speed', 'release_pos_x', 'release_pos_z',
       'is_lhp', 'is_lhb', 'balls', 'strikes', 'pfx_x', 'pfx_z', 'plate_x', 
       'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'is_top', 
       'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'effective_speed', 
       'release_spin_rate', 'release_extension', 'release_pos_y', 'pitch_number',
       'spin_axis', 'spray_angle', 'bat_speed', 'swing_length', 
       'pitch_type_CH', 'pitch_type_CU', 'pitch_type_EP', 'pitch_type_FA',
       'pitch_type_FC', 'pitch_type_FF', 'pitch_type_FO', 'pitch_type_FS',
       'pitch_type_KC', 'pitch_type_KN', 'pitch_type_SC', 'pitch_type_SI',
       'pitch_type_SL', 'pitch_type_ST', 'pitch_type_SV'
    ]
target = 'outcome_code'
## Mimicking the Model
folds = 3
kf = KFold(folds, shuffle=True)
outputs = pd.DataFrame()
for train_idx, test_idx in kf.split(train_df):
    train = train_df.iloc[train_idx]
    test = train_df.iloc[test_idx]
    model = cb.CatBoostClassifier(iterations=1000, verbose=False, loss_function='MultiClassOneVsAll', eval_metric='AUC')
    model.fit(train.loc[:, feats], train[target])
    _df = pd.DataFrame(model.predict_proba(test.loc[:, feats]), index=test.index)
    outputs = pd.concat([outputs, _df])
## Post-Model Processing
train_df = pd.concat([train_df, outputs], axis=1)
train_df['e'] = train_df.loc[:, range(5)].sum(axis=1).sub(1)
train_df[0] = train_df[0].sub(train_df['e'])
## Output Summary
train_df = train_df.rename(columns={0:'out',1:'single',2:'double',3:'triple',4:'home_run'})
print(train_df.loc[:, ['uid','out','single','double','triple','home_run']].sample(10))
print(roc_auc_score(train_df[target], train_df.loc[:, ['out','single','double','triple','home_run']], multi_class='ovr'))