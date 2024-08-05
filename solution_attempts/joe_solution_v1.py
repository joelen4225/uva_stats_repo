# This Attempt Will Focus on Optimizing the Features While Still Using CatBoost
# The First Section, Using All Applicable Features, Produced a Score of 0.7239
# The Second Section, Using Features of Importance 1.5 or Higher, Produced a Score of 0.7282
# The Third Section, Using Significant Features Based on RFE, Produced a Score of 0.7274

## Imports
import os
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV

# Using Only All Applicable Features for Modeling
## Loading in the Training Data
train_df = pd.read_csv('train.csv')
## Formatting Pitch Type Column
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



# Using Only Significant Features for Modeling (CatBoost Method)
## Finding and Pointing Out Important Values
feature_importances = model.get_feature_importance()
feature_names = model.feature_names_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print('---------------------------------------------------')
print(importance_df)
## Setting a Threshold of 1.5 for Importance
threshold = 1.5
selected_features = [feature for feature, importance in zip(importance_df['Feature'], importance_df['Importance']) if importance > threshold]
print(f"Selected Features: {selected_features}")
## Reformatting the Data
train_df = pd.read_csv('train.csv')
## Reformatting Pitch Type Column
one_hot_encoded_df = pd.get_dummies(train_df['pitch_type'], prefix='pitch_type')
train_df = pd.concat([train_df, one_hot_encoded_df], axis=1)
train_df = train_df.drop('pitch_type', axis=1)
## Re-Running the Model
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



# Using Only Significant Features for Modeling (Recursive Feature Elimination)
## Initializing the RFE 
model = cb.CatBoostClassifier(iterations=1000, verbose=False, loss_function='MultiClassOneVsAll', eval_metric='AUC')
rfe = RFE(model, n_features_to_select=20)
rfe = rfe.fit(train_df.loc[:, feats], train_df[target])
## Finding and Pointing Out the T20 Significant Values
selected_features = np.array(feats)[rfe.support_]
print('---------------------------------------------------')
print(f"Selected Features: {selected_features}")
## Reformatting the Data
train_df = pd.read_csv('train.csv')
## Reformatting Pitch Type Column
one_hot_encoded_df = pd.get_dummies(train_df['pitch_type'], prefix='pitch_type')
train_df = pd.concat([train_df, one_hot_encoded_df], axis=1)
train_df = train_df.drop('pitch_type', axis=1)
## Re-Running the Model
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