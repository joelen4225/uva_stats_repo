# This Attempt Will Focus on Using Neural Networks
# The First Section, Using All Applicable Features, Produced a Score of 0.6592

## Imports
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Creating a Function to Build the Neural Network
def build_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Adjust this number
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['AUC'])
    return model

# Using Only All Applicable Features for Modeling
## Loading in the Training Data
train_df = pd.read_csv('train.csv')
## Formatting Pitch Type Column
one_hot_encoded_df = pd.get_dummies(train_df['pitch_type'], prefix='pitch_type')
train_df = pd.concat([train_df, one_hot_encoded_df], axis=1)
train_df = train_df.drop('pitch_type', axis=1)
## Setting the Features
feats = ['release_speed', 'release_pos_x', 'release_pos_z', 'is_lhp', 'is_lhb', 'balls', 'strikes', 'pfx_x', 'pfx_z',
         'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'is_top', 'vx0', 'vy0', 'vz0', 'ax',
         'ay', 'az', 'sz_top', 'sz_bot', 'effective_speed', 'release_spin_rate', 'release_extension', 'release_pos_y',
         'pitch_number', 'spin_axis', 'spray_angle', 'bat_speed', 'swing_length', 'pitch_type_CH', 'pitch_type_CU',
         'pitch_type_EP', 'pitch_type_FA', 'pitch_type_FC', 'pitch_type_FF', 'pitch_type_FO', 'pitch_type_FS',
         'pitch_type_KC', 'pitch_type_KN', 'pitch_type_SC', 'pitch_type_SI', 'pitch_type_SL', 'pitch_type_ST',
         'pitch_type_SV']
target = 'outcome_code'
## Standardizing the Features
scaler = StandardScaler()
train_df[feats] = scaler.fit_transform(train_df[feats])
## Mimicking the Model
num_classes = train_df[target].nunique()
folds = 3
kf = KFold(folds, shuffle=True)
outputs = pd.DataFrame()
for train_idx, test_idx in kf.split(train_df):
    train = train_df.iloc[train_idx]
    test = train_df.iloc[test_idx]
    # One-Hot Encoding the Target within the Loop
    train_target = to_categorical(train[target], num_classes=num_classes)
    test_target = to_categorical(test[target], num_classes=num_classes)
    model = build_model(input_dim=len(feats), num_classes=num_classes)
    model.fit(train.loc[:, feats], train_target, epochs=10, batch_size=32, verbose=0)
    _df = pd.DataFrame(model.predict(test.loc[:, feats]), index=test.index)
    outputs = pd.concat([outputs, _df])
## Post-Model Processing
train_df = pd.concat([train_df, outputs], axis=1)
train_df['e'] = train_df.loc[:, range(5)].sum(axis=1).sub(1)
train_df[0] = train_df[0].sub(train_df['e'])
## Output Summary
train_df = train_df.rename(columns={0: 'out', 1: 'single', 2: 'double', 3: 'triple', 4: 'home_run'})
print(train_df.loc[:, ['uid', 'out', 'single', 'double', 'triple', 'home_run']].sample(10))
print(roc_auc_score(train_df[target], train_df.loc[:, ['out', 'single', 'double', 'triple', 'home_run']], multi_class='ovr'))

# Using Only Significant Features for Modeling (Neural Network Method)
## Finding and Pointing Out Important Values
model = build_model(input_dim=len(feats), num_classes=num_classes)
model.fit(train_df.loc[:, feats], to_categorical(train_df[target]), epochs=10, batch_size=32, verbose=0)
importance_df = pd.DataFrame({'Feature': feats, 'Importance': model.get_weights()[0].sum(axis=1)})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print('---------------------------------------------------')
print(importance_df)
## Setting a Threshold of 1.5 for Importance
threshold = -0.5
selected_features = importance_df[importance_df['Importance'] < threshold]['Feature'].tolist()
print(f"Selected Features: {selected_features}")
## Reformatting the Data
train_df = pd.read_csv('train.csv')
## Reformatting Pitch Type Column
one_hot_encoded_df = pd.get_dummies(train_df['pitch_type'], prefix='pitch_type')
train_df = pd.concat([train_df, one_hot_encoded_df], axis=1)
train_df = train_df.drop('pitch_type', axis=1)
## Standardizing the Features
train_df[selected_features] = scaler.fit_transform(train_df[selected_features])
## One-Hot Encoding the Target
train_df[target] = to_categorical(train_df[target])
## Re-Running the Model
num_classes = train_df[target].nunique()
folds = 3
kf = KFold(folds, shuffle=True)
outputs = pd.DataFrame()
for train_idx, test_idx in kf.split(train_df):
    train = train_df.iloc[train_idx]
    test = train_df.iloc[test_idx]
    # One-Hot Encoding the Target within the Loop
    train_target = to_categorical(train[target], num_classes=num_classes)
    test_target = to_categorical(test[target], num_classes=num_classes)
    model = build_model(input_dim=len(selected_features), num_classes=num_classes)
    model.fit(train.loc[:, selected_features], train_target, epochs=10, batch_size=32, verbose=0)
    _df = pd.DataFrame(model.predict(test.loc[:, selected_features]), index=test.index)
    outputs = pd.concat([outputs, _df])
## Post-Model Processing
train_df = pd.concat([train_df, outputs], axis=1)
train_df['e'] = train_df.loc[:, range(5)].sum(axis=1).sub(1)
train_df[0] = train_df[0].sub(train_df['e'])
## Output Summary
train_df = train_df.rename(columns={0: 'out', 1: 'single', 2: 'double', 3: 'triple', 4: 'home_run'})
print(train_df.loc[:, ['uid', 'out', 'single', 'double', 'triple', 'home_run']].sample(10))
print(roc_auc_score(train_df[target], train_df.loc[:, ['out', 'single', 'double', 'triple', 'home_run']], multi_class='ovr'))

