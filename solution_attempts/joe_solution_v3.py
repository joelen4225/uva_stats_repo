# This Attempt Will Focus on Using Random Forest Modeling
# The First Section, Using All Applicable Features, Produced a Score of 0.6386
# The Second Section, Using Features of Importance 0.03 or Higher, Produced a Score of 0.6358
# The Third Section, Using Significant Features Based on RFE, Produced a Score of 0.6409

## Imports
import numpy as np
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from joblib import parallel_backend
from data_prep import *

# Defining Testing and Training Data
train_df = model_final_df[model_final_df['season'] < 2023]
test_df = model_final_df[model_final_df['season'] >= 2023]

# Defining X & Y Variables
X_train = train_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_train = train_df['team_score']
X_test = test_df.drop(columns=['season', 'week', 'team_abbr', 'opponent', 'team_score'])
y_test = test_df['team_score']

# Ensure No Missing Values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Use a smaller subset for feature selection
subset_size = int(X_train_scaled.shape[0] * 0.1)  # 10% of the data
X_train_subset = X_train_scaled[:subset_size]
y_train_subset = y_train[:subset_size]

# Recursive Feature Elimination with Cross-Validation
base_estimator = LinearRegression()  # Use simpler estimator
rfecv = RFECV(estimator=base_estimator, step=1, cv=2, scoring='neg_mean_absolute_error')
rfecv.fit(X_train_subset, y_train_subset)

# Select the best features
X_train_selected = rfecv.transform(X_train_scaled)
X_test_selected = rfecv.transform(X_test_scaled)

# Define Models
lgbm = LGBMRegressor(verbose=-1)  # Suppress LightGBM warnings
gb = GradientBoostingRegressor()
lr = LinearRegression()

# Creating the Voting Regressor
voting_regressor = VotingRegressor([('lgbm', lgbm), ('rf', base_estimator), ('gb', gb), ('lr', lr)])

# Using a Pipeline to Account for Missing Values and Scaling
pipeline = Pipeline([
    ('voting_regressor', voting_regressor)
])

# Define Parameter Grid
param_grid = {
    'voting_regressor__lgbm__n_estimators': randint(50, 150),
    'voting_regressor__lgbm__learning_rate': uniform(0.01, 0.1),
    'voting_regressor__lgbm__num_leaves': randint(20, 40),
    'voting_regressor__lgbm__max_depth': randint(5, 10),
    'voting_regressor__lgbm__min_child_samples': randint(5, 20),
    'voting_regressor__lgbm__min_split_gain': uniform(0.0, 0.05),
    'voting_regressor__rf__n_estimators': randint(50, 150),
    'voting_regressor__rf__max_features': ['sqrt', 'log2', None],
    'voting_regressor__rf__min_samples_split': randint(2, 10),
    'voting_regressor__rf__min_samples_leaf': randint(1, 5),
    'voting_regressor__rf__max_depth': randint(5, 10),
    'voting_regressor__gb__n_estimators': randint(50, 150),
    'voting_regressor__gb__learning_rate': uniform(0.01, 0.1),
    'voting_regressor__gb__max_depth': randint(3, 10),
    'voting_regressor__gb__min_samples_split': randint(2, 20),
    'voting_regressor__gb__min_samples_leaf': randint(1, 5)
}

# Use RandomizedSearchCV with the Voting Regressor
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=30, cv=3, n_jobs=-1, verbose=1, random_state=42)

# Use joblib to parallelize computations
with parallel_backend('threading', n_jobs=-1):
    random_search.fit(X_train_selected, y_train)

# Predictions
y_pred_test = random_search.best_estimator_.predict(X_test_selected)

# Evaluating the Model
errors = abs(y_pred_test - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Best Parameters:', random_search.best_params_)
