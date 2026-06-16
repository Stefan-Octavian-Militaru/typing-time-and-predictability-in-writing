import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib

def select_features(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    existing_features = [col for col in feature_list if col in df.columns]
    missing_features = [col for col in feature_list if col not in df.columns]
    if len(existing_features) == 1:
        existing_features = existing_features[0]
    if missing_features:
        print(f"Warning: These features were not found: {missing_features}")
    return df[existing_features].copy()

TARGET = "time_taken_seconds"

FEATURES = ['was_deleted', 'backspaces',
            'word_length', 'char_count',
            'is_short', 'is_long', 'vowel_ratio',
            'word_frequency', 'has_period',
            'has_comma', 'is_hyphenated',
            'nr_versions', 'prev_word_length',
            'word_position', 'word_relative_position',
            'keyboard_distance', 'avg_keyboard_distance',
            'sentence_length', 'edit_distance']

TOP_FEATURES = [
    "distance_backspace_interaction",
    "edit_length_ratio",
    "frequency_length_ratio",
    "word_relative_position",
    "edit_distance",
    "nr_versions",
    "keyboard_distance",
    "backspaces",
    "word_frequency"
]

BACKUP_FEATURES = [
    "prev_word_length",
    "has_comma",
    "version_edit_interaction",
    "was_deleted",
    "is_hyphenated",
    "has_period",
    "word_length",
    "word_position"
]

saved_model_name = "typing_time_model_package.pkl"

feature_file = open("word_features.csv", "r")
df = pd.read_csv(feature_file)

working_df = select_features(df, TOP_FEATURES + BACKUP_FEATURES)
targets_df = np.log1p(select_features(df, [TARGET]))
groups_df = select_features(df, ["prompt_id"])


gkf = GroupKFold(n_splits=5)
mae_scores = []
rmse_scores = []

# plt.hist(df["time_taken_seconds"], bins=100)
# plt.show()

model1 = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42
    )
model2 = lgb.LGBMRegressor(
        objective="regression",

        n_estimators=2000,
        learning_rate=0.01,

        num_leaves=31,
        max_depth=-1,

        min_child_samples=10,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=0.1,

        random_state=42,
        n_jobs=-1
    )
model3 = lgb.LGBMRegressor(
    objective="regression",

    # Bigger model
    n_estimators=5000,
    learning_rate=0.02,

    # More expressive trees
    num_leaves=127,
    max_depth=12,

    # Allow finer splits
    min_child_samples=3,
    min_child_weight=1e-3,

    # Less regularization
    reg_alpha=0.0,
    reg_lambda=0.0,

    # Disable aggressive subsampling
    subsample=1.0,
    colsample_bytree=1.0,

    random_state=42,
    n_jobs=-1
)


model4 = HistGradientBoostingRegressor(
    learning_rate=0.03,
    max_iter=5000,
    max_depth=14,
    min_samples_leaf=20,
    l2_regularization=0.1,

    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50,

    random_state=42
)


model = model4
total_error = 0
for fold, (train_idx, test_idx) in enumerate(gkf.split(working_df, targets_df, groups_df)):
    print(f"Currently on fold {fold + 1}")
    X_train = working_df.iloc[train_idx]
    X_test = working_df.iloc[test_idx]
    targets_train = targets_df.iloc[train_idx]
    targets_test = targets_df.iloc[test_idx]



    model.fit(X_train, targets_train)
    log_preds = model.predict(X_test)


    preds = np.expm1(log_preds)
    y_true = np.expm1(targets_test)


    residuals = y_true - preds

    mae = mean_absolute_error(y_true, preds)
    rmse = root_mean_squared_error(y_true, preds)

    train_preds = np.expm1(model.predict(X_train))
    train_true = np.expm1(targets_train)

    train_mae = mean_absolute_error(train_true, train_preds)

    mae_scores.append(mae)
    rmse_scores.append(rmse)

    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test  MAE: {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    for resid in residuals:
        total_error += (resid > 0) * resid + (resid < 0) * -1 * resid

    # if fold == 4:
    #     print(f"Saving model to {saved_model_name}")
    #     model_package = {
    #         "model": model,
    #         "features": list(working_df.columns)
    #     }
    #     joblib.dump(model_package, saved_model_name)
    #plotting residuals
    # plt.scatter(X_test["word_length"], residuals)
    # if fold == 4:
    #     plt.show()
    # plt.scatter(X_test["keyboard_distance"], residuals)
    # if fold == 4:
    #     plt.show()
    # plt.hist(residuals, bins=100)
    # if fold == 4:
    #     plt.show()

print("\n\n\nFINAL RESULTS\n\n\n")

print(f"Mean MAE  : {np.mean(mae_scores):.4f}")
print(f"Std MAE   : {np.std(mae_scores):.4f}")

print(f"Mean RMSE : {np.mean(rmse_scores):.4f}")
print(f"Std RMSE  : {np.std(rmse_scores):.4f}")
print(f"Total Error: {total_error:.4f}")