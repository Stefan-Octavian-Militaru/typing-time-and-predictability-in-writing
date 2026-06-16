import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
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
            'sentence_length', 'edit_distance',
            'distance_backspace_interaction','edit_length_ratio',
            'frequency_length_ratio','position_sentence_interaction','version_edit_interaction']

feature_file = open("word_features.csv", "r")
df = pd.read_csv(feature_file)

working_df = select_features(df, FEATURES)
targets_df = select_features(df, [TARGET])
groups_df = select_features(df, ["prompt_id"])

gkf = GroupKFold(n_splits=5)

mae_scores = []
all_importances = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(working_df, targets_df, groups_df)):
    X_train = working_df.iloc[train_idx]
    X_test = working_df.iloc[test_idx]
    targets_train = targets_df.iloc[train_idx]
    targets_test = targets_df.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, targets_train)

    mae = mean_absolute_error(targets_test, model.predict(X_test))
    mae_scores.append(mae)

    result = permutation_importance(
        model,
        X_test,
        targets_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    fold_importance = pd.DataFrame({
        "feature": working_df.columns,
        "importance": result.importances_mean
    })
    all_importances.append(fold_importance)
    print(f"For fold: {fold}, MAE: {mae:.4f}")

print(f"Mean MAE: {np.mean(mae_scores):.4f}")
print(f"Std MAE: {np.std(mae_scores):.4f}")

importance_df = pd.concat(all_importances)
importance_df = importance_df.groupby("feature")["importance"].mean()
importance_df = importance_df.sort_values(ascending=False)

print("\n\nFEATURE IMPORTANCE LIST")
print(importance_df)

print("\n\nTOP 10 FEATURES")
top_features = importance_df.head(10).index.tolist()
print(top_features)

importance_df = importance_df.sort_values(ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df.index, importance_df.values, color='grey')

plt.xlabel("Importanța")
plt.tight_layout()

plt.show()