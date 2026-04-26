import random
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(csv_path: str, target_column: str = "income"):
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Columns are: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    return X, y


def split_data(X, y, test_size=0.3, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def load_trained_model(model_path: str):
    return keras.models.load_model(model_path)


def row_to_key(row: pd.Series, decimals: int = 6):
    key = []
    for value in row.values:
        if isinstance(value, (float, np.floating)):
            key.append(round(float(value), decimals))
        else:
            key.append(value)
    return tuple(key)


def pair_to_key(a: pd.Series, b: pd.Series):
    return tuple(sorted([row_to_key(a), row_to_key(b)]))


def choose_different_value(current_value, possible_values):
    candidates = [v for v in possible_values if v != current_value]
    return random.choice(candidates) if candidates else current_value


def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns):
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()

    while True:
        sample_a = X_test.iloc[np.random.choice(len(X_test))].copy()
        sample_b = sample_a.copy()

        changed = False
        for col in sensitive_columns:
            vals = X_test[col].dropna().unique()
            new_val = choose_different_value(sample_a[col], vals)
            if new_val != sample_a[col]:
                changed = True
            sample_b[col] = new_val

        if not changed:
            continue

        for col in non_sensitive_columns:
            if col in numeric_cols:
                min_val, max_val = X_test[col].min(), X_test[col].max()
                span = max_val - min_val
                if span == 0:
                    continue

                perturb = np.random.uniform(-0.1 * span, 0.1 * span)
                new_val = np.clip(sample_a[col] + perturb, min_val, max_val)

                if pd.api.types.is_integer_dtype(X_test[col]):
                    new_val = int(round(new_val))

                sample_a[col] = new_val
                sample_b[col] = new_val

        return sample_a, sample_b


def predict_class(model, sample, feature_columns, threshold=0.5):
    x = sample[feature_columns].to_numpy(dtype=np.float32).reshape(1, -1)
    pred = model.predict(x, verbose=0)
    score = float(pred[0][0])
    label = 1 if score >= threshold else 0
    return score, label


def is_discriminatory(model, a, b, features):
    _, la = predict_class(model, a, features)
    _, lb = predict_class(model, b, features)
    return la != lb


def calculate_idi_ratio(model, X_test, sensitive_columns, budget=1000, seed=42):
    set_seed(seed)

    features = X_test.columns.tolist()
    non_sensitive = [c for c in features if c not in sensitive_columns]

    tested, discriminatory = set(), set()

    while len(tested) < budget:
        a, b = generate_sample_pair(X_test, sensitive_columns, non_sensitive)
        key = pair_to_key(a, b)

        if key in tested:
            continue

        tested.add(key)

        if is_discriminatory(model, a, b, features):
            discriminatory.add(key)

    return {
        "budget": budget,
        "idi_ratio": len(discriminatory) / len(tested),
        "seed": seed
    }


def run_wilcoxon_test(results_df):
    print("\nWilcoxon Test Results:")

    for budget in sorted(results_df["budget"].unique()):
        data = results_df[results_df["budget"] == budget]["idi_ratio"].values

        half = len(data) // 2
        first = data[:half]
        second = data[half:half * 2]

        stat, p = wilcoxon(first, second)

        print(f"Budget {budget}: p-value = {p:.4f}")


def run_experiments(csv_path, model_path):
    X, y = load_dataset(csv_path)
    _, X_test, _, _ = split_data(X, y)

    model = load_trained_model(model_path)

    budgets = [100, 300, 500, 1000]
    repeats = 30

    results = []

    for b in budgets:
        for seed in range(repeats):
            res = calculate_idi_ratio(model, X_test, ["age"], b, seed)
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("fairness_results.csv", index=False)

    run_wilcoxon_test(df)


def main():
    run_experiments(
        csv_path="model/processed_kdd.csv",
        model_path="model/model_processed_kdd.h5"
    )


if __name__ == "__main__":
    main()
