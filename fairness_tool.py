import random
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon


# This version is based on the Lab 4 random search baseline.
# I kept the same general idea: generate random pairs, run them through the model,
# and count how many are individual discriminatory instances.
# The main changes are making sure the sensitive feature actually changes,
# avoiding duplicate pairs, and adding repeated runs/statistical testing.


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

    if len(candidates) == 0:
        return current_value

    return random.choice(candidates)


def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns, max_attempts=100):
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()

    for _ in range(max_attempts):
        sample_a = X_test.iloc[np.random.choice(len(X_test))].copy()
        sample_b = sample_a.copy()

        changed = False

        # Same idea as the teacher's baseline random sampling code,
        # but here I force the sensitive feature to actually change.
        for col in sensitive_columns:
            if col not in X_test.columns:
                continue

            vals = X_test[col].dropna().unique()
            new_val = choose_different_value(sample_a[col], vals)

            if new_val != sample_a[col]:
                changed = True

            sample_b[col] = new_val

        if not changed:
            continue

        # Perturb non-sensitive features the same way for both samples.
        # This keeps the comparison focused on the sensitive feature.
        for col in non_sensitive_columns:
            if col not in X_test.columns:
                continue

            if col in numeric_cols:
                min_val = X_test[col].min()
                max_val = X_test[col].max()
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

    raise RuntimeError("Could not generate a valid sample pair.")


def predict_class(model, sample, feature_columns, threshold=0.5):
    x = sample[feature_columns].to_numpy(dtype=np.float32).reshape(1, -1)

    pred = model(x, training=False).numpy()

    if pred.ndim == 2 and pred.shape[1] == 1:
        score = float(pred[0][0])
    elif pred.ndim == 1:
        score = float(pred[0])
    else:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    label = 1 if score >= threshold else 0
    return score, label


def is_discriminatory(model, a, b, features):
    _, label_a = predict_class(model, a, features)
    _, label_b = predict_class(model, b, features)

    return label_a != label_b


def calculate_idi_ratio(model, X_test, sensitive_columns, budget=1000, seed=42):
    set_seed(seed)

    features = X_test.columns.tolist()
    non_sensitive = [c for c in features if c not in sensitive_columns]

    tested: Set[Tuple] = set()
    discriminatory: Set[Tuple] = set()

    attempts = 0
    max_attempts = budget * 20

    while len(tested) < budget and attempts < max_attempts:
        attempts += 1

        a, b = generate_sample_pair(X_test, sensitive_columns, non_sensitive)
        key = pair_to_key(a, b)

        if key in tested:
            continue

        tested.add(key)

        if is_discriminatory(model, a, b, features):
            discriminatory.add(key)

    if len(tested) == 0:
        idi_ratio = 0.0
    else:
        idi_ratio = len(discriminatory) / len(tested)

    return {
        "budget": budget,
        "unique_inputs_tested": len(tested),
        "discriminatory_instances": len(discriminatory),
        "idi_ratio": idi_ratio,
        "seed": seed,
    }


def run_wilcoxon_test():
    print("\nWilcoxon Test Results (Baseline vs Proposed):")

    baseline = pd.read_csv("baseline_results.csv")
    proposed = pd.read_csv("fairness_results.csv")

    stats_results = []

    for budget in sorted(baseline["budget"].unique()):
        b = baseline[baseline["budget"] == budget].sort_values("seed")
        p = proposed[proposed["budget"] == budget].sort_values("seed")

        # make sure they align
        if len(b) != len(p):
            raise ValueError("Mismatch in runs between baseline and proposed")

        stat, p_value = wilcoxon(b["idi_ratio"], p["idi_ratio"])

        stats_results.append({
            "budget": budget,
            "wilcoxon_statistic": stat,
            "p_value": p_value,
        })

        print(f"Budget {budget}: p-value = {p_value:.4f}")

    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv("wilcoxon_results.csv", index=False)

    return stats_df


def run_experiments(csv_path, model_path):
    X, y = load_dataset(csv_path)
    _, X_test, _, _ = split_data(X, y)

    model = load_trained_model(model_path)

    budgets = [100, 300, 500, 1000]
    repeats = 30

    results = []

    for budget in budgets:
        for seed in range(repeats):
            print(f"Running budget={budget}, seed={seed}...")

            result = calculate_idi_ratio(
                model=model,
                X_test=X_test,
                sensitive_columns=["age"],
                budget=budget,
                seed=seed,
            )

            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv("fairness_results.csv", index=False)

    summary_df = (
        results_df.groupby("budget")["idi_ratio"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary_df.to_csv("fairness_summary.csv", index=False)

    print("\nSummary:")
    print(summary_df)

    stats_df = run_wilcoxon_test()

    return results_df, summary_df, stats_df


def main():
    run_experiments(
        csv_path="model/processed_kdd.csv",
        model_path="model/model_processed_kdd.h5",
    )


if __name__ == "__main__":
    main()