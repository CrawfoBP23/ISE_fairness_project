import random
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split


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


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    seed: int = 42,
):
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def load_trained_model(model_path: str):
    return keras.models.load_model(model_path)


def row_to_key(row: pd.Series, decimals: int = 6) -> Tuple:
    key = []
    for value in row.values:
        if isinstance(value, (float, np.floating)):
            key.append(round(float(value), decimals))
        else:
            key.append(value)
    return tuple(key)


def pair_to_key(a: pd.Series, b: pd.Series) -> Tuple[Tuple, Tuple]:
    ka = row_to_key(a)
    kb = row_to_key(b)
    return tuple(sorted([ka, kb]))


def choose_different_value(current_value, possible_values):
    candidates = [v for v in possible_values if v != current_value]
    if not candidates:
        return current_value
    return random.choice(candidates)


def generate_sample_pair(
    X_test: pd.DataFrame,
    sensitive_columns: List[str],
    non_sensitive_columns: List[str],
    max_attempts: int = 100,
):
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()

    for _ in range(max_attempts):
        sample_a = X_test.iloc[np.random.choice(len(X_test))].copy()
        sample_b = sample_a.copy()

        changed_sensitive = False

        # Force the sensitive feature(s) to actually change
        for col in sensitive_columns:
            if col not in X_test.columns:
                continue

            unique_values = X_test[col].dropna().unique()
            new_val = choose_different_value(sample_a[col], unique_values)
            if new_val != sample_a[col]:
                changed_sensitive = True
            sample_b[col] = new_val

        if not changed_sensitive:
            continue

        # Apply the same perturbation to both samples on non-sensitive features
        for col in non_sensitive_columns:
            if col not in X_test.columns:
                continue

            if col in numeric_cols:
                min_val = X_test[col].min()
                max_val = X_test[col].max()

                if min_val == max_val:
                    continue

                span = max_val - min_val
                perturbation = np.random.uniform(-0.1 * span, 0.1 * span)

                new_val = sample_a[col] + perturbation
                new_val = np.clip(new_val, min_val, max_val)

                # If original column is integer-like, round back
                if pd.api.types.is_integer_dtype(X_test[col]):
                    new_val = int(round(new_val))

                sample_a[col] = new_val
                sample_b[col] = new_val

        return sample_a, sample_b

    raise RuntimeError("Could not generate a valid pair after multiple attempts.")


def predict_class(
    model,
    sample: pd.Series,
    feature_columns: List[str],
    threshold: float = 0.5,
):
    x = sample[feature_columns].to_numpy(dtype=np.float32).reshape(1, -1)
    pred = model.predict(x, verbose=0)

    if pred.ndim == 2 and pred.shape[1] == 1:
        score = float(pred[0][0])
    elif pred.ndim == 1:
        score = float(pred[0])
    else:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    label = 1 if score >= threshold else 0
    return score, label


def is_discriminatory(
    model,
    sample_a: pd.Series,
    sample_b: pd.Series,
    feature_columns: List[str],
) -> bool:
    _, label_a = predict_class(model, sample_a, feature_columns)
    _, label_b = predict_class(model, sample_b, feature_columns)
    return label_a != label_b


def calculate_idi_ratio(
    model,
    X_test: pd.DataFrame,
    sensitive_columns: List[str],
    budget: int = 1000,
    seed: int = 42,
):
    set_seed(seed)

    feature_columns = X_test.columns.tolist()
    non_sensitive_columns = [c for c in feature_columns if c not in sensitive_columns]

    tested_pairs: Set[Tuple[Tuple, Tuple]] = set()
    discriminatory_pairs: Set[Tuple[Tuple, Tuple]] = set()

    attempts = 0
    max_attempts = budget * 20

    while len(tested_pairs) < budget and attempts < max_attempts:
        attempts += 1

        sample_a, sample_b = generate_sample_pair(
            X_test,
            sensitive_columns=sensitive_columns,
            non_sensitive_columns=non_sensitive_columns,
        )

        key = pair_to_key(sample_a, sample_b)

        if key in tested_pairs:
            continue

        tested_pairs.add(key)

        if is_discriminatory(model, sample_a, sample_b, feature_columns):
            discriminatory_pairs.add(key)

    total_unique_inputs = len(tested_pairs)
    total_discriminatory = len(discriminatory_pairs)

    idi_ratio = (
        total_discriminatory / total_unique_inputs if total_unique_inputs > 0 else 0.0
    )

    return {
        "budget": budget,
        "unique_inputs_tested": total_unique_inputs,
        "discriminatory_instances": total_discriminatory,
        "idi_ratio": idi_ratio,
        "sensitive_columns": ",".join(sensitive_columns),
        "seed": seed,
    }


def run_experiments(
    csv_path: str,
    model_path: str,
    target_column: str = "income",
    sensitive_columns: List[str] = None,
    budgets: List[int] = None,
    repeats: int = 5,
):
    if sensitive_columns is None:
        sensitive_columns = ["age"]
    if budgets is None:
        budgets = [100, 300, 500]

    print("Loading dataset...")
    X, y = load_dataset(csv_path, target_column=target_column)
    _, X_test, _, _ = split_data(X, y)

    print("Loading model...")
    model = load_trained_model(model_path)

    results = []

    for budget in budgets:
        for seed in range(repeats):
            print(f"Running budget={budget}, seed={seed}...")
            result = calculate_idi_ratio(
                model=model,
                X_test=X_test,
                sensitive_columns=sensitive_columns,
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

    print("\nDetailed results:")
    print(results_df)

    print("\nSummary:")
    print(summary_df)


def main():
    csv_path = "model/processed_kdd.csv"
    model_path = "model/model_processed_kdd.h5"
    target_column = "income"

    sensitive_columns = ["age"]

    budgets = [100, 300, 500, 1000]
    repeats = 30

    run_experiments(
        csv_path=csv_path,
        model_path=model_path,
        target_column=target_column,
        sensitive_columns=sensitive_columns,
        budgets=budgets,
        repeats=repeats,
    )


if __name__ == "__main__":
    main()