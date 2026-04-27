import pandas as pd
import numpy as np
import random
from tensorflow import keras
from sklearn.model_selection import train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["income"])
    y = df["income"]
    return train_test_split(X, y, test_size=0.3, random_state=42)


def load_model(model_path):
    return keras.models.load_model(model_path)


def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns):
    sample_a = X_test.iloc[np.random.choice(len(X_test))].copy()
    sample_b = sample_a.copy()

    # Baseline: randomly choose a sensitive value, not guaranteed to be different
    for col in sensitive_columns:
        if col in X_test.columns:
            unique_values = X_test[col].dropna().unique()
            sample_b[col] = np.random.choice(unique_values)

    # Baseline: perturb non-sensitive features the same way for both samples
    for col in non_sensitive_columns:
        if col in X_test.columns:
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            span = max_val - min_val

            if span == 0:
                continue

            perturb = np.random.uniform(-0.1 * span, 0.1 * span)

            new_a = np.clip(sample_a[col] + perturb, min_val, max_val)
            new_b = np.clip(sample_b[col] + perturb, min_val, max_val)

            # Fix for integer columns
            if pd.api.types.is_integer_dtype(X_test[col]):
                new_a = int(round(new_a))
                new_b = int(round(new_b))

            sample_a[col] = new_a
            sample_b[col] = new_b

    return sample_a, sample_b


def is_discriminatory(model, a, b, threshold=0.05):
    a_array = a.to_numpy(dtype=np.float32).reshape(1, -1)
    b_array = b.to_numpy(dtype=np.float32).reshape(1, -1)

    pred_a = model(a_array, training=False).numpy()[0][0]
    pred_b = model(b_array, training=False).numpy()[0][0]

    return abs(pred_a - pred_b) > threshold


def calculate_idi_ratio(model, X_test, sensitive_columns, budget, seed):
    set_seed(seed)

    non_sensitive = [c for c in X_test.columns if c not in sensitive_columns]

    discriminatory = 0

    for _ in range(budget):
        a, b = generate_sample_pair(X_test, sensitive_columns, non_sensitive)

        if is_discriminatory(model, a, b):
            discriminatory += 1

    return discriminatory / budget


def run_baseline(csv_path, model_path):
    _, X_test, _, _ = load_data(csv_path)
    model = load_model(model_path)

    budgets = [100, 300, 500, 1000]
    repeats = 30

    results = []

    for budget in budgets:
        for seed in range(repeats):
            print(f"[BASELINE] budget={budget}, seed={seed}")

            idi = calculate_idi_ratio(
                model=model,
                X_test=X_test,
                sensitive_columns=["age"],
                budget=budget,
                seed=seed,
            )

            results.append({
                "budget": budget,
                "idi_ratio": idi,
                "seed": seed
            })

    df = pd.DataFrame(results)
    df.to_csv("baseline_results.csv", index=False)

    summary = (
        df.groupby("budget")["idi_ratio"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary.to_csv("baseline_summary.csv", index=False)

    print("\n=== BASELINE SUMMARY ===")
    print(summary)

    return summary


if __name__ == "__main__":
    run_baseline(
        csv_path="model/processed_kdd.csv",
        model_path="model/model_processed_kdd.h5",
    )
