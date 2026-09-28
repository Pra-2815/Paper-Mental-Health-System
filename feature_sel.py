import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = (
    r"..\University Student Stress Dataset\University Student Stress Dataset"
    r"\Cleaned_Data\university_student_stress_dataset.csv"
)

RESULT_PATH = r"..\results\dataset3_feature_selection.csv"


def build_preprocessor(X):
    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    transformers = []

    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            )
        )

    if numerical:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numerical,
            )
        )

    return ColumnTransformer(transformers), categorical, numerical


def aggregate_importance(preprocessor, importances, raw_features):
    names = preprocessor.get_feature_names_out()
    scores = dict.fromkeys(raw_features, 0.0)

    for name, importance in zip(names, importances):
        clean = name.split("__", 1)[1] if "__" in name else name

        for feature in raw_features:
            if clean == feature or clean.startswith(feature + "_"):
                scores[feature] += float(importance)
                break

    return scores


def evaluate_selected(X_train, X_val, y_train, y_val):
    pre, _, _ = build_preprocessor(X_train)

    model = Pipeline([
        ("pre", pre),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(
            y_val, pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_val, pred, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            y_val, pred, average="weighted", zero_division=0
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Stress_Level", "Stress_Score"])
    y = df["Stress_Level"]

    feature_counts = [16, 14, 12, 10, 8, 5]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_results = []
    selected_frequency = {k: {f: 0 for f in X.columns} for k in feature_counts}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Feature ranking is learned ONLY from the training fold.
        pre, _, _ = build_preprocessor(X_train)

        rank_model = Pipeline([
            ("pre", pre),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ])

        rank_model.fit(X_train, y_train)

        fitted_pre = rank_model.named_steps["pre"]
        fitted_rf = rank_model.named_steps["rf"]

        importance = aggregate_importance(
            fitted_pre,
            fitted_rf.feature_importances_,
            X.columns.tolist(),
        )

        ranked_features = sorted(
            importance,
            key=importance.get,
            reverse=True,
        )

        for k in feature_counts:
            selected = ranked_features[:k]

            for feature in selected:
                selected_frequency[k][feature] += 1

            scores = evaluate_selected(
                X_train[selected],
                X_val[selected],
                y_train,
                y_val,
            )

            fold_results.append({
                "fold": fold,
                "features": k,
                **scores,
            })

    results = pd.DataFrame(fold_results)

    summary = (
        results.groupby("features")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
        )
        .reset_index()
    )

    print("\n===== DATASET 3: LEAKAGE-SAFE FEATURE SELECTION =====")
    print("\n5-fold stratified CV")
    print(summary.round(4).to_string(index=False))

    print("\n===== FEATURE SELECTION STABILITY =====")
    for k in feature_counts:
        ranked = sorted(
            selected_frequency[k].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"\nTop features for k={k}:")
        print(", ".join(f"{f} ({n}/5)" for f, n in ranked[:k]))

    output_dir = Path(RESULT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_PATH, index=False)

    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
