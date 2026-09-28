import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from xgboost import XGBClassifier

DATA_PATH = (
    r"..\University Student Stress Dataset\University Student Stress Dataset"
    r"\Cleaned_Data\university_student_stress_dataset.csv"
)
RESULT_PATH = r"..\results\dataset3_model_comparison.csv"

SELECTED_FEATURES = [
    "Age",
    "Study_Hours",
    "Class_Attendance",
    "Exam_Frequency",
    "Assignment_Load",
    "Sleep_Hours",
    "Physical_Exercise",
    "Social_Media_Use",
    "Screen_Time",
    "Family_Income_Level",
    "Peer_Pressure",
    "Family_Support",
    "Anxiety_Level",
    "University_Type",
]


def build_pipeline(model, X):
    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            (
                "num",
                StandardScaler(),
                numerical,
            ),
        ]
    )

    return Pipeline([
        ("pre", preprocessor),
        ("model", model),
    ])


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[SELECTED_FEATURES].copy()

    # XGBoost requires class labels encoded as consecutive integers.
    # This mapping is deterministic and uses only the target labels.
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    y = df["Stress_Level"].map(label_map)

    if y.isna().any():
        raise ValueError(
            "Unexpected Stress_Level values: "
            + str(df.loc[y.isna(), "Stress_Level"].unique())
        )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        ),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            average="weighted",
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            average="weighted",
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            average="weighted",
            zero_division=0,
        ),
    }

    rows = []

    for name, model in models.items():
        print(f"Running: {name}")
        pipe = build_pipeline(model, X)

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        rows.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean(),
        })

    result = pd.DataFrame(rows).sort_values(
        "accuracy_mean",
        ascending=False,
    )

    print("\n===== DATASET 3: MODEL COMPARISON =====")
    print("Features used:", len(SELECTED_FEATURES))
    print("5-fold stratified CV\n")
    print(result.round(4).to_string(index=False))

    output_dir = Path(RESULT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULT_PATH, index=False)

    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
