import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    SCRIPT_DIR.parent
    / "University Student Stress Dataset"
    / "University Student Stress Dataset"
    / "Cleaned_Data"
    / "university_student_stress_dataset.csv"
)

RESULT_DIR = SCRIPT_DIR.parent / "results"

SUMMARY_PATH = RESULT_DIR / "dataset3_class_sensitive_summary.csv"

RESULT_DIR = r"..\results"
SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    "dataset3_class_sensitive_summary.csv"
)


# ============================================================
# SAME 14 FEATURES USED IN PREVIOUS EXPERIMENT
# ============================================================

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


# ============================================================
# PREPROCESSING
# SAME STYLE AS PREVIOUS MODEL COMPARISON
# ============================================================

def build_pipeline(model, X):

    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical = [
        col for col in X.columns
        if col not in categorical
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            (
                "num",
                "passthrough",
                numerical,
            ),
        ]
    )

    return Pipeline(
        [
            ("pre", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 70)
    print("DATASET 3 - CLASS-SENSITIVE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    X = df[SELECTED_FEATURES].copy()
    y = df["Stress_Level"].copy()

    print("\nDataset shape:", df.shape)
    print("Features used:", len(SELECTED_FEATURES))

    print("\nClass distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # MODELS
    # Same important models as previous comparison
    # --------------------------------------------------------

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        "SVM": SVC(),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        ),
    }

    # --------------------------------------------------------
    # 5-FOLD STRATIFIED CV
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    summary_rows = []

    # --------------------------------------------------------
    # MODEL LOOP
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        # Store out-of-fold predictions
        y_true_all = []
        y_pred_all = []

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y), start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            pipeline = build_pipeline(
                model,
                X_train
            )

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            # Save for overall out-of-fold metrics
            y_true_all.extend(y_test)
            y_pred_all.extend(predictions)

            # Fold metrics
            fold_accuracy = accuracy_score(
                y_test,
                predictions
            )

            fold_balanced_accuracy = balanced_accuracy_score(
                y_test,
                predictions
            )

            fold_macro_precision = precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )

            fold_macro_recall = recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )

            fold_macro_f1 = f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )

            fold_results.append([
                fold,
                fold_accuracy,
                fold_balanced_accuracy,
                fold_macro_precision,
                fold_macro_recall,
                fold_macro_f1
            ])

            print(
                f"Fold {fold}: "
                f"Accuracy={fold_accuracy:.4f}, "
                f"Balanced Acc={fold_balanced_accuracy:.4f}, "
                f"Macro F1={fold_macro_f1:.4f}"
            )

        # ----------------------------------------------------
        # OVERALL OUT-OF-FOLD METRICS
        # ----------------------------------------------------

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        overall_accuracy = accuracy_score(
            y_true_all,
            y_pred_all
        )

        balanced_acc = balanced_accuracy_score(
            y_true_all,
            y_pred_all
        )

        macro_precision = precision_score(
            y_true_all,
            y_pred_all,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            y_true_all,
            y_pred_all,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_true_all,
            y_pred_all,
            average="macro",
            zero_division=0
        )

        weighted_precision = precision_score(
            y_true_all,
            y_pred_all,
            average="weighted",
            zero_division=0
        )

        weighted_recall = recall_score(
            y_true_all,
            y_pred_all,
            average="weighted",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_true_all,
            y_pred_all,
            average="weighted",
            zero_division=0
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        labels = ["Low", "Medium", "High"]

        cm = confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=labels
        )

        cm_df = pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )

        cm_path = os.path.join(
            RESULT_DIR,
            f"dataset3_{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
        )

        cm_df.to_csv(cm_path)

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        report = classification_report(
            y_true_all,
            y_pred_all,
            labels=labels,
            zero_division=0
        )

        report_path = os.path.join(
            RESULT_DIR,
            f"dataset3_{model_name.lower().replace(' ', '_')}_classification_report.txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(report)

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print("\nOverall Out-of-Fold Results")
        print("-" * 50)

        print(
            f"Accuracy            : {overall_accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy   : {balanced_acc:.4f}"
        )

        print(
            f"Macro Precision     : {macro_precision:.4f}"
        )

        print(
            f"Macro Recall        : {macro_recall:.4f}"
        )

        print(
            f"Macro F1            : {macro_f1:.4f}"
        )

        print(
            f"Weighted Precision  : {weighted_precision:.4f}"
        )

        print(
            f"Weighted Recall     : {weighted_recall:.4f}"
        )

        print(
            f"Weighted F1         : {weighted_f1:.4f}"
        )

        print("\nConfusion Matrix")
        print(cm_df)

        print("\nClassification Report")
        print(report)

        # ----------------------------------------------------
        # SUMMARY ROW
        # ----------------------------------------------------

        summary_rows.append({
            "Model": model_name,
            "Accuracy": overall_accuracy,
            "Balanced_Accuracy": balanced_acc,
            "Macro_Precision": macro_precision,
            "Macro_Recall": macro_recall,
            "Macro_F1": macro_f1,
            "Weighted_Precision": weighted_precision,
            "Weighted_Recall": weighted_recall,
            "Weighted_F1": weighted_f1
        })

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        "Macro_F1",
        ascending=False
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL CLASS-SENSITIVE COMPARISON")
    print("=" * 70)

    print(
        summary_df.round(4).to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {SUMMARY_PATH}"
    )

    print("\nDataset 3 class-sensitive evaluation completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()