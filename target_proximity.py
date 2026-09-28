import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# =========================
# PATHS
# =========================

DATA_PATH = (
    r"..\University Student Stress Dataset\University Student Stress Dataset"
    r"\Cleaned_Data\university_student_stress_dataset.csv"
)

RESULT_PATH = r"..\results\dataset3_target_proximity_sensitivity.csv"


# =========================
# VARIABLES FROM THE
# AVAILABLE SCORE FORMULA
# =========================

KNOWN_SCORE_COMPONENTS = [
    "Exam_Frequency",
    "Assignment_Load",
    "Anxiety_Level",
    "Peer_Pressure",
    "Screen_Time",
    "Sleep_Hours",
    "Family_Support",
]


TARGET = "Stress_Level"
SCORE = "Stress_Score"


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

all_features = [
    c for c in df.columns
    if c not in [TARGET, SCORE]
]

# Remove the variables appearing in the
# available score-generation example
reduced_features = [
    c for c in all_features
    if c not in KNOWN_SCORE_COMPONENTS
]


print("===== TARGET-PROXIMITY SENSITIVITY =====")

print("\nExcluded target:")
print(SCORE)

print("\nExcluded score-component variables:")
print(KNOWN_SCORE_COMPONENTS)

print("\nRemaining features:")
print(reduced_features)


# =========================
# FEATURES + TARGET
# =========================

X = df[reduced_features]

y = df[TARGET].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})


# =========================
# PREPROCESSING FUNCTION
# =========================

def make_pipeline(model):

    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical = [
        c for c in X.columns
        if c not in categorical
    ]

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        (
            "numerical",
            StandardScaler(),
            numerical
        )
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# =========================
# MODELS
# =========================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=5000,
            random_state=42
        ),

    "SVM":
        SVC(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
}


# =========================
# 5-FOLD CV
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {

    "accuracy": "accuracy",

    "precision": make_scorer(
        precision_score,
        average="weighted",
        zero_division=0
    ),

    "recall": make_scorer(
        recall_score,
        average="weighted",
        zero_division=0
    ),

    "f1": make_scorer(
        f1_score,
        average="weighted",
        zero_division=0
    )
}


# =========================
# RUN EXPERIMENT
# =========================

results = []

for name, model in models.items():

    print(f"\nRunning: {name}")

    pipeline = make_pipeline(model)

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise"
    )

    results.append({

        "model": name,

        "accuracy_mean":
            scores["test_accuracy"].mean(),

        "accuracy_std":
            scores["test_accuracy"].std(),

        "precision_mean":
            scores["test_precision"].mean(),

        "recall_mean":
            scores["test_recall"].mean(),

        "f1_mean":
            scores["test_f1"].mean()
    })


# =========================
# RESULTS
# =========================

result = pd.DataFrame(results)

print("\n===== SENSITIVITY RESULTS =====")

print(
    result.round(4).to_string(index=False)
)


# =========================
# SAVE RESULTS
# =========================

result.to_csv(
    RESULT_PATH,
    index=False
)

print("\nSaved:")
print(RESULT_PATH)