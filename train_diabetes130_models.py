# ============================================================
# CASE STUDY 1: HOSPITAL READMISSION PREDICTION
# Logistic Regression With and Without L2 Regularization
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss
)


# ------------------------------------------------------------
# 2. UPLOAD DATASET
# ------------------------------------------------------------

print("Upload diabetic_data.csv")

uploaded = files.upload()

df = pd.read_csv("diabetic_data.csv")

print("\nDataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n========== DATASET INFORMATION ==========")

print("\nFirst 5 rows:")
display(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 4. ORIGINAL TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n========== ORIGINAL READMISSION DISTRIBUTION ==========")

print(df["readmitted"].value_counts())

print("\nPercentage:")
print(
    df["readmitted"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# 5. REMOVE INVALID / EXPIRED ENCOUNTERS
# ------------------------------------------------------------

# Patients who died during hospitalization cannot experience
# a subsequent 30-day readmission from that encounter.

EXPIRED_CODES = [11, 19, 20, 21]

df = df[
    ~df["discharge_disposition_id"].isin(EXPIRED_CODES)
].copy()

# Remove invalid gender records
df = df[
    df["gender"] != "Unknown/Invalid"
].copy()

print("\nRows after cleaning:", len(df))


# ------------------------------------------------------------
# 6. CREATE BINARY 30-DAY READMISSION TARGET
# ------------------------------------------------------------

# <30 days  -> 1
# NO         -> 0
# >30 days   -> 0

df["readmitted_30d"] = (
    df["readmitted"] == "<30"
).astype(int)

print("\n========== BINARY TARGET ==========")

print(
    df["readmitted_30d"].value_counts()
)

print("\nTarget percentage:")
print(
    df["readmitted_30d"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# 7. TARGET DISTRIBUTION PLOT
# ------------------------------------------------------------

plt.figure(figsize=(6, 4))

df["readmitted_30d"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xticks(
    [0, 1],
    ["No 30-Day Readmission", "30-Day Readmission"],
    rotation=0
)

plt.xlabel("Readmission Status")
plt.ylabel("Number of Encounters")
plt.title("30-Day Readmission Distribution")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. AGE FEATURE ENGINEERING
# ------------------------------------------------------------

# Convert age brackets such as [70-80) into midpoint 75.

def age_midpoint(bracket):

    lo, hi = bracket.strip("[)").split("-")

    return (int(lo) + int(hi)) / 2


df["age_numeric"] = df["age"].apply(age_midpoint)


# ------------------------------------------------------------
# 9. CLEAN RACE
# ------------------------------------------------------------

df["race"] = df["race"].replace(
    "?",
    "Unknown"
)


# ------------------------------------------------------------
# 10. GROUP MEDICAL SPECIALTY
# ------------------------------------------------------------

top_specialties = (
    df["medical_specialty"]
    .value_counts()
    .head(9)
    .index
    .tolist()
)

if "?" in top_specialties:
    top_specialties.remove("?")


df["medical_specialty_grouped"] = (
    df["medical_specialty"]
    .apply(
        lambda x:
        x
        if x in top_specialties and x != "?"
        else (
            "Missing"
            if x == "?"
            else "Other"
        )
    )
)


# ------------------------------------------------------------
# 11. GROUP PRIMARY DIAGNOSIS
# ------------------------------------------------------------

def group_diagnosis(code):

    if pd.isna(code) or code == "?":
        return "Missing"

    code = str(code)

    if code.startswith("V") or code.startswith("E"):
        return "Other"

    try:
        val = float(code)
    except ValueError:
        return "Other"

    if 390 <= val <= 459 or val == 785:
        return "Circulatory"

    if 460 <= val <= 519 or val == 786:
        return "Respiratory"

    if 520 <= val <= 579 or val == 787:
        return "Digestive"

    if val == 250 or (250 <= val < 251):
        return "Diabetes"

    if 800 <= val <= 999:
        return "Injury"

    if 710 <= val <= 739:
        return "Musculoskeletal"

    if 580 <= val <= 629 or val == 788:
        return "Genitourinary"

    if 140 <= val <= 239:
        return "Neoplasms"

    return "Other"


df["diag_1_group"] = (
    df["diag_1"]
    .apply(group_diagnosis)
)


# ------------------------------------------------------------
# 12. MEDICATION FEATURES
# ------------------------------------------------------------

med_cols_all = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]


keep_meds = [
    c for c in med_cols_all
    if df[c].value_counts(
        normalize=True
    ).iloc[0] < 0.999
]

print("\nMedication features retained:")
print(keep_meds)


# ------------------------------------------------------------
# 13. DEFINE NUMERICAL FEATURES
# ------------------------------------------------------------

NUMERIC_FEATURES = [
    "age_numeric",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]


# ------------------------------------------------------------
# 14. DEFINE CATEGORICAL FEATURES
# ------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty_grouped",
    "diag_1_group",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed"
] + keep_meds


# ------------------------------------------------------------
# 15. CONVERT HOSPITAL CODES TO CATEGORICAL VARIABLES
# ------------------------------------------------------------

for c in [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]:

    df[c] = df[c].astype(str)


# ------------------------------------------------------------
# 16. HANDLE MISSING LAB RESULTS
# ------------------------------------------------------------

df["max_glu_serum"] = (
    df["max_glu_serum"]
    .fillna("Not tested")
)

df["A1Cresult"] = (
    df["A1Cresult"]
    .fillna("Not tested")
)


# ------------------------------------------------------------
# 17. CREATE X AND y
# ------------------------------------------------------------

FEATURES = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)

TARGET = "readmitted_30d"

X = df[FEATURES]

y = df[TARGET]

print("\n========== MODEL DATA ==========")

print("X shape:", X.shape)

print("y shape:", y.shape)


# ------------------------------------------------------------
# 18. PATIENT-LEVEL TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------

# patient_nbr identifies individual patients.
# All encounters belonging to the same patient remain
# in either training OR validation.

groups = df["patient_nbr"]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[train_idx]

X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]

y_val = y.iloc[val_idx]


print("\n========== TRAIN / VALIDATION ==========")

print("Training encounters:", len(X_train))

print("Validation encounters:", len(X_val))

print(
    "Training readmission rate:",
    round(y_train.mean(), 4)
)

print(
    "Validation readmission rate:",
    round(y_val.mean(), 4)
)


# Check patient overlap

train_patients = set(
    groups.iloc[train_idx]
)

val_patients = set(
    groups.iloc[val_idx]
)

overlap = train_patients.intersection(
    val_patients
)

print(
    "Patient overlap:",
    len(overlap)
)


# ------------------------------------------------------------
# 19. PREPROCESSING PIPELINE
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            StandardScaler(),
            NUMERIC_FEATURES
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            CATEGORICAL_FEATURES
        )
    ]
)


# ------------------------------------------------------------
# 20. LOGISTIC REGRESSION WITHOUT L2
# ------------------------------------------------------------

model_noreg = Pipeline(
    steps=[

        (
            "preprocess",
            preprocessor
        ),

        (
            "clf",
            LogisticRegression(
                penalty=None,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# 21. LOGISTIC REGRESSION WITH L2
# ------------------------------------------------------------

model_l2 = Pipeline(
    steps=[

        (
            "preprocess",
            preprocessor
        ),

        (
            "clf",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# 22. TRAIN BOTH MODELS
# ------------------------------------------------------------

print("\nTraining model WITHOUT L2...")

model_noreg.fit(
    X_train,
    y_train
)

print("Training model WITH L2...")

model_l2.fit(
    X_train,
    y_train
)

print("\nTraining completed!")


# ------------------------------------------------------------
# 23. PREDICT PROBABILITIES
# ------------------------------------------------------------

prob_noreg = (
    model_noreg
    .predict_proba(X_val)[:, 1]
)

prob_l2 = (
    model_l2
    .predict_proba(X_val)[:, 1]
)


# ------------------------------------------------------------
# 24. ROC-AUC
# ------------------------------------------------------------

auc_noreg = roc_auc_score(
    y_val,
    prob_noreg
)

auc_l2 = roc_auc_score(
    y_val,
    prob_l2
)


print("\n========== ROC-AUC ==========")

print(
    "Without L2:",
    round(auc_noreg, 4)
)

print(
    "With L2:",
    round(auc_l2, 4)
)


# ------------------------------------------------------------
# 25. ROC CURVE
# ------------------------------------------------------------

fpr_noreg, tpr_noreg, _ = roc_curve(
    y_val,
    prob_noreg
)

fpr_l2, tpr_l2, _ = roc_curve(
    y_val,
    prob_l2
)


plt.figure(figsize=(8, 6))


plt.plot(
    fpr_noreg,
    tpr_noreg,
    label=f"No L2 (AUC = {auc_noreg:.3f})"
)


plt.plot(
    fpr_l2,
    tpr_l2,
    label=f"L2 (AUC = {auc_l2:.3f})"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - 30-Day Hospital Readmission"
)

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 26. FUNCTION FOR MODEL EVALUATION
# ------------------------------------------------------------

def evaluate_model(
    model_name,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y_val,
            predictions
        ).ravel()
    )


    auc = roc_auc_score(
        y_val,
        probabilities
    )


    accuracy = accuracy_score(
        y_val,
        predictions
    )


    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )


    specificity = (
        tn / (tn + fp)
    )


    brier = brier_score_loss(
        y_val,
        probabilities
    )


    print("\n===================================")

    print(model_name)

    print("===================================")

    print(
        "ROC-AUC   :",
        round(auc, 4)
    )

    print(
        "Accuracy  :",
        round(accuracy, 4)
    )

    print(
        "Precision :",
        round(precision, 4)
    )

    print(
        "Recall    :",
        round(recall, 4)
    )

    print(
        "Specificity:",
        round(specificity, 4)
    )

    print(
        "F1 Score  :",
        round(f1, 4)
    )

    print(
        "Brier Score:",
        round(brier, 4)
    )

    print("\nConfusion Matrix:")

    print(
        "TN =", tn,
        "| FP =", fp,
        "| FN =", fn,
        "| TP =", tp
    )


    return {
        "Model": model_name,
        "ROC-AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "Brier Score": brier,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ------------------------------------------------------------
# 27. EVALUATE WITHOUT L2
# ------------------------------------------------------------

results_noreg = evaluate_model(
    "Logistic Regression - No L2",
    prob_noreg
)


# ------------------------------------------------------------
# 28. EVALUATE WITH L2
# ------------------------------------------------------------

results_l2 = evaluate_model(
    "Logistic Regression - L2",
    prob_l2
)


# ------------------------------------------------------------
# 29. MODEL COMPARISON TABLE
# ------------------------------------------------------------

comparison = pd.DataFrame(
    [
        results_noreg,
        results_l2
    ]
)


print("\n========== MODEL COMPARISON ==========")

display(comparison)


# ------------------------------------------------------------
# 30. CONFUSION MATRIX - L2 MODEL
# ------------------------------------------------------------

threshold = 0.5

pred_l2 = (
    prob_l2 >= threshold
).astype(int)


cm_l2 = confusion_matrix(
    y_val,
    pred_l2
)


plt.figure(figsize=(6, 5))

plt.imshow(cm_l2)

plt.title(
    "Confusion Matrix - L2 Logistic Regression"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    [
        "No Readmission",
        "Readmission"
    ]
)

plt.yticks(
    [0, 1],
    [
        "No Readmission",
        "Readmission"
    ]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm_l2[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 31. CLASSIFICATION REPORT
# ------------------------------------------------------------

print(
    classification_report(
        y_val,
        pred_l2,
        target_names=[
            "No 30-Day Readmission",
            "30-Day Readmission"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 32. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# ------------------------------------------------------------

tn, fp, fn, tp = (
    confusion_matrix(
        y_val,
        pred_l2
    ).ravel()
)


print("\n========== CLINICAL ERROR ANALYSIS ==========")

print("True Negatives :", tn)

print("False Positives:", fp)

print("False Negatives:", fn)

print("True Positives :", tp)


print("\nFalse Negative:")
print(
    "Actual patient was readmitted within 30 days, "
    "but the model predicted no readmission."
)

print("\nFalse Positive:")
print(
    "Model predicted readmission, "
    "but the patient was not readmitted within 30 days."
)


# ------------------------------------------------------------
# 33. THRESHOLD ANALYSIS
# ------------------------------------------------------------

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


threshold_results = []


for threshold in thresholds:

    predictions = (
        prob_l2 >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y_val,
            predictions
        ).ravel()
    )


    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )


    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )


    threshold_results.append(
        {
            "Threshold": threshold,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "F1": f1,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "TN": tn
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


print("\n========== THRESHOLD ANALYSIS ==========")

display(threshold_df)


# ------------------------------------------------------------
# 34. THRESHOLD PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))


plt.plot(
    threshold_df["Threshold"],
    threshold_df["Sensitivity"],
    marker="o",
    label="Sensitivity"
)


plt.plot(
    threshold_df["Threshold"],
    threshold_df["Specificity"],
    marker="o",
    label="Specificity"
)


plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="o",
    label="F1 Score"
)


plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Threshold Analysis - L2 Logistic Regression"
)

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 35. CLINICAL COST EXAMPLE
# ------------------------------------------------------------

# These are ONLY illustrative weights.
# They are not actual hospital costs.

cost_fn = 5
cost_fp = 1


cost_results = []


for threshold in thresholds:

    predictions = (
        prob_l2 >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y_val,
            predictions
        ).ravel()
    )


    total_cost = (
        cost_fn * fn
        +
        cost_fp * fp
    )


    cost_results.append(
        {
            "Threshold": threshold,
            "FP": fp,
            "FN": fn,
            "Clinical Cost": total_cost
        }
    )


cost_df = pd.DataFrame(
    cost_results
)


print(
    "\n========== ILLUSTRATIVE CLINICAL COST =========="
)

display(cost_df)


# ------------------------------------------------------------
# 36. FINAL SUMMARY
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print(
    f"Without L2 ROC-AUC: {auc_noreg:.4f}"
)

print(
    f"With L2 ROC-AUC   : {auc_l2:.4f}"
)

print(
    f"L2 False Positives: {results_l2['FP']}"
)

print(
    f"L2 False Negatives: {results_l2['FN']}"
)

print(
    f"L2 Sensitivity: {results_l2['Recall']:.4f}"
)

print(
    f"L2 Specificity: {results_l2['Specificity']:.4f}"
)


print("\nCase Study Conclusion:")

print(
    """
The objective of this case study was to predict whether a patient
would be readmitted within 30 days of hospital discharge.

Logistic Regression was implemented both without regularization
and with L2 regularization. Numerical variables were standardized,
while categorical variables were converted using one-hot encoding.

ROC-AUC was used to evaluate the model's ability to distinguish
between patients who were and were not readmitted.

The L2 model applies a penalty to large coefficients, which helps
control model complexity and can improve generalization.

False negatives are clinically important because they represent
patients who were actually readmitted but were predicted as
low-risk. False positives represent patients predicted as
high-risk who were not subsequently readmitted.

Changing the classification threshold creates a trade-off between
false negatives and false positives. Therefore, the final
threshold should be selected according to the clinical consequences
of each type of error and the resources available for intervention.
"""
)
