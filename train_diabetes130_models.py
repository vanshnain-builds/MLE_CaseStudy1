"""
Hospital 30-Day Readmission Prediction -- Diabetes 130-US Hospitals Dataset
=============================================================================
Source: UCI "Diabetes 130-US hospitals for years 1999-2008" (Strack et al.)
101,766 inpatient encounters, 1999-2008, 130 US hospitals.

Trains and evaluates two logistic regression models:
  1. L2-regularized logistic regression (ridge penalty)
  2. Unregularized logistic regression (no penalty)

Target: readmitted within 30 days (binary), derived from the original
3-class `readmitted` column ('<30' -> 1, '>30'/'NO' -> 0).

Outputs -> /home/claude/outputs2/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, precision_score, recall_score,
    accuracy_score, brier_score_loss
)

import os
OUT_DIR = "/home/claude/outputs2"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv("/home/claude/data2/diabetic_data.csv")
print(f"Raw rows: {len(df)}")

# ---------------------------------------------------------------------------
# 2. Clean & filter
#    - Drop encounters where the patient died during the stay (discharge
#      disposition 11/19/20/21 = expired). A deceased patient cannot be
#      readmitted, so including these rows would be a label-definition
#      error, not a modeling choice.
#    - Drop the 3 rows with gender 'Unknown/Invalid'.
# ---------------------------------------------------------------------------
EXPIRED_CODES = [11, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(EXPIRED_CODES)].copy()
df = df[df["gender"] != "Unknown/Invalid"].copy()
print(f"Rows after removing expired/invalid-gender encounters: {len(df)}")

# Binary target: 30-day readmission
df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
print(f"Positive (readmitted <30 days) rate: {df['readmitted_30d'].mean():.4f}")

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

# age: convert '[70-80)' style bracket to its numeric midpoint (75) so the
# natural ordering of age is preserved as a single numeric feature instead
# of 10 unordered one-hot columns.
def age_midpoint(bracket):
    lo, hi = bracket.strip("[)").split("-")
    return (int(lo) + int(hi)) / 2

df["age_numeric"] = df["age"].apply(age_midpoint)

# race: fill '?' as its own category rather than dropping rows (2.2% of data)
df["race"] = df["race"].replace("?", "Unknown")

# medical_specialty: ~49% missing and 70+ raw categories. Keep the top 9
# most frequent specialties explicitly; collapse everything else
# (including '?') into 'Other/Missing' to avoid an unwieldy one-hot block.
top_specialties = df["medical_specialty"].value_counts().head(9).index.tolist()
if "?" in top_specialties:
    top_specialties.remove("?")
df["medical_specialty_grouped"] = df["medical_specialty"].apply(
    lambda x: x if x in top_specialties and x != "?" else ("Missing" if x == "?" else "Other")
)

# ICD-9 primary diagnosis (diag_1) grouped into clinically meaningful
# categories, following the standard grouping used in the original
# Strack et al. (2014) study on this dataset.
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

df["diag_1_group"] = df["diag_1"].apply(group_diagnosis)

# Medication columns: drop near-constant drug columns (>99.9% single value
# -- essentially zero variance, adds noise/dimensionality with no signal).
# Keep the handful of medications prescribed to a meaningful share of patients.
med_cols_all = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
keep_meds = [c for c in med_cols_all if df[c].value_counts(normalize=True).iloc[0] < 0.999]
dropped_meds = [c for c in med_cols_all if c not in keep_meds]
print(f"Keeping {len(keep_meds)} medication columns with real variance: {keep_meds}")
print(f"Dropping {len(dropped_meds)} near-constant medication columns")

NUMERIC_FEATURES = [
    "age_numeric", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
CATEGORICAL_FEATURES = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty_grouped", "diag_1_group",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + keep_meds

# admission_type_id / discharge_disposition_id / admission_source_id are
# integer *codes*, not ordinal quantities -- cast to string so they are
# one-hot encoded rather than treated as numeric magnitude.
for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
    df[c] = df[c].astype(str)

# max_glu_serum / A1Cresult have real missingness (test not ordered) --
# fill with an explicit 'Not tested' category rather than imputing a value.
df["max_glu_serum"] = df["max_glu_serum"].fillna("Not tested")
df["A1Cresult"] = df["A1Cresult"].fillna("Not tested")

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "readmitted_30d"

X = df[FEATURES]
y = df[TARGET]
groups = df["patient_nbr"]  # for a patient-level split

# ---------------------------------------------------------------------------
# 4. Patient-level train/validation split
#    30,248 of the 101,766 encounters belong to patients with >1 encounter.
#    A random row-level split would leak information (the same patient's
#    other visits) between train and validation, inflating apparent
#    performance. GroupShuffleSplit keeps all of a patient's encounters
#    on one side of the split.
# ---------------------------------------------------------------------------
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, val_idx = next(splitter.split(X, y, groups=groups))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

print(f"\nTrain: {len(X_train)} encounters | Validation: {len(X_val)} encounters")
print(f"Train positive rate: {y_train.mean():.4f} | Val positive rate: {y_val.mean():.4f}")
overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
print(f"Patient overlap between train/val: {len(overlap)} (must be 0)")

# ---------------------------------------------------------------------------
# 5. Preprocessing + model pipelines (same design as Case Study 1)
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
    ]
)

model_l2 = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE,
    )),
])

model_noreg = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(
        penalty=None, class_weight="balanced",
        solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE,
    )),
])

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
model_l2.fit(X_train, y_train)
model_noreg.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
def evaluate(model, X_val, y_val, name, threshold=0.5):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_val, proba)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    brier = brier_score_loss(y_val, proba)
    cm = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n=== {name} (threshold={threshold}) ===")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Brier:     {brier:.4f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(classification_report(y_val, preds, target_names=["No 30d Readmit", "30d Readmit"], zero_division=0))

    return {
        "model": name, "roc_auc": auc, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "brier_score": brier,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "proba": proba, "cm": cm
    }

results_l2 = evaluate(model_l2, X_val, y_val, "L2-Regularized Logistic Regression")
results_noreg = evaluate(model_noreg, X_val, y_val, "Unregularized Logistic Regression")

# ---------------------------------------------------------------------------
# 8. Save metrics
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame([
    {k: v for k, v in results_l2.items() if k not in ("proba", "cm")},
    {k: v for k, v in results_noreg.items() if k not in ("proba", "cm")},
])
metrics_df.to_csv(f"{OUT_DIR}/metrics_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 9. ROC curve comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for res, color in [(results_l2, "#2563eb"), (results_noreg, "#dc2626")]:
    fpr, tpr, _ = roc_curve(y_val, res["proba"])
    ax.plot(fpr, tpr, label=f"{res['model']} (AUC={res['roc_auc']:.3f})", color=color, linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curve: 30-Day Readmission -- Diabetes 130-Hospitals Dataset")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 10. Confusion matrices
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, res in zip(axes, [results_l2, results_noreg]):
    cm = res["cm"]
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{res['model']}\n(threshold = 0.5)", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Readmit", "Readmit<30d"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Readmit", "Readmit<30d"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 11. Coefficient comparison (top 25 by |L2 coefficient|)
# ---------------------------------------------------------------------------
feature_names = (
    NUMERIC_FEATURES +
    list(model_l2.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
)
coef_l2 = model_l2.named_steps["clf"].coef_[0]
coef_noreg = model_noreg.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "L2_coef": coef_l2,
    "Unregularized_coef": coef_noreg,
})
coef_df.to_csv(f"{OUT_DIR}/coefficients.csv", index=False)

top_coef = coef_df.reindex(coef_df["L2_coef"].abs().sort_values(ascending=False).index).head(25)
top_coef = top_coef.sort_values("L2_coef")

fig, ax = plt.subplots(figsize=(9, 9))
y_pos = np.arange(len(top_coef))
ax.barh(y_pos - 0.2, top_coef["L2_coef"], height=0.4, label="L2-Regularized", color="#2563eb")
ax.barh(y_pos + 0.2, top_coef["Unregularized_coef"], height=0.4, label="Unregularized", color="#dc2626")
ax.set_yticks(y_pos)
ax.set_yticklabels(top_coef["feature"], fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (log-odds)")
ax.set_title("Top 25 Features by |L2 Coefficient|: L2 vs. Unregularized")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/coefficient_comparison.png", dpi=150)
plt.close()

# Coefficient shrinkage summary: how much does L2 shrink vs unregularized?
l2_norm = np.linalg.norm(coef_l2)
noreg_norm = np.linalg.norm(coef_noreg)
max_abs_l2 = np.abs(coef_l2).max()
max_abs_noreg = np.abs(coef_noreg).max()
print(f"\nL2 coefficient vector norm:            {l2_norm:.3f}")
print(f"Unregularized coefficient vector norm: {noreg_norm:.3f}")
print(f"Max |coefficient|, L2 vs unregularized: {max_abs_l2:.3f} vs {max_abs_noreg:.3f}")

# ---------------------------------------------------------------------------
# 12. Threshold analysis (L2 model)
# ---------------------------------------------------------------------------
thresholds = np.linspace(0.05, 0.95, 19)
prec_list, rec_list, f1_list = [], [], []
for t in thresholds:
    preds_t = (results_l2["proba"] >= t).astype(int)
    prec_list.append(precision_score(y_val, preds_t, zero_division=0))
    rec_list.append(recall_score(y_val, preds_t, zero_division=0))
    f1_list.append(f1_score(y_val, preds_t, zero_division=0))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, prec_list, label="Precision", marker="o", markersize=3)
ax.plot(thresholds, rec_list, label="Recall (Sensitivity)", marker="o", markersize=3)
ax.plot(thresholds, f1_list, label="F1", marker="o", markersize=3)
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6, label="Default threshold (0.5)")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 vs. Threshold (L2 Model)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/threshold_analysis.png", dpi=150)
plt.close()

print(f"\nAll outputs saved to {OUT_DIR}")
print(metrics_df[["model", "roc_auc", "accuracy", "precision", "recall", "f1"]].to_string(index=False))
