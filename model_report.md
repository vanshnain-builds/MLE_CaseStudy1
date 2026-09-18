# Case Study 1 (Real-Data Run): 30-Day Readmission — Diabetes 130-US Hospitals Dataset

**Source:** UCI "Diabetes 130-US hospitals for years 1999–2008" (Strack et al., 2014) — 101,766 inpatient encounters across 130 US hospitals.
**Models:** Logistic Regression with L2 regularization vs. unregularized Logistic Regression (trained from scratch, independent of the earlier synthetic case study).
**Target:** `readmitted_30d` = 1 if the original `readmitted` field was `'<30'` (readmitted within 30 days), else 0.

---

## 1. Data cleaning & filtering

| Step | Rows affected | Reason |
|---|---|---|
| Start | 101,766 | — |
| Drop expired patients (`discharge_disposition_id` ∈ {11, 19, 20, 21}) | −1,652 | A deceased patient cannot be readmitted; keeping them would corrupt the label definition. |
| Drop `gender = 'Unknown/Invalid'` | −3 | Data artifact, negligible volume. |
| **Final** | **100,111** | |

Resulting 30-day readmission rate: **11.3%** (more imbalanced than a typical "any readmission" framing, since `>30` days is bucketed with "not readmitted").

## 2. Feature engineering

| Decision | Rationale |
|---|---|
| `weight` dropped entirely | 96.9% missing — unusable. |
| `payer_code` dropped entirely | 39.6% missing; not a clinically causal predictor of readmission risk. |
| `medical_specialty` → top 9 categories + "Other"/"Missing" | 49.1% missing / 70+ raw categories; collapsing avoids an unwieldy, mostly-empty one-hot block. |
| `age` bracket (`[70-80)`) → numeric midpoint (75) | Preserves the ordinal nature of age as one numeric feature instead of 10 unordered dummy columns. |
| `diag_1` (primary ICD-9 diagnosis) → 9 clinical groups (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other/Missing) | Standard grouping from the original study; ~700 raw ICD-9 codes are far too sparse to one-hot directly. `diag_2`/`diag_3` (secondary/tertiary) dropped for the same reason and to limit dimensionality. |
| 12 of 23 medication columns dropped (e.g. `examide`, `citoglipton`, `acetohexamide`) | >99.9% single value — zero variance, pure noise. 11 medications with real variance were kept (`metformin`, `insulin`, `glipizide`, `glyburide`, `rosiglitazone`, `pioglitazone`, etc.). |
| `admission_type_id` / `discharge_disposition_id` / `admission_source_id` cast to string, one-hot encoded | These are **billing/administrative codes**, not ordered quantities — code 12 is not "more" than code 5. Treating them as categorical is essential. |
| `max_glu_serum` / `A1Cresult` missing → `"Not tested"` category | Missingness here means the test wasn't ordered, which is itself informative, not a value to impute. |
| `encounter_id`, `patient_nbr` excluded as model features | Identifiers, not predictors — but `patient_nbr` is used for the split (next section). |

**Final feature set:** 9 numeric features (age, time in hospital, lab procedures, procedures, medications, outpatient/emergency/inpatient visit counts, number of diagnoses) + 22 categorical features → 1,000+ column pipeline handles the one-hot expansion internally.

## 3. Patient-level train/validation split

30,248 of the 101,766 original encounters belong to a patient who visited more than once. A naive random row-level split would let the same patient appear in both train and validation, leaking information and inflating apparent performance. Instead, the split was done with **`GroupShuffleSplit` on `patient_nbr`**, guaranteeing zero patient overlap between the two sets:

- Train: 80,100 encounters (79,973 unique patients) — 11.33% readmission rate
- Validation: 20,011 encounters — 11.39% readmission rate
- Patient overlap between splits: **0** (verified)

## 4. Results

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 | Brier Score |
|---|---|---|---|---|---|---|
| **L2-Regularized** (`C=1.0`) | **0.655** | 0.673 | 0.181 | 0.530 | 0.270 | 0.225 |
| **Unregularized** | 0.654 | 0.673 | 0.180 | 0.526 | 0.268 | 0.225 |

Confusion matrix at threshold 0.5:
- L2: TN=12,259, FP=5,472, FN=1,071, **TP=1,209**
- Unregularized: TN=12,262, FP=5,469, FN=1,081, **TP=1,199**

Both models land clearly above the random-classifier diagonal (`roc_curves.png`) — a real, if modest, improvement over chance, consistent with published benchmarks on this dataset (logistic regression on this data typically lands in the 0.60–0.68 AUC range; more complex models like gradient boosting add only a few more AUC points, since 30-day readmission is genuinely hard to predict from administrative data alone).

`class_weight="balanced"` was used for both models given the 11.3% positive rate; without it, a naive classifier could reach ~89% accuracy by predicting "no readmission" for everyone while catching zero true readmissions — useless for a prevention program.

## 5. Where L2 regularization actually matters here (unlike the synthetic dataset)

Overall AUC/accuracy are nearly identical between the two models — expected with 80,100 training rows relative to ~1,000 dummy columns, most of which are common categories with plenty of support. But **individual coefficients tell a different story** for the small number of *rare* categories, where L2 clearly does its job of controlling variance:

| Feature | Encounters with this code | L2 coefficient | Unregularized coefficient |
|---|---|---|---|
| `discharge_disposition_id_12` ("Still patient / expected to return for outpatient services") | **3** | 1.02 | **2.81** |
| `admission_type_id_7` ("Trauma Center") | rare | -0.81 | **-2.30** |
| `discharge_disposition_id_9` ("Admitted as inpatient to this hospital" — data artifact code) | **21** | 1.25 | **2.42** |
| `discharge_disposition_id_17` ("Referred to this institution for outpatient services") | 14 | -0.57 | **-1.66** |
| `nateglinide_Up` | rare medication-change value | -0.61 | **-1.33** |

For categories with only a handful of observations, the unregularized model swings toward near-perfect separation and produces inflated, unstable coefficients — exactly the overfitting behavior L2 regularization exists to prevent. This is the direct, visible contrast the case study was designed to surface: **on the earlier synthetic dataset L2 vs. no-L2 made almost no difference (because there was no signal to overfit to); on this real dataset with genuine rare-category sparsity, L2 visibly tempers coefficients on low-support categories while leaving well-supported features (age, time in hospital, number of prior inpatient visits) essentially unchanged.** See `coefficient_comparison.png` for the full picture across the top 25 features by magnitude.

The two models' held-out AUC is nearly identical mainly because these rare categories affect too few validation rows to move the aggregate metric much — the practical benefit of L2 here is a more *stable, trustworthy* coefficient table (useful if a clinician or auditor wants to interpret the model), not necessarily a large AUC gain on this particular validation split.

## 6. Top drivers of 30-day readmission risk (L2 model, well-supported features)

Positive coefficient = higher readmission risk; negative = lower risk.

- **↑ Risk:** discharge to a swing bed, psychiatric facility, rehab facility, or another inpatient care institution (vs. discharge to home) — patients discharged somewhere other than home are systematically sicker/less self-sufficient.
- **↓ Risk:** discharge to hospice/home (code 13) — consistent with hospice care being reserved for end-of-life patients less likely to return for acute readmission.
- **↑ Risk (numeric features, not shown in the table above):** `number_inpatient` (prior inpatient visit count) and `number_emergency` (prior ED visits) are among the strongest numeric predictors — a well-documented finding in readmission literature: **past utilization predicts future utilization** better than almost anything else available in administrative data.

## 7. Clinical cost framing (unchanged from the general principle, now with a real base rate)

At the current 11.3% base rate, this model's default threshold (0.5) catches **53% of true 30-day readmissions** (recall) at the cost of flagging 5,472 patients who won't be readmitted (32% of all non-readmissions) as false positives. Given that a missed readmission (false negative) risks patient harm and CMS penalty exposure, while a false positive costs only an extra follow-up call, this recall/precision balance is a reasonable — arguably still too conservative — starting point; `threshold_analysis.png` shows how lowering the threshold below 0.5 trades additional precision for higher recall if the hospital's cost ratio favors catching more true readmissions.

## 8. Deliverables

| File | Contents |
|---|---|
| `train_diabetes130_models.py` | Full, runnable training/evaluation script |
| `metrics_summary.csv` | Side-by-side metrics for both models |
| `coefficients.csv` | All feature coefficients, both models |
| `roc_curves.png` | ROC curves overlaid |
| `confusion_matrices.png` | Confusion matrices at threshold 0.5 |
| `coefficient_comparison.png` | Top 25 features, L2 vs. unregularized |
| `threshold_analysis.png` | Precision/recall/F1 vs. threshold (L2 model) |

## 9. Caveats & next steps

- This is a **held-out validation split of the training data**, not an external test set — performance should be read as an internal estimate, not a claim about generalization to a different hospital system or time period.
- ICD-9 diagnosis grouping used only the primary diagnosis (`diag_1`); incorporating `diag_2`/`diag_3` (secondary/tertiary) with the same grouping, or diagnosis *interactions*, could add signal at the cost of more features.
- A stronger next step than tuning `C` further would be comparing this linear baseline against a tree-based model (gradient boosting), which typically captures a few more AUC points on this dataset by modeling nonlinearities and interactions logistic regression can't — useful context for whether the interpretability of a linear model is worth the performance gap for this specific use case.
