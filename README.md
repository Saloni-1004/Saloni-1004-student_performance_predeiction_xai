[README.md](https://github.com/user-attachments/files/34440202/README.md)
[README.md](https://github.com/user-attachments/files/34440199/README.md)
# Student Performance Prediction Using Explainable AI

## Do Explanation Methods Agree?

This mini-project trains a model to predict whether a student will pass or
fail the Portuguese-language course in the UCI Student Performance dataset.
It compares SHAP and LIME explanations, measures how often they select the
same features, and provides cautious, rule-based prompts for student support.

The project is for learning and research. It is not intended to grade, rank,
discipline, or deny opportunities to students.

## Dataset and prediction target

- **Dataset:** UCI Student Performance, Portuguese course (`student-por.csv`).
- **Size:** 649 student records.
- **Target:** `Pass` when final grade `G3` is at least 10 on the 0–20 scale;
  otherwise `Fail`. A `G3` of 0 is included as `Fail`.
- **Observed labels:** 549 Pass and 100 Fail.
- **Inputs:** all columns except `G3` and the derived `passed` target.
- **Important scope:** inputs include the earlier course grades `G1` and
  `G2`. The task predicts the final outcome after those grades are available;
  it is not an early prediction made before the course starts.

See [data/README.md](data/README.md) for dataset setup and label details.
The source dataset is available from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance).

## Current results

The baseline is a class-weighted Random Forest with a stratified 80/20
train/test split (`random_state=42`). There are 130 held-out test students,
including 20 Fail cases.

At the default Fail-probability cutoff of 0.50, the held-out test results were:

| Measure | Result |
|---|---:|
| Accuracy | 0.862 |
| ROC AUC | 0.917 |
| Fail average precision | 0.602 |
| Fails caught | 15 / 20 |
| Fails missed | 5 |
| False alarms (Pass predicted as Fail) | 13 |

A cutoff of 0.35 was selected using a validation split from the training data,
by choosing the tested cutoff with the highest F2 score. On the held-out test
set, it caught 16 of 20 Fails, missed 4, and raised 16 false alarms. Compared
with 0.50, this cutoff caught one additional Fail at the cost of three
additional false alarms. The validation selection does not establish that
0.35 is universally better.

### Comparison without earlier grades

A separate model was trained without `G1` and `G2`, using the same held-out
students. At a 0.50 cutoff, it caught 6 of 20 Fails, missed 14, and raised
16 false alarms; Fail ROC AUC was 0.650 and Fail average precision was 0.251.
A 0.30 cutoff was selected using training/validation data. On the held-out
test set, it caught 10 of 20 Fails, missed 10, and raised 36 false alarms.
This comparison suggests that `G1` and `G2` provide substantial predictive
information in this dataset. It does not show that the model can predict
outcomes before the course begins, and the results are from one split.

### Repeated cross-validation

To check whether the one-split comparison was unusual, both feature sets were
evaluated using repeated stratified 5-fold cross-validation (5 repeats, 25
fold evaluations) with a fixed 0.50 cutoff. The reported values are the mean
and standard deviation across folds:

| Feature set | Fail ROC AUC | Fail average precision | Fail recall | Fail F2 |
|---|---:|---:|---:|---:|
| With `G1` and `G2` | 0.965 ± 0.012 | 0.840 ± 0.055 | 0.796 ± 0.113 | 0.778 ± 0.090 |
| Without `G1` and `G2` | 0.834 ± 0.034 | 0.490 ± 0.073 | 0.446 ± 0.114 | 0.455 ± 0.103 |

The repeated-fold results support the observation that earlier grades carry
useful predictive information in this dataset. Fold standard deviations
describe variation across these overlapping repeated splits; they are not
confidence intervals, and this single dataset does not establish
generalization to other student populations.

### Classifier comparison

Logistic Regression, Random Forest, and XGBoost were compared on the training
portion only, using the same repeated stratified 5-fold splits (5 repeats,
25 folds). Fail average precision was the primary ranking metric because it
evaluates Fail ranking across score cutoffs. Recall, precision, and F2 below
use a fixed 0.50 cutoff.

| Model | Fail average precision | Fail ROC AUC | Fail recall at 0.50 | Fail precision at 0.50 | Fail F2 at 0.50 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.879 ± 0.056 | 0.972 ± 0.016 | 0.795 | 0.754 | 0.782 |
| Random Forest | 0.856 ± 0.069 | 0.971 ± 0.013 | 0.818 | 0.762 | 0.801 |
| Logistic Regression | 0.831 ± 0.074 | 0.959 ± 0.021 | 0.835 | 0.630 | 0.781 |

XGBoost had the highest mean Fail average precision, while Random Forest had
the highest mean F2 and precision among these models at the fixed cutoff.
The fold-to-fold standard deviations overlap, so these results do not
establish a decisive winner. The comparison does not use the held-out test
set to rank models. It can be reproduced with `compare_classifiers.py`; its
per-fold results are saved to `reports/classifier-comparison-folds.csv`.

A validation step compared cutoffs for all three models using F2,
which gives Fail recall more weight than precision without simply flagging
every student. Run `evaluate_classifier_thresholds.py` to create
`reports/classifier-threshold-validation.csv`. This uses only a validation
split from the training data. The best cutoff for each model was:

| Model | Selected cutoff | Fail recall | Fail F2 | Caught Fails | Missed Fails | False alarms |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.35 | 0.950 | 0.872 | 19 / 20 | 1 | 10 |
| Logistic Regression | 0.15 | 0.950 | 0.848 | 19 / 20 | 1 | 13 |
| XGBoost | 0.65 | 0.850 | 0.842 | 17 / 20 | 3 | 4 |

Given the project's stated preference to catch as many potential Fails as
possible, Random Forest at a 0.35 cutoff is the current validation-based
choice: it has the highest validation F2 and caught 19 of 20 validation
Fails. XGBoost had the highest mean average precision in repeated
cross-validation, but at its best validation F2 cutoff it caught fewer Fails.
This is a goal-dependent trade-off, not proof that Random Forest is universally
the best algorithm. The existing dashboard already uses Random Forest and
offers the 0.35 cutoff, so no model swap is indicated by these results.

The held-out test set has been examined in earlier Random Forest experiments,
including the 0.35 cutoff result of 16 Fails caught, 4 missed, and 16 false
alarms. Do not describe it as a fresh, untouched confirmation of this new
three-model selection. Independent confirmation would require a new external
student cohort.

## SHAP and LIME agreement

The comparison explains the Fail probability for each of the 130 held-out
students. SHAP contributions for one-hot encoded categories are combined back
to their original input columns so the methods are compared at the same
feature level. The main agreement measure is the Jaccard overlap between each
method's top-10 feature set.

In the current run:

- Mean top-10 Jaccard overlap was **0.350**.
- Mean direction agreement among shared top features was **0.905** for actual
  Fail cases and **0.932** for actual Pass cases.
- Bootstrap intervals for mean Jaccard across students in this test split
  were **0.298–0.375** for Fail and **0.337–0.369** for Pass.
- LIME local R² averaged **0.778** for Fail cases and **0.401** for Pass cases.
- For one selected Fail student, LIME's mean pairwise top-10 Jaccard across
  ten random seeds was **0.573**.

These figures describe one split and the current implementation. They measure
agreement or local approximation—not whether either explanation is correct,
causal, or suitable for making decisions about a student. LIME explanations
can vary with random perturbations and local fit differs between examples.
Create the visual summary after generating the agreement CSV:

```powershell
python .\src\student_performance\plot_explanation_agreement.py
```

This writes `reports\figures\explanation-agreement.png`, showing the overlap
distribution, direction agreement, and LIME local fit by actual outcome.

## Run the project

Use Python 3.11 or newer. From the project folder, open a terminal and create
or activate a virtual environment, then install the project:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Run the teaching notebook

The step-by-step notebook is
`notebooks/student_performance_xai.ipynb`. After installing the development
dependencies above, open it in Jupyter Notebook and select the project's
`.venv` Python kernel. Run the cells from top to bottom. Its dashboard-ready
model-saving cell writes `artifacts\notebook-selected-model.joblib`; run that
cell before starting the Streamlit app.

Place `student-por.csv` in `data\raw\`, then run these commands from the
project folder:

```powershell
python .\src\student_performance\inspect_data.py
python .\src\student_performance\prepare_data.py
python .\src\student_performance\train_model.py
python .\src\student_performance\explain_model.py
python .\src\student_performance\compare_agreement.py
python .\src\student_performance\evaluate_thresholds.py
python .\src\student_performance\evaluate_test.py
python .\src\student_performance\evaluate_without_grades.py
python .\src\student_performance\evaluate_thresholds_without_grades.py
python .\src\student_performance\evaluate_test_without_grades.py
python .\src\student_performance\cross_validate_models.py
python .\src\student_performance\compare_classifiers.py
python .\src\student_performance\evaluate_classifier_thresholds.py
python .\src\student_performance\plot_cv_results.py
python .\src\student_performance\plot_explanation_agreement.py
python -m streamlit run .\src\student_performance\app.py
```

`prepare_data.py` writes `data\processed\student-por-labeled.csv`.
`train_model.py` saves the original script-trained pipeline under
`artifacts\student-performance-model.joblib`. The Streamlit dashboard loads
the notebook-trained pipeline from
`artifacts\notebook-selected-model.joblib`, so run the notebook's model-saving
cell before launching the dashboard. The
explanation agreement script writes per-student results to
`reports\explanation-agreement.csv`. Raw and processed data, model artifacts,
and generated reports are excluded from version control.

The Streamlit dashboard opens in a browser and explains anonymous held-out
examples only. It shows the Pass/Fail prediction, SHAP and LIME feature
contributions, their top-10 overlap, LIME local fit, and cautious discussion
prompts. The dashboard does not display the known outcome for held-out
examples and does not provide a profile form or downloadable explanation
summary.

## Code map

```text
data/
  README.md                         Dataset setup, label definition
src/
  student_performance/
    inspect_data.py                 Inspect the source CSV
    prepare_data.py                 Create the Pass/Fail target
    train_model.py                  Train and evaluate the baseline
    explain_model.py                Explain one example and show support prompts
    compare_agreement.py            Compare SHAP and LIME across test students
    evaluate_thresholds.py          Select a cutoff using validation data
    evaluate_test.py                Compare cutoffs on the held-out test set
    evaluate_without_grades.py      Compare performance without G1 and G2
    evaluate_thresholds_without_grades.py
                                    Select a no-grades cutoff using validation
    evaluate_test_without_grades.py Evaluate no-grades cutoffs on held-out test
    cross_validate_models.py       Compare feature sets over repeated folds
    compare_classifiers.py         Compare Logistic Regression, Random Forest,
                                    and XGBoost using training-fold validation
    evaluate_classifier_thresholds.py
                                    Compare model cutoffs on validation data
    plot_cv_results.py             Plot repeated cross-validation results
    plot_explanation_agreement.py  Plot SHAP/LIME agreement summaries
    app.py                         Interactive Streamlit teacher-demo dashboard
```

Run the full sequence in order. The plotting scripts use the saved
cross-validation values and agreement CSV generated by the preceding scripts.

## Limitations and responsible use

- The dataset is small and has only 100 Fail cases; the held-out test set has
  just 20. Metrics and subgroup comparisons can therefore be uncertain.
- Results are from one random split, not repeated cross-validation or an
  external cohort. They may not generalize to other schools, courses, or
  student populations.
- `G1` and `G2` are strong predictors but mean this is not a beginning-of-course
  early-warning system.
- Features can reflect social and educational context. A predictive
  association is not evidence that a feature caused an outcome.
- Explanations describe model behavior and can disagree or be unstable.
  Neither SHAP nor LIME is ground truth.
- The rule-based support prompts are illustrative suggestions for discussion,
  not validated interventions or guaranteed ways to improve results.
- Do not use this prototype for consequential decisions about students.
