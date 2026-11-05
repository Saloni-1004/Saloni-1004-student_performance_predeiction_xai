from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from scipy import sparse
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "student-por-labeled.csv"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "notebook-selected-model.joblib"
TOP_FEATURES = 10

st.set_page_config(
    page_title="Student Performance XAI",
    page_icon="📘",
    layout="wide",
)


@st.cache_resource
def load_resources(data_path: str, model_path: str) -> dict:
    """Load the saved model and prepare the shared SHAP/LIME inputs."""
    students = pd.read_csv(data_path)
    X = students.drop(columns=["G3", "passed"])
    y = students["passed"]
    X_train, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(model_path)
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    fail_index = classes.index(0)

    categorical_columns = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()
    numeric_columns = [
        column for column in X.columns if column not in categorical_columns
    ]
    categorical_encoder = preprocessor.named_transformers_["categories"]
    categories_by_column = dict(
        zip(categorical_columns, categorical_encoder.categories_)
    )

    X_train_encoded = preprocessor.transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    if sparse.issparse(X_train_encoded):
        X_train_encoded = X_train_encoded.toarray()
    if sparse.issparse(X_test_encoded):
        X_test_encoded = X_test_encoded.toarray()

    shap_explainer = shap.TreeExplainer(
        classifier,
        data=X_train_encoded[: min(100, len(X_train_encoded))],
        feature_perturbation="interventional",
        model_output="probability",
    )

    category_to_code = {
        column: {
            category: code
            for code, category in enumerate(categories)
        }
        for column, categories in categories_by_column.items()
    }
    X_train_lime = X_train.copy()
    for column in categorical_columns:
        X_train_lime[column] = X_train_lime[column].map(
            category_to_code[column]
        )

    return {
        "students": students,
        "X": X,
        "X_train": X_train,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "model": model,
        "preprocessor": preprocessor,
        "classifier": classifier,
        "classes": classes,
        "fail_index": fail_index,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "categories_by_column": categories_by_column,
        "category_to_code": category_to_code,
        "X_train_encoded": X_train_encoded,
        "X_test_encoded": X_test_encoded,
        "X_train_lime": X_train_lime.to_numpy(dtype=float),
        "shap_explainer": shap_explainer,
    }


def shap_values_by_original_feature(resources: dict, profile: pd.DataFrame) -> dict:
    """Sum encoded SHAP contributions back into their source columns."""
    encoded_profile = resources["preprocessor"].transform(profile)
    if sparse.issparse(encoded_profile):
        encoded_profile = encoded_profile.toarray()

    raw_values = resources["shap_explainer"].shap_values(encoded_profile)
    fail_index = resources["fail_index"]

    if isinstance(raw_values, list):
        encoded_values = np.asarray(raw_values[fail_index])[0]
    else:
        values_array = np.asarray(raw_values)
        if values_array.ndim == 3:
            encoded_values = values_array[0, :, fail_index]
        elif values_array.ndim == 2:
            encoded_values = values_array[0]
        else:
            raise ValueError(f"Unexpected SHAP output shape: {values_array.shape}")

    preprocessor = resources["preprocessor"]
    categorical_columns = resources["categorical_columns"]
    numeric_columns = resources["numeric_columns"]
    categories_by_column = resources["categories_by_column"]
    categorical_encoder = preprocessor.named_transformers_["categories"]
    dropped_categories = categorical_encoder.drop_idx_
    original_columns = list(resources["X"].columns)
    grouped_values = {}

    category_offset = preprocessor.output_indices_["categories"].start
    for column, categories, dropped_category in zip(
        categorical_columns,
        categorical_encoder.categories_,
        dropped_categories
        if dropped_categories is not None
        else [None] * len(categorical_columns),
    ):
        encoded_width = len(categories) - int(dropped_category is not None)
        next_offset = category_offset + encoded_width
        grouped_values[column] = float(
            encoded_values[category_offset:next_offset].sum()
        )
        category_offset = next_offset

    if category_offset != preprocessor.output_indices_["numbers"].start:
        raise ValueError(
            "Categorical SHAP values do not match the preprocessor's encoded "
            "feature layout."
        )

    numeric_offset = preprocessor.output_indices_["numbers"].start
    for position, column in enumerate(numeric_columns):
        grouped_values[column] = float(encoded_values[numeric_offset + position])

    return {column: grouped_values[column] for column in original_columns}


def lime_explanation_for_profile(resources: dict, profile: pd.DataFrame):
    """Explain one profile using LIME on original input columns."""
    X = resources["X"]
    categorical_columns = resources["categorical_columns"]
    categories_by_column = resources["categories_by_column"]
    category_to_code = resources["category_to_code"]

    encoded_profile = profile.copy()
    for column in categorical_columns:
        encoded_profile[column] = encoded_profile[column].map(
            category_to_code[column]
        )

    categorical_positions = [
        X.columns.get_loc(column) for column in categorical_columns
    ]
    categorical_names = {
        X.columns.get_loc(column): [
            str(category) for category in categories_by_column[column]
        ]
        for column in categorical_columns
    }

    def predict_from_lime(rows):
        frame = pd.DataFrame(rows, columns=X.columns)
        for column in categorical_columns:
            position = X.columns.get_loc(column)
            categories = categories_by_column[column]
            codes = np.rint(frame[column]).astype(int)
            if ((codes < 0) | (codes >= len(categories))).any():
                raise ValueError(
                    f"LIME produced an unknown category code for {column}."
                )
            frame[column] = [categories[code] for code in codes]
        return resources["model"].predict_proba(frame)

    explainer = LimeTabularExplainer(
        training_data=resources["X_train_lime"],
        feature_names=list(X.columns),
        categorical_features=categorical_positions,
        categorical_names=categorical_names,
        class_names=["Fail", "Pass"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    explanation = explainer.explain_instance(
        data_row=encoded_profile.to_numpy(dtype=float)[0],
        predict_fn=predict_from_lime,
        labels=(resources["fail_index"],),
        num_features=min(TOP_FEATURES, len(X.columns)),
        num_samples=1000,
    )
    return {
        X.columns[position]: float(weight)
        for position, weight in explanation.as_map()[resources["fail_index"]]
    }, float(explanation.score)


def support_prompts(profile: pd.Series) -> list[str]:
    """Return cautious, rule-based topics for a supportive teacher discussion."""
    prompts = []
    if profile["G2"] < 10:
        prompts.append(
            "G2 is below 10: offer a teacher or tutor meeting to review recent "
            "topics and identify where support is needed."
        )
    if profile["G2"] < profile["G1"]:
        prompts.append(
            "G2 is lower than G1: ask whether anything changed recently and "
            "agree on a manageable plan for the next assessment."
        )
    if profile["failures"] > 0:
        prompts.append(
            "Prior course failures are recorded: offer regular, supportive "
            "check-ins with a teacher or academic advisor."
        )
    if profile["absences"] >= 10:
        prompts.append(
            "Many absences are recorded: privately ask whether the student "
            "needs help accessing missed lessons or other support."
        )
    if profile["studytime"] <= 2:
        prompts.append(
            "The reported study-time category is relatively low: discuss a "
            "manageable plan for short, regular study sessions if appropriate."
        )
    return prompts


def render_explanation(resources: dict, profile: pd.DataFrame, cutoff: float):
    """Show prediction, explanation comparison, and optional support prompts."""
    probabilities = resources["model"].predict_proba(profile)[0]
    fail_probability = float(probabilities[resources["fail_index"]])
    predicted_label = "Fail" if fail_probability >= cutoff else "Pass"
    profile_values = profile.iloc[0]

    st.metric("Model prediction", predicted_label)
    st.caption(
        "This is the model's prediction at the selected cutoff, not a guarantee "
        "of the student's actual outcome."
    )

    if predicted_label == "Fail":
        st.warning(
            "The model flags this profile for possible academic support. "
            "Review the context with the student; do not use this result alone."
        )
    else:
        st.success(
            "The model predicts Pass at this cutoff. This is not a guarantee "
            "and does not mean support is unnecessary."
        )

    with st.spinner("Calculating SHAP and LIME explanations..."):
        shap_values = shap_values_by_original_feature(resources, profile)
        lime_values, lime_score = lime_explanation_for_profile(resources, profile)

    shap_ranked = sorted(
        shap_values,
        key=lambda column: abs(shap_values[column]),
        reverse=True,
    )[:TOP_FEATURES]
    lime_ranked = sorted(
        lime_values,
        key=lambda column: abs(lime_values[column]),
        reverse=True,
    )[:TOP_FEATURES]
    shared = set(shap_ranked) & set(lime_ranked)
    union = set(shap_ranked) | set(lime_ranked)
    overlap = len(shared) / len(union) if union else 1.0

    direction_agreement = (
        sum(
            np.sign(shap_values[feature]) == np.sign(lime_values[feature])
            for feature in shared
        ) / len(shared)
        if shared
        else float("nan")
    )

    explanation_checks_passed = (
        overlap >= 0.50
        and np.isfinite(direction_agreement)
        and direction_agreement >= 0.80
        and lime_score >= 0.50
    )

    if explanation_checks_passed:
        st.info(
            "The explanation checks are more consistent for this profile. "
            "This does not guarantee the prediction or explanation is correct; "
            "review the student's context."
        )
    else:
        st.warning(
            "The explanation checks are uncertain for this profile. "
            "Please review the student's context; do not rely on this "
            "explanation alone."
        )

    st.subheader("Why did the model make this prediction?")
    st.write(
        "Each explanation shows how input features influenced the model's "
        "estimated Fail score. A positive value points toward Fail; a negative "
        "value points away from Fail. These are model explanations, not causes."
    )

    chart_left, chart_right = st.columns(2)
    chart_data = [
        ("SHAP", shap_values, shap_ranked, chart_left),
        ("LIME", lime_values, lime_ranked, chart_right),
    ]
    for method_name, value_map, ranked_features, container in chart_data:
        with container:
            st.markdown(f"**{method_name} — top features**")
            chart = pd.DataFrame(
                {
                    "Feature": ranked_features,
                    "Contribution": [value_map[column] for column in ranked_features],
                }
            ).set_index("Feature")
            figure, axis = plt.subplots(figsize=(7, 4.8))
            values = chart["Contribution"]
            colors = ["#c44e52" if value > 0 else "#4c72b0" for value in values]
            axis.barh(chart.index, values, color=colors)
            axis.axvline(0, color="black", linewidth=0.8)
            axis.invert_yaxis()
            axis.set_xlabel("Contribution to model Fail score")
            axis.set_title("Positive = toward Fail; negative = away from Fail")
            axis.grid(axis="x", alpha=0.25)
            figure.tight_layout()
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)
            rows = []
            for column in ranked_features:
                contribution = value_map[column]
                rows.append(
                    {
                        "Feature": column,
                        "Student value": str(profile_values[column]),
                        "Contribution": contribution,
                        "Interpretation": (
                            "Toward Fail" if contribution > 0 else "Away from Fail"
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(rows).style.format({"Contribution": "{:+.4f}"}),
                hide_index=True,
                use_container_width=True,
            )

    metric_left, metric_right = st.columns(2)
    metric_left.metric(
        "Top-10 SHAP/LIME overlap",
        f"{overlap:.3f}",
        help=(
            "Jaccard similarity: shared top features divided by all unique "
            "features across the two top-10 lists."
        ),
    )
    metric_right.metric(
        "LIME local fit (R²)",
        f"{lime_score:.3f}",
        help=(
            "How well LIME approximates the model in this local neighborhood. "
            "This is not prediction confidence or explanation correctness."
        ),
    )
    st.caption(
        "SHAP and LIME use different methods. Low feature overlap is a reason "
        "to be cautious and investigate, not a reason to choose one as truth."
    )

    st.subheader("Support topics to discuss")
    prompts = support_prompts(profile_values)
    if prompts:
        for prompt in prompts:
            st.markdown(f"- {prompt}")
    else:
        st.info(
            "No prototype support prompt was triggered. This does not mean "
            "the student has no support needs."
        )
        
def main():
    st.title("Student Performance: Prediction and Explanation")
    st.markdown(
        "### A trust-aware SHAP and LIME demonstration for the UCI "
        "Portuguese-course dataset"
    )
    
    missing_files = [
        str(path)
        for path in (DATA_FILE, MODEL_FILE)
        if not path.exists()
    ]
    if missing_files:
        st.error(
            "Required project files are missing:\n\n"
            + "\n".join(f"- `{path}`" for path in missing_files)
            + "\n\nRun the notebook through its model-saving cell first."
        )
        return

    resources = load_resources(str(DATA_FILE), str(MODEL_FILE))

    st.sidebar.header("Student profile")
    st.sidebar.caption("Input source: anonymized held-out examples")
    cutoff = st.sidebar.selectbox(
        "Fail prediction cutoff",
        options=[0.35, 0.50],
        index=0,
        format_func=lambda value: f"{value:.2f}",
        help=(
            "0.35 was selected by validation F2; 0.50 is the conventional "
            "default used for baseline model results. Their test-set trade-offs "
            "are documented in the README."
        ),
    )

    X_test = resources["X_test"]
    sample_number = st.sidebar.selectbox(
        "Example record",
        options=list(range(1, len(X_test) + 1)),
        format_func=lambda value: f"Example {value}",
        help=(
            "These are anonymous held-out test records. Their actual final "
            "outcomes are intentionally hidden in this demonstration."
        ),
    )
    profile = X_test.iloc[[sample_number - 1]].copy()
    render_explanation(resources, profile, cutoff)

    with st.expander("How to interpret this screen"):
        st.markdown(
            """
            - **Model prediction:** the Pass/Fail class at the selected cutoff.
            - **SHAP/LIME contributions:** positive values push this model's
              Fail score upward; negative values push it downward.
            - **Top-10 overlap:** the share of unique top features both methods
              selected. It measures agreement, not truth.
            - **LIME local fit:** an approximation-quality measure for the local
              explanation, not model confidence.
            - **Support topics:** rule-based conversation starters, not causal
              recommendations or proven interventions.
            """
        )


if __name__ == "__main__":
    main()
