import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from io import BytesIO
from typing import Optional

# CONFIG
st.set_page_config(
    page_title="PalmPay Adoption Prediction",
    layout="wide"
)

# LOAD MODEL PIPELINE
model = joblib.load("Palmpay_Pipeline1.pkl")
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]


# FEATURE DEFINITIONS (MUST MATCH TRAINING)
CATEGORICAL_COLS = [
    "Gender",
    "device_used",
    "source_of_income",
    "age_bracket",
    "lga_name"
]

NUMERICAL_COLS = [
    "population",
    "bank_charges",
    "multi_app_users",
    "coverage_percent"
]

FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS


# UI HEADER
st.title("PalmPay User Adoption Prediction Dashboard")
st.markdown("""
Upload **RAW LGA-level data** to predict PalmPay adoption probability,
rank LGAs, and understand **why** adoption is high or low.
""")


# TEMPLATE DOWNLOAD
def generate_template() -> pd.DataFrame:
    return pd.DataFrame({col: [] for col in FEATURES})

buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    generate_template().to_excel(writer, index=False)

st.download_button(
    "Download Input Template",
    buffer.getvalue(),
    "palmpay_input_template.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.divider()

# FILE UPLOAD
uploaded_file: Optional[BytesIO] = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# HELPER FUNCTIONS
def kpi_label(p: float) -> str:
    if p >= 0.7:
        return "🟢 High Adoption"
    elif p >= 0.4:
        return "🟡 Medium Adoption"
    return "🔴 Low Adoption"


# MAIN APP LOGIC
if uploaded_file is not None:

    # LOAD DATA
    if uploaded_file.name.endswith(".csv"):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)


    # VALIDATE COLUMNS
    missing = [c for c in FEATURES if c not in df_input.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # PREDICTIONS
    df_input["adoption_prob"] = model.predict_proba(df_input[FEATURES])[:, 1]
    df_input["adoption_class"] = model.predict(df_input[FEATURES])
    df_input["KPI"] = df_input["adoption_prob"].apply(kpi_label)

    st.subheader("Prediction Results")
    st.dataframe(df_input, use_container_width=True)

    st.download_button(
        "Download Predictions",
        df_input.to_csv(index=False),
        "palmpay_predictions.csv",
        "text/csv"
    )


    # LGA RANKING
    st.divider()
    st.subheader("LGA Adoption Ranking")

    leaderboard = (
        df_input[["lga_name", "adoption_prob", "KPI"]]
        .sort_values("adoption_prob", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard["Rank"] = leaderboard.index + 1

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Top 5 LGAs")
        st.dataframe(leaderboard.head(5), use_container_width=True)
    with col2:
        st.markdown("Bottom 5 LGAs")
        st.dataframe(leaderboard.tail(5), use_container_width=True)


    # SHAP EXPLAINABILITY
    st.divider()
    st.subheader("Model Explainability (SHAP)")

    # Transform features
    X_transformed = preprocessor.transform(df_input[FEATURES])

    # Feature names
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_COLS
    )
    feature_names = NUMERICAL_COLS + list(cat_features)

    # SHAP explainer
    explainer = shap.LinearExplainer(
        classifier,
        X_transformed,
        feature_names=feature_names
    )
    shap_values = explainer(X_transformed)


    # GLOBAL SHAP
    st.markdown("Global Feature Importance")
    fig1, _ = plt.subplots()
    shap.plots.bar(shap_values, max_display=10, show=False)
    st.pyplot(fig1)


    # INDIVIDUAL LGA SHAP
    st.markdown("Individual LGA Explanation")

    selected_lga = st.selectbox(
        "Select LGA",
        df_input["lga_name"].unique()
    )

    row = df_input[df_input["lga_name"] == selected_lga][FEATURES]
    row_transformed = preprocessor.transform(row)
    row_shap = explainer(row_transformed)

    # Force dense arrays (critical fix)
    shap_values_dense = row_shap.values[0]

    if hasattr(row_transformed, "toarray"):
        row_data_dense = row_transformed.toarray()[0]
    else:
        row_data_dense = row_transformed[0]

    # Rebuild Explanation object
    row_explanation = shap.Explanation(
        values=shap_values_dense,
        base_values=row_shap.base_values[0],
        data=row_data_dense,
        feature_names=feature_names
    )

    fig2, _ = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(row_explanation, max_display=10, show=False)
    st.pyplot(fig2)


    # BUSINESS EXPLANATION
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "impact": shap_values_dense
    }).sort_values("impact", key=abs, ascending=False)

    st.markdown("Why this prediction was made")

    pos = shap_df[shap_df["impact"] > 0].head(3)
    neg = shap_df[shap_df["impact"] < 0].head(3)

    if not pos.empty:
        st.markdown("**Factors increasing adoption:**")
        for _, r in pos.iterrows():
            st.write(f"• {r['feature']}")

    if not neg.empty:
        st.markdown("**Factors reducing adoption:**")
        for _, r in neg.iterrows():
            st.write(f"• {r['feature']}")


    # SHAP DOWNLOAD
    shap_global_df = pd.DataFrame(
        shap_values.values,
        columns=feature_names
    )

    st.download_button(
        "Download SHAP Values (CSV)",
        shap_global_df.to_csv(index=False),
        "palmpay_shap_values.csv",
        "text/csv"
    )


    # STRATEGIC SUMMARY
    st.divider()
    st.subheader("Strategic Summary")

    st.write(
        f"• Average adoption probability: "
        f"**{df_input['adoption_prob'].mean():.2%}**"
    )
    st.write(
        f"• High adoption LGAs (≥70%): "
        f"**{(df_input['adoption_prob'] >= 0.7).sum()}**"
    )