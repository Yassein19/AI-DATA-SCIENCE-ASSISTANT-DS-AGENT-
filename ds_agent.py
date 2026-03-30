"""
╔══════════════════════════════════════════════════════════════════╗
║           AI Data Science Agent — Streamlit App                  ║
║   Covers: Problem Definition · Data Cleaning · EDA               ║
╚══════════════════════════════════════════════════════════════════╝

Requirements:
    pip install streamlit pandas numpy matplotlib seaborn plotly \
                openpyxl sqlalchemy anthropic

Run:
    streamlit run ds_agent.py
"""

import io
import json
import os
import re
import textwrap

import anthropic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Science Assistant",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Anthropic client (key from env or sidebar)
# ─────────────────────────────────────────────
def get_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


def ask_claude(client: anthropic.Anthropic, system: str, user: str) -> str:
    """Single-turn call to Claude Sonnet 4."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
    except anthropic.AuthenticationError as e:
        st.error("🚨 **Authentication Error**: The Anthropic API key provided in the sidebar is invalid. Please double-check your API key and try again.")
        return "⚠️ *Failed to generate insights due to an invalid API key.*"
    except Exception as e:
        st.error(f"🚨 **An error occurred**: {e}")
        return "⚠️ *Failed to generate insights.*"


# ─────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────
def load_csv_excel(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def load_sql(connection_string: str, query: str) -> pd.DataFrame:
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def load_json(uploaded_file) -> pd.DataFrame:
    data = json.load(uploaded_file)
    if isinstance(data, list):
        return pd.json_normalize(data)
    return pd.json_normalize([data])


def load_text(uploaded_file) -> pd.DataFrame:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    return pd.DataFrame({"text": lines})


# ─────────────────────────────────────────────
#  Helper: concise dataframe summary for Claude
# ─────────────────────────────────────────────
def df_summary(df: pd.DataFrame, max_rows: int = 5) -> str:
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
    buf.write("Columns & dtypes:\n")
    buf.write(df.dtypes.to_string())
    buf.write("\n\nMissing values per column:\n")
    buf.write(df.isnull().sum().to_string())
    buf.write("\n\nDescriptive statistics:\n")
    buf.write(df.describe(include="all").to_string())
    buf.write(f"\n\nFirst {max_rows} rows (sample):\n")
    buf.write(df.head(max_rows).to_string())
    return buf.getvalue()


# ─────────────────────────────────────────────
#  UI — Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Anthropic API Key", type="password",
                             value=os.getenv("ANTHROPIC_API_KEY", ""))
    st.markdown("---")
    st.markdown("### 📂 Data Source")
    source_type = st.selectbox(
        "Select data source",
        ["CSV / Excel", "SQL Database", "JSON / API", "Text / Unstructured"],
    )

# ─────────────────────────────────────────────
#  Main title
# ─────────────────────────────────────────────
st.title("🤖 AI Data Science Assistant")
st.caption("Problem Definition · Data Cleaning · Exploratory Data Analysis")

if not api_key:
    st.warning("⚠️ Enter your Anthropic API key in the sidebar to unlock AI features.")

# ─────────────────────────────────────────────
#  STEP 0 — Data Loading
# ─────────────────────────────────────────────
st.header("📦 Step 1 — Load Your Data")

df: pd.DataFrame | None = None

if source_type == "CSV / Excel":
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded:
        with st.spinner("Loading file…"):
            df = load_csv_excel(uploaded)

elif source_type == "SQL Database":
    conn_str = st.text_input("Connection string",
                              placeholder="sqlite:///mydb.db  |  postgresql://user:pass@host/db")
    query = st.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 500")
    if st.button("Run Query") and conn_str:
        with st.spinner("Querying database…"):
            try:
                df = load_sql(conn_str, query)
                st.success("Query executed successfully!")
            except Exception as e:
                st.error(f"SQL Error: {e}")

elif source_type == "JSON / API":
    uploaded = st.file_uploader("Upload JSON file", type=["json"])
    if uploaded:
        with st.spinner("Parsing JSON…"):
            df = load_json(uploaded)

elif source_type == "Text / Unstructured":
    uploaded = st.file_uploader("Upload text file (.txt)", type=["txt"])
    if uploaded:
        with st.spinner("Loading text…"):
            df = load_text(uploaded)

if df is not None:
    st.success(f"✅ Data loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")
    with st.expander("👀 Preview data"):
        st.dataframe(df.head(20), width="stretch")

# ─────────────────────────────────────────────
#  STEP 1 — Problem Definition
# ─────────────────────────────────────────────
if df is not None:
    st.divider()
    st.header("🎯 Step 2 — Problem Definition")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_problem = st.text_area(
            "Describe your goal or question in plain English",
            placeholder="e.g. I want to predict customer churn. "
                        "The target column is 'churn_flag'.",
            height=120,
        )
    with col2:
        task_type = st.selectbox(
            "Task type (optional hint)",
            ["Auto-detect", "Classification", "Regression",
             "Clustering", "NLP / Text", "Anomaly Detection", "EDA Only"],
        )

    if st.button("🧠 Define Problem with AI", disabled=not api_key) and user_problem:
        client = get_client(api_key)
        with st.spinner("Claude is analysing your problem…"):
            summary = df_summary(df)
            prompt = textwrap.dedent(f"""
                The user has uploaded a dataset with the following profile:

                {summary}

                User's stated goal:
                "{user_problem}"

                Preferred task type: {task_type}

                Please provide:
                1. A concise, precise problem statement
                2. The recommended ML/analysis task type and why
                3. The likely target variable(s) and key features
                4. Potential challenges (class imbalance, data quality, etc.)
                5. Suggested success metrics
                6. A short recommended action plan (bullet points)
            """)
            result = ask_claude(
                client,
                system="You are an expert data scientist. Respond in clear, structured markdown.",
                user=prompt,
            )
        st.markdown(result)
        st.session_state["problem_definition"] = result

# ─────────────────────────────────────────────
#  STEP 2 — Data Cleaning
# ─────────────────────────────────────────────
if df is not None:
    st.divider()
    st.header("🧹 Step 3 — Data Cleaning & Preprocessing")

    tab1, tab2, tab3 = st.tabs(["📊 Quality Report", "🤖 AI Suggestions", "🔧 Auto-Clean"])

    # ── Quality Report ──────────────────────────
    with tab1:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Rows", f"{df.shape[0]:,}")
        col_b.metric("Total Columns", df.shape[1])
        col_c.metric("Missing Cells",
                     f"{df.isnull().sum().sum():,}  ({df.isnull().mean().mean()*100:.1f}%)")
        col_d.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

        st.subheader("Missing Values by Column")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        missing = missing[missing["Missing Count"] > 0].sort_values("Missing %", ascending=False)
        if not missing.empty:
            fig_miss = px.bar(missing, x="Column", y="Missing %",
                              color="Missing %", color_continuous_scale="Reds",
                              title="Missing Value % per Column")
            st.plotly_chart(fig_miss, width="stretch")
        else:
            st.success("🎉 No missing values found!")

        st.subheader("Data Types")
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Column", "Type"]
        dtype_df["Type"] = dtype_df["Type"].astype(str)
        st.dataframe(dtype_df, width="stretch", hide_index=True)

    # ── AI Suggestions ──────────────────────────
    with tab2:
        if st.button("🤖 Get AI Cleaning Suggestions", disabled=not api_key):
            client = get_client(api_key)
            with st.spinner("Claude is reviewing your data quality…"):
                prompt = textwrap.dedent(f"""
                    Here is a profile of the dataset:

                    {df_summary(df)}

                    Provide detailed, actionable data cleaning recommendations:
                    1. How to handle each column with missing values (imputation strategy, drop, flag)
                    2. Columns to drop or transform
                    3. Outlier treatment suggestions for numeric columns
                    4. Encoding recommendations for categorical columns
                    5. Any data type conversions needed
                    6. Feature engineering ideas based on existing columns
                    7. Normalisation / scaling suggestions

                    Format each recommendation as a specific action with reasoning.
                """)
                result = ask_claude(
                    client,
                    system="You are a senior data scientist specialising in data preprocessing. "
                           "Be specific and actionable. Respond in structured markdown.",
                    user=prompt,
                )
            st.markdown(result)

    # ── Auto-Clean ──────────────────────────────
    with tab3:
        st.markdown("Apply common cleaning operations to your dataset:")

        c1, c2 = st.columns(2)
        drop_dups = c1.checkbox("Drop duplicate rows", value=True)
        drop_high_null = c2.checkbox("Drop columns > 70% missing", value=True)
        fill_numeric = c1.checkbox("Fill numeric NaN with median", value=True)
        fill_cat = c2.checkbox("Fill categorical NaN with 'Unknown'", value=True)
        strip_whitespace = c1.checkbox("Strip whitespace from string columns", value=True)

        if st.button("⚡ Apply Cleaning"):
            df_clean = df.copy()
            log = []

            if drop_dups:
                before = len(df_clean)
                df_clean = df_clean.drop_duplicates()
                log.append(f"✅ Dropped {before - len(df_clean)} duplicate rows.")

            if drop_high_null:
                cols_before = df_clean.shape[1]
                df_clean = df_clean.loc[:, df_clean.isnull().mean() < 0.70]
                log.append(f"✅ Dropped {cols_before - df_clean.shape[1]} high-null columns.")

            num_cols = df_clean.select_dtypes(include="number").columns
            cat_cols = df_clean.select_dtypes(include="object").columns

            if fill_numeric:
                df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
                log.append(f"✅ Filled {len(num_cols)} numeric column(s) with median.")

            if fill_cat:
                df_clean[cat_cols] = df_clean[cat_cols].fillna("Unknown")
                log.append(f"✅ Filled {len(cat_cols)} categorical column(s) with 'Unknown'.")

            if strip_whitespace:
                for col in cat_cols:
                    df_clean[col] = df_clean[col].str.strip()
                log.append("✅ Stripped whitespace from string columns.")

            st.session_state["df_clean"] = df_clean

            for msg in log:
                st.success(msg)

            st.info(f"Cleaned dataset: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

            # Download cleaned data
            csv_bytes = df_clean.to_csv(index=False).encode()
            st.download_button("⬇️ Download Cleaned CSV", csv_bytes,
                               file_name="cleaned_data.csv", mime="text/csv")

# ─────────────────────────────────────────────
#  STEP 3 — EDA
# ─────────────────────────────────────────────
if df is not None:
    st.divider()
    st.header("🔍 Step 4 — Exploratory Data Analysis (EDA)")

    # Use cleaned df if available
    eda_df = st.session_state.get("df_clean", df)

    num_cols = eda_df.select_dtypes(include="number").columns.tolist()
    cat_cols = eda_df.select_dtypes(include="object").columns.tolist()

    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "📈 Distributions", "🔗 Correlations", "📦 Categorical", "📉 Scatter", "🤖 AI Insights"
    ])

    # ── Distributions ───────────────────────────
    with eda_tab1:
        if num_cols:
            col_sel = st.selectbox("Select numeric column", num_cols, key="dist_col")
            fig = px.histogram(eda_df, x=col_sel, nbins=40,
                               marginal="box", title=f"Distribution of {col_sel}",
                               color_discrete_sequence=["#636EFA"])
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No numeric columns found.")

    # ── Correlation Heatmap ─────────────────────
    with eda_tab2:
        if len(num_cols) >= 2:
            corr = eda_df[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                 color_continuous_scale="RdBu_r",
                                 title="Correlation Heatmap")
            st.plotly_chart(fig_corr, width="stretch")

            # Top correlations
            st.subheader("🔝 Top Correlations")
            corr_pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    .stack()
                    .reset_index()
            )
            corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]
            corr_pairs["abs"] = corr_pairs["Correlation"].abs()
            corr_pairs = corr_pairs.sort_values("abs", ascending=False).drop(columns="abs")
            st.dataframe(corr_pairs.head(15), width="stretch", hide_index=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # ── Categorical ─────────────────────────────
    with eda_tab3:
        if cat_cols:
            cat_sel = st.selectbox("Select categorical column", cat_cols, key="cat_col")
            top_n = st.slider("Top N values", 5, 30, 10)
            vc = eda_df[cat_sel].value_counts().head(top_n).reset_index()
            vc.columns = [cat_sel, "Count"]
            fig_bar = px.bar(vc, x=cat_sel, y="Count",
                             title=f"Top {top_n} values in '{cat_sel}'",
                             color="Count", color_continuous_scale="Blues")
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("No categorical columns found.")

    # ── Scatter ─────────────────────────────────
    with eda_tab4:
        if len(num_cols) >= 2:
            sc1, sc2, sc3 = st.columns(3)
            x_col = sc1.selectbox("X axis", num_cols, index=0, key="sc_x")
            y_col = sc2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
            color_col = sc3.selectbox("Color by (optional)",
                                      ["None"] + cat_cols + num_cols, key="sc_c")
            color_arg = None if color_col == "None" else color_col
            fig_sc = px.scatter(eda_df, x=x_col, y=y_col, color=color_arg,
                                opacity=0.7, title=f"{y_col} vs {x_col}",
                                trendline="ols" if color_arg is None else None)
            st.plotly_chart(fig_sc, width="stretch")
        else:
            st.info("Need at least 2 numeric columns for scatter plots.")

    # ── AI Insights ─────────────────────────────
    with eda_tab5:
        if st.button("🤖 Generate Full EDA Insights", disabled=not api_key):
            client = get_client(api_key)
            problem_ctx = st.session_state.get(
                "problem_definition", "No problem definition provided yet.")

            with st.spinner("Claude is generating deep EDA insights…"):
                prompt = textwrap.dedent(f"""
                    You are analysing a dataset for a data science project.

                    Problem context:
                    {problem_ctx}

                    Dataset profile:
                    {df_summary(eda_df)}

                    Provide a comprehensive EDA narrative covering:
                    1. **Overview**: What kind of dataset is this? What story does it tell?
                    2. **Key statistics**: Notable patterns in numeric columns
                    3. **Distribution insights**: Skewness, outliers, unusual distributions
                    4. **Correlations**: Which features are most related? Any surprising relationships?
                    5. **Categorical breakdown**: Dominant categories, imbalances
                    6. **Data quality flags**: Remaining issues after cleaning
                    7. **Feature importance hypothesis**: Which features likely matter most for the target?
                    8. **Recommended visualisations**: 3-5 charts that would reveal the most insight
                    9. **Next steps**: What modelling approaches to explore based on EDA findings

                    Be analytical and insightful, not just descriptive.
                """)
                result = ask_claude(
                    client,
                    system="You are a principal data scientist with deep expertise in EDA. "
                           "Be insightful, analytical, and specific. Use structured markdown.",
                    user=prompt,
                )
            st.markdown(result)

        # Interactive Q&A
        st.divider()
        st.subheader("💬 Ask the Agent about your data")
        user_q = st.text_input("Ask any question about your dataset",
                               placeholder="e.g. Which column has the most predictive power for churn?")
        if st.button("Ask", disabled=not api_key) and user_q:
            client = get_client(api_key)
            with st.spinner("Thinking…"):
                result = ask_claude(
                    client,
                    system="You are a data science assistant. Answer questions about the "
                           "dataset concisely and accurately using the provided profile.",
                    user=f"Dataset profile:\n{df_summary(eda_df)}\n\nQuestion: {user_q}",
                )
            st.markdown(result)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("Built with ❤️ using Streamlit · Plotly · Pandas · Claude AI")
