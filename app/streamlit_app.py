
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Project configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retentionai_service import RetentionAIAssistant

st.set_page_config(
    page_title="RetentionAI",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# Load saved project data and the local assistant
# ------------------------------------------------------------

@st.cache_resource
def load_assistant():
    return RetentionAIAssistant(
        processed_dir=PROJECT_ROOT / "data" / "processed"
    )

st.title("RetentionAI")
st.caption(
    "Customer Intelligence | Retention Priorities | Business Q&A"
)

try:
    assistant = load_assistant()
except Exception as error:
    st.error("RetentionAI could not load its project data.")
    st.code(f"{type(error).__name__}: {error}")
    st.stop()

# ------------------------------------------------------------
# Project limitations
# ------------------------------------------------------------

st.warning(
    "This project estimates repeat-purchase risk using Olist "
    "e-commerce data; it is not a subscription-churn model. "
    "Earlier evaluation showed PR-AUC 0.0284 and ROC-AUC 0.5312. "
    "Treat scores as exploratory prioritization signals, not "
    "reliable or confirmed churn probabilities."
)

# ------------------------------------------------------------
# Dashboard navigation
# ------------------------------------------------------------

overview_tab, campaigns_tab, customers_tab, assistant_tab = st.tabs(
    [
        "Overview",
        "Campaign Priorities",
        "Customer Explorer",
        "Ask RetentionAI"
    ]
)

# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with overview_tab:
    st.subheader("Business Overview")

    risk_data = assistant.risk_segments.copy()

    total_customers = int(
        pd.to_numeric(
            risk_data["customer_count"],
            errors="coerce"
        ).fillna(0).sum()
    )

    high_risk_rows = risk_data[
        risk_data["risk_level"].astype(str).str.lower() == "high risk"
    ]

    high_risk_count = int(
        pd.to_numeric(
            high_risk_rows["customer_count"],
            errors="coerce"
        ).fillna(0).sum()
    )

    critical_count = len(assistant.critical_risk_customers)
    campaign_count = len(assistant.campaign_priorities)

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Evaluation Customers", f"{total_customers:,}")
    metric2.metric("High-Risk Customers", f"{high_risk_count:,}")
    metric3.metric("Critical-Risk Segment", f"{critical_count:,}")
    metric4.metric("Retention Campaigns", f"{campaign_count:,}")

    st.subheader("Customer Risk Distribution")

    risk_chart = risk_data[
        ["risk_level", "customer_count"]
    ].copy()

    risk_chart["customer_count"] = pd.to_numeric(
        risk_chart["customer_count"],
        errors="coerce"
    ).fillna(0)

    st.bar_chart(
        risk_chart,
        x="risk_level",
        y="customer_count",
        horizontal=True
    )

    st.subheader("Risk Segment Summary")
    st.dataframe(
        assistant.risk_segments,
        hide_index=True,
        width="stretch"
    )

# ============================================================
# TAB 2: CAMPAIGN PRIORITIES
# ============================================================

with campaigns_tab:
    st.subheader("Campaign Execution Priorities")

    campaigns = assistant.campaign_priorities.copy()

    campaigns["average_priority_score"] = pd.to_numeric(
        campaigns["average_priority_score"],
        errors="coerce"
    )

    campaigns = campaigns.sort_values(
        "average_priority_score",
        ascending=False
    )

    st.bar_chart(
        campaigns,
        x="recommended_action",
        y="customer_count",
        horizontal=True
    )

    display_columns = [
        "recommended_action",
        "campaign_priority",
        "customer_count",
        "customer_percentage",
        "average_priority_score",
        "average_retention_risk",
        "average_monetary"
    ]

    display_columns = [
        column for column in display_columns
        if column in campaigns.columns
    ]

    st.dataframe(
        campaigns[display_columns],
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Priority tiers and scores are internal decision-support "
        "rules. Actual campaign effectiveness has not been measured "
        "because campaign-response outcomes are not included here."
    )

# ============================================================
# TAB 3: CUSTOMER EXPLORER
# ============================================================

with customers_tab:
    st.subheader("Explore Customer Segments")

    segment_choice = st.selectbox(
        "Choose a customer segment",
        [
            "Highest-Risk Examples (Top 20)",
            "High-Value / High-Risk Segment",
            "Critical-Risk Segment"
        ]
    )

    if segment_choice == "Highest-Risk Examples (Top 20)":
        customer_data = assistant.high_risk_customers.copy()

    elif segment_choice == "High-Value / High-Risk Segment":
        customer_data = assistant.high_value_customers.copy()

    else:
        customer_data = assistant.critical_risk_customers.copy()

    if "monetary" in customer_data.columns:
        customer_data["monetary"] = pd.to_numeric(
            customer_data["monetary"],
            errors="coerce"
        )

    if segment_choice == "High-Value / High-Risk Segment":
        customer_data = customer_data.sort_values(
            "monetary",
            ascending=False
        )

        st.caption(
            f"Showing the top 100 by historical monetary value "
            f"from a segment of {len(customer_data):,} customers."
        )

        customer_data = customer_data.head(100)

    elif segment_choice == "Highest-Risk Examples (Top 20)":
        st.caption(
            "These are saved high-risk examples, not the entire "
            "customer population."
        )

    else:
        st.caption(
            f"Customers in the exported critical-risk segment: "
            f"{len(customer_data):,}."
        )

    search_id = st.text_input(
        "Filter by customer ID (optional)"
    )

    if search_id.strip() and "customer_unique_id" in customer_data:
        customer_data = customer_data[
            customer_data["customer_unique_id"]
            .astype(str)
            .str.contains(search_id.strip(), case=False, regex=False)
        ]

    preferred_columns = [
        "customer_unique_id",
        "business_priority_score",
        "priority_tier",
        "retention_risk",
        "repeat_purchase_probability",
        "monetary",
        "average_review_score",
        "recency_days",
        "frequency",
        "canceled_orders",
        "unavailable_orders",
        "delivery_success_rate",
        "recommended_action"
    ]

    visible_columns = [
        column for column in preferred_columns
        if column in customer_data.columns
    ]

    if customer_data.empty:
        st.info("No customer records match that filter.")
    else:
        st.dataframe(
            customer_data[visible_columns],
            hide_index=True,
            width="stretch"
        )

        st.caption(
            f"Records displayed: {len(customer_data):,}"
        )

# ============================================================
# TAB 4: ASK RETENTIONAI
# ============================================================

with assistant_tab:
    st.subheader("Business Question Assistant")

    st.write(
        "Ask about risk segments, high-risk customers, critical-risk "
        "customers, high-value customers, or campaign priorities."
    )

    with st.form("retentionai_question_form"):
        question = st.text_input(
            "Your business question",
            placeholder=(
                "Which retention campaigns should we prioritize?"
            )
        )

        submitted = st.form_submit_button(
            "Get Answer",
            type="primary"
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a business question.")
        else:
            with st.spinner("Searching RetentionAI data..."):
                result = assistant.ask(question)

            st.markdown("### Answer")
            st.write(result["answer"])

            st.caption(
                f"Response mode: {result['response_source']}"
            )

            retrieved = result["retrieved_documents"]

            st.markdown("### Retrieved Knowledge")

            if retrieved.empty:
                st.info(
                    "No knowledge document met the retrieval "
                    "threshold for this question."
                )
            else:
                evidence_view = retrieved[
                    ["topic", "similarity_score"]
                ].copy()

                st.dataframe(
                    evidence_view,
                    hide_index=True,
                    width="stretch"
                )

                with st.expander("View retrieved evidence"):
                    for _, row in retrieved.iterrows():
                        st.markdown(f"**{row['topic']}**")
                        st.write(row["content"])

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.title("RetentionAI")
    st.caption("Local application mode")

    st.success("Local data and RAG enabled")
    st.info("Gemini API calls are disabled in this interface.")

    st.markdown("---")
    st.markdown("**Project components**")
    st.write("• Machine learning risk estimates")
    st.write("• Business priority scoring")
    st.write("• Retention campaign summaries")
    st.write("• TF-IDF knowledge retrieval")
    st.write("• Natural-language business questions")

    st.markdown("---")
    st.caption("Built with Python, pandas, scikit-learn, and Streamlit.")
