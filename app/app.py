
import streamlit as st
import requests
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Kestrel | AI Routing",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
API_URL = "http://127.0.0.1:8000/predict"

# ---------- SIDEBAR ----------
st.sidebar.title("KESTREL")
st.sidebar.caption("AI SERVICE OPERATIONS")
st.sidebar.divider()

st.sidebar.subheader("System Status")

try:
    health = requests.get(
        "http://127.0.0.1:8000/",
        timeout=3
    )
    api_online = health.status_code == 200
except requests.RequestException:
    api_online = False

if api_online:
    st.sidebar.success("Routing API Online")
else:
    st.sidebar.error("Routing API Offline")

st.sidebar.divider()
st.sidebar.caption("Model: TF-IDF + Linear SVM")
st.sidebar.caption("Version: Local trained model")

# ---------- HEADER ----------
st.title("Service Request Routing Dashboard")
st.caption(
    "AI-powered classification and routing "
    "for Kestrel Home Appliances"
)

# ---------- METRICS ----------
m1, m2, m3, m4 = st.columns(4)

m1.metric("Test Requests", "2,178")
m2.metric("Validation Accuracy", "95.20%")
m3.metric("Macro F1 Score", "95.00%")
m4.metric("Service Teams", "7")

st.divider()

# ---------- DASHBOARD TABS ----------
tab1, tab2, tab3 = st.tabs([
    "New Service Request",
    "Team Distribution",
    "Model Performance"
])

# ---------- REQUEST FORM ----------
with tab1:
    st.subheader("Customer Service Request")
    st.write("Enter the customer's details to predict the appropriate service team.")

    with st.form("routing_form"):
        left, right = st.columns(2)

        with left:
            request_text = st.text_area(
                "Customer issue",
                placeholder="Example: My washing machine is not starting...",
                height=140
            )

            product_family = st.selectbox(
                "Product Category",
                [
                    "Washing Machine",
                    "Refrigerator",
                    "Air Conditioner",
                    "Water Purifier",
                    "Microwave",
                    "Other"
                ]
            )

        with right:
            warranty_status = st.selectbox(
                "Warranty Status",
                ["In Warranty", "Out of Warranty", "Unknown"]
            )

            channel = st.selectbox(
                "Request Channel",
                ["web", "phone", "email", "app"]
            )

            source = st.selectbox(
                "Request Source",
                ["crm", "legacy_zoho"]
            )

        submitted = st.form_submit_button(
            "Predict Service Team",
            use_container_width=True
        )

    if submitted:
        if not request_text.strip():
            st.warning("Please enter the customer issue.")
        elif not api_online:
            st.error("Routing API is offline. Start FastAPI first.")
        else:
            payload = {
                "request_text": request_text,
                "product_family": product_family,
                "warranty_status": warranty_status,
                "channel": channel,
                "source": source
            }

            try:
                with st.spinner("Analysing customer request..."):
                    response = requests.post(
                        API_URL,
                        json=payload,
                        timeout=30
                    )

                response.raise_for_status()
                result = response.json()

                st.success("Request classified successfully")

                st.markdown("### Predicted Routing Team")

                st.info(
                    f"**{result['predicted_team']}**"
                )

                st.markdown("### Request Information")

                c1, c2 = st.columns(2)

                c1.write(f"**Product:** {product_family}")
                c1.write(f"**Warranty:** {warranty_status}")

                c2.write(f"**Channel:** {channel}")
                c2.write(f"**Source:** {source}")

                st.markdown("### Routing Details")

                for reason in result.get("reasons", []):
                    st.write("•", reason)

                st.caption(
                    "The details above summarize the request inputs. "
                    "They are not independent explanations of model decisions."
                )

            except requests.RequestException as error:
                st.error(f"Request failed: {error}")

# ---------- TEAM DISTRIBUTION ----------
with tab2:
    st.subheader("Predicted Team Distribution")

    prediction_file = BASE_DIR / "predictions.csv"

    if prediction_file.exists():
        predictions = pd.read_csv(prediction_file)

        counts = (
            predictions["team"]
            .value_counts()
            .rename_axis("Service Team")
            .reset_index(name="Requests")
        )

        st.bar_chart(
            counts.set_index("Service Team")
        )

        st.dataframe(
            counts,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Distribution of predicted teams across the "
            "2,178 unlabelled test requests."
        )
    else:
        st.warning("predictions.csv was not found.")

# ---------- MODEL PERFORMANCE ----------
with tab3:
    st.subheader("Model Evaluation")

    st.write(
        "Chronological validation using historical "
        "service requests."
    )

    a, b = st.columns(2)

    a.metric("Accuracy", "95.20%")
    b.metric("Macro F1", "95.00%")

    report_file = BASE_DIR / "evidence" / "evaluation_report.txt"
    matrix_file = BASE_DIR / "evidence" / "confusion_matrix.png"

    if report_file.exists():
        st.markdown("### Classification Report")
        st.code(
            report_file.read_text(encoding="utf-8"),
            language="text"
        )

    if matrix_file.exists():
        st.markdown("### Confusion Matrix")
        st.image(
            str(matrix_file),
            use_container_width=True
        )

st.divider()

st.caption(
    "Kestrel Home Appliances | AI Service Routing Prototype"
)
