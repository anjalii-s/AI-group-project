import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Ethereum Fraud Detection – Analyst Dashboard",
    layout="wide"
)

# ======================================
# SIDEBAR NAVIGATION
# ======================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Dashboard", "Business Understanding", "Data Understanding",
     "Modeling Approach", "Model Performance", "Explainability (SHAP)",
     "Big Data Architecture", "Limitations & Future Work"]
)

# ======================================
# DASHBOARD PAGE
# ======================================
if page == "Dashboard":
    st.title("Ethereum Fraud Detection – Analyst Dashboard")

    st.sidebar.subheader("Analyst Controls")
    threshold = st.sidebar.slider(
        "Fraud Risk Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    st.sidebar.write(f"Current threshold: {threshold}")

    st.info("""
    Lowering the threshold increases fraud detection (recall) but may raise false positives.
    Raising the threshold reduces false alerts but increases the risk of missed fraud.
    """)

    st.header("Operational Risk Metrics")

    if threshold <= 0.4:
        st.metric("Fraud Recall", "↑ High", delta="Fewer missed fraud cases")
        st.metric("False Positives", "↑ Increased")
    elif threshold <= 0.6:
        st.metric("Fraud Recall", "Balanced")
        st.metric("False Positives", "Balanced")
    else:
        st.metric("Fraud Recall", "↓ Lower")
        st.metric("False Positives", "↓ Reduced")

    st.header("Account-Level Risk Exploration")

    account_id = st.selectbox(
        "Select Ethereum Account ID",
        options=["Account_101", "Account_205", "Account_309"]
    )

    st.write(f"Selected account: {account_id}")

    st.subheader("Why is this account risky?")
    st.write("""
    Key contributing factors identified by SHAP:
    • High ERC20 transaction frequency  
    • Abnormally short transaction intervals  
    • Large Ether value movements  
    • High interaction with multiple addresses  
    """)

    # ======================================
    # DYNAMIC CONFUSION MATRIX
    # ======================================
    st.header("Dynamic Confusion Matrix")

    try:
        pred_df = pd.read_csv("predictions_with_probs.csv")

        # Apply threshold
        pred_df["predicted"] = (pred_df["fraud_prob"] >= threshold).astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(pred_df["FLAG"], pred_df["predicted"])

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (Threshold = {threshold})")

        st.pyplot(fig)

    except Exception as e:
        st.warning("Dynamic confusion matrix could not be generated. Ensure predictions_with_probs.csv is uploaded.")
        st.text(str(e))

# ======================================
# BUSINESS UNDERSTANDING
# ======================================
elif page == "Business Understanding":
    st.title("Business Understanding")
    st.write("""
    Ethereum fraud (e.g., phishing, Ponzi schemes) causes significant financial loss.
    Due to the large volume and velocity of blockchain transactions, scalable Big Data
    analytics are required.

    This project applies the CRISP-DM lifecycle and Big Data technologies to detect
    fraudulent Ethereum accounts.
    """)

# ======================================
# DATA UNDERSTANDING
# ======================================
elif page == "Data Understanding":
    st.title("Data Understanding")
    st.write("""
    Dataset: Ethereum Transaction Dataset (Kaggle)

    • Total records: 9,841  
    • Target variable: FLAG (1 = Fraud, 0 = Legitimate)  
    • Class imbalance: ~15% fraudulent accounts  
    • High-dimensional transactional and ERC20 features  
    """)

# ======================================
# MODELING APPROACH
# ======================================
elif page == "Modeling Approach":
    st.title("Modeling Approach")
    st.write("""
    • Big Data processing using Apache Spark  
    • Feature engineering via VectorAssembler  
    • Random Forest classifier for fraud detection  
    • Evaluation using AUC, Precision, Recall, and F1-score  

    Random Forest was selected due to its robustness, interpretability, and suitability
    for imbalanced transaction data.
    """)

# ======================================
# MODEL PERFORMANCE
# ======================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("""
    **Evaluation Results (Test Set):**

    • AUC: ~0.99  
    • Precision (Fraud): 0.98  
    • Recall (Fraud): 0.88  
    • F1-score (Fraud): 0.93  

    High AUC indicates strong class separation, while recall ensures effective fraud detection.
    """)

# ======================================
# EXPLAINABILITY
# ======================================
elif page == "Explainability (SHAP)":
    st.title("Model Explainability (SHAP)")
    st.write("""
    SHAP (SHapley Additive exPlanations) was used to interpret the Random Forest model.

    Key fraud indicators identified:
    • Time difference between first and last transaction  
    • ERC20 transaction activity  
    • Ether value received and sent  
    • Unique interacting addresses  

    These features align with known Ethereum fraud behaviors.
    """)

# ======================================
# BIG DATA ARCHITECTURE
# ======================================
elif page == "Big Data Architecture":
    st.title("Big Data Architecture")
    st.write("""
    **System Architecture:**

    1. Data ingestion using Spark CSV reader  
    2. Distributed preprocessing and feature assembly  
    3. Spark ML Random Forest training  
    4. Evaluation and explainability (SHAP)  
    5. Deployment layer via Streamlit UI  

    This architecture supports scalability, transparency, and trust.
    """)

# ======================================
# LIMITATIONS & FUTURE WORK
# ======================================
elif page == "Limitations & Future Work":
    st.title("Limitations and Future Work")
    st.write("""
    • Dataset is pre-aggregated and may contain feature correlations  
    • No synthetic oversampling (SMOTE) was applied to avoid unrealistic transactions  
    • Future work could include:
      – Streaming data ingestion  
      – Advanced ensemble models  
      – Real-time fraud detection dashboards  
    """)
