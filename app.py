import streamlit as st
import joblib
import pandas as pd
# ==========================
# LOAD TRAINED MODEL
# ==========================
model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)
# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("📊 Customer Churn Prediction")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 About Project")

st.sidebar.write("""
This application predicts whether a telecom customer is likely to churn using Machine Learning.
""")

st.sidebar.markdown("### 🤖 Best Model")
st.sidebar.success("Random Forest")

st.sidebar.markdown("### 🎯 Model Accuracy")
st.sidebar.info("≈ 80%")

st.sidebar.markdown("### 🛠 Technologies")

st.sidebar.write("""
- Python
- Pandas
- Scikit-learn
- Streamlit
- Random Forest
""")

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Mohana Sushma")

# ==========================
# TITLE
# ==========================
st.title("📊 Customer Churn Prediction Dashboard")
st.caption(
    "Predict customer churn using a Machine Learning model trained on the Telco Customer Churn dataset."
)
st.markdown("""
Predict whether a telecom customer is likely to churn based on customer demographics,
service usage, and account information.

Fill in the details below and click **Predict Churn**.
""")


st.divider()

# ==========================
# CUSTOMER INFORMATION
# ==========================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col2:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )
# ==========================
# SERVICE INFORMATION
# ==========================

st.divider()
st.subheader("📶 Service Information")

col1, col2 = st.columns(2)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )
# ==========================
# ACCOUNT INFORMATION
# ==========================

st.divider()
st.subheader("💳 Account Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=850.0
    )
st.divider()

if st.button("🔍 Predict Customer Churn"):

    # Create a dataframe with all features initialized to 0
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    st.write(input_df.dtypes)
    st.write(type(monthly_charges), monthly_charges)
    # ==========================
    # Binary Features
    # ==========================

    input_df.loc[0, "SeniorCitizen"] = 1 if senior == "Yes" else 0
    input_df.loc[0, "Partner"] = 1 if partner == "Yes" else 0
    input_df.loc[0, "Dependents"] = 1 if dependents == "Yes" else 0
    input_df.loc[0, "PhoneService"] = 1 if phone_service == "Yes" else 0
    input_df.loc[0, "PaperlessBilling"] = 1 if paperless_billing == "Yes" else 0

    # Numerical Features
    try:
        input_df.loc[0, "tenure"] = tenure
        st.success("✅ tenure assigned")

        input_df.loc[0, "MonthlyCharges"] = monthly_charges
        st.success("✅ MonthlyCharges assigned")

        input_df.loc[0, "TotalCharges"] = total_charges
        st.success("✅ TotalCharges assigned")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Gender
    if gender == "Male":
        input_df.loc[0, "gender_Male"] = 1

    # Multiple Lines
    if multiple_lines == "Yes":
        input_df.loc[0, "MultipleLines_Yes"] = 1
    elif multiple_lines == "No phone service":
        input_df.loc[0, "MultipleLines_No phone service"] = 1

    # Internet Service
    if internet_service == "Fiber optic":
        input_df.loc[0, "InternetService_Fiber optic"] = 1
    elif internet_service == "No":
        input_df.loc[0, "InternetService_No"] = 1

    # Online Security
    if online_security == "Yes":
        input_df.loc[0, "OnlineSecurity_Yes"] = 1
    elif online_security == "No internet service":
        input_df.loc[0, "OnlineSecurity_No internet service"] = 1

    # Online Backup
    if online_backup == "Yes":
        input_df.loc[0, "OnlineBackup_Yes"] = 1
    elif online_backup == "No internet service":
        input_df.loc[0, "OnlineBackup_No internet service"] = 1

    # Device Protection
    if device_protection == "Yes":
        input_df.loc[0, "DeviceProtection_Yes"] = 1
    elif device_protection == "No internet service":
        input_df.loc[0, "DeviceProtection_No internet service"] = 1

    # Tech Support
    if tech_support == "Yes":
        input_df.loc[0, "TechSupport_Yes"] = 1
    elif tech_support == "No internet service":
        input_df.loc[0, "TechSupport_No internet service"] = 1

    # Streaming TV
    if streaming_tv == "Yes":
        input_df.loc[0, "StreamingTV_Yes"] = 1
    elif streaming_tv == "No internet service":
        input_df.loc[0, "StreamingTV_No internet service"] = 1

    # Streaming Movies
    if streaming_movies == "Yes":
        input_df.loc[0, "StreamingMovies_Yes"] = 1
    elif streaming_movies == "No internet service":
        input_df.loc[0, "StreamingMovies_No internet service"] = 1

    # Contract
    if contract == "One year":
        input_df.loc[0, "Contract_One year"] = 1
    elif contract == "Two year":
        input_df.loc[0, "Contract_Two year"] = 1

    # Payment Method
    if payment_method == "Credit card (automatic)":
        input_df.loc[0, "PaymentMethod_Credit card (automatic)"] = 1
    elif payment_method == "Electronic check":
        input_df.loc[0, "PaymentMethod_Electronic check"] = 1
    elif payment_method == "Mailed check":
        input_df.loc[0, "PaymentMethod_Mailed check"] = 1

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # Get churn probability FIRST
    churn_probability = probability[1]
    # Calculate Risk Level
    if churn_probability < 0.30:
        risk_level = "🟢 Low"
    elif churn_probability < 0.70:
        risk_level = "🟡 Medium"
    else:
        risk_level = "🔴 High"
    st.divider()
    st.subheader("📊 Prediction Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.metric("Customer Status", "Churn")
        else:
            st.metric("Customer Status", "Stay")

    with col2:
        st.metric(
            "Churn Probability",
            f"{churn_probability*100:.2f}%"
        )

    with col3:
        if churn_probability < 0.30:
            st.metric("Risk Level", "🟢 Low")
        elif churn_probability < 0.70:
            st.metric("Risk Level", "🟡 Medium")
        else:
            st.metric("Risk Level", "🔴 High")

    # Final message
    if prediction == 1:
        st.error("🚨 Customer is Likely to Churn")
    else:
        st.success("🎉 Customer is Likely to Stay")

    # Confidence section
    st.subheader("📈 Prediction Confidence")
    st.progress(float(churn_probability))
    st.write(f"**Churn Probability:** {churn_probability*100:.2f}%")
     
    # ==========================
    # RECOMMENDATION
    # ==========================
    if risk_level == "🟢 Low":

        recommendation = """
    Customer has a low probability of churning.

    Continue providing quality service and maintain customer satisfaction.
    """

    elif risk_level == "🟡 Medium":

        recommendation = """
    Customer has a moderate chance of churning.

    Consider offering personalized discounts or loyalty rewards.
    """

    else:

        recommendation = """
    Customer has a high probability of churning.

    Immediate customer retention strategies are recommended.
    """
    st.subheader("💡Business Recommendation")

    st.info(recommendation)
    # ==========================
    # POSSIBLE REASONS FOR CHURN
    # ==========================

    st.subheader("🔍 Possible Reasons for Churn")

    reasons = []

    # Contract
    if contract == "Month-to-month":
        reasons.append("📄 Customer has a Month-to-month contract, which is associated with higher churn.")

    # Internet Service
    if internet_service == "Fiber optic":
        reasons.append("🌐 Customer uses Fiber Optic Internet, where churn is generally higher.")

    # Monthly Charges
    if monthly_charges > 80:
        reasons.append("💰 Monthly charges are relatively high.")

    # Tech Support
    if tech_support == "No":
        reasons.append("🛠 Customer does not have Tech Support.")

    # Online Security
    if online_security == "No":
        reasons.append("🔒 Customer does not have Online Security.")

    # Tenure
    if tenure < 12:
        reasons.append("📅 Customer has been with the company for less than one year.")

    # Display reasons
    if reasons:
        for reason in reasons:
            st.write(reason)
    else:
        st.success("✅ No major churn indicators were identified based on the provided information.")
    # ==========================
    # RETENTION STRATEGIES
    # ==========================

    st.subheader("💼 Suggested Retention Strategies")

    strategies = []

    if contract == "Month-to-month":
        strategies.append("✅ Encourage the customer to switch to a long-term contract.")

    if monthly_charges > 80:
        strategies.append("💰 Consider offering a discount or promotional plan.")

    if tech_support == "No":
        strategies.append("🛠 Provide a free trial of Technical Support.")

    if online_security == "No":
        strategies.append("🔒 Offer Online Security as a value-added service.")

    if tenure < 12:
        strategies.append("🎁 Provide a loyalty reward for new customers.")

    if strategies:
        for strategy in strategies:
            st.write(strategy)
    else:
        st.success("🎉 Continue maintaining customer satisfaction with regular engagement.")
    st.divider()

    st.caption(
        "This prediction is generated using a Random Forest machine learning model. "
        "It is intended to support decision-making and should not replace business judgment."
    )   
    # ==========================
    # PREDICTION SUMMARY
    # ==========================

    st.divider()
    st.subheader("📋 Prediction Summary")

    status = "Churn" if prediction == 1 else "Stay"

    st.write(f"**Customer Status:** {status}")
    st.write(f"**Risk Level:** {risk_level}")
    st.write(f"**Churn Probability:** {churn_probability*100:.2f}%")

    # Explain why the customer is likely to stay
    if prediction == 0:

        st.subheader("✅ Why this customer is likely to stay")

        stay_reasons = []

        if contract != "Month-to-month":
            stay_reasons.append("📄 Customer has a long-term contract.")

        if monthly_charges <= 80:
            stay_reasons.append("💰 Monthly charges are within a reasonable range.")

        if tech_support == "Yes":
            stay_reasons.append("🛠 Customer has Technical Support.")

        if online_security == "Yes":
            stay_reasons.append("🔒 Customer has Online Security.")

        if tenure >= 12:
            stay_reasons.append("📅 Customer has been with the company for a long time.")

        if stay_reasons:
            for reason in stay_reasons:
                st.write(reason)

        else:
            st.write("Customer shows several characteristics associated with long-term retention.")