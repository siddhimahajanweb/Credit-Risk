import streamlit as st
import pickle
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Credit Risk Assistant",
    page_icon="🏦",
    layout="wide"
)


# -----------------------------------
# LOAD MACHINE LEARNING MODEL
# -----------------------------------

with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)


# -----------------------------------
# LOAD RAG VECTORSTORE
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# -----------------------------------
# TITLE
# -----------------------------------

st.title("🏦 Credit Risk & Loan Approval Assistant")

st.write(
    "Predict loan approval using Machine Learning "
    "and retrieve credit-risk information using RAG."
)


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "Loan Prediction",
        "Credit Risk Assistant"
    ]
)


# -----------------------------------
# LOAN PREDICTION
# -----------------------------------

if page == "Loan Prediction":

    st.header("🏦 Loan Approval Prediction")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0
        )

    with col2:

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0
        )

        loan_term = st.number_input(
            "Loan Amount Term",
            min_value=0
        )

        credit_history = st.selectbox(
            "Credit History",
            [1, 0]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Rural", "Semiurban"]
        )

    predict_button = st.button(
    "🔍 Predict Loan Status"
)

if predict_button:

    # Convert user inputs to model format

    gender_value = 1 if gender == "Male" else 0

    married_value = 1 if married == "Yes" else 0

    dependents_value = 3 if dependents == "3+" else int(dependents)

    education_value = 1 if education == "Graduate" else 0

    self_employed_value = 1 if self_employed == "Yes" else 0

    property_area_value = {
        "Urban": 1,
        "Rural": 0,
        "Semiurban": 2
    }[property_area]

    # Create input DataFrame

    input_data = pd.DataFrame([[
        gender_value,
        married_value,
        dependents_value,
        education_value,
        self_employed_value,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area_value
    ]])

    # Make prediction

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # Display result

    if prediction == 1:

        st.success("✅ Loan is likely to be APPROVED")

    else:

        st.error("❌ Loan is likely to be REJECTED")

    st.write(
        f"Approval Probability: **{probability:.2%}**"
    )


# -----------------------------------
# RAG ASSISTANT
# -----------------------------------

else:

    st.header("🤖 Credit Risk Assistant")

    st.write(
        "Ask a question about loan approval, credit risk, "
        "eligibility, or related topics."
    )

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("🔎 Search Knowledge"):

        if question:

            results = vectorstore.similarity_search(
                question,
                k=3
            )

            st.subheader("📚 Retrieved Information")

            for i, result in enumerate(results):

                st.markdown(
                    f"### Result {i + 1}"
                )

                st.write(result.page_content)

        else:

            st.warning(
                "Please enter a question."
            )