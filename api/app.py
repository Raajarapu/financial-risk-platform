 import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify, render_template

from sklearn.base import BaseEstimator, TransformerMixin

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import pipeline


# ============================================================
# Financial Feature Engineering
# ============================================================

class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["Income_Per_Dependent"] = (
            X["Monthly_Income"] /
            (X["Number_of_Dependents"].fillna(0) + 1)
        )

        X["Loan_to_Income"] = (
            X["Loan_Amount"] /
            (X["Annual_Income"] + 1)
        )

        X["Debt_to_Income"] = (
            X["Total_Debt"] /
            (X["Annual_Income"] + 1)
        )

        X["Available_Credit"] = (
            X["Credit_Limit"] *
            (1 - X["Credit_Utilization"])
        )

        X["Monthly_Loan_Burden"] = (
            X["Loan_Amount"] /
            X["Loan_Term_Months"].clip(lower=1)
        )

        X["Delinquency_Score"] = (
            X["Times_30_59_Days_Past_Due"]
            + 2 * X["Times_60_89_Days_Past_Due"]
            + 3 * X["Times_90_Plus_Days_Past_Due"]
            + 4 * X["Times_120_Plus_Days_Past_Due"]
        )

        X["Credit_Utilization_Risk"] = (
            X["Credit_Utilization"] > 0.75
        ).astype(int)

        X["Recent_Late_Payment_Rate"] = (
            X["Late_Payment_Count"] /
            (X["Credit_History_Years"].fillna(0) + 1)
        )

        X["Credit_Account_Density"] = (
            X["Total_Credit_Accounts"] /
            (X["Credit_History_Years"].fillna(0) + 1)
        )

        return X


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "financial_risk_model.joblib"
)

RAG_PATH = os.path.join(
    BASE_DIR,
    "rag",
    "documents"
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "rag",
    "chroma_db"
)


# ============================================================
# Load ML Model
# ============================================================

artifact = joblib.load(
    MODEL_PATH
)

ml_pipeline = artifact["model"]


# ============================================================
# RAG Setup
# ============================================================

loader = DirectoryLoader(
    RAG_PATH,
    glob="*.txt"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

chunks = splitter.split_documents(
    documents
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)


def retrieve_context(query, k=4):

    docs = vector_db.similarity_search(
        query,
        k=k
    )

    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


# ============================================================
# Local LLM
# ============================================================

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=-1
)


def generate_response(prompt):

    result = llm(
        prompt,
        max_new_tokens=300,
        do_sample=False,
        return_full_text=False
    )

    return result[0]["generated_text"]


# ============================================================
# Flask
# ============================================================

app = Flask(
    __name__,
    template_folder="templates"
)


# ============================================================
# Web UI
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# Health Check
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "service": "financial-risk-api"
    })


# ============================================================
# Model Information
# ============================================================

@app.route("/model-info", methods=["GET"])
def model_info():

    return jsonify({
        "model": "RandomForest",
        "version": artifact["version"],
        "target": artifact["target"]
    })


# ============================================================
# Prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    applicant_df = pd.DataFrame(
        [data]
    )

    probability = ml_pipeline.predict_proba(
        applicant_df
    )[0, 1]

    risk_level = (
        "High"
        if probability >= 0.60
        else "Medium"
        if probability >= 0.30
        else "Low"
    )

    return jsonify({
        "risk_probability": round(
            float(probability),
            4
        ),
        "risk_level": risk_level
    })


# ============================================================
# Complete AI Analysis
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    # ========================================================
    # Required categorical fields
    # ========================================================
    # The web UI sends the core numeric applicant fields.
    # The trained ML pipeline also expects these fields.
    # Therefore, provide safe default values when they
    # are not supplied by the client.
    # ========================================================

    data.setdefault(
        "Marital_Status",
        "Single"
    )

    data.setdefault(
        "Education_Level",
        "Bachelor"
    )

    data.setdefault(
        "Employment_Status",
        "Employed"
    )

    data.setdefault(
        "Home_Ownership",
        "Rent"
    )

    data.setdefault(
        "Loan_Purpose",
        "Personal"
    )

    data.setdefault(
        "Region",
        "South"
    )

    # This feature is numeric in the trained model.
    data.setdefault(
        "Delinquency_History",
        0
    )

    data.setdefault(
        "Has_Employment_Income",
        1
    )

    applicant_df = pd.DataFrame(
        [data]
    )

    # ========================================================
    # ML Prediction
    # ========================================================

    risk_probability = ml_pipeline.predict_proba(
        applicant_df
    )[0, 1]

    risk_level = (
        "High"
        if risk_probability >= 0.60
        else "Medium"
        if risk_probability >= 0.30
        else "Low"
    )

    # ========================================================
    # RAG Retrieval
    # ========================================================

    context = retrieve_context(
        "loan default risk debt credit utilization delinquency",
        k=4
    )

    # ========================================================
    # LLM Prompt
    # ========================================================

    prompt = f"""
You are a financial risk analysis assistant.

Analyze the following loan applicant.

Applicant:
{data}

ML default risk probability:
{risk_probability:.2%}

Risk level:
{risk_level}

Retrieved financial knowledge:
{context}

Provide:

1. Risk Level
2. Key Risk Factors
3. Explanation of the ML prediction
4. Recommended Action

Use only information provided.
Do not invent applicant information.
Do not make a guaranteed lending decision.
"""

    # ========================================================
    # LLM Generation
    # ========================================================

    answer = generate_response(
        prompt
    )

    # ========================================================
    # Final Response
    # ========================================================

    return jsonify({
        "risk_probability": round(
            float(risk_probability),
            4
        ),
        "risk_level": risk_level,
        "analysis": answer
    })


# ============================================================
# Application Entry Point
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
