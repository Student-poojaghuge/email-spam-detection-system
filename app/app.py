import streamlit as st
import joblib
import re
import os
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

model_path = os.path.join(
    BASE_DIR,
    "models",
    "spam_model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR,
    "models",
    "vectorizer.pkl"
)

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


# ==========================================
# NLP PREPROCESSING
# ==========================================

stop_words = set(
    stopwords.words("english")
)

stemmer = PorterStemmer()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


# ==========================================
# HEADER
# ==========================================

st.title("📧 Spam Detection System")

st.write(
    "Machine Learning + NLP based system "
    "for detecting spam messages."
)

st.divider()


# ==========================================
# SINGLE MESSAGE PREDICTION
# ==========================================

st.subheader("🔍 Check a Message")

message = st.text_area(
    "Enter your message",
    height=160,
    placeholder=(
        "Example: Congratulations! "
        "You have won a free prize..."
    )
)


if st.button(
    "🚀 Check Message",
    use_container_width=True
):

    if not message.strip():

        st.warning(
            "Please enter a message."
        )

    else:

        # Preprocess
        cleaned_message = preprocess_text(
            message
        )

        # TF-IDF
        message_tfidf = vectorizer.transform(
            [cleaned_message]
        )

        # Prediction
        prediction = model.predict(
            message_tfidf
        )[0]

        probability = model.predict_proba(
            message_tfidf
        )[0]

        spam_probability = (
            probability[1] * 100
        )

        ham_probability = (
            probability[0] * 100
        )


        # ==================================
        # RESULT
        # ==================================

        if prediction == 1:

            st.error(
                "🚨 SPAM MESSAGE"
            )

        else:

            st.success(
                "✅ NOT SPAM"
            )


        # ==================================
        # PROBABILITY
        # ==================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Ham Probability",
                f"{ham_probability:.2f}%"
            )


        st.progress(
            int(spam_probability)
        )


        # ==================================
        # MESSAGE STATISTICS
        # ==================================

        st.subheader(
            "📊 Message Statistics"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Characters",
                len(message)
            )

        with col2:

            st.metric(
                "Words",
                len(message.split())
            )

        with col3:

            st.metric(
                "Processed Words",
                len(cleaned_message.split())
            )


# ==========================================
# BATCH CSV PREDICTION
# ==========================================

st.divider()

st.subheader(
    "📂 Batch Prediction"
)

st.write(
    "Upload a CSV file containing a column named "
    "`message`."
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(
        uploaded_file
    )

    if "message" not in data.columns:

        st.error(
            "CSV must contain a 'message' column."
        )

    else:

        cleaned_messages = data[
            "message"
        ].astype(str).apply(
            preprocess_text
        )

        tfidf_data = vectorizer.transform(
            cleaned_messages
        )

        predictions = model.predict(
            tfidf_data
        )

        probabilities = model.predict_proba(
            tfidf_data
        )[:, 1] * 100

        data["prediction"] = [
            "Spam" if p == 1
            else "Not Spam"
            for p in predictions
        ]

        data["spam_probability"] = (
            probabilities.round(2)
        )

        st.dataframe(
            data,
            use_container_width=True
        )

        csv = data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Results",
            csv,
            "spam_predictions.csv",
            "text/csv"
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Built using Python, NLP, TF-IDF, "
    "Machine Learning and Streamlit."
)