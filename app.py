import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Prevents TensorFlow-related import issues

import streamlit as st
import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(page_title="SHL GenAI Assessment Recommender", layout="wide")

# Load CSV data
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "dataset", "shl_catalog.csv")
    if not os.path.exists(csv_path):
        st.error("dataset/shl_catalog.csv not found. Please make sure the file exists.")
        st.stop()
    return pd.read_csv(csv_path)


# Load local embedding model
@st.cache_resource
def load_local_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Local embedding
def get_local_embedding(texts, model):
    return model.encode(texts)

# OpenAI embedding
def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Main app
def main():
    st.title("SHL GenAI Assessment Recommendation Tool")
    st.markdown("This tool recommends SHL assessments based on your job description using Retrieval-Augmented Generation (RAG).")

    # Sidebar
    st.sidebar.title("Settings")
    use_openai = st.sidebar.checkbox("Use OpenAI Embeddings (Needs API Key)")
    if use_openai:
        openai.api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    job_description = st.text_area("Paste the Job Description here:")

    df = load_data()
    st.subheader("Available SHL Assessments")
    st.dataframe(df.drop(columns=["url"]))

    if st.button("Recommend Assessments"):
        if not job_description.strip():
            st.warning("Please enter a job description first.")
            return

        with st.spinner("Analyzing and generating recommendations..."):
            corpus = df["description"].tolist()

            try:
                if use_openai and openai.api_key:
                    query_embedding = get_openai_embedding(job_description)
                    corpus_embeddings = [get_openai_embedding(desc) for desc in corpus]
                else:
                    model = load_local_model()
                    query_embedding = get_local_embedding([job_description], model)[0]
                    corpus_embeddings = get_local_embedding(corpus, model)
            except Exception as e:
                st.error(f" Embedding failed: {e}")
                return

            similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
            df["similarity"] = similarities
            top_matches = df.sort_values("similarity", ascending=False).head(3)

            st.subheader("Top Recommended Assessments")
            for _, row in top_matches.iterrows():
                st.markdown(f"### [{row['name']}]({row['url']})")
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Remote Testing:** {row['remote_testing']}")
                st.write(f"**Adaptive Support:** {row['adaptive_support']}")
                st.write(f"**Duration:** {row['duration']}")
                st.progress(float(row["similarity"]))

if __name__ == "__main__":
    main()
