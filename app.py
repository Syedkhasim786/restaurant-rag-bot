import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai

# Load Gemini API key
api_key = st.secrets["GEMINI_API_KEY"]

# Create Gemini client
client = genai.Client(api_key=api_key)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load PDF documents
def load_documents(folder_path):

    docs = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            reader = PdfReader(os.path.join(folder_path, file))

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    docs.append(text)

    return docs


# Build FAISS index
@st.cache_resource
def build_index(folder_path):

    docs = load_documents(folder_path)

    embeddings = embedding_model.encode(docs)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, docs


# Search function
def search(query, index, docs, k=3):

    query_vector = embedding_model.encode([query])

    distances, indices = index.search(np.array(query_vector), k)

    results = [docs[i] for i in indices[0]]

    return results


# Generate answer using Gemini
def generate_answer(query, context):

    prompt = f"""
You are a helpful restaurant assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly.
"""

    try:

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"Error from Gemini API: {str(e)}"


# Streamlit UI
st.title("🍽 Restaurant AI Bot")

folder_path = "restaurant_docs"

index, docs = build_index(folder_path)

query = st.text_input("Ask about the restaurant")

if query:

    results = search(query, index, docs)

    context = "\n".join(results)

    answer = generate_answer(query, context)

    st.write(answer)
    
