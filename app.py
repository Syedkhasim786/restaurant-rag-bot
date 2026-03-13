import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai

# Gemini API
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


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
def build_index(folder_path):

    docs = load_documents(folder_path)

    embeddings = model.encode(docs)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, docs


# Search function
def search(query, index, docs, k=3):

    query_vector = model.encode([query])

    distances, indices = index.search(np.array(query_vector), k)

    results = [docs[i] for i in indices[0]]

    return results


# Generate answer from Gemini
def generate_answer(query, context):

    prompt = f"""
You are a helpful restaurant assistant.

Restaurant information:
{context}

Customer question:
{query}

Give a clear and friendly answer.
"""

    response = client.models.generate_content(
    model="gemini-1.5-flash-latest"
        contents=prompt
    )

    return response.text


# Streamlit UI
st.title("🍽 Restaurant AI Bot")

folder_path = "restaurant_docs"

index, docs = build_index(folder_path)

query = st.text_input("Ask about the restaurant")

if query:

    results = search(query, index, docs)

    context = "\n".join(results)

    answer = generate_answer(query, context)

    st.write("🤖 Bot:", answer)
