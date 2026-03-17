import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai

# --- 1. CACHED INITIALIZATION ---
# Using @st.cache_resource ensures these only load once per session
@st.cache_resource
def get_gemini_client():
    api_key = st.secrets["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

client = get_gemini_client()
embedding_model = get_embedding_model()

# --- 2. DOCUMENT LOADING ---
def load_documents(folder_path):
    docs = []
    if not os.path.exists(folder_path):
        st.error(f"Folder '{folder_path}' not found.")
        return []
        
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    docs.append(text)
    return docs

# --- 3. INDEXING ---
@st.cache_resource
def build_index(folder_path):
    docs = load_documents(folder_path)
    if not docs:
        return None, []
        
    embeddings = embedding_model.encode(docs)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, docs

# --- 4. SEARCH & GENERATION ---
def search(query, index, docs, k=3):
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    results = [docs[i] for i in indices[0]]
    return results

# @st.cache_data prevents hitting the API for the exact same question twice
@st.cache_data
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

# --- 5. STREAMLIT UI ---
st.title("🍽 Restaurant AI Bot")

folder_path = "restaurant_docs"
index, docs = build_index(folder_path)

if index is None:
    st.warning("Please add PDF files to the 'restaurant_docs' folder to start.")
else:
    # Use a form to prevent API calls on every keystroke
    with st.form("query_form"):
        query = st.text_input("Ask about the restaurant (e.g., 'What is on the menu?')")
        submit_button = st.form_submit_button("Ask Gemini")

    if submit_button and query:
        with st.spinner("Searching documents and generating answer..."):
            results = search(query, index, docs)
            context = "\n".join(results)
            answer = generate_answer(query, context)
            
            st.markdown("### Answer:")
            st.write(answer)
