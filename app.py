import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai

# --- 1. INITIALIZATION ---
@st.cache_resource
def get_client():
    # Make sure 'GEMINI_API_KEY' is set in your Streamlit/GitHub secrets
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

client = get_client()
embedding_model = get_embedding_model()

# --- 2. DOCUMENT PROCESSING ---
def load_documents(folder_path):
    docs = []
    if not os.path.exists(folder_path):
        st.error(f"Folder '{folder_path}' not found. Create it and add your PDFs.")
        return []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                text = page.extract_text()
                if text: docs.append(text)
    return docs

@st.cache_resource
def build_index(folder_path):
    docs = load_documents(folder_path)
    if not docs: return None, []
    embeddings = embedding_model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, docs

# --- 3. CORE LOGIC ---
def search(query, index, docs, k=3):
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    return [docs[i] for i in indices[0]]

def generate_answer(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer clearly based only on the context."
    try:
        # UPDATED MODEL FOR 2026
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- 4. UI ---
st.set_page_config(page_title="Restaurant Bot", page_icon="🍽")
st.title("🍽 Restaurant AI Assistant")

index, docs = build_index("restaurant_docs")

if index:
    # FORM PREVENTS 429 ERRORS BY WAITING FOR BUTTON CLICK
    with st.form("chat_form"):
        user_query = st.text_input("Ask a question about the menu or services:")
        submitted = st.form_submit_button("Ask")

    if submitted and user_query:
        with st.spinner("Consulting documents..."):
            context_results = search(user_query, index, docs)
            context_text = "\n".join(context_results)
            answer = generate_answer(user_query, context_text)
            st.markdown("### Answer")
            st.write(answer)
else:
    st.info("Upload PDF files to the 'restaurant_docs' folder to begin.")
