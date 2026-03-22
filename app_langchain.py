import streamlit as st
import os

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Gemini
from google import genai

# -------------------------------
# 1. INITIALIZATION
# -------------------------------
@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

client = get_client()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# 2. LOAD DOCUMENTS
# -------------------------------
def load_documents(folder_path):
    all_docs = []

    if not os.path.exists(folder_path):
        st.error(f"Folder '{folder_path}' not found.")
        return []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            all_docs.extend(docs)

    return all_docs

# -------------------------------
# 3. SPLIT DOCUMENTS
# -------------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

# -------------------------------
# 4. BUILD VECTOR STORE
# -------------------------------
def build_vectorstore(folder_path):
    documents = load_documents(folder_path)

    if not documents:
        return None

    split_docs = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore

# -------------------------------
# 5. RETRIEVE DOCUMENTS
# -------------------------------
def retrieve_docs(query, vectorstore):
    docs = vectorstore.similarity_search(query, k=6)

    if "drink" in query.lower():
        extra_docs = vectorstore.similarity_search("drinks beverages cool drinks", k=4)
        docs.extend(extra_docs)

    return docs

# -------------------------------
# 6. GENERATE ANSWER
# -------------------------------
def generate_answer(query, context):
    if "drink" in query.lower() and "drink" not in context.lower():
        return """
Drinks are not clearly listed in the menu.

However, we typically offer:
- Soft drinks (Coke, Pepsi, Sprite)
- Tea & Coffee
- Fresh juices
"""

    prompt = f"""
You are a helpful restaurant assistant.

Rules:
- Answer from the context
- If not found, give a helpful suggestion

Context:
{context}

Question:
{query}

Answer clearly:
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# 7. STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Restaurant Bot", page_icon="🍽")
st.title("🍽 Restaurant AI Assistant (LangChain Version)")

# Filters
food_type = st.selectbox("Select food type", ["All", "Veg", "Non-Veg"])
category = st.selectbox("Select category", ["All", "Starters", "Main Course", "Desserts", "Drinks", "Services"])

# Chat history display
st.markdown("### 💬 Chat History")
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")

vectorstore = build_vectorstore("restaurant_docs")

if vectorstore:
    with st.form("chat_form"):
        user_query = st.text_input("Ask a question about the menu or services:")
        submitted = st.form_submit_button("Ask")

    if submitted and user_query:
        # Apply filters
        if food_type != "All":
            user_query = f"{food_type} {user_query}"
        if category != "All":
            user_query = f"{category} {user_query}"

        user_query = user_query + " restaurant menu services food items"

        # ✅ FIXED INDENTATION HERE
        with st.spinner("Searching and generating answer..."):
            docs = retrieve_docs(user_query, vectorstore)

            context = "\n".join([doc.page_content for doc in docs])

            # Debug
            print("USER QUERY:", user_query)
            print("CONTEXT:", context)

            answer = generate_answer(user_query, context)

            # Save chat
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("Bot", answer))

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            for doc in docs:
                source = doc.metadata.get("source", "Unknown file")
                page = doc.metadata.get("page", "N/A")
                st.write(f"{source} - Page {page}")

else:
    st.info("Please add PDF files to the 'restaurant_docs' folder.")
