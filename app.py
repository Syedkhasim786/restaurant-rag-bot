import streamlit as st
import os

st.title("🍽 Restaurant AI Bot")

query = st.text_input("Ask about the restaurant")

if query:
    results = search(query, index, docs)
    context = "\n".join(results)

    answer = generate_answer(query, context)

    st.write("🤖 Bot:", answer)
