import streamlit as st
from langchain_helper import get_qa_chain

st.title("Rent A Car")

question = st.text_input("Ask a question: ")
if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])