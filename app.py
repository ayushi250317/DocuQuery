import streamlit as st
from embeddings import vector_embedding_for_pdf, vector_embedding_for_text, vector_embedding_for_web
from config import load_environment, llm, prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

# Load environment variables
load_environment()

# Sidebar for input type selection
st.sidebar.title("Options")
option = st.sidebar.selectbox("Select Input Type", ("Text File", "URL", "PDF"))

if option == "Text File":
    text_file = st.sidebar.file_uploader("Upload Text File", type=["txt"])
    if text_file is not None:
        if st.sidebar.button("Submit Text"):
            vector_embedding_for_text(text_file)
            st.sidebar.success("Text file processed and embedded.")

elif option == "URL":
    url = st.sidebar.text_input("Enter URL")
    if st.sidebar.button("Submit URL"):
        vector_embedding_for_web(url)
        st.sidebar.success("URL processed and embedded.")

elif option == "PDF":
    pdf_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
    if pdf_file is not None:
        if st.sidebar.button("Submit PDF"):
            vector_embedding_for_pdf(pdf_file)
            st.sidebar.success("PDF file processed and embedded.")

# Main page for question input and results
st.title("DocuQuery: Intelligent Document Q&A")
st.markdown("### Provide your question below and get the answer based on the uploaded documents or URL.")
prompt1 = st.text_input("Input your prompt here")

# Question answering based on the prompt and selected input
if st.button("Submit Prompt") and prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.markdown("#### Answer")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.error("Please embed documents first.")
