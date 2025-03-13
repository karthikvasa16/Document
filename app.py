import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Set up Streamlit UI
st.title("Chat with Your Documents - RAG Application")

uploaded_files = st.file_uploader("Upload multiple documents (PDF, DOCX, TXT)", accept_multiple_files=True)

if uploaded_files:
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.error("Unsupported file format.")
            continue
        
        documents.extend(loader.load())
    
    st.success("Documents uploaded and processed successfully!")
    
    # Create embeddings and FAISS index
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    user_query = st.text_input("Ask a question about your documents:")
    if user_query:
        response = qa_chain.run(user_query)
        st.write("### Response:")
        st.write(response)
