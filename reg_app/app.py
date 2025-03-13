import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv
import os
import tempfile


load_dotenv()

# Set up the Streamlit app title
st.title("Retrieval Augmented Generation (RAG) Application with Gemini")
st.sidebar.title("Document Chatbot")
st.sidebar.write("Upload documents and chat with them using RAG and Gemini.")


uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Configure Gemini API
genai.configure(api_key=os.getenv("AIzaSyBNMBgfQrvWGo-TR8b1k8nQfZDu_NgydC0"))  # Load API key from .env file

# Function to save uploaded files to temporary files
def save_uploaded_files(uploaded_files):
    temp_files = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            temp_file.write(file.getvalue())
            temp_files.append(temp_file.name)
    return temp_files

# Function to load and split documents
def load_and_split_documents(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.error(f"Unsupported file type: {file_path}")
            continue
        documents.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Function to create embeddings and vector store
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use Hugging Face embeddings
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Function to retrieve relevant passages
def retrieve_relevant_passages(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 passages
    relevant_docs = retriever.get_relevant_documents(query)
    return relevant_docs

# Function to generate a response using Gemini
def generate_response(query, relevant_docs):
    try:
        # Combine retrieved passages into a single context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Use Gemini to generate a response
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to handle the chat interface
def chat_interface(vector_store):
    st.header("Chat with Your Documents")
    query = st.text_input("Ask a question about the documents:")
    
    if query:
        # Retrieve relevant passages
        relevant_docs = retrieve_relevant_passages(vector_store, query)
        
        # Display relevant passages
        st.write("**Relevant Passages:**")
        for doc in relevant_docs:
            st.write(doc.page_content)
        
        # Generate a response using Gemini
        response = generate_response(query, relevant_docs)
        
        # Display the response
        st.write("**Response:**")
        st.write(response)

# Main function to run the app
def main():
    if uploaded_files:
        # Save uploaded files to temporary files
        temp_file_paths = save_uploaded_files(uploaded_files)
        
        # Load and split documents
        texts = load_and_split_documents(temp_file_paths)
        st.success(f"Loaded and split {len(texts)} document chunks.")

        # Create embeddings and vector store
        vector_store = create_vector_store(texts)
        st.success("Created embeddings and vector store.")

        # Display the chat interface
        chat_interface(vector_store)
    else:
        st.info("Please upload documents to get started.")

# Run the app
if __name__ == "__main__":
    main()