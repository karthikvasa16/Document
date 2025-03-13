from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            loader = PyPDFLoader(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(file)
        elif file.type == "text/plain":
            loader = TextLoader(file)
        else:
            raise ValueError(f"Unsupported file type: {file.type}")
        documents.extend(loader.load())
    return documents