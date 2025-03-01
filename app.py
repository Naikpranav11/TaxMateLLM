import streamlit as st
from langchain_groq import ChatGroq
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
import concurrent.futures

# Initialize Streamlit app
st.title("💰 Tax-Consultant")

# Load and process PDF files from the Data folder
with st.expander("Q&A", expanded=True):
    def load_pdfs_from_folder(folder_path):
        documents = []
        pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
        
        def process_pdf(file_path):
            loader = PyPDFLoader(file_path)
            return loader.load()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_pdf, pdf_files))
            for result in results:
                documents.extend(result)
        
        return documents

    # Define data folder path
    data_folder = "Data"

    if os.path.exists(data_folder):
        documents = load_pdfs_from_folder(data_folder)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Larger chunks
            chunk_overlap=100  # Less overlap
        )
        texts = text_splitter.split_documents(documents)
        
        # Use a faster embedding model
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        
        # Use persistent vector storage
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = Qdrant.from_documents(
                documents=texts,
                embedding=embeddings,
                path="qdrant_db"  # Persistent storage
            )
        
        st.success("Tax documents loaded successfully!")
    else:
        st.error("Data folder not found! Please make sure the 'Data' folder with PDFs exists.")

    # Question input
    if 'vector_store' in st.session_state:
        question = st.text_input("Ask a tax-related question:")
        
        if question:
            # Use the hidden API key from Streamlit Secrets
            api_key = st.secrets["GROQ_API_KEY"]
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile", 
                temperature=0.7,
                api_key=api_key  # Provide the API key securely
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever()
            )
            
            with st.spinner("Generating answer..."):
                response = qa_chain.run(question)
                st.write("Answer:", response)