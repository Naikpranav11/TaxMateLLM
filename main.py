# import os
# from langchain.embeddings.base import Embeddings
# from langchain.document_loaders import PyMuPDFLoader  # Use this for loading PDFs
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.llms import GroqAPI  # Groq API LLM
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams

# # Define Groq Embeddings Class
# class GroqEmbeddings(Embeddings):
#     def embed_documents(self, texts):
#         # Implement your Groq embedding API call here
#         return [self.embed_query(text) for text in texts]

#     def embed_query(self, text):
#         # Replace this with Groq API's embedding endpoint call
#         import requests
#         response = requests.post(
#             "https://api.groq.com/embeddings",
#             json={"text": text},
#             headers={"Authorization": "Bearer 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"}
#         )
#         response.raise_for_status()
#         return response.json()["embedding"]

# # Set your API key for Groq
# os.environ["GROQ_API_KEY"] = "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"

# # Directory containing multiple tax documents (in PDF format)
# DATA_DIR = "Data"

# # Qdrant client initialization
# qdrant_client = QdrantClient(url="http://localhost:6333")  # Assuming Qdrant is running locally

# # Step 1: Load and process PDF documents
# def preprocess_documents(data_dir):
#     all_documents = []
#     for filename in os.listdir(data_dir):
#         if filename.endswith(".pdf"):  # Check if the file is a PDF
#             loader = PyMuPDFLoader(os.path.join(data_dir, filename))  # Load PDF using PyMuPDFLoader
#             documents = loader.load()  # Extract text from PDF
#             all_documents.extend(documents)  # Add extracted documents to the list
#     return all_documents

# # Step 2: Vectorize documents and store in Qdrant
# def create_vectorstore(documents, collection_name="tax_collection"):
#     # Split text into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = text_splitter.split_documents(documents)

#     # Create embeddings using Groq
#     embeddings = GroqEmbeddings()
#     vectors = [embeddings.embed_query(doc.page_content) for doc in docs]

#     # Create a new collection in Qdrant or overwrite an existing one
#     qdrant_client.recreate_collection(
#         collection_name=collection_name,
#         vector_params=VectorParams(size=len(vectors[0]), distance="Cosine")
#     )

#     # Upload vectors to Qdrant
#     qdrant_client.upload_collection(
#         collection_name=collection_name,
#         points=vectors,
#         payload=[{"text": doc.page_content} for doc in docs]
#     )

# # Step 3: Set up QA pipeline using Qdrant for retrieval
# def setup_qa_pipeline(collection_name="tax_collection"):
#     retriever = qdrant_client.as_retriever(collection_name=collection_name)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=GroqAPI(model="groq-chat-llm"),  # Replace with your Groq LLM model
#         retriever=retriever,
#         return_source_documents=True
#     )
#     return qa_chain

# if __name__ == "__main__":
#     # Preprocess and vectorize documents
#     print("Processing documents...")
#     documents = preprocess_documents(DATA_DIR)
#     create_vectorstore(documents)
#     print("Vectorstore created and stored in Qdrant!")

#     # Initialize QA pipeline
#     print("Setting up QA pipeline...")
#     qa_chain = setup_qa_pipeline()
#     print("QA pipeline ready!")
