# TaxConsultantLLM

## Overview
The Tax Q&A Bot is a Streamlit-based application that utilizes LangChain and Groq-powered LLMs to provide intelligent answers to tax-related questions. By processing tax documents in PDF format, the bot creates a searchable knowledge base, enabling users to interact with it seamlessly.

---

## Features
1. **Document Ingestion**:
   - Automatically loads and processes PDF documents from the `Data` folder.
   - Splits documents into manageable chunks for effective querying.

2. **Persistent Vector Storage**:
   - Uses Qdrant for storing vectorized document embeddings persistently.

3. **Fast and Efficient Embeddings**:
   - Leverages the `intfloat/multilingual-e5-small` Hugging Face model for embedding generation.

4. **Question Answering**:
   - Integrates Groq-powered LLMs for precise and context-aware answers to tax-related queries.

---

## Requirements

### Python Dependencies
- `streamlit`
- `langchain`
- `langchain-groq`
- `qdrant-client`
- `huggingface-hub`
- `pypdf2`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `Data` folder in the root directory and add PDF files with tax-related information.

4. Set up Streamlit secrets for the Groq API key. Add a `secrets.toml` file under the `.streamlit` folder:
   ```toml
   [secrets]
   GROQ_API_KEY = "your_api_key_here"
   ```

### Run the Application
Start the Streamlit app:
```bash
streamlit run app.py
```

---

## Code Breakdown

### 1. **Initialization**
The app starts by setting up a Streamlit interface:
```python
st.title("💰 Tax Q&A Bot (Groq-powered)")
```

### 2. **Document Loading**
The `load_pdfs_from_folder` function processes all PDFs in the `Data` folder:
```python
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
```

### 3. **Document Splitting**
Documents are split into chunks using the `RecursiveCharacterTextSplitter`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Larger chunks
    chunk_overlap=100  # Less overlap
)
texts = text_splitter.split_documents(documents)
```

### 4. **Embeddings and Vector Store**
Embeddings are generated and stored in Qdrant:
```python
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = Qdrant.from_documents(
        documents=texts,
        embedding=embeddings,
        path="qdrant_db"  # Persistent storage
    )
```

### 5. **Question Answering**
User questions are processed, and answers are generated:
```python
if 'vector_store' in st.session_state:
    question = st.text_input("Ask a tax-related question:")
    
    if question:
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
```

---

## Usage Instructions
1. **Upload Tax Documents**:
   Place relevant PDFs in the `Data` folder.

2. **Ask Questions**:
   Type your tax-related question in the input field and press Enter.

3. **View Results**:
   The bot will generate and display the answer based on the uploaded documents.

---

## FAQ

### 1. What type of documents are supported?
Currently, only PDF files are supported. Ensure the files contain clear and structured tax information.

### 2. Can the bot process large documents?
Yes, documents are split into chunks to enable efficient processing and querying.

### 3. How is data stored?
The vectorized data is stored persistently using Qdrant, allowing quick retrieval even after restarting the app.

### 4. What is the purpose of Groq-powered LLM?
The Groq-powered LLM provides high-quality, contextually accurate answers by leveraging advanced natural language processing capabilities.

---

## Future Enhancements
- Add support for additional document formats.
- Enable multi-language question answering.
- Integrate advanced summarization capabilities.
- Enhance user interface for better accessibility and experience.

---

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Description of feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Support
For any questions or issues, feel free to open an issue in the repository or contact the project maintainer.


