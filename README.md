# RAG Log Analysis Application

## 1. Project Overview

This project is a **Retrieval-Augmented Generation (RAG)** application designed to analyze log files from various sources. The system can intelligently answer questions about the logs by leveraging a Large Language Model (LLM) combined with a robust retrieval system. This submission demonstrates a strong foundation in building RAG pipelines, even though the core logic for providing consistently accurate answers is still under development.

The primary goal of this project was to build a system that can:
- Ingest and parse logs from multiple formats (JSONL, pretty-printed, Apache).
- Use a hybrid search mechanism (semantic + keyword) to find the most relevant log entries.
- Generate a structured, evidence-based answer to a user's query.

## 2. Architecture and Design Choices

The application is built using a modular architecture, with distinct components for each stage of the RAG pipeline. This design makes the system easy to understand, maintain, and extend.

### Key Components:

-   **LogProcessor**: Responsible for reading and parsing log files. It's designed to be resilient, handling multiple formats and ingesting unknown formats as generic messages to ensure no data is lost.
-   **RAGIngestor**: Transforms the parsed logs into a format suitable for the RAG pipeline. It creates descriptive, contextual chunks to improve retrieval accuracy and then ingests them into a ChromaDB vector store.
-   **RAGQueryEngine**: The core of the application. It uses a hybrid search approach, combining the strengths of semantic search (for understanding intent) and keyword search (for finding specific terms). This is achieved using LangChain's `EnsembleRetriever`.
-   **Structured Output**: The LLM is prompted to return a structured JSON object, which is validated using a Pydantic model. This ensures that the output is predictable and can be easily used by other systems.

### Technology Stack:

-   **Language**: Python
-   **Core Framework**: LangChain
-   **LLM**: Ollama (specifically `llama3.1:8b`)
-   **Vector Store**: ChromaDB
-   **Embeddings**: `nomic-embed-text`
-   **Keyword Search**: `rank-bm25`

## 3. How to Run the Application

### Prerequisites:

1.  **Python 3.8+**: Ensure you have a recent version of Python installed.
2.  **Ollama**: The application requires a running Ollama server. You can download it from [ollama.ai](https://ollama.ai/).
3.  **LLM and Embedding Models**: Before running the application, you need to pull the necessary models:
    ```bash
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    ```

### Installation:

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application:

1.  **Start the Ollama server**:
    ```bash
    ollama serve
    ```
2.  **Run the main script**:
    ```bash
    python main.py
    ```

The script will first perform a one-time ingestion of the log data from the `logs` directory into the ChromaDB vector store. On subsequent runs, it will use the existing database and proceed directly to the querying stage.

## 4. Current Status and Future Work

This project is a proof-of-concept and serves as a strong foundation for a more advanced log analysis tool. While the core components are in place, the system does not yet provide consistently accurate answers to all types of queries.

### Future Work:

-   **Improve Retrieval Accuracy**: The current hybrid search is a good start, but it could be improved by fine-tuning the retriever weights and experimenting with different chunking strategies.
-   **Enhance Log Parsing**: The `LogProcessor` could be extended to support more log formats and to extract more structured information from each log entry.
-   **Advanced Querying**: Implement a more sophisticated query parser that can handle complex questions and break them down into smaller, more manageable sub-queries.
-   **Frontend and API**: Build a user-friendly frontend and a RESTful API to make the application more accessible to users.
-   **Evaluation Framework**: Develop a comprehensive evaluation framework to measure the accuracy of the RAG pipeline and track improvements over time.

## 5. Conclusion

This project successfully demonstrates the core principles of building a RAG pipeline for a real-world use case. While there is room for improvement, the current implementation showcases a solid understanding of the technologies and a clear vision for future development.
