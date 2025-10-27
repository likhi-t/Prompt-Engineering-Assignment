"""
Main script for the RAG Log Analysis Application.

This script defines and orchestrates a Retrieval-Augmented Generation (RAG) pipeline
for analyzing log files. The process includes:
1.  **Configuration**: Sets up paths, models, and other constants.
2.  **Log Parsing**: The `LogProcessor` reads log files in various formats (JSONL,
    pretty-printed, Apache) and unifies them into a common structure.
3.  **RAG Ingestion**: The `RAGIngestor` takes the unified logs, creates descriptive
    text chunks for better retrieval, and ingests them into a ChromaDB vector store.
    It also creates a cache of documents for the keyword retriever.
4.  **Query Engine**: The `RAGQueryEngine` sets up a hybrid search retriever (semantic
    + keyword) and a question-answering chain using an Ollama LLM. It is
    designed to return structured JSON output.
5.  **Execution**: The main block (`if __name__ == '__main__':`) runs the ingestion
    process if no vector store exists and then executes a sample query.
"""

import os
import re
import json
import pickle
import traceback
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

# --- RAG/LLM Core Libraries ---
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Pydantic for Structured Output ---
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# --- LangChain Components ---
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# ====================================================================
# --- 1. CONFIGURATION ---
# ====================================================================

LOGS_DIR = Path('logs')
VECTOR_DB_PATH = "./chroma_log_db_v4"
DOCS_CACHE_PATH = "./rag_documents_v4.pkl"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
INGESTION_BATCH_SIZE = 500

# --- Pydantic Model for Structured Output ---
class StructuredLogAnalysis(BaseModel):
    """
    Defines the structured output for the log analysis LLM.
    This ensures the model's response is in a predictable, machine-readable format.
    """
    service: str = Field(description="The name of the service (pod) identified in the log entry.")
    node: Optional[str] = Field(description="The node where the event occurred, if available in the log.", default=None)
    evidence: str = Field(description="The complete, unmodified JSON log entry that definitively supports the conclusion.")
    conclusion: str = Field(description="A concise, one-sentence summary directly answering the user's question based on the evidence.")

# ====================================================================
# --- 2. LOG PARSING AND UNIFICATION ---
# ====================================================================

class LogProcessor:
    """
    Handles the ingestion and parsing of log files from various formats.

    This class reads all `.log` files from a specified directory, attempts to parse
    each line using a series of format-specific parsers, and unifies them into a
    list of dictionaries.
    """
    def __init__(self, log_directory: Path):
        """
        Initializes the LogProcessor.

        Args:
            log_directory (Path): The path to the directory containing log files.
        """
        self.log_directory = log_directory
        self.unified_logs: List[Dict[str, Any]] = []

    def _parse_jsonl_log(self, line: str) -> Union[Dict[str, Any], None]:
        """Parses a single line of a JSONL-formatted log."""
        try:
            data = json.loads(line)
            data['original_message'] = line.strip()
            return data
        except json.JSONDecodeError:
            return None

    def _parse_pretty_log(self, line: str) -> Union[Dict[str, Any], None]:
        """Parses a single line of a 'pretty' formatted log using regex."""
        match = re.search(r"\[(?P<level>\w+)\]\s+\[node:\s+(?P<node>[^\]]+)\]\s+(?P<message>.+)", line)
        if match:
            data = match.groupdict()
            data['original_message'] = line
            return data
        return None

    def _parse_apache_log(self, line: str) -> Union[Dict[str, Any], None]:
        """Parses a single line of a common Apache access log format."""
        match = re.search(r'"(?P<service>[^"]+)"', line)
        if match:
            data = match.groupdict()
            data['original_message'] = line.strip()
            return data
        return None

    def run_ingestion(self, limit: int = None):
        """
        Reads and processes all log files in the specified directory.

        It iterates through each file, line by line, and applies a series of
        parsers. If a line cannot be parsed by any specific method, it's ingested
        as a generic message to ensure no data is lost.

        Args:
            limit (int, optional): The maximum number of log entries to process.
                                   Defaults to None (no limit).
        """
        print(f"--- Starting log ingestion from '{self.log_directory}' ---")
        if not self.log_directory.exists():
            print(f"ERROR: Log directory '{self.log_directory}' not found.")
            return

        for log_file in self.log_directory.glob('*.log'):
            print(f"Processing file: {log_file.name}")
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if limit and len(self.unified_logs) >= limit:
                        print(f"Reached ingestion limit of {limit} records.")
                        break

                    # Try each parser in order until one succeeds
                    parsed_line = None
                    for parser in [self._parse_jsonl_log, self._parse_pretty_log, self._parse_apache_log]:
                        parsed_line = parser(line)
                        if parsed_line:
                            break

                    if parsed_line:
                        self.unified_logs.append(parsed_line)
                    else:
                        self.unified_logs.append({'original_message': line.strip()})

            if limit and len(self.unified_logs) >= limit:
                break

        print(f"✅ Total unified records: {len(self.unified_logs)}")

# ====================================================================
# --- 3. RAG INGESTION ---
# ====================================================================

class RAGIngestor:
    """
    Handles the transformation and ingestion of unified logs into the RAG system.

    This class takes the parsed log data, creates rich, descriptive text chunks
    to improve retrieval accuracy, and then ingests these chunks into a ChromaDB
    vector store.
    """
    def __init__(self, unified_logs: List[Dict[str, Any]]):
        """
        Initializes the RAGIngestor.

        Args:
            unified_logs (List[Dict[str, Any]]): A list of parsed log dictionaries.
        """
        self.unified_logs = unified_logs
        self.final_chunks: List[Dict[str, Any]] = []

    def create_contextual_chunks(self) -> List[Dict[str, Any]]:
        """
        Creates rich, descriptive text chunks from the unified logs.

        This process transforms structured log data into natural language sentences,
        which helps the semantic search model better understand the context of each
        log entry. The original log line is preserved in the metadata as 'evidence'.

        Returns:
            List[Dict[str, Any]]: A list of chunks, each with 'text' and 'metadata'.
        """
        self.final_chunks = []
        for log in self.unified_logs:
            evidence = log.get('original_message', '')
            readable_text = ""

            if 'lat_ms' in log:
                service = log.get('service', 'an unnamed service')
                readable_text = f"Latency event recorded for service '{service}' with a latency of {log['lat_ms']} milliseconds."
            elif log.get('level') == 'error':
                service = log.get('service', 'an unnamed service')
                node = f"on node '{log['node']}'" if 'node' in log else ""
                readable_text = f"An error event occurred for service '{service}' {node}. The message is: {log.get('message', '')}"
            else:
                readable_text = f"A general log event was recorded. The content is: {evidence}"

            metadata = {
                'req_id': log.get('req', 'N/A'),
                'node': log.get('node', 'N/A'),
                'evidence': evidence,
            }
            self.final_chunks.append({'text': readable_text, 'metadata': metadata})

        print(f"✅ Created {len(self.final_chunks)} descriptive contextual chunks for RAG.")
        return self.final_chunks

    def ingest(self):
        """
        Ingests the contextual chunks into the ChromaDB vector store.

        This method handles batching to efficiently process a large number of documents.
        It also saves the processed documents to a pickle file to be used by the
        keyword retriever.
        """
        if not self.final_chunks:
            print("🛑 No chunks to ingest. Run create_contextual_chunks first.")
            return

        print(f"\n--- Ingesting to ChromaDB with {EMBEDDING_MODEL} (Batching: {INGESTION_BATCH_SIZE}) ---")

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

        documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in self.final_chunks]

        # Batch processing for efficiency
        print(f"Preparing to ingest {len(documents)} documents...")
        for i in range(0, len(documents), INGESTION_BATCH_SIZE):
            batch = documents[i:i + INGESTION_BATCH_SIZE]
            if i == 0:
                vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=VECTOR_DB_PATH
                )
            else:
                vector_store.add_documents(documents=batch)
            print(f"✅ Ingested batch {i // INGESTION_BATCH_SIZE + 1}/{((len(documents) - 1) // INGESTION_BATCH_SIZE) + 1}")

        print("✅ Ingestion complete. Vector store saved.")

        # Cache documents for the keyword retriever
        with open(DOCS_CACHE_PATH, 'wb') as f:
            pickle.dump(documents, f)
        print(f"✅ Saved {len(documents)} documents to cache for keyword search at {DOCS_CACHE_PATH}")

# ====================================================================
# --- 4. RAG QUERY ENGINE ---
# ====================================================================

def format_docs(docs: List[Document]) -> str:
    """Helper function to format retrieved documents for the LLM prompt."""
    return "\n\n".join(doc.metadata.get('evidence', doc.page_content) for doc in docs)

def _sanitize_llm_output(text: str) -> str:
    """
    Extracts a valid JSON block from the LLM's raw output.

    This function is a safeguard against conversational filler or markdown
    formatting that the LLM might add to its response.
    """
    match = re.search(r"```(json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(2)
    return text

class RAGQueryEngine:
    """
    Manages the RAG query process, from retrieval to generation.

    This class initializes a hybrid search retriever (combining semantic and
    keyword search) and sets up a LangChain Expression Language (LCEL) chain
    to process user queries, retrieve relevant context, and generate a
    structured answer.
    """
    def __init__(self):
        """Initializes the RAG Query Engine."""
        print(f"\n--- Initializing RAG Query Engine with {LLM_MODEL} ---")
        self._check_ollama_connection()

        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        self.vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=self.embeddings)

        print("--- Setting up Hybrid Search (Semantic + Keyword) ---")
        self._setup_hybrid_retriever()
        self.qa_chain = self._setup_qa_chain()

    def _check_ollama_connection(self):
        """Performs a sanity check to ensure the Ollama server is accessible."""
        print(f"--- Checking connection to Ollama at {OLLAMA_BASE_URL} ---")
        try:
            requests.get(OLLAMA_BASE_URL, timeout=5)
            print("✅ Ollama connection successful.")
        except requests.RequestException as e:
            print("\n🛑 CRITICAL ERROR: Could not connect to the Ollama server.")
            raise ConnectionError("Ollama server not found.") from e

    def _setup_hybrid_retriever(self):
        """Sets up the hybrid retriever combining keyword and semantic search."""
        if not os.path.exists(DOCS_CACHE_PATH):
            raise FileNotFoundError(f"Cache not found at '{DOCS_CACHE_PATH}'. Run ingestion first.")

        with open(DOCS_CACHE_PATH, 'rb') as f:
            cached_docs = pickle.load(f)

        keyword_retriever = BM25Retriever.from_documents(cached_docs)
        keyword_retriever.k = 10
        semantic_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[keyword_retriever, semantic_retriever],
            weights=[0.7, 0.3]  # Prioritize keyword search
        )

    def _setup_qa_chain(self):
        """Builds the main LCEL question-answering chain."""
        llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
        parser = PydanticOutputParser(pydantic_object=StructuredLogAnalysis)

        QA_PROMPT_TEMPLATE = """
        You are a Log Analysis AI Assistant. Your ONLY output MUST be a single, valid JSON object.

        **CONTEXTUAL LOGS:**
        ---
        {context}
        ---

        USER QUESTION: {question}

        **Instructions for JSON Output Generation**:
        1.  Analyze the logs to find the evidence that answers the question.
        2.  Create a one-sentence conclusion. If no answer is found, state: "The answer could not be determined from the provided logs."
        3.  Extract entities as specified in the format instructions.
        4.  Your final JSON output MUST conform to the schema.

        {format_instructions}
        """
        qa_prompt = PromptTemplate.from_template(
            QA_PROMPT_TEMPLATE,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        return (
            {"context": self.ensemble_retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | RunnableLambda(_sanitize_llm_output)
            | parser
        )

    def query(self, question: str) -> str:
        """
        Executes a query against the RAG pipeline.

        Args:
            question (str): The user's question.

        Returns:
            str: A formatted string containing the structured analysis.
        """
        print(f"\n--- Running Query: {question} ---")
        result = self.qa_chain.invoke(question)

        if isinstance(result, StructuredLogAnalysis):
            node_info = f"**Node:** {result.node}\n" if result.node else ""
            return (f"**Service:** {result.service}\n"
                    f"{node_info}"
                    f"**Conclusion:** {result.conclusion}\n"
                    f"**--- Evidence Log Entry ---**\n"
                    f"{result.evidence}")
        return str(result)

# ====================================================================
# --- 5. MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    """
    Main execution block.

    This part of the script runs when executed directly. It first checks if a
    vector store exists. If not, it runs the full ingestion pipeline. Then,
    it initializes the query engine and runs a sample query.
    """
    print("--- RAG Log Analysis Application Start ---")
    print("Ensure Ollama is running (`ollama serve`) and models are pulled!")

    # Step 1: Ingest data if the vector store doesn't exist
    if not os.path.exists(VECTOR_DB_PATH):
        print("--- No existing vector store found. Starting ingestion process. ---")
        processor = LogProcessor(LOGS_DIR)
        processor.run_ingestion(limit=1000)

        ingestor = RAGIngestor(processor.unified_logs)
        chunks = ingestor.create_contextual_chunks()
        ingestor.ingest()
    else:
        print(f"--- Using existing vector store at {VECTOR_DB_PATH}. Skipping ingestion. ---")

    # Step 2: Initialize the query engine and run a query
    try:
        query_engine = RAGQueryEngine()
        test_query = "Which service has the highest latency?"
        response = query_engine.query(test_query)

        print("\n=============================================")
        print(f"QUESTION: {test_query}")
        print("=============================================")
        print(response.strip())
        print("=============================================")
    except Exception as e:
        print("\n🛑 CRITICAL QUERY ERROR:")
        traceback.print_exc()
        print("\nHINT: Ensure Ollama is running and initial ingestion was successful.")
