import os
import sys
import logging
from pathlib import Path
import re
import textwrap
from dotenv import load_dotenv
from typing import Union # <--- IMPORT Union for older Python versions

# MCP Imports
from mcp.server.fastmcp import FastMCP

# IBM Watsonx.ai SDK Imports
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Vector DB Imports
import chromadb
from pypdf import PdfReader

# --- Configuration and Setup ---

# Load .env variables
load_dotenv()

# Watsonx Credentials and Config
API_KEY    = os.getenv("WATSONX_APIKEY")
URL        = os.getenv("WATSONX_URL")
PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_ID   = os.getenv("MODEL_ID", "ibm/granite-13b-instruct-v2")

# Vector DB Config
DOCS_FOLDER = Path(os.getenv("DOCS_FOLDER", "documents"))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_data"))
COLLECTION_NAME = "pdf_documents_chunked_watsonx"
NUM_RESULTS_RAG = 3
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

# --- Global Variables (Using typing.Union for compatibility) ---
# Holds the initialized ChromaDB collection
pdf_collection: Union[chromadb.Collection, None] = None # <--- FIX: Use Union
# Holds the persistent ChromaDB client instance
chroma_client: Union[chromadb.PersistentClient, None] = None # <--- FIX: Use Union

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Environment Variable Validation ---
for name, val in [
    ("WATSONX_APIKEY", API_KEY),
    ("WATSONX_URL", URL),
    ("PROJECT_ID", PROJECT_ID)
]:
    if not val:
        raise RuntimeError(f"{name} is not set. Please add it to your .env file.")

# --- Vector DB Helper Functions (Unchanged: pdf_to_text, chunk_text) ---

def pdf_to_text(folder: Path) -> dict:
    """Extract text from every PDF in `folder`, preserving structure."""
    texts = {}
    if not folder.is_dir():
        logging.error("Folder not found: %s", folder)
        return texts
    logging.info("Scanning folder '%s' for PDFs...", folder)
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        logging.warning("No PDF files found in folder '%s'.", folder)
        return texts
    for pdf_path in pdf_files:
        try:
            logging.info("Attempting to read %s", pdf_path.name)
            reader = PdfReader(str(pdf_path))
            if reader.is_encrypted:
                 logging.warning("PDF is encrypted: %s. Attempting default decryption.", pdf_path.name)
                 try: reader.decrypt(''); logging.info("Successfully decrypted %s.", pdf_path.name)
                 except Exception as de: logging.error("Failed decrypt %s: %s.", pdf_path.name, de); continue
            # Combine text extraction and joining for brevity
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
            # Apply regex substitutions efficiently
            content = re.sub(r'\n{3,}', '\n\n', content.strip())
            if content:
                texts[pdf_path.name] = content
                logging.info("Extracted %d chars from %s", len(content), pdf_path.name)
            else:
                logging.warning("Extracted no text from %s.", pdf_path.name)
        except Exception as e:
            logging.error("Error reading %s: %s", pdf_path.name, e)
    if not texts:
        logging.warning("No text extracted from any PDF.")
    return texts

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursively splits text into chunks of a target size with overlap."""
    if not text: return []
    separators = ["\n\n", "\n", ". ", " ", ""] # Prioritized separators

    def split(text_to_split, current_separators):
        # Base case: If text is small enough or cannot be split further by separators
        if not text_to_split or len(text_to_split) <= chunk_size:
             return [text_to_split] if text_to_split and text_to_split.strip() else []

        current_separator = current_separators[0]
        next_separators = current_separators[1:]

        # Handle the case of splitting by character if no other separators work
        if current_separator == "":
            chunks = []
            start = 0
            while start < len(text_to_split):
                end = start + chunk_size
                chunk = text_to_split[start:end]
                if chunk.strip():
                    chunks.append(chunk)
                # Move start for the next chunk, considering overlap
                start += chunk_size - chunk_overlap
                if start >= len(text_to_split): # Avoid infinite loop if overlap is too large
                     break
            return chunks

        # Try splitting with the current separator
        splits = [s for s in text_to_split.split(current_separator) if s.strip()]
        if not splits: # If splitting results in nothing, try next separator
             return split(text_to_split, next_separators) if next_separators else split(text_to_split, [""])


        final_chunks = []
        current_chunk_parts = []
        current_length = 0
        separator_len = len(current_separator)

        for i, part in enumerate(splits):
            part_len = len(part)

            # Calculate length if this part is added
            potential_length = current_length + part_len + (separator_len if current_chunk_parts else 0)

            if potential_length <= chunk_size:
                # Add part to the current chunk
                current_chunk_parts.append(part)
                current_length = potential_length
            else:
                # Current chunk is full, or the part itself is too large
                if current_chunk_parts:
                    # Finalize the current chunk
                    chunk = current_separator.join(current_chunk_parts)
                    if chunk.strip(): final_chunks.append(chunk)

                    # Start new chunk with overlap logic (simplified for clarity)
                    # Basic overlap: take the last part(s) or tail of the finished chunk
                    overlap_text = chunk[-(min(chunk_overlap, len(chunk))):]
                    # Try to make overlap cleaner (optional)
                    last_space = overlap_text.rfind(' ')
                    if last_space != -1: overlap_text = overlap_text[last_space+1:]

                    # Start next chunk with the current part potentially prefixed by overlap
                    # Be careful not to duplicate content excessively
                    # A simpler restart: just start new chunk with current part
                    current_chunk_parts = [part]
                    current_length = part_len

                    # Check if the current part itself is too large
                    if part_len > chunk_size:
                         # Split this large part further down
                         final_chunks.extend(split(part, next_separators if next_separators else [""]))
                         # Reset since the large part was handled
                         current_chunk_parts = []
                         current_length = 0

                else: # current_chunk_parts is empty, meaning part itself > chunk_size
                     final_chunks.extend(split(part, next_separators if next_separators else [""]))
                     current_chunk_parts = [] # Ensure reset
                     current_length = 0


            # If it's the last part, add any remaining assembled chunk
            if i == len(splits) - 1 and current_chunk_parts:
                 chunk = current_separator.join(current_chunk_parts)
                 if chunk.strip(): final_chunks.append(chunk)


        return [c for c in final_chunks if c.strip()] # Final cleanup

    # Start the splitting process
    return split(text, separators)


# --- Function to Build Vector DB if Needed (Unchanged) ---

def build_new_vector_db(client: chromadb.PersistentClient, docs_folder: Path) -> bool:
    """Extracts, chunks, and indexes PDFs into the specified collection."""
    global pdf_collection # To assign the newly created/updated collection

    logging.info("--- Starting Vector DB Creation/Update Process ---")
    try:
        extracted = pdf_to_text(docs_folder)
        if not extracted:
            logging.error("No text extracted, cannot build vector DB.")
            return False

        logging.info("Accessing collection '%s'...", COLLECTION_NAME)
        # get_or_create handles creation if it doesn't exist
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logging.info("Using collection '%s'. Current item count: %d", COLLECTION_NAME, collection.count())

        docs, metas, ids = [], [], []
        logging.info("Chunking text with target size ~%d chars, overlap %d chars...", CHUNK_SIZE, CHUNK_OVERLAP)
        total_chunks = 0
        for fname, text in extracted.items():
            if not text.strip(): continue
            file_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            logging.info("Generated %d chunks for %s.", len(file_chunks), fname)
            if not file_chunks: continue
            for idx, chunk in enumerate(file_chunks, start=1):
                docs.append(chunk)
                metas.append({"source": fname, "chunk_idx": idx})
                # Sanitize filename more robustly for ID
                safe_fname = re.sub(r'[^\w-]', '_', fname) # Replace non-alphanumeric/hyphen/underscore
                unique_id = f"{safe_fname}_chunk_{idx}"
                ids.append(unique_id)
            total_chunks += len(file_chunks)

        if not docs:
            logging.warning("No text chunks generated for indexing.")
            pdf_collection = collection # Assign empty collection
            return True

        logging.info("Prepared %d total chunks for indexing.", total_chunks)
        batch_size = 100
        num_batches = (len(docs) + batch_size - 1) // batch_size
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            batch_metas = metas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            try:
                # Use add() for simplicity assuming mostly new data on build
                collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                logging.info("Indexed batch %d/%d (size %d chunks).",
                             (i // batch_size) + 1, num_batches, len(batch_docs))
            except Exception as e:
                 # Consider logging which IDs failed if possible
                 logging.error("Error adding batch starting at index %d: %s", i, e)
                 # Optionally add a flag to stop on error or continue

        final_count = collection.count()
        logging.info("Finished indexing. Collection '%s' now contains %d items.",
                     collection.name, final_count)
        pdf_collection = collection # Assign populated collection
        return True

    except Exception as e:
        logging.error("Failed during Vector DB build process: %s", e, exc_info=True)
        return False


# --- Initialization Function (Using list_collections check - Unchanged) ---
def initialize_vector_db(persist_path: Path, docs_folder_path: Path):
    """Initializes a persistent ChromaDB client and ensures the collection exists."""
    global pdf_collection, chroma_client
    logging.info("--- Initializing Persistent Vector DB ---")
    logging.info("Storage directory: %s", persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    try:
        chroma_client = chromadb.PersistentClient(path=str(persist_path))
        logging.info("ChromaDB Persistent Client initialized.")
    except Exception as e:
        logging.error("Fatal: Failed to initialize ChromaDB Persistent Client: %s", e, exc_info=True)
        return False

    try:
        logging.info("Checking for existing collection: '%s'", COLLECTION_NAME)
        existing_collections = chroma_client.list_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in existing_collections)

        if collection_exists:
            logging.info("Collection '%s' found. Getting handle.", COLLECTION_NAME)
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            count = collection.count()
            if count > 0:
                logging.info("Existing collection has %d items. Using existing.", count)
                pdf_collection = collection
                return True
            else:
                logging.warning("Found existing collection '%s', but it is empty. Will attempt to build.", COLLECTION_NAME)
                return build_new_vector_db(chroma_client, docs_folder_path)
        else:
            logging.info("Collection '%s' not found. Proceeding to build a new one.", COLLECTION_NAME)
            return build_new_vector_db(chroma_client, docs_folder_path)

    except Exception as e:
        logging.error("Error during Vector DB initialization/check: %s", e, exc_info=True)
        return False


# --- Watsonx.ai Initialization (Unchanged) ---
try:
    creds  = Credentials(url=URL, api_key=API_KEY)
    client = APIClient(credentials=creds, project_id=PROJECT_ID)
    model = ModelInference(
        model_id=MODEL_ID,
        credentials=creds,
        project_id=PROJECT_ID
    )
    logging.info(
        f"Initialized Watsonx.ai model '{MODEL_ID}' for project '{PROJECT_ID}'."
    )
except Exception as e:
    logging.error("Failed to initialize Watsonx.ai client/model: %s", e, exc_info=True)
    sys.exit(1)

# --- MCP Server Setup (Unchanged) ---
mcp = FastMCP("Watsonx RAG Chatbot Server")

# --- RAG Tool Definition (Unchanged logic) ---
@mcp.tool()
def chat_with_manual(query: str) -> str:
    """
    Answers questions about drone manuals using RAG with Watsonx.ai.
    Relies on the globally initialized pdf_collection.
    """
    global pdf_collection
    logging.info("Received RAG query: %r", query)

    if pdf_collection is None:
        logging.error("Vector DB collection is not available. Initialization might have failed.")
        return "Error: The document database is not ready. Please check server logs."

    try:
        logging.info("Querying Vector DB ('%s') for top %d results...", pdf_collection.name, NUM_RESULTS_RAG)
        results = pdf_collection.query(
            query_texts=[query],
            n_results=NUM_RESULTS_RAG,
            include=['documents'] # Requesting documents for context
        )
    except Exception as e:
        logging.error("Error querying ChromaDB: %s", e, exc_info=True)
        return f"Error: Could not retrieve information from the document database."

    # Process results to create context string
    retrieved_docs = results.get('documents', [[]])[0] # Get list of docs for the first query
    # Inside chat_with_manual, after getting retrieved_docs
    logging.info(f"Retrieved documents: {retrieved_docs}") # Log the actual content


    if not retrieved_docs:
        logging.warning("No relevant documents found in Vector DB for query: %r", query)
        # Decide fallback: Answer without context or return specific message
        # context_string = "No relevant context found in the provided documents."
        # For this example, let's return a message indicating no context was found
        return "I couldn't find specific information about that in the available documents."
    else:
        logging.info("Retrieved %d document chunks.", len(retrieved_docs))
        context_string = "\n\n---\n\n".join(retrieved_docs) # Join chunks with separator

    # Construct the prompt for the language model
    prompt_template = f"""
    You are a helpful assistant that answers questions based *only* on the information provided from the manual below.

    --- Manual Context ---
    {context_string}
    --- End of Context ---

    Using only the context above, please answer the following question clearly and accurately.

    Question: {query}

    Answer:
    """
    logging.info("Constructed prompt for Watsonx.ai (length: %d chars)", len(prompt_template))
    # Avoid logging excessively long prompts unless debugging
    logging.debug("Prompt:\n%s", prompt_template)
    logging.info("Prompt:\n%s", prompt_template)
    # Define generation parameters for Watsonx.ai
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS:  300,
        GenParams.MIN_NEW_TOKENS:  10,
        # GenParams.TEMPERATURE: 0.7, # Example: uncomment for less deterministic output
        GenParams.STOP_SEQUENCES: ["\n\n", "---", "Question:", "Context:"] # Add more stop sequences
    }
    logging.info("Sending request to Watsonx.ai model '%s'...", MODEL_ID)

    # Call Watsonx.ai model
    try:
        resp = model.generate_text(prompt=prompt_template, params=params, raw_response=True)
        # Check response structure carefully
        if resp and isinstance(resp, dict) and "results" in resp and isinstance(resp["results"], list) and len(resp["results"]) > 0:
            first_result = resp["results"][0]
            if isinstance(first_result, dict) and "generated_text" in first_result:
                answer = first_result["generated_text"].strip()
                logging.info("Received Watsonx.ai response: %r", answer)
                # Simple post-processing: remove incomplete sentences if ending with '...' or similar common truncation markers
                if answer.endswith("..."):
                     last_period = answer.rfind('.')
                     if last_period != -1:
                         answer = answer[:last_period+1]
                return answer
            else:
                 logging.error("Watsonx.ai response structure unexpected ('generated_text' missing): %s", first_result)
                 return "Error: Received an unexpected response format from the AI model (Detail 1)."
        else:
             logging.error("Watsonx.ai response structure unexpected (main structure): %s", resp)
             return "Error: Received an unexpected response format from the AI model (Detail 2)."
    except Exception as e:
        logging.error("Watsonx.ai inference error: %s", e, exc_info=True)
        return f"Error: Failed to generate an answer due to an AI model issue."

# --- Main Execution (Unchanged) ---
if __name__ == "__main__":
    # Initialize/Load the Persistent Vector DB *before* starting the server
    if not initialize_vector_db(CHROMA_PERSIST_DIR, DOCS_FOLDER):
         logging.error("CRITICAL: Failed to initialize Vector DB. Exiting.")
         sys.exit(1) # Stop server if DB isn't ready

    # Start the MCP server (this call is blocking)
    logging.info("Starting MCP server on STDIO transport...")
    mcp.run()
    logging.info("MCP server stopped.")