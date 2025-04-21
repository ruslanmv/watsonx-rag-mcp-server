import os
import sys
import logging
from pathlib import Path
import re
import textwrap
from dotenv import load_dotenv

# MCP Imports
from mcp.server.fastmcp import FastMCP
# Note: base prompt template import removed as we create a custom one

# IBM Watsonx.ai SDK Imports
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Vector DB Imports
import chromadb
from pypdf import PdfReader

# --- Configuration and Setup ---

# Load .env variables (ensure .env file exists with your credentials)
load_dotenv()

# Watsonx Credentials and Config
API_KEY    = os.getenv("WATSONX_APIKEY")
URL        = os.getenv("WATSONX_URL")
PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_ID   = os.getenv("MODEL_ID", "ibm/granite-13b-instruct-v2") # Default model

# Vector DB Config
DOCS_FOLDER = Path(os.getenv("DOCS_FOLDER", "documents")) # Folder containing PDFs
COLLECTION_NAME = "pdf_documents_chunked_watsonx"
NUM_RESULTS_RAG = 3 # Number of chunks to retrieve for RAG context
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

# Validate essential env vars
for name, val in [
    ("WATSONX_APIKEY", API_KEY),
    ("WATSONX_URL", URL),
    ("PROJECT_ID", PROJECT_ID)
]:
    if not val:
        raise RuntimeError(f"{name} is not set. Please add it to your .env file.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Vector DB Functions (Copied and adapted from vector.py) ---

def pdf_to_text(folder: Path) -> dict:
    """Extract text from every PDF in `folder`, preserving structure."""
    # ... (Keep the exact pdf_to_text function from the previous version) ...
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
                 try:
                     reader.decrypt('')
                     logging.info("Successfully decrypted %s with empty password.", pdf_path.name)
                 except Exception as decrypt_e:
                     logging.error("Failed to decrypt %s: %s. Skipping.", pdf_path.name, decrypt_e)
                     continue

            content_list = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    cleaned_text = re.sub(r' +', ' ', page_text)
                    content_list.append(cleaned_text) # Removed page break marker for simplicity

            content = "\n".join(content_list).strip()
            content = re.sub(r'\n{3,}', '\n\n', content)

            if content:
                texts[pdf_path.name] = content
                logging.info("Extracted %d chars from %s", len(content), pdf_path.name)
            else:
                logging.warning("Extracted no text from %s (content is empty).", pdf_path.name)

        except Exception as e:
            logging.error("Error reading %s: %s", pdf_path.name, e)

    if not texts:
            logging.warning("No text extracted from any PDF after processing.")
    return texts


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursively splits text into chunks of a target size with overlap."""
    # ... (Keep the exact chunk_text function from the previous version) ...
    if not text:
        return []
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split(text_to_split, current_separators):
        if not text_to_split: return []
        if len(text_to_split) <= chunk_size:
             if text_to_split.strip(): return [text_to_split]
             else: return []

        current_separator = current_separators[0]
        next_separators = current_separators[1:]

        if current_separator == "":
            chunks = []
            for i in range(0, len(text_to_split), chunk_size - chunk_overlap):
                 chunk = text_to_split[i : i + chunk_size]
                 if chunk.strip(): chunks.append(chunk)
            return chunks

        splits = [s for s in text_to_split.split(current_separator) if s.strip()]
        final_chunks = []
        current_chunk_parts = []
        current_length = 0

        for part in splits:
            part_len = len(part)
            separator_len = len(current_separator)

            if current_length + part_len + (separator_len if current_chunk_parts else 0) <= chunk_size:
                current_chunk_parts.append(part)
                current_length += part_len + (separator_len if len(current_chunk_parts) > 1 else 0)
            else:
                if current_chunk_parts:
                    chunk = current_separator.join(current_chunk_parts)
                    if chunk.strip(): final_chunks.append(chunk)
                    overlap_text = ""
                    if chunk_overlap > 0:
                         overlap_text = chunk[-chunk_overlap:]
                         last_space = overlap_text.rfind(' ')
                         if last_space != -1: overlap_text = overlap_text[last_space+1:]
                    current_chunk_parts = [overlap_text + part] if overlap_text else [part]
                    current_length = len(current_chunk_parts[0])
                else:
                    if len(next_separators) > 0:
                        final_chunks.extend(split(part, next_separators))
                    else:
                        final_chunks.extend(split(part, [""]))
                    current_chunk_parts = []
                    current_length = 0
        if current_chunk_parts:
            chunk = current_separator.join(current_chunk_parts)
            if chunk.strip(): final_chunks.append(chunk)
        return [c for c in final_chunks if c.strip()]

    return split(text, separators)


def create_chromadb_from_text(extracted: dict, collection_name: str, chunk_size: int, chunk_overlap: int) -> chromadb.Collection:
    """Build a ChromaDB collection using improved chunking."""
    # ... (Keep the exact create_chromadb_from_text function, but make collection_name, chunk_size, chunk_overlap parameters) ...
    client_chroma = chromadb.Client() # Using default in-memory ephemeral client

    try:
        # Check if collection exists before trying to delete
        existing_collections = [c.name for c in client_chroma.list_collections()]
        if collection_name in existing_collections:
             client_chroma.delete_collection(collection_name)
             logging.info("Deleted existing collection '%s'", collection_name)
        else:
             logging.info("Collection '%s' did not exist.", collection_name)
    except Exception as e:
        logging.warning("Could not delete collection '%s' (may not exist or error): %s", collection_name, e)

    # Explicitly create with default embedding function (adjust if using specific models)
    collection = client_chroma.get_or_create_collection(
        name=collection_name
    )
    logging.info("Using collection '%s'. Initial count: %d", collection_name, collection.count())

    docs, metas, ids = [], [], []
    logging.info("Chunking text with target size ~%d chars, overlap %d chars...", chunk_size, chunk_overlap)
    total_chunks = 0
    for fname, text in extracted.items():
        if not text.strip():
            logging.warning("Skipping empty document: %s", fname)
            continue
        file_chunks = chunk_text(text, chunk_size, chunk_overlap)
        logging.info("Generated %d chunks for %s.", len(file_chunks), fname)
        if not file_chunks:
             logging.warning("No chunks generated for %s after splitting.", fname)
             continue
        for idx, chunk in enumerate(file_chunks, start=1):
            docs.append(chunk)
            metas.append({"source": fname, "chunk_idx": idx})
            # Sanitize filename for ID
            safe_fname = re.sub(r'[^a-zA-Z0-9_-]', '_', fname)
            unique_id = f"{safe_fname}_chunk_{idx}"
            ids.append(unique_id)
        total_chunks += len(file_chunks)

    if docs:
        logging.info("Prepared %d total chunks for indexing.", total_chunks)
        batch_size = 100 # Smaller batch size for in-memory potentially
        num_batches = (len(docs) + batch_size - 1) // batch_size
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            batch_metas = metas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            try:
                collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                logging.info("Indexed batch %d/%d (size %d chunks).",
                             (i // batch_size) + 1, num_batches, len(batch_docs))
            except Exception as e:
                 logging.error("Error adding batch starting at index %d: %s", i, e)
        logging.info("Finished indexing. Collection '%s' now contains %d items.",
                     collection.name, collection.count())
    else:
        logging.warning("No text chunks extracted or processed for indexing.")
    return collection

# --- Global Variables ---
pdf_collection: chromadb.Collection | None = None # Holds the initialized ChromaDB collection

# --- Initialization Function ---
def initialize_vector_db(docs_folder_path: Path):
    """Processes PDFs and initializes the ChromaDB collection."""
    global pdf_collection
    logging.info("--- Starting Vector DB Creation Process ---")
    if not docs_folder_path.is_dir():
        logging.error("Documents folder '%s' not found. Cannot build vector DB.", docs_folder_path)
        return False

    extracted = pdf_to_text(docs_folder_path)
    if not extracted:
        logging.error("No text extracted from any PDFs in %s. Cannot build vector DB.", docs_folder_path)
        return False

    logging.info("Extracted text from %d files.", len(extracted))
    logging.info("Starting ChromaDB collection creation '%s'...", COLLECTION_NAME)
    try:
        pdf_collection = create_chromadb_from_text(
            extracted,
            collection_name=COLLECTION_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        collection_item_count = pdf_collection.count()
        logging.info("ChromaDB setup complete. Collection '%s' contains %d items.",
                     pdf_collection.name, collection_item_count)
        if collection_item_count == 0:
             logging.warning("Vector DB collection is empty after initialization.")
             return False
        return True
    except Exception as e:
        logging.error("Failed to create ChromaDB collection: %s", e, exc_info=True)
        pdf_collection = None # Ensure it's None if setup failed
        return False

# --- Watsonx.ai Initialization ---
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
    sys.exit(1) # Exit if Watsonx cannot be initialized

# --- MCP Server Setup ---
mcp = FastMCP("Watsonx RAG Chatbot Server")

# --- RAG Tool Definition ---
@mcp.tool()
def chat_with_manual(query: str) -> str:
    """
    Answers questions about drone manuals using RAG with Watsonx.ai.

    Retrieves relevant text chunks from the indexed PDF manual(s),
    constructs a context, and asks the Watsonx.ai model to answer based
    on that context.

    :param query: The user's question about the drone manual.
    :return: The answer generated by Watsonx.ai based on the retrieved context.
    """
    global pdf_collection
    logging.info("Received RAG query: %r", query)

    # 1. Check if Vector DB is initialized
    if pdf_collection is None or pdf_collection.count() == 0:
        logging.error("Vector DB not initialized or empty. Cannot perform RAG.")
        return "Error: The document database is not ready. Please ensure documents are processed."

    # 2. Retrieve relevant chunks (R part of RAG)
    try:
        logging.info("Querying Vector DB for top %d results...", NUM_RESULTS_RAG)
        results = pdf_collection.query(
            query_texts=[query],
            n_results=NUM_RESULTS_RAG,
            include=['documents'] # Only need the text content for the context
        )
    except Exception as e:
        logging.error("Error querying ChromaDB: %s", e, exc_info=True)
        return f"Error: Could not retrieve information from the document database."

    # 3. Check and Format Context
    retrieved_docs = results.get('documents', [[]])[0] # Get list of docs for the first query
    if not retrieved_docs:
        logging.warning("No relevant documents found in Vector DB for query: %r", query)
        # Optionally, you could still try asking the LLM without context,
        # or return a specific message.
        # Let's try asking the LLM directly as a fallback for this example.
        context_string = "No relevant context found in the provided documents."
        # return "I couldn't find specific information about that in the documents."
    else:
        logging.info("Retrieved %d document chunks.", len(retrieved_docs))
        # Concatenate the retrieved documents into a single context string
        context_string = "\n\n---\n\n".join(retrieved_docs) # Separator between chunks


    # 4. Construct Prompt for LLM (A and G parts of RAG)
    prompt_template = f"""Context from the drone manual:
--- Start of Context ---
{context_string}
--- End of Context ---

Based *only* on the context provided above, please answer the following question:
Question: {query}

Answer:"""

    logging.info("Constructed prompt for Watsonx.ai (length: %d chars)", len(prompt_template))
    # Avoid logging the full potentially large prompt unless debugging:
    # logging.debug("Prompt:\n%s", prompt_template)

    # 5. Generate Answer with Watsonx.ai
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS:  300, # Allow slightly longer answers for RAG
        GenParams.MIN_NEW_TOKENS:  10,
        # Add other params like temperature if needed, e.g., GenParams.TEMPERATURE: 0.7
        GenParams.STOP_SEQUENCES: ["\n\n"] # Stop if it generates double newlines maybe?
    }

    logging.info("Sending request to Watsonx.ai model '%s'...", MODEL_ID)
    try:
        resp = model.generate_text(
            prompt=prompt_template,
            params=params,
            raw_response=True # Get the full JSON
        )

        # Extract the generated text
        if resp and resp.get("results") and resp["results"][0].get("generated_text"):
            answer = resp["results"][0]["generated_text"].strip()
            logging.info("Received Watsonx.ai response: %r", answer)
            # Optional: Add post-processing to clean up the answer if needed
            return answer
        else:
             logging.error("Watsonx.ai response format unexpected or empty: %s", resp)
             return "Error: Received an unexpected response from the AI model."

    except Exception as e:
        logging.error("Watsonx.ai inference error: %s", e, exc_info=True)
        return f"Error: Failed to generate an answer due to an AI model issue."

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize the Vector DB *before* starting the server
    if not initialize_vector_db(DOCS_FOLDER):
         logging.error("Failed to initialize Vector DB. Exiting.")
         sys.exit(1)

    # Start the MCP server (this call is blocking)
    logging.info("Starting MCP server on STDIO transport...")
    mcp.run()
    logging.info("MCP server stopped.")