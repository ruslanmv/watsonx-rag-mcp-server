import logging
import chromadb
from pathlib import Path
from pypdf import PdfReader # Assuming PdfReader comes from pypdf
import sys
import textwrap # Import textwrap for pretty printing/indenting
import re # Import regex for sentence splitting


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


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
                 try:
                     reader.decrypt('')
                     logging.info("Successfully decrypted %s with empty password.", pdf_path.name)
                 except Exception as decrypt_e:
                     logging.error("Failed to decrypt %s: %s. Skipping.", pdf_path.name, decrypt_e)
                     continue

            # Extract text page by page to potentially add page metadata later if needed
            # For now, concatenate all pages but preserve newlines better.
            content_list = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Basic cleaning: replace multiple spaces with one, but keep newlines
                    cleaned_text = re.sub(r' +', ' ', page_text)
                    # Add page breaks as a potential separator, can be useful for chunking
                    content_list.append(cleaned_text + "\n--- Page Break ---") # Optional page marker

            content = "\n".join(content_list).strip() # Join pages with newlines

            # Replace multiple consecutive newlines with a maximum of two (\n\n)
            content = re.sub(r'\n{3,}', '\n\n', content)

            if content:
                texts[pdf_path.name] = content
                logging.info("Extracted %d chars from %s (incl. potential page markers)", len(content), pdf_path.name)
            else:
                logging.warning("Extracted no text from %s (content is empty).", pdf_path.name)

        except Exception as e:
            logging.error("Error reading %s: %s", pdf_path.name, e)

    if not texts:
            logging.warning("No text extracted from any PDF after processing.")
    return texts

# --- Improved Chunking Logic ---
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Recursively splits text into chunks of a target size with overlap.
    Tries splitting by paragraph, then newline, then sentence, then word.
    """
    if not text:
        return []

    # Define separators in order of preference
    separators = ["\n\n", "\n", ". ", " ", ""] # Paragraph, newline, sentence, word, character

    def split(text_to_split, current_separators):
        if not text_to_split:
            return []
            
        if len(text_to_split) <= chunk_size:
             # If the remaining text is small enough, return it as a single chunk
             # unless it's only whitespace
             if text_to_split.strip():
                 return [text_to_split]
             else:
                 return []


        current_separator = current_separators[0]
        next_separators = current_separators[1:]

        # Handle the case of splitting by character if no other separators work
        if current_separator == "":
            # Split into chunks of size `chunk_size` directly
            chunks = []
            for i in range(0, len(text_to_split), chunk_size - chunk_overlap):
                 chunk = text_to_split[i : i + chunk_size]
                 if chunk.strip():
                     chunks.append(chunk)
            return chunks

        # Try splitting with the current separator
        splits = text_to_split.split(current_separator)
        
        # Filter out empty strings that can result from splitting
        splits = [s for s in splits if s.strip()] 

        final_chunks = []
        current_chunk_parts = []
        current_length = 0

        for part in splits:
            part_len = len(part)
            separator_len = len(current_separator)

            if current_length + part_len + (separator_len if current_chunk_parts else 0) <= chunk_size:
                # Add part to the current chunk
                current_chunk_parts.append(part)
                current_length += part_len + (separator_len if len(current_chunk_parts) > 1 else 0)
            else:
                # Current chunk is full (or the part itself is too large)
                if current_chunk_parts:
                    # Finalize the current chunk
                    chunk = current_separator.join(current_chunk_parts)
                    if chunk.strip(): # Add only if not just whitespace
                       final_chunks.append(chunk)
                    
                    # Start a new chunk with overlap
                    # Try to find a suitable overlap point
                    overlap_text = ""
                    if chunk_overlap > 0:
                         # Take the end of the finalized chunk as overlap
                         overlap_text = chunk[-chunk_overlap:]
                         # Try to find a natural break point (e.g., space) within the overlap
                         last_space = overlap_text.rfind(' ')
                         if last_space != -1:
                              overlap_text = overlap_text[last_space+1:]
                         else: # No space, keep the raw overlap tail
                             pass 
                    
                    # Reset for the next chunk, starting with the current part
                    # Add overlap prefix if available
                    current_chunk_parts = [overlap_text + part] if overlap_text else [part]
                    current_length = len(current_chunk_parts[0])
                    
                else:
                    # The part itself is larger than chunk_size, split it further
                    if len(next_separators) > 0:
                        # Use the next level of separators
                        final_chunks.extend(split(part, next_separators))
                    else:
                        # Cannot split further with separators, force split by size
                        final_chunks.extend(split(part, [""])) # Use character split

                    # Reset after handling the large part
                    current_chunk_parts = []
                    current_length = 0

        # Add the last assembled chunk if any parts remain
        if current_chunk_parts:
            chunk = current_separator.join(current_chunk_parts)
            if chunk.strip():
                final_chunks.append(chunk)

        # Post-process: Ensure chunks are not just whitespace
        return [c for c in final_chunks if c.strip()]

    # Start the splitting process
    initial_chunks = split(text, separators)
    # Filter out potentially very small chunks resulting from splits if needed,
    # but often better to keep them for completeness unless they are trivial.
    # Example filter (optional):
    # min_final_chunk_len = 50
    # initial_chunks = [c for c in initial_chunks if len(c) >= min_final_chunk_len]
    return initial_chunks


def create_chromadb_from_text(extracted: dict) -> chromadb.Collection:
    """Build a ChromaDB collection using improved chunking."""
    client_chroma = chromadb.Client() # In-memory client

    collection_name = "pdf_documents_chunked" # Use a new name to avoid conflicts

    try:
        client_chroma.delete_collection(collection_name)
        logging.info("Deleted existing collection '%s'", collection_name)
    except Exception:
        logging.info("Collection '%s' did not exist or could not be deleted (ok).", collection_name)

    # Explicitly create with default embedding function (adjust if using specific models)
    # Check ChromaDB docs for specifying models like sentence-transformers
    collection = client_chroma.get_or_create_collection(
        name=collection_name
        # embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction() # Example explicit default
        )
    logging.info("Using collection '%s'. Initial count: %d", collection_name, collection.count())

    docs, metas, ids = [], [], []
    chunk_size = 800 # Target characters per chunk
    chunk_overlap = 150 # Characters overlap between chunks

    logging.info("Chunking text with target size ~%d chars, overlap %d chars...", chunk_size, chunk_overlap)

    total_chunks = 0
    for fname, text in extracted.items():
        if not text.strip():
            logging.warning("Skipping empty document: %s", fname)
            continue

        # Use the improved chunking function
        file_chunks = chunk_text(text, chunk_size, chunk_overlap)
        logging.info("Generated %d chunks for %s.", len(file_chunks), fname)

        if not file_chunks:
             logging.warning("No chunks generated for %s after splitting.", fname)
             continue

        for idx, chunk in enumerate(file_chunks, start=1):
            docs.append(chunk)
            metas.append({"source": fname, "chunk_idx": idx})
            unique_id = f"{fname.replace('.', '_').replace(' ', '_').replace('-', '_')}_chunk_{idx}"
            ids.append(unique_id)
        total_chunks += len(file_chunks)

    if docs:
        logging.info("Prepared %d total chunks for indexing.", total_chunks)
        batch_size = 500 # Keep batching for large numbers of chunks
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
                 # Consider adding more details like the IDs that failed if possible

        logging.info("Finished indexing. Collection '%s' now contains %d items.",
                     collection.name, collection.count())
    else:
        logging.warning("No text chunks extracted or processed for indexing.")

    return collection


if __name__ == "__main__":
    docs_folder = Path("documents")

    logging.info("--- Starting Vector DB Creation Process ---")
    extracted = pdf_to_text(docs_folder)

    if not extracted:
        logging.error("No text extracted from any PDFs in %s – Cannot build vector DB. Exiting.", docs_folder)
        sys.exit(1)

    logging.info("Extracted text from %d files.", len(extracted))
    logging.info("Starting ChromaDB collection creation with improved chunking...")
    pdf_collection = create_chromadb_from_text(extracted)

    collection_item_count = pdf_collection.count() # Check count *after* creation/indexing
    logging.info("ChromaDB setup complete. Collection '%s' contains %d items.",
                 pdf_collection.name, collection_item_count)

    # 2. Query the vector database
    logging.info("--- Starting Vector DB Query Process ---")

    if collection_item_count == 0:
        logging.warning("Collection is empty. Cannot perform query.")
    else:
        query_text = "flight modes of this drone" # Example query
        # Retrieve more results to increase chances of finding the best one
        num_results = 3 # Let's try getting the top 3 chunks

        logging.info("Querying for: '%s' (Top %d results)", query_text, num_results)

        try:
            query_results = pdf_collection.query(
                query_texts=[query_text],
                n_results=num_results,
                include=['documents', 'metadatas', 'distances'] # Include IDs implicitly
            )

            logging.info("Query executed successfully.")

            # Check if the results structure is as expected
            if (query_results and
                query_results.get('ids') and
                query_results['ids'][0] and # Check if the list of IDs for the first query is not empty
                query_results.get('documents') and
                query_results['documents'][0]): # Check if the list of documents for the first query is not empty

                logging.info("\n--- Top %d Most Similar Chunks ---", len(query_results['ids'][0]))

                for i in range(len(query_results['ids'][0])): # Iterate based on the number of results received
                    doc = query_results['documents'][0][i]
                    meta = query_results['metadatas'][0][i]
                    distance = query_results['distances'][0][i]
                    result_id = query_results['ids'][0][i] # Get ID from the results

                    logging.info(f"\nResult {i+1}:")
                    logging.info(f"  Source File: {meta.get('source', 'N/A')}")
                    logging.info(f"  Chunk Index (within file): {meta.get('chunk_idx', 'N/A')}")
                    logging.info(f"  Similarity Distance: {distance:.4f}")
                    logging.info(f"  Chunk ID: {result_id}")
                    logging.info("  Content:")

                    # Print the full content of the retrieved chunk, indented
                    # Use textwrap to handle potentially long lines gracefully
                    wrapped_content = textwrap.fill(doc, width=100, initial_indent='    ', subsequent_indent='    ')
                    print(wrapped_content)
                    # Or just log it:
                    # logging.info("    " + doc.replace("\n", "\n    ")) # Basic indentation for logging

                logging.info("\n--- End of Results ---")

            else:
                logging.warning("Query returned no results or results format unexpected.")
                logging.debug("Raw query results: %s", query_results) # Log raw results for debugging

        except Exception as e:
            logging.error("An error occurred during the query: %s", e, exc_info=True) # Log traceback
            logging.error("Ensure the ChromaDB collection has an embedding function configured (default is used if none specified at creation) and items indexed.")

    logging.info("--- Script finished ---")