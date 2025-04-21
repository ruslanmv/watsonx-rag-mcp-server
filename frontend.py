import os
import sys
import asyncio
import logging
import uuid # To generate unique session IDs if needed, but Flask session handles this
from flask import Flask, render_template, request, redirect, url_for, session

# --- MCP Client Imports ---
# Assuming mcp library is installed: pip install meta-compute-protocol
# Make sure server.py is in the same directory or Python path
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    # from mcp.common.content import TextContent # Might be needed depending on exact response wrapping
except ImportError:
    print("ERROR: 'meta-compute-protocol' library not found.")
    print("Please install it: pip install meta-compute-protocol")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Frontend] %(levelname)s: %(message)s')

# --- Flask App Setup ---
app = Flask(__name__)
# IMPORTANT: Change this to a random secret key for production!
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-replace-me")

# --- Conversation Memory Configuration ---
MAX_HISTORY_CHARS = 4000 # Approximate token limit (adjust as needed) ~1000 tokens

# --- Helper Function for MCP Interaction ---

async def async_mcp_call(user_query: str):
    """
    Starts server.py, connects via MCP, calls the tool, and returns the response.
    This runs the full client logic for each call - inefficient but simpler for Flask.
    """
    server_params = StdioServerParameters(command=sys.executable, args=["server.py"]) # Use sys.executable for portability
    logging.info("Attempting to start and connect to MCP server ('server.py')...")
    response_text = None
    error_message = None

    try:
        async with stdio_client(server_params) as (reader, writer):
            logging.info("Connected to MCP server. Initializing session...")
            async with ClientSession(reader, writer) as mcp_session:
                await mcp_session.initialize()
                logging.info("MCP session initialized.")
                logging.info("Calling tool 'chat_with_manual' with query: %r", user_query)

                try:
                    response = await mcp_session.call_tool(
                        "chat_with_manual",
                        arguments={"query": user_query}
                    )
                    logging.info("Received response from server.")

                    # Process response (handle potential wrapping)
                    final_answer = response
                    # Example check if response is wrapped like TextContent (adjust based on actual MCP response)
                    if hasattr(response, 'content') and response.content and isinstance(response.content, list):
                         first_content = response.content[0]
                         if hasattr(first_content, 'text'):
                             final_answer = first_content.text
                    elif isinstance(response, str): # If it's already a string
                         final_answer = response

                    # Ensure we have a string before returning
                    response_text = str(final_answer) if final_answer is not None else "Received an empty response."


                except Exception as tool_call_e:
                    logging.error("Error calling tool 'chat_with_manual': %s", tool_call_e, exc_info=True)
                    error_message = f"Error calling backend tool: {tool_call_e}"

    except Exception as connection_e:
        logging.error("Failed to connect or communicate with the MCP server: %s", connection_e, exc_info=True)
        error_message = f"Could not connect to or run the server: {connection_e}"

    # Ensure server process is terminated (stdio_client context manager handles this)
    logging.info("MCP client connection closed.")
    return response_text, error_message


def get_rag_response(user_query: str) -> tuple[str | None, str | None]:
    """
    Synchronous wrapper to run the asynchronous MCP call.
    Returns (response_text, error_message)
    """
    # --- Running asyncio logic within Flask ---
    # Note: Running asyncio.run() inside a synchronous Flask route handler
    # is generally discouraged for production scalability, but works for simple demos.
    # More robust solutions involve running the server independently (e.g., via TCP)
    # or using async Flask extensions.
    try:
        # For Python 3.7+
        response_text, error_message = asyncio.run(async_mcp_call(user_query))
        return response_text, error_message
    except RuntimeError as e:
        # Handle cases like "asyncio.run() cannot be called from a running event loop"
        # This might happen in certain deployment scenarios or with specific Flask extensions.
        logging.error("Asyncio runtime error: %s. Trying get_event_loop().run_until_complete()", e)
        try:
            loop = asyncio.get_event_loop()
            response_text, error_message = loop.run_until_complete(async_mcp_call(user_query))
            return response_text, error_message
        except Exception as fallback_e:
             logging.error("Fallback asyncio execution failed: %s", fallback_e)
             return None, f"Internal server error during async execution: {fallback_e}"
    except Exception as e:
        logging.error("Unexpected error during RAG response retrieval: %s", e)
        return None, f"Unexpected error: {e}"


# --- Conversation History Management ---

def trim_history(history: list, max_chars: int) -> list:
    """Removes oldest messages if total character count exceeds max_chars."""
    current_chars = sum(len(msg.get('content', '')) for msg in history)
    # Keep removing the oldest messages (index 1 and 2, skipping potential system prompt at 0)
    # until the total character count is below the limit.
    while current_chars > max_chars and len(history) > 2: # Always keep at least one pair if possible
        # Remove the first user message and the first assistant response
        removed_user = history.pop(0) # Assuming user message is older
        current_chars -= len(removed_user.get('content', ''))
        if history: # Ensure there's another message to remove (the assistant's reply)
            removed_assistant = history.pop(0)
            current_chars -= len(removed_assistant.get('content',''))
        logging.info(f"History trimmed. Current chars: {current_chars}")

    # Simplified trimming (removes oldest message regardless of role):
    # while current_chars > max_chars and len(history) > 1:
    #     removed_message = history.pop(0) # Remove the very first message
    #     current_chars -= len(removed_message.get('content', ''))
    #     logging.info(f"History trimmed. Current chars: {current_chars}")
    return history


# --- Flask Routes ---

@app.route('/')
def index():
    """Displays the chat interface."""
    if 'history' not in session:
        session['history'] = [] # Initialize history for new session
        # Optional: Add an initial system message
        # session['history'].append({"role": "system", "content": "You are chatting with PDF manuals."})
    return render_template('chat.html', history=session['history'])

@app.route('/chat', methods=['POST'])
def chat():
    """Handles user messages and gets bot responses."""
    user_message = request.form.get('message')

    if not user_message:
        # Handle empty submission if required='required' is removed from input
        return redirect(url_for('index'))

    # Ensure history exists in session
    if 'history' not in session:
        session['history'] = []

    # Add user message to history
    session['history'].append({"role": "user", "content": user_message})

    # --- Get Response from RAG Backend ---
    bot_response, error = get_rag_response(user_message)
    # ------------------------------------

    if error:
        session['history'].append({"role": "error", "content": error})
    elif bot_response:
        session['history'].append({"role": "assistant", "content": bot_response})
    else:
        # Handle case where bot gives no response and no error
        session['history'].append({"role": "error", "content": "No response received from the backend."})

    # Trim history after adding new messages
    session['history'] = trim_history(session['history'], MAX_HISTORY_CHARS)

    # Mark session as modified since we changed a mutable list inside it
    session.modified = True

    return redirect(url_for('index')) # Redirect back to display the updated chat

@app.route('/clear')
def clear_chat():
    """Clears the chat history from the session."""
    session.pop('history', None) # Remove history safely
    logging.info("Chat history cleared.")
    return redirect(url_for('index'))

# --- Run the App ---
if __name__ == '__main__':
    # Make sure the static folder is correctly configured relative to this script
    # app.static_folder = 'static'
    # Consider using waitress or gunicorn for production instead of Flask's dev server
    app.run(debug=True, host='0.0.0.0', port=5001) # Run on port 5001 to avoid conflict if server.py uses 5000