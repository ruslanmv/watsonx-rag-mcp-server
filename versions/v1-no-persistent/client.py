import asyncio
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure basic logging for the client
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Client] %(message)s')

async def main():
    # Command to start the server.py script
    # Ensure 'python' is in your PATH or provide the full path.
    server_params = StdioServerParameters(command="python", args=["server.py"])
    logging.info("Attempting to start and connect to MCP server...")

    try:
        async with stdio_client(server_params) as (reader, writer):
            logging.info("Connected to MCP server. Initializing session...")
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                logging.info("MCP session initialized.")

                # The question you want to ask the RAG system
                user_msg = "Give me the flight modes of this drone please"
                # Or try another question: "What is the maximum flight time?"

                logging.info("Calling tool 'chat_with_manual' with query: %r", user_msg)

                try:
                    # Call the specific RAG tool defined in server.py
                    response = await session.call_tool(
                        "chat_with_manual",
                        arguments={"query": user_msg}
                    )
                    logging.info("Received response from server.")
                    print("\n" + "="*20 + " Query " + "="*20)
                    print(f"Your Question: {user_msg}")
                    print("\n" + "="*20 + " Answer " + "="*20)
                    print(f"Bot Answer:\n{response}") # Print the final answer from the RAG tool
                    print("\n" + "="*50)

                except Exception as e:
                    logging.error("Error calling tool 'chat_with_manual': %s", e, exc_info=True)
                    print(f"\nError interacting with the chatbot: {e}")

    except Exception as e:
        logging.error("Failed to connect or communicate with the MCP server: %s", e, exc_info=True)
        print(f"\nCould not connect to the server: {e}")

if __name__ == "__main__":
    asyncio.run(main())