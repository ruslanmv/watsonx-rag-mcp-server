# Containerizing and Running the Flask RAG Chatbot

This guide explains how to build a Docker image for the Flask RAG chatbot application and run it as a container.


## Prerequisites

- **Docker:**  
  You need Docker installed and running on your system. Download it from [Docker's website](https://www.docker.com/products/docker-desktop/).

- **Project Files:**  
  Ensure you have the following files and directories structured correctly:

  ```
  ├── Dockerfile
  ├── requirements.txt
  ├── frontend.py
  ├── server.py
  ├── templates/
  │   ├── base.html
  │   └── chat.html
  ├── static/
  │   └── assets/
  │       └── watsonx-wallpaper.jpg
  ├── .env
  ├── documents/
  │   └── ... (your PDFs)
  └── chroma_db_data/
  ```

---

## Step 1: Create `requirements.txt`

Ensure you have a `requirements.txt` in the project root that includes:

- `Flask`
- `meta-compute-protocol`
- `python-dotenv`
- `ibm-watsonx-ai`
- `chromadb`
- `pypdf`
- *(and any other dependencies)*

---

## Step 2: Create `Dockerfile`

Create a file named `Dockerfile` (no extension) in your project root and paste in the Dockerfile contents. This sets up the Python environment, installs dependencies, and configures the application.

---

## Step 3: Build the Docker Image

Navigate to your project directory and run:

```bash
docker build -t watsonx-rag-chatbot .
```

**Command Breakdown:**

- `docker build`: Builds the image.
- `-t watsonx-rag-chatbot`: Tags the image.
- `.`: Uses the current directory as the build context.

> **Note:** First-time builds may take longer due to downloading the Python base image and dependencies.

---

## Step 4: Run the Docker Container

### On Linux/macOS:

```bash
docker run --rm -p 5001:5001 \
  -v "$(pwd)/.env:/app/.env" \
  -v "$(pwd)/documents:/app/documents" \
  -v "$(pwd)/chroma_db_data:/app/chroma_db_data" \
  --name rag-chat-container \
  watsonx-rag-chatbot
```

### On Windows (Command Prompt):

```cmd
docker run --rm -p 5001:5001 ^
  -v "%cd%/.env:/app/.env" ^
  -v "%cd%/documents:/app/documents" ^
  -v "%cd%/chroma_db_data:/app/chroma_db_data" ^
  --name rag-chat-container ^
  watsonx-rag-chatbot
```

### On Windows (PowerShell):

```powershell
docker run --rm -p 5001:5001 `
  -v "${PWD}/.env:/app/.env" `
  -v "${PWD}/documents:/app/documents" `
  -v "${PWD}/chroma_db_data:/app/chroma_db_data" `
  --name rag-chat-container `
  watsonx-rag-chatbot
```

**Command Breakdown:**

- `docker run`: Start a container from an image.
- `--rm`: Remove container after it exits.
- `-p 5001:5001`: Port mapping for localhost access.
- `-v <host>:<container>`: Mount local folders into container:
  - `.env`: For environment variables.
  - `documents`: For source PDFs.
  - `chroma_db_data`: For persistent vector DB storage.
- `--name`: Assigns a readable container name.
- `watsonx-rag-chatbot`: The image name.

---

## Step 5: Access the Application

Once running, open your browser and go to:

```
http://localhost:5001
```

You should see the RAG chatbot interface.

---

## Notes and Considerations

- **`.env` Security:**  
  Be careful not to expose sensitive values. For production, consider alternatives like `docker run -e` or Docker secrets.

- **Persistence:**  
  The `chroma_db_data` volume ensures your vector database persists across container restarts.

- **Stopping the Container:**
  - If running in the foreground, use `Ctrl+C`.
  - For background mode (`-d`), stop with:

    ```bash
    docker stop rag-chat-container
    ```

- **Removing the Container:**
  - If `--rm` wasn’t used, remove with:

    ```bash
    docker rm rag-chat-container
    ```

- **Production Deployment:**
  - Replace Flask's dev server with a WSGI server like Gunicorn or Waitress.
  - Update your `Dockerfile` and `requirements.txt` accordingly.