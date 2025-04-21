# Containerizing and Running the Flask RAG Chatbot

This guide explains how to build a Docker image for the Flask RAG chatbot application, run it as a container ensuring the `.env` file is loaded correctly, and push/pull the image to/from a registry.



## Prerequisites

- **Docker:**  
  You need Docker installed and running on your system. Download from [Docker's website](https://www.docker.com/products/docker-desktop/).

- **Docker Hub Account (Optional):**  
  If you want to push/pull images, you'll need an account on [Docker Hub](https://hub.docker.com/) or another container registry.

- **Project Files:**  
  Ensure the following file structure (updated to **not copy `.env`** into the image):

  ```
  /watsonx-ragmcp-server
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


## Step 1: Create `requirements.txt`

List all Python dependencies in a `requirements.txt` file. Example:

```
python-dotenv>=0.21.0
PyPDF2>=3.0.0
chromadb>=0.4.0
ibm-watsonx-ai>=1.3.8
mcp[cli]>=1.6.0
pypdf
pycryptodome
flask==3.1.0 
```


## Step 2: Create `Dockerfile`

Create a `Dockerfile` in your project root using a version that **does not copy the `.env` file**. This ensures sensitive data remains outside the image and is only mounted at runtime.



## Step 3: Build the Docker Image

In the terminal, run the following in your project directory:

```bash
docker build -t watsonx-rag-chatbot .
```

**Command Breakdown:**

- `docker build`: Builds a new image.
- `-t watsonx-rag-chatbot`: Tags the image.
- `.`: Uses the current directory as the build context.

> This build excludes the `.env` file.



## Step 4: Run the Docker Container (Including `.env`)

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

- `--rm`: Automatically removes the container after it stops.
- `-p 5001:5001`: Maps local port to container port.
- `-v "$(pwd)/.env:/app/.env"`: Mounts the local `.env` file at runtime.
- Other volume mounts allow document access and persistent DB storage.
- `--name`: Names the container.
- `watsonx-rag-chatbot`: Uses the built image.



## Step 5: Access the Application

Visit [http://localhost:5001](http://localhost:5001) in your browser to access the chatbot UI.

---

## Step 6: Pushing the Image to a Registry (Optional)

To share your image or deploy elsewhere:

### 1. Log In to Docker Hub

```bash
docker login
```

Enter your Docker Hub credentials when prompted.

### 2. Tag the Image

```bash
docker tag watsonx-rag-chatbot ruslanmv/watsonx-rag-chatbot:latest
```

- Replace `ruslanmv` with your Docker Hub username.

### 3. Push the Image

```bash
docker push ruslanmv/watsonx-rag-chatbot:latest
```


## Step 7: Pulling and Running the Image Elsewhere (Optional)

On another machine:

### 1. Pull the Image

```bash
docker pull ruslanmv/watsonx-rag-chatbot:latest
```

### 2. Run the Container

Ensure local copies of `.env`, `documents`, and `chroma_db_data` exist.

```bash
docker run --rm -p 5001:5001 \
  -v "$(pwd)/.env:/app/.env" \
  -v "$(pwd)/documents:/app/documents" \
  -v "$(pwd)/chroma_db_data:/app/chroma_db_data" \
  --name rag-chat-container \
  ruslanmv/watsonx-rag-chatbot:latest
```

> Adjust volume paths for Windows as shown earlier.



## Notes and Considerations

- **`.env` Security:**  
  Mounting via volume is common in development. For production, consider using `-e VARIABLE=value` or Docker secrets.

- **Persistence:**  
  The `chroma_db_data` volume allows vector index reuse across runs.

- **Stopping/Removing:**  
  - Use `Ctrl+C` to stop foreground containers.  
  - Use `docker stop rag-chat-container` and `docker rm rag-chat-container` if run detached.