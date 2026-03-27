# 🦙 LLM Hello World - Project Context

This project is a local LLM-powered question-answering system featuring **TinyLlama 1.1B** integrated via **LangChain** and **LlamaCpp**. It is designed to run entirely offline on standard hardware (CPU-only support included).

## 🏗️ Architecture & Core Components

The project follows a modular, stateless design where each component has a specific responsibility:

-   **`config.py`**: The **Single Source of Truth**. Contains all parameters for model selection (`ModelConfig`), hardware resources (`InputParams`), inference behavior (`InferenceParams`), and chat personality (`ChatConfig`).
-   **`model_loader.py`**: Responsible for lazy-loading the model. It automatically downloads the GGUF file from HuggingFace (TheBloke repo) if not present and initializes the `LlamaCpp` instance.
-   **`chat_context.py`**: Manages the application state. It uses a **sliding window** mechanism for chat history to stay within the 2048-token context limit and formats prompts using the **ChatML** template (`<|system|>`, `<|user|>`, `<|assistant|>`).
-   **`llm_runner.py`**: Orchestrates the 5-step inference loop (Input → Context → Generate → Collect → Store) and provides the CLI interactive chat interface.
-   **`main.py`**: A FastAPI-based web server that exposes the LLM capabilities via REST endpoints (currently a template ready for implementation).

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Recommended: `venv` or `conda` environment.

### Setup & Installation
1.  **Initialize Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  **Install Dependencies**:
    - **CPU-only**: `pip install -r requirements.txt`
    - **NVIDIA GPU (CUDA)**: 
      ```bash
      CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.90
      pip install -r requirements.txt --ignore-installed llama-cpp-python
      ```

### Running the Application
-   **CLI Interface**: `python llm_runner.py`
    -   *Note: Automatically downloads the ~670MB model on the first run.*
-   **FastAPI Server**: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
    -   Swagger Docs: `http://localhost:8000/docs`

## 🛠️ Development Conventions

### 1. Configuration First
Always modify `config.py` to adjust model behavior (e.g., `temperature`, `max_tokens`, `n_gpu_layers`). Avoid hardcoding these values in logic files.

### 2. Stateless Design
The system is designed to be stateless. Sessions are currently stored in-memory (`SessionRegistry` in `chat_context.py`). For production, this should be extended to use Redis or a database.

### 3. Prompting Standard
The project strictly follows the **ChatML** format required by TinyLlama 1.1B Chat:
```text
<|system|>
{system_prompt}</s>
<|user|>
{user_query}</s>
<|assistant|>
```

### 4. Inference Loop Steps
When modifying the generation logic, adhere to the established lifecycle:
1.  **Prepare Prompt**: Build ChatML string from history.
2.  **Invoke LLM**: Generate tokens (supporting streaming).
3.  **Collect Result**: Gather text and performance metadata (tokens/s, latency).
4.  **Store History**: Update the `ChatSession` with the assistant's response.
5.  **Output**: Deliver the response to the user.

## 🗺️ Roadmap & TODOs
- [ ] Implement the logic for `/chat` and `/chat/stream` endpoints in `main.py`.
- [ ] Add persistence for chat sessions (e.g., SQLite or Redis).
- [ ] Implement Docker support for easier deployment.
- [ ] Add support for alternative GGUF models via `config.py`.
