# 🚀 LLM Hello World: TinyLlama Chat

This project provides a full-stack AI Chat application, featuring a **TinyLlama** model served via **FastAPI** on the backend and a modern **Mesop** sidebar interface on the frontend.

## 📋 Prerequisites

- **Python: 3.10 to 3.12** is recommended for compatibility with `llama-cpp-python` and `mesop`.
- **System**: macOS, Linux, or Windows (via WSL2 recommended for backend).

---

## 🛠️ Step 1: Backend Setup (API Server)

The backend uses `llama-cpp-python` to serve a TinyLlama GGUF model through a FastAPI service.

1.  **Navigate to the backend directory**:
    ```bash
    cd backend
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    > **Tip**: If you have an NVIDIA GPU, follow the instructions in `backend/requirements.txt` to enable CUDA acceleration.

4.  **Download the model**:
    Run the following command to download the required TinyLlama GGUF model:
    ```bash
    python3 model_loader.py
    ```

5.  **Start the API server**:
    ```bash
    python3 main.py
    ```
    The server will be available at `http://localhost:8000`. You can visit `http://localhost:8000/docs` for the interactive API documentation.

---

## 🎨 Step 2: Frontend Setup (Mesop UI)

The frontend provides a polished interface with a sidebar for managing chat sessions.

1.  **Open a new terminal window** and navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start the Mesop UI**:
    ```bash
    mesop main.py
    ```
    The UI will be accessible at `http://localhost:32123`.

---

## 📁 Project Overview

- **`backend/`**: FastAPI server, model loading logic, and session state management.
- **`frontend/`**: Modular Mesop components (`sidebar`, `session_row`, `chat_main_area`) and API integration.

## ✨ Features

- **Local LLM**: TinyLlama running entirely on your machine.
- **Persistent Sessions**: Create multiple chat rooms and switch between them.
- **Real-time Streaming**: Watch the AI response generate token-by-token.
- **Automatic History Sync**: Chat history is fetched and restored automatically when switching rooms.
