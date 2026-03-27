# 🦙 LLM Hello World

Chương trình hỏi đáp sử dụng **TinyLlama 1.1B** qua **LangChain + LlamaCpp**.  
Chạy hoàn toàn **local**, không cần internet sau khi download model, không cần GPU.

---

## 📁 Cấu trúc project

```
llm-hello-world/
│
├── config.py          # ⚙️  Toàn bộ tham số: model, input, inference, output, chat
├── model_loader.py    # 📦  Download model từ HuggingFace + khởi tạo LlamaCpp
├── chat_context.py    # 🧠  Application Context: ChatSession, lịch sử hội thoại
├── llm_runner.py      # 🚀  Inference loop + CLI chat
├── main.py            # 🌐  FastAPI server template
│
├── models/            # 📂  Thư mục lưu file GGUF (tự tạo, không commit)
├── requirements.txt
├── .gitignore
└── README.md
```

### Vai trò từng file

| File | Trách nhiệm |
|------|-------------|
| `config.py` | **Single source of truth** cho mọi tham số. Thay đổi model/temperature/max_tokens ở đây. |
| `model_loader.py` | Download GGUF + khởi tạo `LlamaCpp`. Tách ra để dễ swap model. |
| `chat_context.py` | Quản lý `ChatSession` stateless: lịch sử hội thoại (sliding window), build prompt ChatML. |
| `llm_runner.py` | Vòng lặp inference 5 bước + CLI chat loop. Entry point chính. |
| `main.py` | FastAPI server: khai báo endpoints, schema, sẵn sàng để implement. |

---

## 🚀 Cài đặt & Chạy

### 1. Tạo virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# hoặc: venv\Scripts\activate   # Windows
```

### 2. Cài dependencies

```bash
# CPU-only (máy không có GPU):
pip install -r requirements.txt

# Nếu có NVIDIA GPU (CUDA):
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.90
pip install -r requirements.txt --ignore-installed llama-cpp-python
```

### 3. Chạy chương trình CLI (recommended để thử trước)

```bash
python3 llm_runner.py
```

Lần đầu chạy sẽ tự động download model (~670MB). Sau đó bắt đầu chat.

### 4. Chạy FastAPI server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập: `http://localhost:8000/docs`

### 5. Chạy bằng Docker (Khuyên dùng cho Deployment)

Dockerfile đã được cấu hình để tự động download model trong quá trình build, giúp container có thể chạy ngay lập tức mà không cần internet.

**Build image:**
```bash
docker build -t llm-backend .
```

**Chạy container:**
```bash
docker run -d -p 8000:8000 --name ai-chat-be llm-backend
```


---

## 💬 Sử dụng CLI

```
═══════════════════════════════════════════════════════════
  🦙 LLM Hello World  |  TinyLlama 1.1B Chat
═══════════════════════════════════════════════════════════
  Session ID : a3f2b1c4
  Temperature: 0.7
  Max tokens : 512
  Streaming  : ✅
───────────────────────────────────────────────────────────

👤 Bạn: Hà Nội có gì hay?

🤖 Assistant: Hà Nội là thủ đô của Việt Nam với nhiều điểm tham quan...

  ──────────────────────────────────────────────────────
  📊 Tokens: 145 prompt + 87 completion = 232 total
  ⚡ Tốc độ: 18.3 tok/s  |  Latency: 4.75s  |  Finish: stop
```

### Lệnh đặc biệt

| Lệnh | Chức năng |
|------|-----------|
| `/help` | Xem danh sách lệnh |
| `/reset` | Xóa lịch sử, bắt đầu hội thoại mới |
| `/history` | Xem lịch sử hội thoại |
| `/stats` | Xem thống kê session (tokens, turns) |
| `/quit` | Thoát chương trình |

---

## ⚙️ Cấu hình

Tất cả tham số nằm trong **`config.py`**:

```python
# Đổi model
model_config.hf_filename = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"  # chất lượng cao hơn

# Tăng sáng tạo
inference_params.temperature = 0.9

# Tăng độ dài trả lời
inference_params.max_tokens = 1024

# Dùng GPU (nếu có)
input_params.n_gpu_layers = 32
```

---

## 🧠 Kiến trúc Inference Loop

```
User Input
    │
    ▼
[1] ChatSession.prepare_prompt()
    ├── Thêm user message vào history
    └── Build ChatML prompt (system + history + câu hỏi mới)
    │
    ▼
[2] LlamaCpp.invoke(prompt)
    ┌─────────────────────────────────┐
    │  VÒNG LẶP INFERENCE             │
    │  for token in [1..max_tokens]:  │
    │    logits = forward_pass()      │
    │    token  = sample(temperature) │
    │    if stop_token: break         │
    │    yield token  ← streaming     │
    └─────────────────────────────────┘
    │
    ▼
[3] Thu thập InferenceResult
    ├── text, finish_reason
    └── tokens, latency, speed
    │
    ▼
[4] ChatSession.add_assistant_response()
    └── Lưu vào history (cho lượt tiếp theo)
    │
    ▼
[5] Hiển thị cho user
```

---

## 📦 Model

| | TinyLlama Q4_K_M |
|---|---|
| **Kích thước** | ~670 MB |
| **RAM cần** | ~1.5 GB |
| **Tốc độ (CPU)** | 15–25 tok/s |
| **Context** | 2048 tokens |
| **Phù hợp** | Máy yếu, laptop cũ, không GPU |

---

## 🗺️ Roadmap

- [x] CLI chat với lịch sử hội thoại
- [x] FastAPI server template
- [ ] Implement `/chat` endpoint
- [ ] Streaming SSE với `/chat/stream`
- [ ] Tích hợp Redis để persist sessions
- [ ] Docker support

---

## 📝 License

MIT
