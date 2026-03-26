"""
main.py
=======
FastAPI server — API layer cho LLM Hello World.

Đây là entry point khi chạy ứng dụng dưới dạng web service.
Hiện tại là TEMPLATE — các endpoint được khai báo với placeholder,
sẵn sàng để implement logic thực tế từ llm_runner.py.

Kiến trúc API:
  POST /chat              → Gửi tin nhắn, nhận câu trả lời (stateful session)
  POST /chat/stream       → Gửi tin nhắn, nhận stream response (SSE)
  GET  /sessions/{id}     → Lấy thông tin session
  DELETE /sessions/{id}   → Xóa session
  GET  /health            → Health check
  GET  /model/info        → Thông tin model đang chạy

Chạy server:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Truy cập docs:
  http://localhost:8000/docs       (Swagger UI)
  http://localhost:8000/redoc      (ReDoc)
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import model_config, inference_params, input_params, chat_config


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMAS — Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body cho POST /chat"""
    message: str = Field(
        ...,
        description="Câu hỏi / tin nhắn của người dùng",
        example="Hà Nội có gì hay?",
        min_length=1,
        max_length=2000,
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID để tiếp tục hội thoại cũ. "
                    "Bỏ trống để tạo session mới.",
        example="abc12345",
    )


class ChatResponse(BaseModel):
    """Response body cho POST /chat"""
    session_id: str = Field(description="Session ID (dùng cho các request tiếp theo)")
    message: str = Field(description="Câu trả lời của AI")
    finish_reason: str = Field(description="'stop' hoặc 'length'")
    usage: dict = Field(description="Thống kê token usage")
    latency_seconds: float = Field(description="Thời gian inference (giây)")


class SessionInfo(BaseModel):
    """Response body cho GET /sessions/{session_id}"""
    session_id: str
    turn_count: int
    max_turns: int
    total_tokens_estimate: int
    uptime_seconds: float
    history: list


class ModelInfo(BaseModel):
    """Response body cho GET /model/info"""
    name: str
    filename: str
    n_ctx: int
    n_threads: int
    n_gpu_layers: int
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int


class HealthResponse(BaseModel):
    """Response body cho GET /health"""
    status: str
    model_loaded: bool
    active_sessions: int
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# APP STATE — Giữ LLM instance và session registry
# ---------------------------------------------------------------------------

class AppState:
    """
    Giữ các object cần dùng chung trong suốt lifetime của app.
    Được khởi tạo trong lifespan và gắn vào app.state.
    """
    llm = None           # LlamaCpp instance
    runner = None        # LLMRunner instance


# ---------------------------------------------------------------------------
# LIFESPAN — Khởi tạo và dọn dẹp khi server start/stop
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    FastAPI lifespan handler:
      - startup : load model vào memory
      - shutdown: giải phóng tài nguyên
    """
    # ── STARTUP ────────────────────────────────────────────────────────────
    print("[main] 🚀 Server đang khởi động...")
    print("[main] ℹ️  Model sẽ được load khi có request đầu tiên (lazy loading)")
    print("[main]    Hoặc uncomment đoạn dưới để load ngay khi start:\n")
    print("[main]    # from model_loader import load_model")
    print("[main]    # from llm_runner import LLMRunner")
    print("[main]    # app.state.llm = load_model()")
    print("[main]    # app.state.runner = LLMRunner(app.state.llm)")
    print()

    # Eager loading model khi server start:
    from model_loader import load_model
    from llm_runner import LLMRunner
    app.state.llm    = load_model()
    app.state.runner = LLMRunner(app.state.llm)

    app.state.model_loaded = True

    yield  # Server đang chạy

    # ── SHUTDOWN ───────────────────────────────────────────────────────────
    print("[main] 🛑 Server đang shutdown...")
    # Cleanup nếu cần (đóng DB connections, etc.)


# ---------------------------------------------------------------------------
# FASTAPI APP INIT
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Hello World API",
    description=(
        "API server cho LLM Hello World sử dụng TinyLlama + LangChain LlamaCpp.\n\n"
        "Hỗ trợ hội thoại có nhớ lịch sử theo session."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — cho phép frontend (React, Vue...) gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Production: thay bằng domain cụ thể
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Dùng để load balancer / monitoring kiểm tra server còn sống không.
    """
    from chat_context import session_registry

    return HealthResponse(
        status="ok",
        model_loaded=getattr(app.state, "model_loaded", False),
        active_sessions=session_registry.active_sessions,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["System"])
async def get_model_info():
    """
    Trả về thông tin cấu hình model hiện tại.
    Client dùng để hiển thị model đang dùng và các tham số.
    """
    return ModelInfo(
        name=model_config.name,
        filename=model_config.hf_filename,
        n_ctx=input_params.n_ctx,
        n_threads=input_params.n_threads,
        n_gpu_layers=input_params.n_gpu_layers,
        temperature=inference_params.temperature,
        max_tokens=inference_params.max_tokens,
        top_p=inference_params.top_p,
        top_k=inference_params.top_k,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    x_session_id: Optional[str] = Header(None, description="Session ID từ header"),
):
    """
    Endpoint chính để hội thoại với LLM.

    - Nếu `session_id` trong body hoặc header `X-Session-Id` → tiếp tục hội thoại cũ
    - Nếu không có → tạo session mới
    - Trả về `session_id` để client dùng cho request tiếp theo

    **TODO:** Implement logic thực tế:
    ```python
    session = session_registry.get_or_create(request.session_id or x_session_id)
    result  = app.state.runner.run_inference(session, request.message)
    return ChatResponse(session_id=session.session_id, ...)
    ```
    """
    from chat_context import session_registry

    # 1. Xác định Session
    session_id = request.session_id or x_session_id
    session    = session_registry.get_or_create(session_id)

    # 2. Chạy inference
    # runner đã được khởi tạo trong lifespan
    result = app.state.runner.run_inference(session, request.message)

    # 3. Trả về kết quả
    return ChatResponse(
        session_id      = session.session_id,
        message         = result.text,
        finish_reason   = result.finish_reason,
        usage           = {
            "prompt_tokens"     : result.prompt_tokens,
            "completion_tokens" : result.completion_tokens,
            "total_tokens"      : result.total_tokens,
        },
        latency_seconds = result.latency_seconds,
    )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    x_session_id: Optional[str] = Header(None),
):
    from chat_context import session_registry

    # 1. Xác định Session
    session_id = request.session_id or x_session_id
    session    = session_registry.get_or_create(session_id)

    # 2. Chuẩn bị prompt
    prompt = session.prepare_prompt(request.message)

    # 3. Định nghĩa generator cho SSE
    async def token_generator():
        full_response = ""
        # Sử dụng astream của LangChain
        async for chunk in app.state.llm.astream(prompt):
            full_response += chunk
            # Format SSE: data: <content>\n\n
            yield f"data: {chunk}\n\n"

        # Sau khi stream xong, lưu vào history
        session.add_assistant_response(full_response)
        yield "data: [DONE]\n\n"

    # 4. Trả về StreamingResponse
    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session.session_id,
        },
    )


@app.get("/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def get_session(session_id: str):
    """
    Lấy thông tin chi tiết của một session.
    Bao gồm lịch sử hội thoại và thống kê token.
    """
    from chat_context import session_registry

    session = session_registry._sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' không tồn tại hoặc đã hết hạn."
        )

    stats = session.get_stats()
    return SessionInfo(
        session_id=stats["session_id"],
        turn_count=stats["turn_count"],
        max_turns=stats["max_turns"],
        total_tokens_estimate=stats["total_tokens_est"],
        uptime_seconds=stats["uptime_seconds"],
        history=session.history.to_list(),
    )


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    """
    Xóa session khỏi memory.
    Dùng khi user muốn bắt đầu hội thoại mới hoặc logout.
    """
    from chat_context import session_registry

    deleted = session_registry.delete(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' không tìm thấy."
        )

    return {"message": f"Session '{session_id}' đã được xóa.", "status": "deleted"}


@app.delete("/sessions/{session_id}/history", tags=["Sessions"])
async def reset_session_history(session_id: str):
    """
    Xóa lịch sử hội thoại nhưng giữ session.
    Tương đương '/reset' trong CLI.
    """
    from chat_context import session_registry

    session = session_registry._sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' không tồn tại."
        )

    session.reset()
    return {"message": "Lịch sử đã được xóa.", "session_id": session_id}


# ---------------------------------------------------------------------------
# DEVELOPMENT ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  LLM Hello World — FastAPI Server")
    print("=" * 60)
    print("  Docs: http://localhost:8000/docs")
    print("  API : http://localhost:8000")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,         # Auto-reload khi code thay đổi (dev mode)
        log_level="info",
    )
