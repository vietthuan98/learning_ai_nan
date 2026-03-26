"""
config.py
=========
Trung tâm cấu hình toàn bộ chương trình LLM Hello World.

Phân loại rõ ràng 4 nhóm tham số:
  1. ModelConfig     - Tên model, đường dẫn, nguồn download
  2. InputParams     - Tham số khởi tạo LlamaCpp (context, threads, GPU)
  3. InferenceParams - Tham số điều khiển quá trình sinh token (temperature, top_p...)
  4. OutputParams    - Tham số định dạng đầu ra (streaming, stop tokens)
  5. ChatConfig      - Cấu hình hội thoại (system prompt, lịch sử)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# 1. MODEL CONFIG
#    Chọn TinyLlama 1.1B Q4_K_M — mô hình nhỏ gọn (~670MB), chạy tốt trên CPU
#    yếu, phù hợp laptop không có GPU rời.
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Tên định danh nội bộ
    name: str = "TinyLlama-1.1B-Chat-v1.0"

    # HuggingFace repo chứa file GGUF (TheBloke là kho quantized models phổ biến)
    hf_repo: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"

    # Tên file GGUF cụ thể trong repo
    # Q4_K_M = quantize 4-bit, cân bằng tốt giữa tốc độ và chất lượng
    hf_filename: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    # Thư mục lưu model trên máy local
    model_dir: Path = Path("./models")

    @property
    def model_path(self) -> Path:
        """Đường dẫn đầy đủ tới file GGUF trên local."""
        return self.model_dir / self.hf_filename


# ---------------------------------------------------------------------------
# 2. INPUT PARAMS (Tham số đầu vào — nạp khi khởi tạo LlamaCpp)
#    Đây là các tham số ảnh hưởng đến BỘ NHỚ và TÀI NGUYÊN phần cứng.
# ---------------------------------------------------------------------------
@dataclass
class InputParams:
    # Kích thước cửa sổ ngữ cảnh (context window)
    # 2048 tokens ≈ ~1500 từ tiếng Anh — đủ cho hội thoại vừa
    n_ctx: int = 2048

    # Số luồng CPU dùng cho inference
    # Nên đặt bằng số physical cores, không phải logical (hyperthreading)
    n_threads: int = 4

    # Số lớp transformer chạy trên GPU (VRAM)
    # 0 = chạy hoàn toàn trên CPU — phù hợp máy không có GPU
    n_gpu_layers: int = 0

    # Kích thước batch khi decode (ảnh hưởng tốc độ)
    n_batch: int = 512

    # Tắt log verbose của llama.cpp để output gọn gàng
    verbose: bool = False


# ---------------------------------------------------------------------------
# 3. INFERENCE PARAMS (Tham số inference — điều khiển quá trình sinh token)
#    Đây là các tham số ảnh hưởng đến CHẤT LƯỢNG và TÍNH CÁCH của câu trả lời.
# ---------------------------------------------------------------------------
@dataclass
class InferenceParams:
    # Độ ngẫu nhiên khi chọn token tiếp theo
    # 0.0 = deterministic (luôn chọn token có xác suất cao nhất)
    # 1.0 = rất ngẫu nhiên, sáng tạo
    # 0.7 = cân bằng tốt cho chatbot
    temperature: float = 0.7

    # Số token tối đa trong một câu trả lời
    # 512 tokens ≈ ~380 từ — đủ cho câu trả lời vừa phải
    max_tokens: int = 512

    # Nucleus sampling: chỉ lấy tập token có tổng xác suất <= top_p
    # Loại bỏ các token rất hiếm, tránh "ảo giác" (hallucination)
    top_p: float = 0.95

    # Top-K sampling: chỉ chọn trong K token xác suất cao nhất
    top_k: int = 40

    # Phạt khi mô hình lặp lại token đã dùng
    # 1.0 = không phạt, >1.0 = phạt càng mạnh
    repeat_penalty: float = 1.1

    # Danh sách chuỗi khiến inference dừng lại (stop sequences)
    # Quan trọng: ngăn model "bịa" thêm lượt hội thoại của user
    stop_sequences: List[str] = field(default_factory=lambda: [
        "Human:",
        "User:",
        "<|user|>",
        "</s>",
    ])


# ---------------------------------------------------------------------------
# 4. OUTPUT PARAMS (Tham số đầu ra)
#    Điều khiển CÁCH TRẢ VỀ kết quả cho người dùng.
# ---------------------------------------------------------------------------
@dataclass
class OutputParams:
    # Streaming: trả token về ngay khi vừa sinh — UX tốt hơn, thấy output "gõ ra"
    # False = đợi xong mới trả toàn bộ
    streaming: bool = True

    # In prompt gốc vào output hay không
    echo: bool = False

    # Hiển thị metadata (token count, speed) sau mỗi lượt trả lời
    show_metadata: bool = True


# ---------------------------------------------------------------------------
# 5. CHAT CONFIG (Cấu hình hội thoại)
#    Điều khiển NGỮ CẢNH và BỘ NHỚ hội thoại (stateless — lưu trong memory).
# ---------------------------------------------------------------------------
@dataclass
class ChatConfig:
    # System prompt — định nghĩa "nhân cách" và vai trò của AI
    system_prompt: str = (
        "You are a helpful, friendly, and concise AI assistant. "
        "Answer clearly and accurately. "
        "If you don't know something, say so honestly. "
        "Respond in the same language the user uses."
    )

    # Số lượt hội thoại (user + assistant) tối đa lưu trong history
    # Giới hạn để không vượt quá n_ctx
    # 1 turn = 1 tin nhắn user + 1 tin nhắn assistant
    max_history_turns: int = 10

    # Template định dạng prompt theo chuẩn TinyLlama Chat
    # TinyLlama dùng chuẩn ChatML: <|system|>...<|user|>...<|assistant|>
    chat_template: str = "chatml"


# ---------------------------------------------------------------------------
# SINGLETON INSTANCES — Import trực tiếp từ các module khác
# ---------------------------------------------------------------------------
model_config = ModelConfig()
input_params = InputParams()
inference_params = InferenceParams()
output_params = OutputParams()
chat_config = ChatConfig()
