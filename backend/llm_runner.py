"""
llm_runner.py
=============
Chạy vòng lặp inference LLM và quản lý luồng hội thoại CLI.

Module này là "trái tim" của chương trình — nơi mọi thành phần hội tụ:
  config.py       → tham số đầu vào / inference / đầu ra
  model_loader.py → LlamaCpp instance
  chat_context.py → ChatSession (history + prompt building)

Vòng lặp inference (inference loop) gồm 5 bước:
  1. [INPUT]    Nhận câu hỏi từ người dùng
  2. [CONTEXT]  Chuẩn bị prompt (lịch sử + câu hỏi mới)
  3. [GENERATE] LlamaCpp sinh token từng cái một (autoregressive)
  4. [COLLECT]  Thu thập toàn bộ output + metadata
  5. [STORE]    Lưu câu trả lời vào history để lần sau có context

Luồng dữ liệu:
  User input
    → ChatSession.prepare_prompt()
    → LlamaCpp.invoke() [vòng lặp token generation]
    → InferenceResult (text + metadata)
    → ChatSession.add_assistant_response()
    → Hiển thị cho user
"""

import time
from dataclasses import dataclass
from typing import Optional

from langchain_community.llms import LlamaCpp

from config import inference_params, output_params, chat_config
from chat_context import ChatSession, session_registry


# ---------------------------------------------------------------------------
# OUTPUT DATA MODEL — Tham số đầu ra có cấu trúc
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """
    Đóng gói toàn bộ OUTPUT của một lần inference.

    Không chỉ là chuỗi text — còn bao gồm metadata hữu ích
    cho monitoring, debug, và billing (nếu dùng cloud LLM).
    """
    # ── Nội dung chính ────────────────────────────────────────────────────
    text: str                       # Câu trả lời của LLM

    # ── Metadata thống kê ─────────────────────────────────────────────────
    prompt_tokens: int              # Số token của prompt đầu vào
    completion_tokens: int          # Số token trong câu trả lời
    total_tokens: int               # Tổng = prompt + completion

    # ── Thông tin hiệu năng ───────────────────────────────────────────────
    latency_seconds: float          # Tổng thời gian từ gửi đến nhận xong
    tokens_per_second: float        # Tốc độ sinh token

    # ── Trạng thái kết thúc ───────────────────────────────────────────────
    # "stop"   = gặp stop token/sequence (hoàn chỉnh)
    # "length" = đạt max_tokens (bị cắt ngắn)
    finish_reason: str

    def display_metadata(self):
        """In metadata sau mỗi lượt trả lời."""
        print(
            f"\n  {'─'*50}\n"
            f"  📊 Tokens: {self.prompt_tokens} prompt + "
            f"{self.completion_tokens} completion = {self.total_tokens} total\n"
            f"  ⚡ Tốc độ: {self.tokens_per_second:.1f} tok/s  |  "
            f"Latency: {self.latency_seconds:.2f}s  |  "
            f"Finish: {self.finish_reason}\n"
        )


# ---------------------------------------------------------------------------
# LLM RUNNER — Thực hiện inference và quản lý vòng lặp
# ---------------------------------------------------------------------------

class LLMRunner:
    """
    Điều phối toàn bộ quá trình inference.

    Thiết kế:
      - Nhận LlamaCpp instance (đã được load bởi model_loader.py)
      - Phơi ra run_inference() để llm_runner.py và FastAPI đều dùng được
      - Không giữ state — state nằm trong ChatSession
    """

    def __init__(self, llm: LlamaCpp):
        """
        Args:
            llm: LlamaCpp instance đã được khởi tạo bởi model_loader.build_llm()
        """
        self.llm = llm

    def run_inference(
        self,
        session: ChatSession,
        user_message: str,
    ) -> InferenceResult:
        """
        Thực hiện một lượt inference hoàn chỉnh.

        Đây là nơi VÒNG LẶP INFERENCE xảy ra:
          Bên trong LlamaCpp.invoke(), mô hình thực hiện:
            for each token in [1..max_tokens]:
                logits = forward_pass(current_context)  # Feed-forward
                token  = sample(logits, temperature, top_p, top_k)
                if token in stop_sequences: break
                yield token  # Stream ra client
                current_context.append(token)  # Autoregressive: token mới làm input tiếp

        Args:
            session:      ChatSession hiện tại (chứa history)
            user_message: Câu hỏi mới từ người dùng

        Returns:
            InferenceResult: Câu trả lời + metadata đầy đủ
        """
        # ── BƯỚC 2: CHUẨN BỊ PROMPT ──────────────────────────────────────
        # ChatSession kết hợp system prompt + history + câu hỏi mới
        # thành một chuỗi ChatML format đúng với TinyLlama
        prompt = session.prepare_prompt(user_message)

        # Ước tính token của prompt (để tính completion tokens sau)
        prompt_token_estimate = len(prompt.split()) * 4 // 3  # rough estimate

        # ── BƯỚC 3: GỌI LLM (INFERENCE LOOP bên trong LlamaCpp) ──────────
        print("\n🤖 Assistant: ", end="", flush=True)
        start_time = time.time()

        # LlamaCpp.invoke() chạy vòng lặp token generation:
        # - Streaming=True → token được stream ra stdout qua callback
        # - Kết thúc khi gặp stop sequence hoặc đạt max_tokens
        raw_response = self.llm.invoke(prompt)

        elapsed = time.time() - start_time

        # ── BƯỚC 4: THU THẬP OUTPUT + METADATA ───────────────────────────
        # Clean response: loại bỏ leading/trailing whitespace
        response_text = raw_response.strip()

        # Tính completion tokens
        completion_tokens = len(response_text.split()) * 4 // 3
        total_tokens = prompt_token_estimate + completion_tokens

        # Xác định finish_reason
        finish_reason = "length" if completion_tokens >= inference_params.max_tokens * 0.95 else "stop"

        # Tốc độ sinh token
        tokens_per_second = completion_tokens / elapsed if elapsed > 0 else 0

        result = InferenceResult(
            text=response_text,
            prompt_tokens=prompt_token_estimate,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=elapsed,
            tokens_per_second=tokens_per_second,
            finish_reason=finish_reason,
        )

        # ── BƯỚC 5: LƯU VÀO HISTORY ──────────────────────────────────────
        # Quan trọng: lưu response vào session để lần sau có context
        session.add_assistant_response(response_text)

        return result


# ---------------------------------------------------------------------------
# CLI CHAT LOOP — Chạy chương trình hỏi đáp trên terminal
# ---------------------------------------------------------------------------

def run_cli_chat(llm: LlamaCpp):
    """
    Vòng lặp hội thoại CLI (Command Line Interface).

    Đây là outer loop của chương trình:
      while True:
          input = read_user_input()
          if quit: break
          result = run_inference(input)
          display(result)

    Args:
        llm: LlamaCpp instance đã load
    """
    runner = LLMRunner(llm)

    # Tạo session mới cho cuộc hội thoại này
    session = ChatSession()

    _print_banner(session)

    # ── OUTER LOOP: Vòng lặp hội thoại ─────────────────────────────────────
    # Mỗi iteration = một lượt user hỏi + AI trả lời
    while True:
        try:
            # ── BƯỚC 1: NHẬN INPUT ────────────────────────────────────────
            print()
            user_input = input("👤 Bạn: ").strip()

            if not user_input:
                print("  (Nhập câu hỏi hoặc '/help' để xem lệnh)")
                continue

            # ── XỬ LÝ LỆNH ĐẶC BIỆT ──────────────────────────────────────
            if user_input.lower() in ("/quit", "/exit", "/q"):
                print("\n👋 Tạm biệt! Kết thúc phiên hội thoại.")
                break

            if user_input.lower() in ("/help", "/h"):
                _print_help()
                continue

            if user_input.lower() in ("/reset", "/clear"):
                session.reset()
                print("🔄 Đã xóa lịch sử hội thoại. Bắt đầu lại từ đầu.")
                continue

            if user_input.lower() in ("/stats", "/info"):
                _print_stats(session)
                continue

            if user_input.lower() == "/history":
                _print_history(session)
                continue

            # ── BƯỚC 2-5: CHẠY INFERENCE ──────────────────────────────────
            result = runner.run_inference(session, user_input)

            # ── HIỂN THỊ METADATA (nếu bật) ──────────────────────────────
            if output_params.show_metadata:
                result.display_metadata()

            # Cảnh báo nếu response bị cắt
            if result.finish_reason == "length":
                print("  ⚠️  Câu trả lời bị cắt ngắn (đạt max_tokens). "
                      "Nhập '/continue' để tiếp tục.")

        except KeyboardInterrupt:
            print("\n\n👋 Nhấn Ctrl+C lần nữa để thoát, hoặc tiếp tục nhập.")
            try:
                input()
            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break


# ---------------------------------------------------------------------------
# HELPER DISPLAY FUNCTIONS
# ---------------------------------------------------------------------------

def _print_banner(session: ChatSession):
    """In banner khởi động chương trình."""
    print("\n" + "═" * 60)
    print("  🦙 LLM Hello World  |  TinyLlama 1.1B Chat")
    print("═" * 60)
    print(f"  Session ID : {session.session_id}")
    print(f"  Model      : TinyLlama-1.1B-Chat-v1.0 (Q4_K_M)")
    print(f"  Context    : {chat_config.max_history_turns} turns lịch sử")
    print(f"  Temperature: {inference_params.temperature}")
    print(f"  Max tokens : {inference_params.max_tokens}")
    print(f"  Streaming  : {'✅' if output_params.streaming else '❌'}")
    print("─" * 60)
    print("  Gõ '/help' để xem các lệnh  |  '/quit' để thoát")
    print("═" * 60)


def _print_help():
    """In danh sách lệnh."""
    print("""
  📋 Các lệnh hỗ trợ:
  ─────────────────────────────────────────
  /help    — Hiển thị trợ giúp này
  /reset   — Xóa lịch sử, bắt đầu hội thoại mới
  /history — Xem lịch sử hội thoại hiện tại
  /stats   — Xem thống kê session
  /quit    — Thoát chương trình
  ─────────────────────────────────────────
    """)


def _print_stats(session: ChatSession):
    """In thống kê session."""
    stats = session.get_stats()
    print(f"""
  📊 Thống kê session:
  ─────────────────────────────────────────
  Session ID : {stats['session_id']}
  Số lượt    : {stats['turn_count']} / {stats['max_turns']} turns
  Tokens ước : ~{stats['total_tokens_est']} tokens trong context
  Uptime     : {stats['uptime_seconds']}s
  ─────────────────────────────────────────
    """)


def _print_history(session: ChatSession):
    """In lịch sử hội thoại."""
    messages = session.history.messages
    non_system = [m for m in messages if m.role.value != "system"]

    if not non_system:
        print("  (Chưa có lịch sử hội thoại)")
        return

    print(f"\n  📜 Lịch sử ({len(non_system)} messages):")
    print("  " + "─" * 50)
    for msg in non_system:
        icon = "👤" if msg.role.value == "user" else "🤖"
        content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"  {icon} [{msg.role.value:10s}] {content_preview}")
    print("  " + "─" * 50)


# ---------------------------------------------------------------------------
# STANDALONE ENTRY POINT — chạy trực tiếp file này
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from model_loader import load_model

    print("[llm_runner] Đang khởi động LLM Hello World...")

    # Load model (download nếu cần + khởi tạo LlamaCpp)
    llm = load_model()

    # Bắt đầu vòng lặp hội thoại CLI
    run_cli_chat(llm)
