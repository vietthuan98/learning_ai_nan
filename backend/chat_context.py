"""
chat_context.py
===============
Quản lý Application Context cho chương trình hỏi đáp LLM.

Thiết kế STATELESS (không lưu trạng thái vào DB/file):
  - Lịch sử hội thoại sống trong RAM, gắn với một ChatSession object
  - Mỗi lần gọi API → client gửi lại toàn bộ history (hoặc server giữ trong memory)
  - Session hết khi process kết thúc hoặc client không gửi nữa

Trách nhiệm của module này:
  1. ChatMessage     — Data class đại diện một tin nhắn
  2. ChatHistory     — Danh sách tin nhắn có giới hạn (sliding window)
  3. PromptBuilder   — Chuyển history → formatted prompt string cho LlamaCpp
  4. ChatSession     — Kết hợp history + builder, là "context" của một cuộc trò chuyện

Tại sao tách riêng?
  - Logic context/memory không phụ thuộc vào LlamaCpp hay LangChain
  - Dễ thay thế bằng vector store memory sau này
  - Unit testable độc lập
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from config import chat_config, ChatConfig


# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """Các vai trong hội thoại."""
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """
    Đại diện một tin nhắn đơn trong hội thoại.

    Attributes:
        role:       Ai nói — system / user / assistant
        content:    Nội dung tin nhắn
        timestamp:  Unix timestamp khi tạo
        token_count: Ước tính số token (lazy, tính khi cần)
    """
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: Optional[int] = None

    def estimate_tokens(self) -> int:
        """
        Ước tính số token (quy tắc thực tế: 1 token ≈ 0.75 từ tiếng Anh).
        Dùng để theo dõi context window usage.
        """
        if self.token_count is None:
            # Rough estimate: words / 0.75
            word_count = len(self.content.split())
            self.token_count = int(word_count / 0.75) + 4  # +4 cho special tokens
        return self.token_count

    def to_dict(self) -> dict:
        """Serialize để trả về API response."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# CHAT HISTORY — Sliding window memory
# ---------------------------------------------------------------------------

class ChatHistory:
    """
    Lưu trữ lịch sử hội thoại dưới dạng sliding window.

    Stateless design:
      - Không ghi vào DB hay file
      - Khi đạt giới hạn max_turns → tự động xóa lượt cũ nhất
      - System message luôn được giữ (không bị xóa)

    Tại sao cần sliding window?
      LlamaCpp có giới hạn n_ctx tokens. Nếu history quá dài → overflow.
      Sliding window đảm bảo prompt luôn nằm trong giới hạn n_ctx.
    """

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: số lượt (user+assistant) tối đa lưu trong memory.
                       1 turn = 1 user message + 1 assistant message = 2 messages.
        """
        self.max_turns = max_turns
        self._messages: List[ChatMessage] = []

    def add_message(self, role: Role, content: str) -> ChatMessage:
        """Thêm một tin nhắn mới vào history."""
        msg = ChatMessage(role=role, content=content.strip())
        self._messages.append(msg)
        self._enforce_limit()
        return msg

    def _enforce_limit(self):
        """
        Giữ history trong giới hạn max_turns.
        System message LUÔN được bảo vệ — không bao giờ bị xóa.
        Xóa từ tin nhắn cũ nhất (index 1 nếu có system, index 0 nếu không).
        """
        # Tách system message ra khỏi danh sách để bảo vệ
        system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
        non_system  = [m for m in self._messages if m.role != Role.SYSTEM]

        # max_turns * 2 vì mỗi turn = 2 messages (user + assistant)
        max_non_system = self.max_turns * 2
        if len(non_system) > max_non_system:
            # Cắt bỏ tin nhắn cũ nhất (đầu danh sách)
            non_system = non_system[-max_non_system:]

        # Ghép lại: system message luôn đứng đầu
        self._messages = system_msgs + non_system

    @property
    def messages(self) -> List[ChatMessage]:
        """Trả về toàn bộ messages (read-only view)."""
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        """Số lượt hội thoại thực tế (không tính system message)."""
        user_msgs = [m for m in self._messages if m.role == Role.USER]
        return len(user_msgs)

    @property
    def total_estimated_tokens(self) -> int:
        """Tổng ước tính token của toàn bộ history."""
        return sum(m.estimate_tokens() for m in self._messages)

    def clear(self, keep_system: bool = True):
        """
        Xóa toàn bộ lịch sử.

        Args:
            keep_system: Giữ lại system message (True) hay xóa luôn (False)
        """
        if keep_system:
            self._messages = [m for m in self._messages if m.role == Role.SYSTEM]
        else:
            self._messages = []

    def to_list(self) -> List[dict]:
        """Serialize toàn bộ history để trả về API."""
        return [m.to_dict() for m in self._messages]


# ---------------------------------------------------------------------------
# PROMPT BUILDER — Chuyển history thành formatted string
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Chuyển đổi ChatHistory → formatted prompt string cho LlamaCpp.

    TinyLlama dùng ChatML format:
        <|system|>
        {system_content}</s>
        <|user|>
        {user_content}</s>
        <|assistant|>
        {assistant_content}</s>
        <|user|>
        {next_user_content}</s>
        <|assistant|>

    Dấu </s> là EOS token của TinyLlama — báo hiệu kết thúc một "turn".
    Prompt kết thúc bằng <|assistant|> để kích hoạt model bắt đầu generate.
    """

    # Mapping Role → ChatML tag
    _ROLE_TAGS = {
        Role.SYSTEM:    "<|system|>",
        Role.USER:      "<|user|>",
        Role.ASSISTANT: "<|assistant|>",
    }

    def build(self, history: ChatHistory) -> str:
        """
        Xây dựng full prompt từ toàn bộ chat history.

        Args:
            history: ChatHistory object chứa tất cả messages

        Returns:
            str: Formatted prompt sẵn sàng nạp vào LlamaCpp
        """
        parts = []

        for msg in history.messages:
            tag = self._ROLE_TAGS[msg.role]
            # Mỗi message: <|role|>\n{content}</s>\n
            parts.append(f"{tag}\n{msg.content}</s>")

        # Kết thúc bằng <|assistant|> để model bắt đầu generate
        parts.append("<|assistant|>")

        return "\n".join(parts)

    def build_with_new_message(
        self,
        history: ChatHistory,
        user_message: str
    ) -> str:
        """
        Xây dựng prompt với tin nhắn user mới (chưa add vào history).
        Dùng khi muốn preview prompt trước khi thực sự inference.

        Args:
            history:      History hiện tại (không bao gồm tin nhắn mới)
            user_message: Câu hỏi mới của user

        Returns:
            str: Formatted prompt bao gồm tin nhắn mới
        """
        # Tạo bản copy tạm thời của history để không làm ô nhiễm state
        temp_messages = list(history.messages)
        temp_messages.append(
            ChatMessage(role=Role.USER, content=user_message.strip())
        )

        parts = []
        for msg in temp_messages:
            tag = self._ROLE_TAGS[msg.role]
            parts.append(f"{tag}\n{msg.content}</s>")

        parts.append("<|assistant|>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# CHAT SESSION — Kết hợp history + builder = Application Context
# ---------------------------------------------------------------------------

class ChatSession:
    """
    Application Context cho một phiên hội thoại.

    Đây là object trung tâm mà llm_runner.py sẽ tương tác.

    Stateless design:
      - Không ghi DB/file
      - Session ID dùng để phân biệt các session trong memory (dict)
      - Khi process restart → tất cả sessions mất, phải tạo lại

    Typical usage:
        session = ChatSession()
        prompt = session.prepare_prompt("Hà Nội có gì hay?")
        response = llm.invoke(prompt)
        session.add_assistant_response(response)
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        cfg: ChatConfig = chat_config,
    ):
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self.cfg = cfg
        self.created_at: float = time.time()

        # Khởi tạo history với sliding window
        self.history = ChatHistory(max_turns=cfg.max_history_turns)

        # Prompt builder cho TinyLlama ChatML format
        self.builder = PromptBuilder()

        # Thêm system message đầu tiên
        self.history.add_message(Role.SYSTEM, cfg.system_prompt)

    def prepare_prompt(self, user_message: str) -> str:
        """
        Bước 1 của inference loop:
          - Thêm tin nhắn user vào history
          - Build và trả về formatted prompt

        Args:
            user_message: Câu hỏi / input của người dùng

        Returns:
            str: Full prompt sẵn sàng nạp vào LlamaCpp
        """
        # Lưu user message vào history
        self.history.add_message(Role.USER, user_message)

        # Build prompt từ toàn bộ history
        prompt = self.builder.build(self.history)

        return prompt

    def add_assistant_response(self, response: str) -> ChatMessage:
        """
        Bước cuối của inference loop:
          - Lưu câu trả lời của assistant vào history
          - Để lần hỏi tiếp theo có context đầy đủ

        Args:
            response: Văn bản LLM vừa generate ra

        Returns:
            ChatMessage: Tin nhắn assistant vừa được tạo
        """
        return self.history.add_message(Role.ASSISTANT, response)

    def get_stats(self) -> dict:
        """Trả về thống kê session để debug / monitoring."""
        return {
            "session_id":      self.session_id,
            "turn_count":      self.history.turn_count,
            "max_turns":       self.cfg.max_history_turns,
            "total_tokens_est": self.history.total_estimated_tokens,
            "uptime_seconds":  round(time.time() - self.created_at, 1),
        }

    def reset(self):
        """Xóa history, giữ lại system prompt. Bắt đầu hội thoại mới."""
        self.history.clear(keep_system=True)


# ---------------------------------------------------------------------------
# SESSION REGISTRY — Quản lý nhiều session song song (cho FastAPI)
# ---------------------------------------------------------------------------

class SessionRegistry:
    """
    Dictionary đơn giản lưu các ChatSession đang active trong memory.

    Dùng trong FastAPI để:
      - Client gửi session_id trong request header
      - Server lookup session tương ứng
      - Nếu không có → tạo mới

    Stateless: tất cả sessions mất khi server restart.
    Production: thay bằng Redis hoặc database.
    """

    def __init__(self):
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> ChatSession:
        """
        Lấy session theo ID hoặc tạo mới nếu không tồn tại.

        Args:
            session_id: ID session từ client (None → tạo session mới)

        Returns:
            ChatSession: Session hiện tại hoặc mới tạo
        """
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        # Tạo session mới
        session = ChatSession(session_id=session_id)
        self._sessions[session.session_id] = session
        return session

    def delete(self, session_id: str) -> bool:
        """Xóa session khỏi registry."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    @property
    def active_sessions(self) -> int:
        """Số session đang active."""
        return len(self._sessions)


# Global registry — dùng chung trong ứng dụng
session_registry = SessionRegistry()


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Test chat_context standalone ===\n")

    session = ChatSession()

    # Simulate 3 turns of conversation
    conversations = [
        "Xin chào! Bạn là ai?",
        "Hà Nội có những điểm tham quan nào nổi tiếng?",
        "Cảm ơn, tôi sẽ ghé thăm Hồ Hoàn Kiếm!",
    ]

    for user_input in conversations:
        print(f"👤 User: {user_input}")
        prompt = session.prepare_prompt(user_input)
        fake_response = f"[Fake LLM response to: {user_input[:30]}...]"
        session.add_assistant_response(fake_response)
        print(f"🤖 Assistant: {fake_response}")
        print()

    print("📊 Session stats:", session.get_stats())
    print("\n📜 Full history:")
    for msg in session.history.messages:
        print(f"  [{msg.role.value:10s}] {msg.content[:60]}...")

    print("\n📝 Sample prompt preview:")
    print("─" * 50)
    sample_prompt = session.prepare_prompt("Bạn có biết Văn Miếu không?")
    print(sample_prompt)
    print("─" * 50)
