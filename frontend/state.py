import mesop as me
from dataclasses import field
from api_client import api_client
from mesop.labs.chat import State as ChatState
import mesop.labs as mel

@me.stateclass
class State:
  session_id: str = ""
  sessions: list[str] = field(default_factory=list)

def update_chat_history(session_id: str):
  """Fetches history from backend and updates mesop's chat state."""
  chat_state = me.state(ChatState)
  if not session_id:
    chat_state.output = []
    return
    
  try:
    info = api_client.get_session_info(session_id)
    chat_state.output = [
        mel.ChatMessage(role=m["role"], content=m["content"])
        for m in info.get("history", [])
        if m["role"] in ["user", "assistant"]
    ]
  except Exception as ex:
    print(f"Could not load history for {session_id}: {ex}")
    chat_state.output = []
