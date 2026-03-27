import mesop as me
import mesop.labs as mel
import uuid
from api_client import api_client
from mesop.labs.chat import State as ChatState
from dataclasses import field

@me.stateclass
class State:
  session_id: str = ""
  sessions: list[str] = field(default_factory=list)

# --- Helpers ---
def update_chat_history(session_id: str):
  """Fetches history from backend and updates mesop's chat state."""
  chat_state = me.state(ChatState)
  if not session_id:
    chat_state.output = []
    return
    
  try:
    info = api_client.get_session_info(session_id)
    # Backend returns history as list of dicts with role and content
    chat_state.output = [
        mel.ChatMessage(role=m["role"], content=m["content"])
        for m in info.get("history", [])
        if m["role"] in ["user", "assistant"]
    ]
  except Exception as ex:
    # Most likely session not found on backend (e.g. brand new session)
    print(f"Could not load history for {session_id}: {ex}")
    chat_state.output = []

# --- Event Handlers ---
def on_load(e: me.LoadEvent):
  me.set_theme_mode("system")
  state = me.state(State)
  try:
      # Fetch the list of active sessions from the backend on startup
      state.sessions = api_client.list_sessions()
      if state.sessions:
        state.session_id = state.sessions[0]
        update_chat_history(state.session_id)
      else:
        state.session_id = ""
  except Exception as ex:
      print(f"Error loading sessions: {ex}")

def on_click_session(e: me.ClickEvent):
  state = me.state(State)
  state.session_id = e.key
  update_chat_history(e.key)

def on_click_new_chat(e: me.ClickEvent):
  state = me.state(State)
  new_session_id = str(uuid.uuid4())[:8]
  state.session_id = new_session_id # Reset to start a new chat
  state.sessions.append(new_session_id)
  update_chat_history("") # Clear any existing history for the new chat

def on_click_delete(e: me.ClickEvent):
  state = me.state(State)
  # The key for delete is "del-{sid}", so we need to extract the sid
  sid = e.key.replace("del-", "")
  try:
    if api_client.delete_session(sid):
      # Find the index of the session before deleting it
      try:
        idx = state.sessions.index(sid)
      except ValueError:
        idx = -1
      
      state.sessions = [s for s in state.sessions if s != sid]
      
      # If we just deleted the active session, pick a neighbor
      if state.session_id == sid:
        if state.sessions:
          # Select neighbor: previous one (idx - 1) or next one (now at 0)
          new_idx = max(0, idx - 1)
          state.session_id = state.sessions[new_idx]
          update_chat_history(state.session_id)
        else:
          state.session_id = ""
          update_chat_history("")
  except Exception as ex:
    print(f"Error deleting session: {ex}")



def transform(input: str, history: list[mel.ChatMessage]):
  state = me.state(State)
  
  # Use the streaming service to get chunks from the backend
  gen = api_client.stream_chat(input, session_id=state.session_id)
  
  try:
    while True:
      yield next(gen)
  except StopIteration as e:
    # Capture the final session_id returned by the generator
    if e.value:
      state.session_id = e.value
      # If it's a new session, add it to the list
      if state.session_id not in state.sessions:
        state.sessions.append(state.session_id)

# --- Persistent Styles ---
SIDEBAR_STYLE = me.Style(
    width="280px",
    height="100vh",
    background="#f1f3f5",
    border=me.Border(right=me.BorderSide(width=1, style="solid", color="#dee2e6")),
    display="flex",
    flex_direction="column",
    padding=me.Padding.all(16),
    flex_shrink=0,
)

# --- Page Layout ---
@me.page(
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://mesop-dev.github.io"],
    dangerously_disable_trusted_types=True
  ),
  path="/",
  title="AI Chat with Sidebar",
  on_load=on_load,
)
def page():
  state = me.state(State)
  print(state)

  with me.box(style=me.Style(display="flex", height="100vh", overflow_y="hidden")):
    # --- Sidebar ---
    with me.box(style=SIDEBAR_STYLE):
      with me.box(style=me.Style(display="flex", justify_content="space-between", align_items="center", margin=me.Margin(bottom=20))):
        me.text("Chat Rooms", style=me.Style(font_weight="bold", font_size=18))
      
      # Sessions List
      with me.box(style=me.Style(flex_grow=1, overflow_y="auto")):
        if not state.sessions:
          me.text("No active chats", style=me.Style(font_style="italic", color="#888", padding=me.Padding.all(8)))
        
        for sid in state.sessions:
          is_active = sid == state.session_id
          with me.box(
            style=me.Style(
              display="flex",
              justify_content="space-between",
              align_items="center",
              margin=me.Margin(bottom=4),
              border_radius=8,
              background="#e9ecef" if is_active else "transparent",
            )
          ):
            # Session click area
            with me.box(
              key=sid,
              on_click=on_click_session,
              style=me.Style(
                flex_grow=1,
                padding=me.Padding.symmetric(vertical=10, horizontal=12),
                cursor="pointer",
              )
            ):
              me.text(f"Session {sid}", style=me.Style(font_size=14))
            
            # Delete button (separate clickable area)
            with me.box(
              key=f"del-{sid}",
              on_click=on_click_delete,
              style=me.Style(
                padding=me.Padding.symmetric(vertical=10, horizontal=12),
                cursor="pointer",
              )
            ):
              me.icon("delete", style=me.Style(font_size=18, color="#dc3545" if is_active else "#888"))


      # New Chat Button
      with me.box(style=me.Style(padding=me.Padding(top=16), border=me.Border(top=me.BorderSide(width=1, style="solid", color="#ddd")))):
        me.button("New Chat", on_click=on_click_new_chat, type="flat", style=me.Style(width="100%", background="#007bff", color="white"))

    # --- Main Chat Area ---
    # Keying the chat component by session_id forces a fresh component when switching
    with me.box(style=me.Style(flex_grow=1, height="100vh", display="flex", flex_direction="column"), key=state.session_id):
      mel.chat(transform, title="TinyLlama AI", bot_user="TinyLlama Bot")