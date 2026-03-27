import mesop as me
import uuid
from state import State, update_chat_history
from session_row import session_row

SIDEBAR_STYLE = me.Style(
    width="280px",
    height="100vh",
    background="#f8f9fa",
    border=me.Border(right=me.BorderSide(width=1, style="solid", color="#e0e0e0")),
    display="flex",
    flex_direction="column",
    padding=me.Padding.all(16),
    flex_shrink=0,
)

def on_click_new_chat(e: me.ClickEvent):
  state = me.state(State)
  new_session_id = str(uuid.uuid4())[:8]
  state.session_id = new_session_id
  state.sessions.append(new_session_id)
  update_chat_history("")

def sidebar():
  state = me.state(State)
  with me.box(style=SIDEBAR_STYLE):
      # Header
      with me.box(style=me.Style(display="flex", justify_content="space-between", align_items="center", margin=me.Margin(bottom=24))):
        me.text("Chat Rooms", style=me.Style(font_weight="bold", font_size=20, color="#2c3e50"))
      
      # Scrollable Session List
      with me.box(style=me.Style(flex_grow=1, overflow_y="auto")):
        if not state.sessions:
          me.text("No active chats", style=me.Style(font_style="italic", color="#999", padding=me.Padding.all(8)))
        
        for sid in state.sessions:
          session_row(sid)

      # Footer with New Chat button
      with me.box(style=me.Style(padding=me.Padding(top=16), border=me.Border(top=me.BorderSide(width=1, style="solid", color="#eee")))):
          me.button(
              "New Chat", 
              on_click=on_click_new_chat, 
              type="flat", 
              style=me.Style(width="100%", background="#007bff", color="white", border_radius=8, padding=me.Padding.all(12))
          )
