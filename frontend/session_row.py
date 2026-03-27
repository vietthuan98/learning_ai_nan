import mesop as me
from state import State, update_chat_history
from api_client import api_client

def on_click_session(e: me.ClickEvent):
  state = me.state(State)
  state.session_id = e.key
  update_chat_history(e.key)

def on_click_delete(e: me.ClickEvent):
  state = me.state(State)
  sid = e.key.replace("del-", "")
  try:
    if api_client.delete_session(sid):
      try:
        idx = state.sessions.index(sid)
      except ValueError:
        idx = -1
      
      state.sessions = [s for s in state.sessions if s != sid]
      
      if state.session_id == sid:
        if state.sessions:
          new_idx = max(0, idx - 1)
          state.session_id = state.sessions[new_idx]
          update_chat_history(state.session_id)
        else:
          state.session_id = ""
          update_chat_history("")
  except Exception as ex:
    print(f"Error deleting session: {ex}")

def session_row(sid: str):
  state = me.state(State)
  is_active = sid == state.session_id
  
  with me.box(
    style=me.Style(
      display="flex",
      justify_content="space-between",
      align_items="center",
      margin=me.Margin(bottom=4),
      border_radius=8,
      background="#e3f2fd" if is_active else "transparent",
    )
  ):
    # Clickable session area
    with me.box(
      key=sid,
      on_click=on_click_session,
      style=me.Style(
        flex_grow=1,
        padding=me.Padding.symmetric(vertical=12, horizontal=16),
        cursor="pointer",
      )
    ):
      me.text(
          f"Session {sid}", 
          style=me.Style(font_size=15, color="#1e88e5" if is_active else "#444", font_weight="500" if is_active else "400")
      )
    
    # Delete button
    with me.box(
      key=f"del-{sid}",
      on_click=on_click_delete,
      style=me.Style(
        padding=me.Padding.symmetric(vertical=12, horizontal=16),
        cursor="pointer",
      )
    ):
      me.icon("delete", style=me.Style(font_size=20, color="#e53935" if is_active else "#bbb"))
