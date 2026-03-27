import mesop as me
from state import State, update_chat_history
from api_client import api_client
from sidebar import sidebar
from chat_main_area import chat_main_area

# --- Styling Constants ---
ROOT_STYLE = me.Style(
    display="flex",
    height="100vh",
    overflow_y="hidden",
    font_family="Roboto, sans-serif"
)

# --- Event Handlers ---
def on_load(e: me.LoadEvent):
  me.set_theme_mode("system")
  state = me.state(State)
  try:
      state.sessions = api_client.list_sessions()
      if state.sessions:
        state.session_id = state.sessions[0]
        update_chat_history(state.session_id)
      else:
        state.session_id = ""
  except Exception as ex:
      print(f"Error loading sessions: {ex}")

# --- Page Definition ---
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
  with me.box(style=ROOT_STYLE):
    sidebar()
    chat_main_area()