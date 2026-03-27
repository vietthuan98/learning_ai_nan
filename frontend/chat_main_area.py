import mesop as me
import mesop.labs as mel
from state import State
from api_client import api_client

MAIN_CONTENT_STYLE = me.Style(
    flex_grow=1,
    height="100vh",
    display="flex",
    flex_direction="column"
)

def transform(input: str, history: list[mel.ChatMessage]):
  state = me.state(State)
  gen = api_client.stream_chat(input, session_id=state.session_id)
  try:
    while True:
      yield next(gen)
  except StopIteration as e:
    if e.value:
      state.session_id = e.value
      if state.session_id not in state.sessions:
        state.sessions.append(state.session_id)

def chat_main_area():
  state = me.state(State)
  with me.box(style=MAIN_CONTENT_STYLE, key=state.session_id):
      mel.chat(transform, title="TinyLlama AI", bot_user="TinyLlama Bot")
