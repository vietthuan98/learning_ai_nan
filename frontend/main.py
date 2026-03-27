import mesop as me
import mesop.labs as mel
from api_client import api_client

@me.stateclass
class State:
  session_id: str = ""

def on_load(e: me.LoadEvent):
  me.set_theme_mode("system")


@me.page(
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://mesop-dev.github.io"]
  ),
  path="/",
  title="Real Chat Integration",
  on_load=on_load,
)
def page():
  mel.chat(transform, title="AI Chat App", bot_user="TinyLlama Bot")


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