import httpx
import json
import os
from typing import Generator, Optional

class ChatClient:
    """
    A service class to interact with the LLM backend.
    Focuses on streaming responses for a modern chat experience.
    """
    
    def __init__(self, base_url: str = None):
        # Use provided URL or fallback to environment variable, then to localhost
        self.base_url = (base_url or os.getenv("BACKEND_URL", "http://localhost:8000")).rstrip("/")


    def stream_chat(self, message: str, session_id: Optional[str] = None) -> Generator[str, None, str]:
        """
        Sends a message to the backend and yields tokens as they arrive.
        Returns the final session_id.
        """
        url = f"{self.base_url}/chat/stream"
        payload = {
            "message": message,
            "session_id": session_id
        }
        
        # We'll return the session_id at the end of the generator
        final_session_id = session_id

        with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                yield f"Error: Backend returned {response.status_code}"
                return session_id

            # Extract session ID from headers if provided by backend
            if "X-Session-Id" in response.headers:
                final_session_id = response.headers["X-Session-Id"]

            for line in response.iter_lines():
                if not line:
                    continue
                
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    
                    if data == "[DONE]":
                        break
                    
                    yield data
        
        return final_session_id

    def send_chat(self, message: str, session_id: Optional[str] = None) -> dict:
        """
        Sends a message to the backend and returns the full response (non-streaming).
        """
        url = f"{self.base_url}/chat"
        payload = {
            "message": message,
            "session_id": session_id
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    def list_sessions(self) -> list[str]:
        """Fetches the list of active session IDs."""
        url = f"{self.base_url}/sessions"
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session from the backend."""
        url = f"{self.base_url}/sessions/{session_id}"
        with httpx.Client(timeout=10.0) as client:
            response = client.delete(url)
            return response.status_code == 200

    def get_session_info(self, session_id: str) -> dict:
        """Fetches detailed information (including history) for a session."""
        url = f"{self.base_url}/sessions/{session_id}"
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()

# Create a singleton instance for ease of use
api_client = ChatClient()
