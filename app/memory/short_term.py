"""
Short-term memory: per-session conversation buffer.

Maintained as a plain list of {"role": ..., "content": ...} dicts in the
caller's session context (Streamlit st.session_state or equivalent).
This avoids depending on deprecated LangChain memory classes and keeps
the code simple and portable.
"""
from __future__ import annotations

from collections import deque


class SessionMemory:
    """
    Sliding-window conversation buffer that retains the last `k` turns (user + assistant pairs).
    Each turn is stored as two entries: one user message and one assistant message.
    """

    def __init__(self, k: int = 10) -> None:
        self._k = k
        # Store up to k*2 messages (k turns × 2 roles)
        self._messages: deque[dict] = deque(maxlen=k * 2)

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Append a user → assistant turn to the buffer."""
        self._messages.append({"role": "user", "content": user_message})
        self._messages.append({"role": "assistant", "content": assistant_message})

    def get_history(self) -> list[dict]:
        """Return the current window as a list of {role, content} dicts."""
        return list(self._messages)

    def clear(self) -> None:
        """Reset the buffer."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


def create_session_memory(k: int = 10) -> SessionMemory:
    """Create a fresh session memory with a k-turn sliding window."""
    return SessionMemory(k=k)
