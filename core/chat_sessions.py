from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


MAX_SESSION_MESSAGES = 400


class ChatSessionStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def list_sessions(self) -> list[dict[str, Any]]:
        data = self._load_data()
        out: list[dict[str, Any]] = []
        for session in data.get("sessions", []):
            if not isinstance(session, dict):
                continue
            out.append(
                {
                    "session_id": str(session.get("session_id") or ""),
                    "title": str(session.get("title") or "Workflow Chat"),
                    "updated_at": str(session.get("updated_at") or ""),
                    "message_count": len(session.get("messages") or []),
                }
            )

        out.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return out

    def create_session(self, title: str = "") -> dict[str, Any]:
        with self._lock:
            data = self._load_data_unlocked()
            sid = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            now = datetime.now().isoformat(timespec="seconds")
            session = {
                "session_id": sid,
                "title": (title or "Workflow Chat").strip()[:120] or "Workflow Chat",
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
            data.setdefault("sessions", []).append(session)
            self._save_data_unlocked(data)
            return session

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        data = self._load_data()
        for session in data.get("sessions", []):
            if str(session.get("session_id") or "") == session_id:
                return session
        return None

    def append_messages(self, session_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            data = self._load_data_unlocked()
            found = None
            for session in data.get("sessions", []):
                if str(session.get("session_id") or "") == session_id:
                    found = session
                    break

            if found is None:
                found = self.create_session("Workflow Chat")
                data = self._load_data_unlocked()
                for session in data.get("sessions", []):
                    if str(session.get("session_id") or "") == str(found.get("session_id") or ""):
                        found = session
                        break

            safe_messages = [self._sanitize_message(item) for item in messages if isinstance(item, dict)]
            found.setdefault("messages", []).extend(safe_messages)
            if len(found["messages"]) > MAX_SESSION_MESSAGES:
                found["messages"] = found["messages"][-MAX_SESSION_MESSAGES:]

            found["title"] = self._derive_title(found)
            found["updated_at"] = datetime.now().isoformat(timespec="seconds")

            self._save_data_unlocked(data)
            return found

    def update_title(self, session_id: str, title: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._load_data_unlocked()
            for session in data.get("sessions", []):
                if str(session.get("session_id") or "") != session_id:
                    continue
                clean = (title or "").strip()[:120] or "Workflow Chat"
                session["title"] = clean
                session["updated_at"] = datetime.now().isoformat(timespec="seconds")
                self._save_data_unlocked(data)
                return session
        return None

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            data = self._load_data_unlocked()
            before = len(data.get("sessions", []))
            data["sessions"] = [
                session for session in data.get("sessions", []) if str(session.get("session_id") or "") != session_id
            ]
            if len(data["sessions"]) == before:
                return False
            self._save_data_unlocked(data)
            return True

    def _load_data(self) -> dict[str, Any]:
        with self._lock:
            return self._load_data_unlocked()

    def _load_data_unlocked(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return {"sessions": []}
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
                return payload
        except Exception:
            pass
        return {"sessions": []}

    def _save_data_unlocked(self, data: dict[str, Any]) -> None:
        self.file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _sanitize_message(message: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": str(message.get("role") or "assistant"),
            "content": str(message.get("content") or ""),
            "time": str(message.get("time") or datetime.now().strftime("%H:%M:%S")),
            "runtime": str(message.get("runtime") or ""),
            "model": str(message.get("model") or ""),
            "citations": str(message.get("citations") or ""),
            "reasoning_chain": [str(item) for item in message.get("reasoning_chain", [])],
        }

    @staticmethod
    def _derive_title(session: dict[str, Any]) -> str:
        current = str(session.get("title") or "").strip()
        if current and current.lower() != "workflow chat":
            return current[:120]

        for msg in session.get("messages", []):
            if str(msg.get("role") or "").lower() != "user":
                continue
            content = " ".join(str(msg.get("content") or "").split())
            if not content:
                continue
            return (content[:64] + "...") if len(content) > 64 else content
        return "Workflow Chat"
