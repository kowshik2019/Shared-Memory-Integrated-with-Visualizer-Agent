"""
shared_memory.py
----------------
Shared memory store — the campfire where all agents gather,
share context, and remember what other agents found.

This is the backbone of inter-agent communication.
Every agent reads from and writes to this shared context,
enabling collaborative intelligence across the system.

Key Concepts:
    - SharedMemory: Thread-safe singleton dictionary that persists
      across all agent invocations within a single run.
    - Each agent appends its findings under a namespaced key.
    - Downstream agents read upstream context before acting.
    - Full history is preserved for auditability.
"""

import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """Single piece of information stored by an agent."""
    agent: str               # which agent wrote this
    key: str                 # topic / data label
    value: Any               # the payload (string, dict, list, etc.)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SharedMemory:
    """
    Thread-safe shared memory for multi-agent collaboration.

    Usage:
        mem = SharedMemory()
        mem.store("resume_analyzer", "skills", ["Python", "SQL", "Spark"])
        mem.store("jd_parser", "required_skills", ["Python", "AWS"])
        skills = mem.recall("resume_analyzer", "skills")
        all_context = mem.get_full_context()     # every agent can read everything
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton — all agents share the same memory instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._entries: List[MemoryEntry] = []
                cls._instance._store: Dict[str, Dict[str, Any]] = {}
            return cls._instance

    def store(self, agent: str, key: str, value: Any) -> None:
        """
        Store a value under agent namespace.
        Example: store("resume_analyzer", "candidate_skills", [...])
        """
        with self._lock:
            if agent not in self._store:
                self._store[agent] = {}
            self._store[agent][key] = value
            self._entries.append(MemoryEntry(agent=agent, key=key, value=value))

    def recall(self, agent: str, key: str) -> Optional[Any]:
        """
        Recall a specific value stored by a specific agent.
        Returns None if not found.
        """
        with self._lock:
            return self._store.get(agent, {}).get(key)

    def recall_all_from_agent(self, agent: str) -> Dict[str, Any]:
        """Get everything a specific agent has stored."""
        with self._lock:
            return dict(self._store.get(agent, {}))

    def get_full_context(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the FULL shared context — everything every agent has stored.
        This is what makes collaboration work: any agent can read any
        other agent's findings.
        """
        with self._lock:
            return {agent: dict(data) for agent, data in self._store.items()}

    def get_context_summary(self) -> str:
        """
        Human-readable summary of everything in memory.
        Useful for injecting into LLM prompts so agents are aware
        of what other agents have found.
        """
        with self._lock:
            if not self._store:
                return "No shared context available yet."
            lines = []
            for agent, data in self._store.items():
                lines.append(f"\n[{agent}]")
                for k, v in data.items():
                    if isinstance(v, list):
                        v_str = ", ".join(str(i) for i in v[:20])
                        if len(v) > 20:
                            v_str += f" ... (+{len(v)-20} more)"
                    elif isinstance(v, dict):
                        v_str = str(v)[:500]
                    else:
                        v_str = str(v)[:500]
                    lines.append(f"  {k}: {v_str}")
            return "\n".join(lines)

    def get_history(self) -> List[Dict]:
        """Full chronological log of every store() call."""
        with self._lock:
            return [
                {
                    "agent": e.agent,
                    "key": e.key,
                    "value": e.value,
                    "timestamp": e.timestamp,
                }
                for e in self._entries
            ]

    def reset(self) -> None:
        """Clear all memory — used between runs."""
        with self._lock:
            self._entries.clear()
            self._store.clear()


# Module-level singleton — import this everywhere
memory = SharedMemory()
