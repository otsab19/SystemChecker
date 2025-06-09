from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import os
from langchain.memory import (
    ConversationBufferMemory, ConversationBufferWindowMemory,
    ConversationSummaryMemory, ConversationSummaryBufferMemory
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from config.settings import settings


class MemoryManager:
    """Advanced memory management for agents"""

    def __init__(self):
        self.memory_store = {}
        self.session_memory = ConversationBufferWindowMemory(k=20)
        self.long_term_memory_file = "data/long_term_memory.json"
        self._load_long_term_memory()

    def _load_long_term_memory(self):
        """Load long-term memory from file"""
        if os.path.exists(self.long_term_memory_file):
            try:
                with open(self.long_term_memory_file, 'r') as f:
                    self.long_term_memory = json.load(f)
            except Exception:
                self.long_term_memory = {}
        else:
            self.long_term_memory = {}

    def _save_long_term_memory(self):
        """Save long-term memory to file"""
        os.makedirs(os.path.dirname(self.long_term_memory_file), exist_ok=True)
        with open(self.long_term_memory_file, 'w') as f:
            json.dump(self.long_term_memory, f, indent=2)

    def add_interaction(self, user_input: str, ai_response: str, context: Dict[str, Any] = None):
        """Add an interaction to memory"""
        timestamp = datetime.now().isoformat()

        # Add to session memory
        self.session_memory.chat_memory.add_user_message(user_input)
        self.session_memory.chat_memory.add_ai_message(ai_response)

        # Add to long-term memory
        interaction = {
            "timestamp": timestamp,
            "user_input": user_input,
            "ai_response": ai_response,
            "context": context or {}
        }

        if "interactions" not in self.long_term_memory:
            self.long_term_memory["interactions"] = []

        self.long_term_memory["interactions"].append(interaction)

        # Keep only last 1000 interactions
        if len(self.long_term_memory["interactions"]) > 1000:
            self.long_term_memory["interactions"] = self.long_term_memory["interactions"][-1000:]

        self._save_long_term_memory()

    def get_relevant_context(self, query: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """Get relevant context from memory based on query"""
        if "interactions" not in self.long_term_memory:
            return []

        # Simple keyword-based relevance (could be enhanced with embeddings)
        query_words = set(query.lower().split())
        relevant_interactions = []

        for interaction in self.long_term_memory["interactions"]:
            input_words = set(interaction["user_input"].lower().split())
            response_words = set(interaction["ai_response"].lower().split())

            # Calculate relevance score
            relevance = len(query_words.intersection(input_words.union(response_words)))

            if relevance > 0:
                relevant_interactions.append({
                    **interaction,
                    "relevance_score": relevance
                })

        # Sort by relevance and return top items
        relevant_interactions.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_interactions[:max_items]

    def get_session_summary(self) -> str:
        """Get summary of current session"""
        messages = self.session_memory.chat_memory.messages
        if not messages:
            return "No previous interactions in this session."

        summary = f"Session started with {len(messages) // 2} interactions:\n"
        for i in range(0, min(len(messages), 6), 2):  # Show last 3 interactions
            if i + 1 < len(messages):
                summary += f"User: {messages[i].content[:100]}...\n"
                summary += f"AI: {messages[i + 1].content[:100]}...\n\n"

        return summary