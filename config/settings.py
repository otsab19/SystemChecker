import os
from enum import Enum
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv()


class AgentPattern(Enum):
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    MULTI_AGENT = "multi_agent"
    CONVERSATIONAL = "conversational"
    STRUCTURED_CHAT = "structured_chat"
    SELF_ASK = "self_ask"
    OPENAI_FUNCTIONS = "openai_functions"


class AgentMode(Enum):
    INTERACTIVE = "interactive"
    AUTONOMOUS = "autonomous"
    SUPERVISED = "supervised"


class Settings:
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Agent Configuration
    DEFAULT_AGENT_PATTERN = AgentPattern.PLAN_EXECUTE
    AGENT_MODE = AgentMode.INTERACTIVE
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

    # Vector Store Configuration
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vector_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "system_info")
    EMBEDDING_MODEL = "models/embedding-001"
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Data Collection Configuration
    COLLECTION_INTERVAL_HOURS = int(os.getenv("COLLECTION_INTERVAL_HOURS", "1"))
    MAX_LOG_ENTRIES = int(os.getenv("MAX_LOG_ENTRIES", "100"))
    ENABLE_BACKGROUND_COLLECTION = os.getenv("ENABLE_BACKGROUND_COLLECTION", "true").lower() == "true"

    # Security Configuration
    REQUIRE_CONFIRMATION = os.getenv("REQUIRE_CONFIRMATION", "true").lower() == "true"
    SAFE_MODE = os.getenv("SAFE_MODE", "true").lower() == "true"
    ALLOWED_COMMANDS = os.getenv("ALLOWED_COMMANDS", "ps,top,free,df,uptime,whoami").split(",")

    # Multi-Agent Configuration
    ENABLE_SPECIALIST_AGENTS = os.getenv("ENABLE_SPECIALIST_AGENTS", "true").lower() == "true"
    AGENT_COLLABORATION_MODE = os.getenv("AGENT_COLLABORATION_MODE", "hierarchical")  # hierarchical, peer-to-peer

    # Performance Configuration
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

    @classmethod
    def get_agent_config(cls, pattern: AgentPattern) -> Dict[str, Any]:
        """Get configuration specific to agent pattern"""
        configs = {
            AgentPattern.REACT: {
                "max_iterations": cls.MAX_ITERATIONS,
                "early_stopping_method": "generate",
                "handle_parsing_errors": True,
                "verbose": True
            },
            AgentPattern.PLAN_EXECUTE: {
                "max_iterations": cls.MAX_ITERATIONS,
                "max_execution_time": 300,
                "return_intermediate_steps": True
            },
            AgentPattern.MULTI_AGENT: {
                "coordination_strategy": "hierarchical",
                "max_rounds": 5,
                "consensus_threshold": 0.8
            },
            AgentPattern.CONVERSATIONAL: {
                "memory_key": "chat_history",
                "return_messages": True,
                "max_token_limit": 4000
            }
        }
        return configs.get(pattern, {})


settings = Settings()