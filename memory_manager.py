import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid
import logging

# Add these imports
from langchain.memory import BaseMemory  # Changed from langchain_core.memory
from langchain.schema import HumanMessage, AIMessage  # Changed from langchain_core.messages
from langchain.memory import BaseChatMessageHistory  # Changed from langchain_core.chat_history

from redis import Redis
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

class RedisMessageHistory(BaseChatMessageHistory):
    """Manages conversation history in Redis."""
    
    def __init__(self, session_id: str, redis_client: Redis, ttl: int = 86400):
        self.session_id = session_id
        self.redis_client = redis_client
        self.ttl = ttl  # Default TTL of 24 hours
        self.messages: List[Dict[str, Any]] = []
    
    def add_user_message(self, message: str, metadata: Dict[str, Any] = None) -> None:
        """Add a user message to the history."""
        self.add_message({
            'type': 'human',
            'content': message,
            'metadata': metadata or {}
        })

    def add_ai_message(self, message: str, metadata: Dict[str, Any] = None) -> None:
        """Add an AI message to the history."""
        self.add_message({
            'type': 'ai',
            'content': message,
            'metadata': metadata or {}
        })
        
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the Redis store."""
        messages = self.get_messages()
        messages.append(message)
        
        # Store messages with TTL
        self.redis_client.setex(
            f"chat:history:{self.session_id}",
            self.ttl,
            json.dumps(messages)
        )
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Retrieve messages from Redis."""
        messages_json = self.redis_client.get(f"chat:history:{self.session_id}")
        if messages_json:
            return json.loads(messages_json)
        return []

    def clear(self) -> None:
        """Clear message history from Redis."""
        self.redis_client.delete(f"chat:history:{self.session_id}")

class PostgresMessageHistory:
    """Manages conversation history in PostgreSQL."""
    
    def __init__(self, db_params: Dict[str, str]):
        self.db_params = db_params
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize PostgreSQL tables."""
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        session_id VARCHAR(36) PRIMARY KEY,
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ended_at TIMESTAMP,
                        metadata JSONB
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(36) REFERENCES conversations(session_id),
                        type VARCHAR(20),
                        content TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)
                conn.commit()
    
    def store_conversation(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Store a complete conversation in PostgreSQL."""
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                # Update or insert conversation record
                cur.execute("""
                    INSERT INTO conversations (session_id, ended_at)
                    VALUES (%s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id) 
                    DO UPDATE SET ended_at = CURRENT_TIMESTAMP
                """, (session_id,))
                
                # Store messages
                for message in messages:
                    cur.execute("""
                        INSERT INTO messages (session_id, type, content, metadata)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        session_id,
                        message['type'],
                        message['content'],
                        Json(message.get('metadata', {}))
                    ))
                conn.commit()
    
    def get_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve a conversation from PostgreSQL."""
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT type, content, timestamp, metadata
                    FROM messages
                    WHERE session_id = %s
                    ORDER BY timestamp ASC
                """, (session_id,))
                
                messages = []
                for msg_type, content, timestamp, metadata in cur.fetchall():
                    messages.append({
                        'type': msg_type,
                        'content': content,
                        'timestamp': timestamp.isoformat(),
                        'metadata': metadata
                    })
                return messages

class HybridMemory(BaseMemory):
    """Implements hybrid memory management using Redis and PostgreSQL."""
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        redis_client: Optional[Redis] = None,
        postgres_client: Optional[PostgresMessageHistory] = None
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.redis_client = redis_client or Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        
        db_params = {
            'dbname': os.getenv('POSTGRES_DB', 'ecommerce'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432))
        }
        self.postgres_client = postgres_client or PostgresMessageHistory(db_params)
        self.redis_history = RedisMessageHistory(self.session_id, self.redis_client)
    
    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables required by this memory object."""
        return ["history"]
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a message to the hybrid memory system."""
        message = {
            'type': 'human' if role == 'user' else 'ai',
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.redis_history.add_message(message)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Retrieve conversation history from Redis or PostgreSQL."""
        messages = self.redis_history.get_messages()
        
        if not messages:
            # Fallback to PostgreSQL if Redis is empty
            messages = self.postgres_client.get_conversation(self.session_id)
        
        return messages
    
    def persist_conversation(self) -> None:
        """Persist conversation from Redis to PostgreSQL."""
        messages = self.redis_history.get_messages()
        if messages:
            self.postgres_client.store_conversation(self.session_id, messages)
            # Clear Redis after successful persistence
            self.redis_history.clear()
    
    def clear(self) -> None:
        """Clear the conversation history from Redis."""
        self.redis_history.clear()
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables."""
        return {"history": self.get_conversation_history()}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context from this conversation."""
        if "input" in inputs:
            self.add_message("user", inputs["input"])
        if "output" in outputs:
            self.add_message("ai", outputs["output"])