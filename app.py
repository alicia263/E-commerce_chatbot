import os
import time
import uuid
from typing import Dict, Any, Tuple, List, Optional
import logging
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from groq import Groq
from dotenv import load_dotenv
from redis import Redis
import psycopg2
from psycopg2.extras import DictCursor

# Basic configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Environment variables
model_name = os.getenv('MODEL_NAME', 'multi-qa-MiniLM-L6-cos-v1')
es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
groq_api_key = os.getenv('GROQ_API_KEY')
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
TZ_INFO = os.getenv("TZ", "Europe/Berlin")
tz = ZoneInfo(TZ_INFO)

# Initialize models and clients
model = SentenceTransformer(model_name)
es_client = Elasticsearch(es_url)

def get_db_connection():
    """Create a connection to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        return None

def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")
            
            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    openai_cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        conn.rollback()
    finally:
        conn.close()

class RedisMemoryManager:
    """Manages conversation history in Redis with automatic expiration."""
    
    def __init__(self, expiration_time: int = 3600):
        self.redis_client = Redis.from_url(redis_url, decode_responses=True)
        self.session_id = str(uuid.uuid4())
        self.expiration_time = expiration_time
    
    def _get_key(self) -> str:
        return f"conversation:{self.session_id}"
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any]) -> None:
        key = self._get_key()
        message = {
            "role": role,
            "content": content,
            "metadata": metadata,
            "timestamp": datetime.now(tz).isoformat()
        }
        self.redis_client.rpush(key, json.dumps(message))
        self.redis_client.expire(key, self.expiration_time)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        key = self._get_key()
        messages = self.redis_client.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in messages]
    
    def clear_history(self) -> None:
        key = self._get_key()
        self.redis_client.delete(key)

def elastic_search_hybrid(query: str, index_name: str = "ecommerce-products") -> list:
    """Performs a hybrid search combining vector similarity and keyword matching."""
    query_vector = model.encode(query).tolist()
    
    search_query = {
        'size': 5,
        'query': {
            'bool': {
                'must': [
                    {
                        'script_score': {
                            'query': {'match_all': {}},
                            'script': {
                                'source': "cosineSimilarity(params.query_vector, 'combined_vector') + 1.0",
                                'params': {'query_vector': query_vector}
                            }
                        }
                    }
                ],
                'should': [
                    {
                        'multi_match': {
                            'query': query,
                            'fields': ['productName^3', 'category^2', 'productDescription'],
                            'type': 'cross_fields',
                            'operator': 'and'
                        }
                    }
                ]
            }
        }
    }
    
    try:
        results = es_client.search(index=index_name, body=search_query)
        return [hit['_source'] for hit in results['hits']['hits']]
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return []

def build_product_context(search_results: list) -> str:
    """Builds a context string from product search results."""
    context_items = []
    for product in search_results:
        final_price = product['price'] * (1 - product.get('discount', 0)/100)
        context_items.append(
            f"Product: {product['productName']}\n"
            f"Category: {product['category']}\n"
            f"Price: ${product['price']:.2f}\n"
            f"Final Price: ${final_price:.2f}\n"
            f"Colors: {', '.join(product['availableColours'])}\n"
            f"Sizes: {', '.join(product['sizes'])}\n"
            f"Description: {product['productDescription']}\n"
        )
    return "\n---\n".join(context_items)

def build_prompt(query: str, search_results: list, conversation_history: List[Dict[str, Any]]) -> str:
    """Builds a prompt including conversation history and product information."""
    prompt_template = """You are a knowledgeable and helpful e-commerce shopping assistant. Using the provided product information 
    and conversation history, answer the customer's question accurately and professionally. Maintain context from the previous 
    conversation when relevant. If referring to previous interactions, be explicit about what was discussed before.

    Previous Conversation:
    {history}

    Product Information:
    {context}

    Current Question: {question}

    Please provide a helpful, accurate, and natural response that takes into account both the conversation history and the 
    available product information. If referring to previously discussed items, make those references clear."""

    context = build_product_context(search_results)
    history_lines = []
    for msg in conversation_history:
        history_lines.append(f"{msg['role']}: {msg['content']}")
    history = "\n".join(history_lines)
    
    return prompt_template.format(
        history=history,
        context=context,
        question=query
    )

def llm(prompt: str, model: str = 'llama-3.1-70b-versatile') -> Tuple[str, Dict[str, Any], float]:
    """Generates a response using the Groq LLM."""
    client = Groq()
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    end_time = time.time()
    
    return response.choices[0].message.content, response.usage.to_dict(), end_time - start_time

def evaluate_relevance(question: str, answer: str) -> Tuple[str, str, Dict[str, int]]:
    """Evaluates the relevance of the generated answer to the given question.

    Args:
        question (str): The input question.
        answer (str): The generated answer to evaluate.

    Returns:
        Tuple[str, str, Dict[str, int]]: The relevance rating, explanation, and token usage.
    """
    logging.info("Evaluating relevance of generated answer...")
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system...
    (continued)
    """
    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt)

    try:
        json_eval = json.loads(evaluation)
        logging.info(f"Evaluation result: {json_eval}")
        return json_eval['Relevance'], json_eval['Explanation'], tokens
    except json.JSONDecodeError:
        logging.error("Failed to parse evaluation JSON.")
        return "UNKNOWN", "Failed to parse evaluation", tokens

class EnhancedEcommerceAssistant:
    """Enhanced version of EcommerceAssistant with Redis memory and PostgreSQL storage."""
    
    def __init__(self, memory_expiration: int = 3600):
        self.memory = RedisMemoryManager(expiration_time=memory_expiration)
        self.session_id = self.memory.session_id
        logging.info(f"Initialized new session with ID: {self.session_id}")
    
    def rag(self, query: str, model: str = 'llama-3.1-70b-versatile') -> Dict[str, Any]:
        """Executes the RAG pipeline with memory management and storage."""
        conversation_id = str(uuid.uuid4())
        search_results = elastic_search_hybrid(query)
        
        metadata = {
            'conversation_id': conversation_id,
            'search_count': len(search_results),
            'timestamp': datetime.now(tz).isoformat()
        }
        
        self.memory.add_message("user", query, metadata)
        
        if not search_results:
            response = "I apologize, but I couldn't find any relevant products matching your query. Could you please try rephrasing your question or provide more specific details?"
            self.memory.add_message("assistant", response, metadata)
            response_data = {
                "id": conversation_id,
                "session_id": self.session_id,
                "question": query,
                "answer": response,
                "model_used": model,
                "response_time": 0,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "search_results": []
            }
            save_conversation_to_db(conversation_id, query, response_data)
            return response_data
        
        conversation_history = self.memory.get_conversation_history()
        prompt = build_prompt(query, search_results, conversation_history)
        answer, tokens, response_time = llm(prompt, model=model)
        
        metadata.update({
            'response_time': response_time,
            'tokens': tokens
        })
        self.memory.add_message("assistant", answer, metadata)
        
        response_data = {
            "id": conversation_id,
            "session_id": self.session_id,
            "question": query,
            "answer": answer,
            "model_used": model,
            "response_time": response_time,
            "tokens": tokens,
            "search_results": search_results
        }
        
        save_conversation_to_db(conversation_id, query, response_data)
        return response_data
    
    def add_feedback(self, conversation_id: str, feedback: int) -> None:
        """Add user feedback for a conversation."""
        save_feedback(conversation_id, feedback)
        logging.info(f"Feedback {feedback} added for conversation {conversation_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return get_usage_stats()
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """Get recent conversations with feedback."""
        return get_recent_conversations(limit)
    
    def reset_conversation(self) -> Dict[str, Any]:
        """Reset conversation and start new session."""
        self.memory.clear_history()
        self.memory = RedisMemoryManager()
        self.session_id = self.memory.session_id
        
        return {
            "message": "Conversation reset successfully",
            "new_session_id": self.session_id
        }
    
    def end_session(self) -> None:
        """End the current session and clear Redis memory."""
        self.memory.clear_history()
        logging.info(f"Session {self.session_id} ended and cleared from Redis")

def format_response(response_data: Dict[str, Any]) -> str:
    """Formats the response data for display."""
    return f"""
Answer: {response_data['answer']}

Response Time: {response_data['response_time']:.2f} seconds
Total Tokens: {response_data['tokens']['total_tokens']}
Session ID: {response_data['session_id']}
"""

if __name__ == "__main__":
    assistant = EnhancedEcommerceAssistant()
    print("E-commerce Product Assistant - Ask me anything about our products!")
    print("(Type 'quit' to exit, 'reset' to start a new conversation, 'stats' to see usage statistics)")
    
    try:
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() == 'quit':
                assistant.end_session()
                break
            elif question.lower() == 'reset':
                result = assistant.reset_conversation()
                print(f"\n{result['message']}")
                print(f"New session started with ID: {result['new_session_id']}")
                continue
            elif question.lower() == 'stats':
                stats = assistant.get_stats()
                print("\nUsage Statistics:")
                print(f"Average Response Time: {stats['avg_response_time']:.2f} seconds")
                print(f"Total Tokens: {stats['total_tokens']}")
                print(f"Positive Feedback: {stats['positive_feedback']}")
                print(f"Negative Feedback: {stats['negative_feedback']}")
                continue
                
            answer_data = assistant.rag(question)
            print(format_response(answer_data))
            
            # Ask for feedback
            feedback = input("\nWas this response helpful? (y/n): ").strip().lower()
            if feedback in ['y', 'n']:
                feedback_value = 1 if feedback == 'y' else -1
                assistant.add_feedback(answer_data['id'], feedback_value)
    
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        assistant.end_session()
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        assistant.end_session()
        raise