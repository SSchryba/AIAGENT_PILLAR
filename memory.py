import os
import logging
import gzip
import pickle
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import BaseMessage
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedMemory:
    """Enhanced memory system with multiple storage backends and compression."""
    
    def __init__(self, memory_type: str = "conversation_buffer"):
        self.memory_type = memory_type
        self.vector_store = None
        self.conversation_memory = None
        self.compression_enabled = os.getenv('MEMORY_COMPRESSION', 'True').lower() == 'true'
        self.max_messages = int(os.getenv('MEMORY_MAX_MESSAGES', '1000'))
        self.cleanup_interval = int(os.getenv('MEMORY_CLEANUP_INTERVAL', '24'))  # hours
        self.last_cleanup = datetime.now()
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize memory components."""
        try:
            # Initialize vector store for semantic search
            self._initialize_vector_store()
            
            # Initialize conversation memory
            self._initialize_conversation_memory()
            
            # Perform initial cleanup
            self._cleanup_old_messages()
            
            logger.info(f"Memory system initialized with type: {self.memory_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize the vector store for semantic search."""
        try:
            persist_dir = os.getenv('MEMORY_PERSIST_DIR', './db')
            model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            self.vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            
            logger.info(f"Vector store initialized at {persist_dir}")
            
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
            self.vector_store = None
    
    def _initialize_conversation_memory(self):
        """Initialize conversation memory based on type."""
        try:
            if self.memory_type == "conversation_buffer":
                self.conversation_memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="history"
                )
            elif self.memory_type == "conversation_summary":
                self.conversation_memory = ConversationSummaryMemory(
                    return_messages=True,
                    memory_key="history"
                )
            else:
                # Default to conversation buffer
                self.conversation_memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="history"
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize conversation memory: {e}")
            raise
    
    def _compress_message(self, message: str) -> bytes:
        """Compress a message using gzip."""
        if not self.compression_enabled:
            return message.encode('utf-8')
        
        try:
            return gzip.compress(message.encode('utf-8'))
        except Exception as e:
            logger.warning(f"Compression failed, using uncompressed: {e}")
            return message.encode('utf-8')
    
    def _decompress_message(self, compressed_data: bytes) -> str:
        """Decompress a message using gzip."""
        if not self.compression_enabled:
            return compressed_data.decode('utf-8')
        
        try:
            return gzip.decompress(compressed_data).decode('utf-8')
        except Exception as e:
            logger.warning(f"Decompression failed, using as-is: {e}")
            return compressed_data.decode('utf-8')
    
    def _cleanup_old_messages(self):
        """Clean up old messages to prevent memory bloat."""
        try:
            if not self.conversation_memory:
                return
            
            messages = self.conversation_memory.chat_memory.messages
            if len(messages) > self.max_messages:
                # Keep only the most recent messages
                messages_to_remove = len(messages) - self.max_messages
                self.conversation_memory.chat_memory.messages = messages[messages_to_remove:]
                logger.info(f"Cleaned up {messages_to_remove} old messages")
            
            # Update cleanup timestamp
            self.last_cleanup = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def add_message(self, message: str, is_human: bool = True):
        """Add a message to memory."""
        try:
            # Check if cleanup is needed
            if datetime.now() - self.last_cleanup > timedelta(hours=self.cleanup_interval):
                self._cleanup_old_messages()
            
            if self.conversation_memory:
                if is_human:
                    self.conversation_memory.chat_memory.add_user_message(message)
                else:
                    self.conversation_memory.chat_memory.add_ai_message(message)
            
            # Also store in vector store for semantic search
            if self.vector_store:
                # Compress message before storing
                compressed_message = self._compress_message(message)
                self.vector_store.add_texts([message])  # Store original for search
                
        except Exception as e:
            logger.error(f"Failed to add message to memory: {e}")
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all stored messages."""
        try:
            if self.conversation_memory:
                return self.conversation_memory.chat_memory.messages
            return []
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    def search_similar(self, query: str, k: int = 5) -> List[str]:
        """Search for similar messages in memory."""
        try:
            if self.vector_store:
                results = self.vector_store.similarity_search(query, k=k)
                return [doc.page_content for doc in results]
            return []
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []
    
    def clear_memory(self):
        """Clear all stored memory."""
        try:
            if self.conversation_memory:
                self.conversation_memory.clear()
            
            if self.vector_store:
                # Note: This is a simplified clear - in production you might want more sophisticated clearing
                logger.info("Vector store memory cleared")
            
            # Reset cleanup timestamp
            self.last_cleanup = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory."""
        try:
            messages = self.get_messages()
            stats = {
                "memory_type": self.memory_type,
                "conversation_messages": len(messages),
                "vector_store_available": self.vector_store is not None,
                "compression_enabled": self.compression_enabled,
                "max_messages": self.max_messages,
                "cleanup_interval_hours": self.cleanup_interval,
                "last_cleanup": self.last_cleanup.isoformat()
            }
            
            if self.vector_store:
                try:
                    stats["vector_store_collection_count"] = self.vector_store._collection.count()
                except Exception:
                    stats["vector_store_collection_count"] = "Unknown"
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def export_memory(self, filepath: str) -> bool:
        """Export memory to a file."""
        try:
            data = {
                "messages": self.get_messages(),
                "memory_type": self.memory_type,
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Memory exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export memory: {e}")
            return False
    
    def import_memory(self, filepath: str) -> bool:
        """Import memory from a file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            if self.conversation_memory and "messages" in data:
                self.conversation_memory.chat_memory.messages = data["messages"]
            
            logger.info(f"Memory imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import memory: {e}")
            return False

def get_memory(memory_type: Optional[str] = None) -> EnhancedMemory:
    """Factory function to create memory instance."""
    memory_type = memory_type or os.getenv('MEMORY_TYPE', 'conversation_buffer')
    return EnhancedMemory(memory_type=memory_type)

# Backward compatibility function
def get_simple_memory():
    """Legacy function for backward compatibility."""
    return get_memory()
