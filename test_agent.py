#!/usr/bin/env python3
"""
Test suite for AI Agent
Comprehensive testing for all components of the AI agent system.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import AIAgent, ResponseCache
from memory import EnhancedMemory
import agent_config as config

class TestResponseCache(unittest.TestCase):
    """Test the response caching system."""
    
    def setUp(self):
        self.cache = ResponseCache(ttl_seconds=1)
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get functionality."""
        message = "Hello, world!"
        response = "Hi there!"
        
        # Initially should be None
        self.assertIsNone(self.cache.get(message))
        
        # Set cache
        self.cache.set(message, response)
        
        # Should get cached response
        self.assertEqual(self.cache.get(message), response)
    
    def test_cache_expiration(self):
        """Test that cache entries expire after TTL."""
        message = "Test message"
        response = "Test response"
        
        self.cache.set(message, response)
        self.assertEqual(self.cache.get(message), response)
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        # Should be None after expiration
        self.assertIsNone(self.cache.get(message))
    
    def test_cache_clear(self):
        """Test cache clearing functionality."""
        message = "Test message"
        response = "Test response"
        
        self.cache.set(message, response)
        self.assertEqual(self.cache.get(message), response)
        
        self.cache.clear()
        self.assertIsNone(self.cache.get(message))
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        self.assertIn('cache_size', stats)
        self.assertIn('ttl_seconds', stats)
        self.assertEqual(stats['cache_size'], 0)
        self.assertEqual(stats['ttl_seconds'], 1)

class TestEnhancedMemory(unittest.TestCase):
    """Test the enhanced memory system."""
    
    def setUp(self):
        # Use a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_persist_dir = os.getenv('MEMORY_PERSIST_DIR')
        os.environ['MEMORY_PERSIST_DIR'] = self.temp_dir
        
        # Mock the vector store to avoid actual database operations
        with patch('memory.Chroma') as mock_chroma:
            self.mock_vector_store = Mock()
            mock_chroma.return_value = self.mock_vector_store
            self.memory = EnhancedMemory()
    
    def tearDown(self):
        # Clean up
        if self.original_persist_dir:
            os.environ['MEMORY_PERSIST_DIR'] = self.original_persist_dir
        else:
            del os.environ['MEMORY_PERSIST_DIR']
        
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_initialization(self):
        """Test memory system initialization."""
        self.assertIsNotNone(self.memory.conversation_memory)
        self.assertEqual(self.memory.memory_type, "conversation_buffer")
    
    def test_add_and_get_messages(self):
        """Test adding and retrieving messages."""
        user_message = "Hello, agent!"
        ai_message = "Hello, human!"
        
        # Add messages
        self.memory.add_message(user_message, is_human=True)
        self.memory.add_message(ai_message, is_human=False)
        
        # Get messages
        messages = self.memory.get_messages()
        self.assertEqual(len(messages), 2)
        
        # Check message content
        self.assertIn(user_message, str(messages[0]))
        self.assertIn(ai_message, str(messages[1]))
    
    def test_memory_clear(self):
        """Test memory clearing functionality."""
        # Add some messages
        self.memory.add_message("Test message", is_human=True)
        self.assertEqual(len(self.memory.get_messages()), 1)
        
        # Clear memory
        self.memory.clear_memory()
        self.assertEqual(len(self.memory.get_memory_stats()['conversation_messages']), 0)
    
    def test_memory_stats(self):
        """Test memory statistics."""
        stats = self.memory.get_memory_stats()
        
        required_keys = [
            'memory_type', 'conversation_messages', 'vector_store_available',
            'compression_enabled', 'max_messages', 'cleanup_interval_hours'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
    
    def test_memory_export_import(self):
        """Test memory export and import functionality."""
        # Add some messages
        self.memory.add_message("Test message 1", is_human=True)
        self.memory.add_message("Test response 1", is_human=False)
        
        # Export memory
        export_file = os.path.join(self.temp_dir, "memory_export.pkl")
        self.assertTrue(self.memory.export_memory(export_file))
        
        # Clear memory
        self.memory.clear_memory()
        self.assertEqual(len(self.memory.get_messages()), 0)
        
        # Import memory
        self.assertTrue(self.memory.import_memory(export_file))
        self.assertEqual(len(self.memory.get_messages()), 2)

class TestAIAgent(unittest.TestCase):
    """Test the AI agent functionality."""
    
    def setUp(self):
        # Mock the LLM to avoid actual API calls
        with patch('agent.OpenAI') as mock_openai:
            self.mock_llm = Mock()
            self.mock_llm.predict.return_value = "Mock response"
            mock_openai.return_value = self.mock_llm
            
            # Mock the conversation chain
            with patch('agent.ConversationChain') as mock_chain:
                self.mock_conversation = Mock()
                self.mock_conversation.predict.return_value = "Mock conversation response"
                mock_chain.return_value = self.mock_conversation
                
                self.agent = AIAgent()
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent.conversation)
        self.assertIsNotNone(self.agent.memory)
        self.assertIsNotNone(self.agent.cache)
    
    def test_chat_functionality(self):
        """Test chat functionality."""
        message = "Hello, agent!"
        response = self.agent.chat(message)
        
        # Should return a response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Should have called the conversation chain
        self.mock_conversation.predict.assert_called_once_with(input=message)
    
    def test_cache_integration(self):
        """Test that caching works with chat."""
        message = "Test message for caching"
        
        # First call should hit the LLM
        response1 = self.agent.chat(message)
        self.mock_conversation.predict.assert_called_once()
        
        # Reset the mock
        self.mock_conversation.predict.reset_mock()
        
        # Second call should use cache
        response2 = self.agent.chat(message)
        self.mock_conversation.predict.assert_not_called()
        
        # Responses should be the same
        self.assertEqual(response1, response2)
    
    def test_memory_summary(self):
        """Test memory summary functionality."""
        summary = self.agent.get_memory_summary()
        
        required_keys = ['memory_type', 'memory_size', 'cache']
        for key in required_keys:
            self.assertIn(key, summary)
    
    def test_cache_clear(self):
        """Test cache clearing functionality."""
        # Add something to cache
        self.agent.chat("Test message")
        
        # Clear cache
        self.agent.clear_cache()
        
        # Cache should be empty
        stats = self.agent.cache.get_stats()
        self.assertEqual(stats['cache_size'], 0)

class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_structure(self):
        """Test that configuration has the expected structure."""
        config_dict = config.get_config()
        
        required_sections = [
            'agent', 'response', 'memory', 'web', 'security',
            'logging', 'model', 'tools', 'templates', 'performance'
        ]
        
        for section in required_sections:
            self.assertIn(section, config_dict)
    
    def test_agent_config(self):
        """Test agent-specific configuration."""
        agent_config = config.get_config()['agent']
        
        required_keys = ['name', 'version', 'description', 'tone', 'personality_traits', 'goals']
        for key in required_keys:
            self.assertIn(key, agent_config)
    
    def test_performance_config(self):
        """Test performance configuration."""
        perf_config = config.get_config()['performance']
        
        required_keys = ['enable_caching', 'cache_ttl', 'max_concurrent_requests']
        for key in required_keys:
            self.assertIn(key, perf_config)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        # Set up a temporary environment for integration tests
        self.temp_dir = tempfile.mkdtemp()
        self.original_persist_dir = os.getenv('MEMORY_PERSIST_DIR')
        os.environ['MEMORY_PERSIST_DIR'] = self.temp_dir
    
    def tearDown(self):
        # Clean up
        if self.original_persist_dir:
            os.environ['MEMORY_PERSIST_DIR'] = self.original_persist_dir
        else:
            del os.environ['MEMORY_PERSIST_DIR']
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('agent.OpenAI')
    @patch('memory.Chroma')
    def test_full_conversation_flow(self, mock_chroma, mock_openai):
        """Test a complete conversation flow."""
        # Mock dependencies
        mock_llm = Mock()
        mock_llm.predict.return_value = "Mock AI response"
        mock_openai.return_value = mock_llm
        
        mock_vector_store = Mock()
        mock_chroma.return_value = mock_vector_store
        
        # Create agent
        agent = AIAgent()
        
        # Simulate conversation
        user_message = "What's the weather like?"
        response = agent.chat(user_message)
        
        # Verify response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Verify memory was updated
        memory_stats = agent.get_memory_summary()
        self.assertIn('memory_size', memory_stats)
        
        # Verify cache was updated
        cache_stats = agent.cache.get_stats()
        self.assertGreater(cache_stats['cache_size'], 0)

def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestResponseCache,
        TestEnhancedMemory,
        TestAIAgent,
        TestConfiguration,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == '__main__':
    print("Running AI Agent Test Suite")
    print("=" * 50)
    
    result = run_tests()
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 