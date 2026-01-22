"""Shared fixtures and mocks for RAG chatbot tests"""

import pytest
from unittest.mock import Mock
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import SearchResults


@pytest.fixture
def sample_search_results():
    """Create sample successful search results"""
    return SearchResults(
        documents=["Lesson content about machine learning basics"],
        metadata=[{
            "course_title": "ML Fundamentals",
            "lesson_number": 1,
            "chunk_index": 0
        }],
        distances=[0.15]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Create search results with error"""
    return SearchResults.empty("No course found matching 'NonExistent'")


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Create a mock VectorStore with working configuration"""
    store = Mock()
    store.max_results = 5
    store.search = Mock(return_value=sample_search_results)
    store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
    return store


@pytest.fixture
def mock_vector_store_zero_results():
    """Create a mock VectorStore simulating MAX_RESULTS=0 bug"""
    store = Mock()
    store.max_results = 0
    store.search = Mock(return_value=SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    ))
    store.get_lesson_link = Mock(return_value=None)
    return store


@pytest.fixture
def mock_anthropic_response_with_tool_use():
    """Create mock Anthropic response that requests tool use"""
    response = Mock()
    response.stop_reason = "tool_use"

    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_123"
    tool_use_block.input = {"query": "machine learning"}

    response.content = [tool_use_block]
    return response


@pytest.fixture
def mock_anthropic_response_text():
    """Create mock Anthropic response with text only"""
    response = Mock()
    response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = "Here is the answer about machine learning."

    response.content = [text_block]
    return response
