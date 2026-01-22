"""Tests for AIGenerator - Claude API integration with tool calling"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class TestAIGeneratorResponse:
    """Test AIGenerator.generate_response() method"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_direct_answer(self, mock_anthropic_class):
        """Test direct response without tool use"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Direct answer")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response("Hello")

        assert result == "Direct answer"
        mock_client.messages.create.assert_called_once()

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_class):
        """Test response that uses tools"""
        mock_client = Mock()

        # First response requests tool use
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "machine learning"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_block]

        # Second response after tool execution
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(type="text", text="Answer based on search")]

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic_class.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "[Course Content] ML basics..."

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="What is ML?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        assert result == "Answer based on search"
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning"
        )

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_receives_empty_results_from_bug(self, mock_anthropic_class):
        """CRITICAL: Test behavior when tool returns empty due to MAX_RESULTS=0

        When the search tool returns "No relevant content found" (due to
        MAX_RESULTS=0 bug), the AI has no context and returns an unhelpful
        response.
        """
        mock_client = Mock()

        # First response requests tool use
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "machine learning"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_block]

        # Second response after empty tool result
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(
            type="text",
            text="I couldn't find any relevant content in the course materials."
        )]

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic_class.return_value = mock_client

        # Mock tool manager returning empty results (the bug)
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "No relevant content found."

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="What is ML?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # The AI has no context and gives unhelpful response
        assert "couldn't find" in result.lower()

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response with conversation context"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Context-aware answer")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="Follow up question",
            conversation_history="User: Previous question\nAssistant: Previous answer"
        )

        # Verify history was included in system prompt
        call_args = mock_client.messages.create.call_args
        assert "Previous question" in call_args.kwargs["system"]
        assert result == "Context-aware answer"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_passes_tools_to_api(self, mock_anthropic_class):
        """Test that tools are correctly passed to Claude API"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(type="text", text="Answer")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        tools = [
            {"name": "search_course_content", "description": "Search courses"},
            {"name": "get_course_outline", "description": "Get outline"}
        ]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Test", tools=tools)

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["tools"] == tools
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}


class TestAIGeneratorToolExecution:
    """Test _handle_tool_execution() method"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_multiple_tool_calls(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        mock_client = Mock()

        # Response with multiple tool calls
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "topic A"}

        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"course_name": "ML Course"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_block_1, tool_block_2]

        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(type="text", text="Combined answer")]

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="Complex question",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager
        )

        assert result == "Combined answer"
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_results_sent_back_to_api(self, mock_anthropic_class):
        """Test that tool results are correctly sent back to Claude"""
        mock_client = Mock()

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_abc"
        tool_block.input = {"query": "test"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_block]

        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(type="text", text="Final answer")]

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool output here"

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="Test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Check the second API call includes tool results
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs["messages"]

        # Should have: user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_abc"
        assert messages[2]["content"][0]["content"] == "Tool output here"
