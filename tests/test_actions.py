#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Test suite for ActionManager functionality.

This module tests the ActionManager class which handles execution of actions
during conversation flows. Tests cover:
- Built-in actions (TTS, end conversation)
- Custom action registration and execution
- Error handling and validation
- Action sequencing
- Frame queueing

The tests use unittest.IsolatedAsyncioTestCase for async support and include
mocked dependencies for PipelineTask.
"""

import unittest
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import EndFrame, TTSSpeakFrame

from pipecat_flows.actions import ActionManager
from pipecat_flows.exceptions import ActionError


class TestActionManager(unittest.IsolatedAsyncioTestCase):
    """Test suite for ActionManager class.

    Tests functionality of ActionManager including:
    - Built-in action handlers:
        - TTS speech synthesis
        - Conversation ending
    - Custom action registration
    - Action execution sequencing
    - Error handling:
        - Invalid actions
        - Failed handlers
    - Multiple action execution
    - Frame queueing validation

    Each test uses mocked dependencies to verify:
    - Correct frame generation
    - Proper service calls
    - Error handling behavior
    - Action sequencing
    """

    def setUp(self):
        """Set up test fixtures before each test.

        Creates:
        - Mock PipelineTask for frame queueing
        - ActionManager instance with mocked dependencies
        """
        self.mock_task = AsyncMock()
        self.mock_task.queue_frame = AsyncMock()
        self.action_manager = ActionManager(self.mock_task)

    async def test_initialization(self):
        """Test ActionManager initialization and default handlers."""
        # Verify built-in action handlers are registered
        self.assertIn("tts_say", self.action_manager.action_handlers)
        self.assertIn("end_conversation", self.action_manager.action_handlers)

    async def test_tts_action(self):
        """Test basic TTS action execution."""
        action = {"type": "tts_say", "text": "Hello"}
        await self.action_manager.execute_actions([action])

        # Verify TTSSpeakFrame was queued with correct text
        self.mock_task.queue_frame.assert_called_once()
        frame = self.mock_task.queue_frame.call_args[0][0]
        self.assertIsInstance(frame, TTSSpeakFrame)
        self.assertEqual(frame.text, "Hello")

    @patch("loguru.logger.error")
    async def test_tts_action_no_text(self, mock_logger):
        """Test TTS action with missing text field."""
        action = {"type": "tts_say"}  # Missing text field

        # The implementation logs error but doesn't raise
        await self.action_manager.execute_actions([action])

        # Verify error was logged
        mock_logger.assert_called_with("TTS action missing 'text' field")

        # Verify no frame was queued
        self.mock_task.queue_frame.assert_not_called()

    async def test_end_conversation_action(self):
        """Test basic end conversation action."""
        action = {"type": "end_conversation"}
        await self.action_manager.execute_actions([action])

        # Verify EndFrame was queued
        self.mock_task.queue_frame.assert_called_once()
        frame = self.mock_task.queue_frame.call_args[0][0]
        self.assertIsInstance(frame, EndFrame)

    async def test_end_conversation_with_goodbye(self):
        """Test end conversation action with goodbye message."""
        action = {"type": "end_conversation", "text": "Goodbye!"}
        await self.action_manager.execute_actions([action])

        # Verify both frames were queued in correct order
        self.assertEqual(self.mock_task.queue_frame.call_count, 2)

        # Verify TTSSpeakFrame
        first_frame = self.mock_task.queue_frame.call_args_list[0][0][0]
        self.assertIsInstance(first_frame, TTSSpeakFrame)
        self.assertEqual(first_frame.text, "Goodbye!")

        # Verify EndFrame
        second_frame = self.mock_task.queue_frame.call_args_list[1][0][0]
        self.assertIsInstance(second_frame, EndFrame)

    async def test_custom_action(self):
        """Test registering and executing custom actions."""
        mock_handler = AsyncMock()
        self.action_manager._register_action("custom", mock_handler)

        # Verify handler was registered
        self.assertIn("custom", self.action_manager.action_handlers)

        # Execute custom action
        action = {"type": "custom", "data": "test"}
        await self.action_manager.execute_actions([action])

        # Verify handler was called with correct data
        mock_handler.assert_called_once_with(action)

    async def test_invalid_action(self):
        """Test handling invalid actions."""
        # Test missing type
        with self.assertRaises(ActionError) as context:
            await self.action_manager.execute_actions([{}])
        self.assertIn("missing required 'type' field", str(context.exception))

        # Test unknown action type
        with self.assertRaises(ActionError) as context:
            await self.action_manager.execute_actions([{"type": "invalid"}])
        self.assertIn("No handler registered", str(context.exception))

    async def test_multiple_actions(self):
        """Test executing multiple actions in sequence."""
        actions = [
            {"type": "tts_say", "text": "First"},
            {"type": "tts_say", "text": "Second"},
        ]
        await self.action_manager.execute_actions(actions)

        # Verify frames were queued in correct order
        self.assertEqual(self.mock_task.queue_frame.call_count, 2)

        first_frame = self.mock_task.queue_frame.call_args_list[0][0][0]
        self.assertEqual(first_frame.text, "First")

        second_frame = self.mock_task.queue_frame.call_args_list[1][0][0]
        self.assertEqual(second_frame.text, "Second")

    def test_register_invalid_handler(self):
        """Test registering invalid action handlers."""
        # Test non-callable handler
        with self.assertRaises(ValueError) as context:
            self.action_manager._register_action("invalid", "not_callable")
        self.assertIn("must be callable", str(context.exception))

        # Test None handler
        with self.assertRaises(ValueError) as context:
            self.action_manager._register_action("invalid", None)
        self.assertIn("must be callable", str(context.exception))

    async def test_none_or_empty_actions(self):
        """Test handling None or empty action lists."""
        # Test None actions
        await self.action_manager.execute_actions(None)
        self.mock_task.queue_frame.assert_not_called()

        # Test empty list
        await self.action_manager.execute_actions([])
        self.mock_task.queue_frame.assert_not_called()

    @patch("loguru.logger.error")
    async def test_action_error_handling(self, mock_logger):
        """Test error handling during action execution."""
        # Configure task mock to raise an error
        self.mock_task.queue_frame.side_effect = Exception("Frame error")

        action = {"type": "tts_say", "text": "Hello"}
        await self.action_manager.execute_actions([action])

        # Verify error was logged
        mock_logger.assert_called_with("TTS error: Frame error")

    async def test_action_execution_error_handling(self):
        """Test error handling during action execution."""
        # Test action with missing handler
        with self.assertRaises(ActionError):
            await self.action_manager.execute_actions([{"type": "nonexistent_action"}])

        # Test action handler that raises an exception
        async def failing_handler(action):
            raise Exception("Handler error")

        self.action_manager._register_action("failing_action", failing_handler)

        with self.assertRaises(ActionError):
            await self.action_manager.execute_actions([{"type": "failing_action"}])
