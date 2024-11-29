import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import (
    EndFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
)
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.google import GoogleLLMService
from pipecat.services.openai import OpenAILLMService

from pipecat_flows import FlowManager
from pipecat_flows.exceptions import (
    FlowTransitionError,
    InvalidFunctionError,
)


class TestFlowManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test fixtures before each test"""
        # Reset mock call counts for each test
        self.mock_task = AsyncMock()
        self.mock_task.queue_frame = AsyncMock()
        self.mock_tts = AsyncMock()
        self.mock_tts.say = AsyncMock()

        # Create fresh LLM mocks for each test
        self.mock_openai = MagicMock(spec=OpenAILLMService)
        self.mock_anthropic = MagicMock(spec=AnthropicLLMService)
        self.mock_gemini = MagicMock(spec=GoogleLLMService)

        # Provider-specific flow configurations
        self.openai_config = {
            "initial_node": "start",
            "nodes": {
                "start": {
                    "messages": [{"role": "system", "content": "Start node"}],
                    "functions": [
                        {
                            "type": "function",
                            "function": {
                                "name": "process",
                                "description": "Process data",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"data": {"type": "string"}},
                                },
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "middle",
                                "description": "Go to middle",
                                "parameters": {},
                            },
                        },
                    ],
                    "pre_actions": [{"type": "tts_say", "text": "Starting..."}],
                },
                "middle": {
                    "messages": [{"role": "system", "content": "Middle node"}],
                    "functions": [
                        {
                            "type": "function",
                            "function": {
                                "name": "end",
                                "description": "End conversation",
                                "parameters": {},
                            },
                        }
                    ],
                    "post_actions": [{"type": "tts_say", "text": "Processing complete"}],
                },
                "end": {
                    "messages": [{"role": "system", "content": "End node"}],
                    "functions": [],
                    "pre_actions": [
                        {"type": "tts_say", "text": "Goodbye!"},
                        {"type": "end_conversation"},
                    ],
                },
            },
        }

        self.anthropic_config = {
            "initial_node": "start",
            "nodes": {
                "start": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Start node"}]}
                    ],
                    "functions": [
                        {
                            "name": "process",
                            "description": "Process data",
                            "input_schema": {
                                "type": "object",
                                "properties": {"data": {"type": "string"}},
                            },
                        },
                        {"name": "middle", "description": "Go to middle", "input_schema": {}},
                    ],
                    "pre_actions": [{"type": "tts_say", "text": "Starting..."}],
                },
                "middle": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Middle node"}]}
                    ],
                    "functions": [
                        {"name": "end", "description": "End conversation", "input_schema": {}}
                    ],
                },
                "end": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "End node"}]}
                    ],
                    "functions": [],
                },
            },
        }

        self.gemini_config = {
            "initial_node": "start",
            "nodes": {
                "start": {
                    "messages": [{"role": "system", "content": "Start node"}],
                    "functions": [
                        {
                            "function_declarations": [
                                {
                                    "name": "process",
                                    "description": "Process data",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"data": {"type": "string"}},
                                    },
                                },
                                {"name": "middle", "description": "Go to middle", "parameters": {}},
                            ]
                        }
                    ],
                    "pre_actions": [{"type": "tts_say", "text": "Starting..."}],
                },
                "middle": {
                    "messages": [{"role": "system", "content": "Middle node"}],
                    "functions": [
                        {
                            "function_declarations": [
                                {"name": "end", "description": "End conversation", "parameters": {}}
                            ]
                        }
                    ],
                },
                "end": {"messages": [{"role": "system", "content": "End node"}], "functions": []},
            },
        }

    async def test_initialization_all_providers(self):
        """Test initialization with all LLM providers"""
        initial_messages = [{"role": "system", "content": "Initial context"}]

        for config, llm in [
            (self.openai_config, self.mock_openai),
            (self.anthropic_config, self.mock_anthropic),
            (self.gemini_config, self.mock_gemini),
        ]:
            # Reset mock call counts
            self.mock_task.reset_mock()

            flow_manager = FlowManager(config, self.mock_task, llm, self.mock_tts)
            await flow_manager.initialize(initial_messages)

            # Verify initialization state
            self.assertTrue(flow_manager.initialized)

            # Verify frames were queued
            calls = self.mock_task.queue_frame.call_args_list
            self.assertEqual(len(calls), 2)  # Should have exactly 2 calls

            # Verify first call is LLMMessagesUpdateFrame
            self.assertIsInstance(calls[0].args[0], LLMMessagesUpdateFrame)
            self.assertEqual(
                calls[0].args[0].messages, initial_messages + config["nodes"]["start"]["messages"]
            )

            # Verify second call is LLMSetToolsFrame
            self.assertIsInstance(calls[1].args[0], LLMSetToolsFrame)
            self.assertEqual(calls[1].args[0].tools, config["nodes"]["start"]["functions"])

    async def test_function_registration_all_providers(self):
        """Test function registration for all providers"""
        for config, llm in [
            (self.openai_config, self.mock_openai),
            (self.anthropic_config, self.mock_anthropic),
            (self.gemini_config, self.mock_gemini),
        ]:
            # Reset mock
            llm.register_function.reset_mock()

            flow_manager = FlowManager(config, self.mock_task, llm, self.mock_tts)
            await flow_manager.initialize([])

            # Verify edge functions were registered with wrapper function
            llm.register_function.assert_any_call(
                "middle",
                unittest.mock.ANY,  # Accept any function since it's wrapped
            )

    async def test_transitions_all_providers(self):
        """Test transitions with all providers"""
        for config, llm in [
            (self.openai_config, self.mock_openai),
            (self.anthropic_config, self.mock_anthropic),
            (self.gemini_config, self.mock_gemini),
        ]:
            # Reset mock
            self.mock_task.reset_mock()

            flow_manager = FlowManager(config, self.mock_task, llm, self.mock_tts)
            await flow_manager.initialize([])

            # Test valid transition
            await flow_manager.handle_transition("middle")

            # Get all calls after initialization
            calls = self.mock_task.queue_frame.call_args_list

            # Verify LLMMessagesAppendFrame and LLMSetToolsFrame were called
            append_frames = [c for c in calls if isinstance(c.args[0], LLMMessagesAppendFrame)]
            tools_frames = [c for c in calls if isinstance(c.args[0], LLMSetToolsFrame)]

            self.assertTrue(len(append_frames) > 0)
            self.assertTrue(len(tools_frames) > 0)

            # Verify content of last frames
            last_append = append_frames[-1]
            self.assertEqual(last_append.args[0].messages, config["nodes"]["middle"]["messages"])

    async def test_complex_flow_all_providers(self):
        """Test complete flow sequences with all providers"""
        for config, llm in [
            (self.openai_config, self.mock_openai),
            (self.anthropic_config, self.mock_anthropic),
            (self.gemini_config, self.mock_gemini),
        ]:
            # Reset mocks
            self.mock_task.reset_mock()
            self.mock_tts.reset_mock()

            flow_manager = FlowManager(config, self.mock_task, llm, self.mock_tts)
            await flow_manager.initialize([])

            # Execute complete flow
            await flow_manager.handle_transition("middle")
            await flow_manager.handle_transition("end")

            # Add small delay to allow actions to complete
            await asyncio.sleep(0.1)

            # Verify key frames were queued
            calls = self.mock_task.queue_frame.call_args_list

            # Print actual calls for debugging
            print("\nActual calls:")
            for call in calls:
                print(f"- {type(call.args[0]).__name__}")

            # Verify message and tool frames
            self.assertTrue(
                any(isinstance(call.args[0], LLMMessagesAppendFrame) for call in calls),
                "No LLMMessagesAppendFrame found",
            )
            self.assertTrue(
                any(isinstance(call.args[0], LLMSetToolsFrame) for call in calls),
                "No LLMSetToolsFrame found",
            )

            # Verify end conversation actions were executed
            if "pre_actions" in config["nodes"]["end"]:
                end_actions = config["nodes"]["end"]["pre_actions"]
                if any(action["type"] == "end_conversation" for action in end_actions):
                    # Either verify EndFrame was queued
                    end_frame_queued = any(isinstance(call.args[0], EndFrame) for call in calls)
                    # Or verify end_conversation action was registered
                    action_registered = any(
                        "end_conversation" in str(call)
                        for call in self.mock_task.queue_frame.mock_calls
                    )
                    self.assertTrue(
                        end_frame_queued or action_registered, "No end conversation action found"
                    )

    async def test_error_handling_all_providers(self):
        """Test error handling for all providers"""
        for config, llm in [
            (self.openai_config, self.mock_openai),
            (self.anthropic_config, self.mock_anthropic),
            (self.gemini_config, self.mock_gemini),
        ]:
            # Test uninitialized transition
            flow_manager = FlowManager(config, self.mock_task, llm, self.mock_tts)
            with self.assertRaises(FlowTransitionError):
                await flow_manager.handle_transition("middle")

            # Initialize and try invalid transition
            await flow_manager.initialize([])
            with self.assertRaises(InvalidFunctionError):
                await flow_manager.handle_transition("nonexistent")

            # Test double initialization
            flow_manager = FlowManager(config, self.mock_task, llm, self.mock_tts)
            await flow_manager.initialize([])
            # Just verify it doesn't raise an error
            await flow_manager.initialize([])  # Should log warning but not raise

    async def test_action_execution_all_providers(self):
        """Test action execution for all providers"""
        for config, llm in [
            (self.openai_config, self.mock_openai),
            (self.anthropic_config, self.mock_anthropic),
            (self.gemini_config, self.mock_gemini),
        ]:
            # Reset mocks
            self.mock_tts.reset_mock()
            self.mock_task.reset_mock()

            flow_manager = FlowManager(config, self.mock_task, llm, self.mock_tts)
            await flow_manager.initialize([])

            # Print debug information
            print(
                f"\nTesting config with pre_actions: {config['nodes']['start'].get('pre_actions')}"
            )
            print(f"TTS mock calls: {self.mock_tts.say.mock_calls}")
            print(f"TTS mock called: {self.mock_tts.say.called}")

            # Test start node pre-actions
            if "pre_actions" in config["nodes"]["start"]:
                pre_actions = config["nodes"]["start"]["pre_actions"]
                for action in pre_actions:
                    if action["type"] == "tts_say":
                        # Verify the action handler was registered
                        self.assertIn(
                            "tts_say",
                            flow_manager.action_manager.action_handlers,
                            "TTS action handler not registered",
                        )

                        # Execute the action explicitly
                        await flow_manager.action_manager.execute_actions([action])

                        # Verify TTS was called with correct text
                        self.assertTrue(
                            self.mock_tts.say.called, f"TTS say not called with action: {action}"
                        )
                        self.mock_tts.say.assert_called_with(action["text"])

            # Test middle node post-actions
            if "post_actions" in config["nodes"]["middle"]:
                # Reset TTS mock for middle node actions
                self.mock_tts.reset_mock()

                # Transition to middle
                await flow_manager.handle_transition("middle")

                post_actions = config["nodes"]["middle"]["post_actions"]
                for action in post_actions:
                    if action["type"] == "tts_say":
                        self.assertTrue(
                            self.mock_tts.say.called,
                            f"TTS say not called with post-action: {action}",
                        )
                        self.mock_tts.say.assert_called_with(action["text"])

            # Test end node pre-actions
            if "pre_actions" in config["nodes"]["end"]:
                # Reset TTS mock for end node actions
                self.mock_tts.reset_mock()

                # Transition to end from current node
                await flow_manager.handle_transition("end")

                pre_actions = config["nodes"]["end"]["pre_actions"]
                for action in pre_actions:
                    if action["type"] == "tts_say":
                        self.assertTrue(
                            self.mock_tts.say.called,
                            f"TTS say not called with end pre-action: {action}",
                        )
                        self.mock_tts.say.assert_called_with(action["text"])

            # Add debug output for state transitions
            print(f"\nFinal node: {flow_manager.flow.current_node}")
            print(f"Available functions: {flow_manager.flow.get_available_function_names()}")

    async def test_action_manager_setup(self):
        """Test that action manager is properly initialized"""
        flow_manager = FlowManager(
            self.openai_config, self.mock_task, self.mock_openai, self.mock_tts
        )

        # Verify action manager exists
        self.assertIsNotNone(flow_manager.action_manager)  # Changed from _action_manager

        # Verify built-in actions are registered
        self.assertIn("tts_say", flow_manager.action_manager.action_handlers)
        self.assertIn("end_conversation", flow_manager.action_manager.action_handlers)

        # Verify TTS service is properly set
        self.assertEqual(flow_manager.action_manager.tts, self.mock_tts)

    @patch("loguru.logger.debug")
    async def test_logging_all_providers(self, mock_logger):
        """Test logging for all providers"""
        for config, llm in [
            (self.openai_config, self.mock_openai),
            (self.anthropic_config, self.mock_anthropic),
            (self.gemini_config, self.mock_gemini),
        ]:
            # Reset mock
            mock_logger.reset_mock()

            flow_manager = FlowManager(config, self.mock_task, llm, self.mock_tts)
            await flow_manager.initialize([])
            await flow_manager.handle_transition("middle")

            # Verify transition logging
            mock_logger.assert_any_call("Attempting transition from start to middle")

    async def test_null_and_empty_cases(self):
        """Test handling of null and empty values"""
        # Test with empty messages
        flow_manager = FlowManager(
            self.openai_config, self.mock_task, self.mock_openai, self.mock_tts
        )
        await flow_manager.initialize([])

        # Test with empty functions
        config = self.openai_config.copy()
        config["nodes"]["start"]["functions"] = []
        flow_manager = FlowManager(config, self.mock_task, self.mock_openai, self.mock_tts)
        await flow_manager.initialize([])

        # Test with missing optional fields
        config = self.openai_config.copy()
        del config["nodes"]["start"]["pre_actions"]
        flow_manager = FlowManager(config, self.mock_task, self.mock_openai, self.mock_tts)
        await flow_manager.initialize([])
