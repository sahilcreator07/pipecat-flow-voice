import unittest
from unittest.mock import MagicMock, patch

from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.google import GoogleLLMService
from pipecat.services.openai import OpenAILLMService

from pipecat_flows.adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter
from pipecat_flows.state import FlowState


class TestFlowState(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test cases with configs for different LLM providers"""
        # Create mock LLM services
        self.mock_openai = MagicMock(spec=OpenAILLMService)
        self.mock_anthropic = MagicMock(spec=AnthropicLLMService)
        self.mock_gemini = MagicMock(spec=GoogleLLMService)

        # OpenAI format config
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
                                "description": "Process node function",
                                "parameters": {},
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "middle",
                                "description": "Transition to middle",
                                "parameters": {},
                            },
                        },
                    ],
                },
                "middle": {
                    "messages": [{"role": "system", "content": "Middle node"}],
                    "functions": [
                        {
                            "type": "function",
                            "function": {
                                "name": "end",
                                "description": "Transition to end",
                                "parameters": {},
                            },
                        }
                    ],
                },
                "end": {"messages": [{"role": "system", "content": "End node"}], "functions": []},
            },
        }

        # Anthropic format config
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
                            "description": "Process node function",
                            "input_schema": {},
                        },
                        {
                            "name": "middle",
                            "description": "Transition to middle",
                            "input_schema": {},
                        },
                    ],
                },
                "middle": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Middle node"}]}
                    ],
                    "functions": [
                        {"name": "end", "description": "Transition to end", "input_schema": {}}
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

        # Gemini format config
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
                                    "description": "Process node function",
                                    "parameters": {},
                                },
                                {
                                    "name": "middle",
                                    "description": "Transition to middle",
                                    "parameters": {},
                                },
                            ]
                        }
                    ],
                },
                "middle": {
                    "messages": [{"role": "system", "content": "Middle node"}],
                    "functions": [
                        {
                            "function_declarations": [
                                {
                                    "name": "end",
                                    "description": "Transition to end",
                                    "parameters": {},
                                }
                            ]
                        }
                    ],
                },
                "end": {"messages": [{"role": "system", "content": "End node"}], "functions": []},
            },
        }

    def test_initialization_with_different_llms(self):
        """Test initialization with different LLM providers"""
        # OpenAI
        flow_openai = FlowState(self.openai_config, self.mock_openai)
        self.assertEqual(flow_openai.current_node, "start")
        functions = flow_openai.get_current_functions()
        self.assertTrue(all("type" in f for f in functions))

        # Anthropic
        flow_anthropic = FlowState(self.anthropic_config, self.mock_anthropic)
        functions = flow_anthropic.get_current_functions()
        self.assertTrue(all("input_schema" in f for f in functions))

        # Gemini
        flow_gemini = FlowState(self.gemini_config, self.mock_gemini)
        functions = flow_gemini.get_current_functions()
        self.assertTrue("function_declarations" in functions[0])

    @patch("pipecat_flows.state.create_adapter")
    def test_initialization_errors(self, mock_create_adapter):
        """Test initialization error cases"""
        mock_create_adapter.return_value = OpenAIAdapter()

        # Test missing initial_node
        invalid_config = {"nodes": {}}
        with self.assertRaises(ValueError) as context:
            FlowState(invalid_config, self.mock_openai)
        self.assertEqual(str(context.exception), "Flow config must specify 'initial_node'")

        # Test missing nodes
        invalid_config = {"initial_node": "start"}
        with self.assertRaises(ValueError) as context:
            FlowState(invalid_config, self.mock_openai)
        self.assertEqual(str(context.exception), "Flow config must specify 'nodes'")

        # Test initial node not in nodes
        invalid_config = {"initial_node": "invalid", "nodes": {}}
        with self.assertRaises(ValueError) as context:
            FlowState(invalid_config, self.mock_openai)
        self.assertEqual(str(context.exception), "Initial node 'invalid' not found in nodes")

    def test_get_current_messages(self):
        """Test retrieving messages for current node"""
        flow = FlowState(self.openai_config, self.mock_openai)
        messages = flow.get_current_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "Start node")

    def test_get_current_functions(self):
        """Test retrieving functions for current node"""
        flow = FlowState(self.openai_config, self.mock_openai)
        functions = flow.get_current_functions()
        self.assertEqual(len(functions), 2)
        self.assertTrue(all("type" in f for f in functions))
        self.assertTrue(all("function" in f for f in functions))

    def test_get_available_function_names(self):
        """Test retrieving available function names"""
        flow = FlowState(self.openai_config, self.mock_openai)
        names = flow.get_available_function_names()
        self.assertEqual(names, {"process", "middle"})

    @patch("pipecat_flows.state.create_adapter")
    def test_adapter_creation(self, mock_create_adapter):
        """Test that the correct adapter is created for each LLM type"""
        # Configure mock adapters
        mock_create_adapter.return_value = OpenAIAdapter()

        FlowState(self.openai_config, self.mock_openai)
        mock_create_adapter.assert_called_once_with(self.mock_openai)

    @patch("pipecat_flows.state.create_adapter")
    def test_function_call_parsing(self, mock_create_adapter):
        """Test parsing function calls for different LLM formats"""
        # OpenAI
        mock_create_adapter.return_value = OpenAIAdapter()
        flow = FlowState(self.openai_config, self.mock_openai)

        openai_call = {
            "type": "function",
            "function": {"name": "process"},
            "arguments": {"data": "test"},  # Arguments at top level
        }
        self.assertEqual(flow.get_function_name_from_call(openai_call), "process")
        self.assertEqual(flow.get_function_args_from_call(openai_call), {"data": "test"})

        # Anthropic
        mock_create_adapter.return_value = AnthropicAdapter()
        flow = FlowState(self.anthropic_config, self.mock_anthropic)

        anthropic_call = {"name": "process", "arguments": {"data": "test"}}
        self.assertEqual(flow.get_function_name_from_call(anthropic_call), "process")
        self.assertEqual(flow.get_function_args_from_call(anthropic_call), {"data": "test"})

        # Gemini
        mock_create_adapter.return_value = GeminiAdapter()
        flow = FlowState(self.gemini_config, self.mock_gemini)

        gemini_call = {"name": "process", "args": {"data": "test"}}
        self.assertEqual(flow.get_function_name_from_call(gemini_call), "process")
        self.assertEqual(flow.get_function_args_from_call(gemini_call), {"data": "test"})

    def test_transition_with_different_formats(self):
        """Test transitions work correctly with different function formats"""
        # OpenAI
        flow_openai = FlowState(self.openai_config, self.mock_openai)
        self.assertEqual(flow_openai.transition("middle"), "middle")

        # Anthropic
        flow_anthropic = FlowState(self.anthropic_config, self.mock_anthropic)
        self.assertEqual(flow_anthropic.transition("middle"), "middle")

        # Gemini
        flow_gemini = FlowState(self.gemini_config, self.mock_gemini)
        self.assertEqual(flow_gemini.transition("middle"), "middle")

    def test_transition_edge_cases(self):
        """Test transition edge cases and error conditions"""
        flow = FlowState(self.openai_config, self.mock_openai)

        # Test transition with non-existent function
        result = flow.transition("non_existent")
        self.assertIsNone(result)
        self.assertEqual(flow.get_current_node(), "start")

        # Test transition with node function (shouldn't change state)
        result = flow.transition("process")
        self.assertIsNone(result)
        self.assertEqual(flow.get_current_node(), "start")

        # Test multiple valid transitions
        self.assertEqual(flow.transition("middle"), "middle")
        self.assertEqual(flow.get_current_node(), "middle")
        self.assertEqual(flow.transition("end"), "end")
        self.assertEqual(flow.get_current_node(), "end")

    def test_get_all_available_function_names(self):
        """Test retrieving all function names across all nodes"""
        flow = FlowState(self.openai_config, self.mock_openai)
        all_names = flow.get_all_available_function_names()
        self.assertEqual(all_names, {"process", "middle", "end"})
