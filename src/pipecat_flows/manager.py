#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Callable, List

from loguru import logger
from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
)

from .actions import ActionManager
from .state import FlowState


class FlowManager:
    """Manages conversation flows in a Pipecat pipeline.

    This manager handles the progression through a flow defined by nodes, where each node
    represents a state in the conversation. Each node has:
    - Messages for the LLM (in provider-specific format)
    - Available functions that can be called (in provider-specific format)
    - Optional pre-actions to execute before LLM inference
    - Optional post-actions to execute after LLM inference

    The flow is defined by a configuration that specifies:
    - Initial node
    - Available nodes and their configurations
    - Transitions between nodes via function calls

    Function handling is split between:
    - Node functions: Registered directly with the LLM before flow initialization
    - Edge functions: Registered by FlowManager during initialization

    While all functions are registered with the LLM, only functions defined in the
    current node's configuration are available for use at any given time.
    """

    def __init__(self, flow_config: dict, task, llm, tts=None):
        """Initialize the flow manager.

        Args:
            flow_config: Dictionary containing the flow configuration
            task: PipelineTask instance used to queue frames
            llm: LLM service for handling functions
            tts: Optional TTS service for voice actions
        """
        self.flow = FlowState(flow_config, llm)
        self.initialized = False
        self.task = task
        self.llm = llm
        self.action_manager = ActionManager(task, tts)

    async def initialize(self, initial_messages: List[dict]):
        """Initialize the flow with starting messages and functions.

        This method:
        1. Registers edge functions with the LLM (node functions should already be registered)
        2. Sets up the initial context with system messages and node messages
        3. Sets available tools based on the initial node's configuration

        Args:
            initial_messages: List of initial messages (typically system messages)
                            to include in the context
        """
        if not self.initialized:
            await self.register_functions()

            messages = initial_messages + self.flow.get_current_messages()
            await self.task.queue_frame(LLMMessagesUpdateFrame(messages=messages))
            await self.task.queue_frame(LLMSetToolsFrame(tools=self.flow.get_current_functions()))
            self.initialized = True
            logger.debug(f"Initialized flow at node: {self.flow.current_node}")
        else:
            logger.warning("Attempted to initialize FlowManager multiple times")

    def register_action(self, action_type: str, handler: Callable) -> None:
        """Register a handler for a specific action type.

        Args:
            action_type: String identifier for the action (e.g., "tts_say")
            handler: Async or sync function that handles the action

        Example:
            async def custom_notification(action: dict):
                notification_text = action.get("notification_text", "")
                # Custom notification logic here

            flow_manager.register_action("send_notification", custom_notification)

            # Can then be used in flow configuration:
            {
                "pre_actions": [
                    {
                        "type": "send_notification",
                        "notification_text": "Starting process..."
                    }
                ]
            }
        """
        self.action_manager._register_action(action_type, handler)

    async def register_functions(self):
        """Register edge functions from the flow configuration with the LLM service.

        This method:
        1. Gets all available function names across all nodes using the format parser
        2. For node functions (names that don't match node names):
            - Expects them to be already registered with the LLM
            - Logs their presence but doesn't register them
        3. For edge functions (names that match node names):
            - Registers them with the LLM using handle_edge_function
            - These trigger state transitions when called
        """
        registered_handlers = set()

        async def handle_edge_function(
            function_name, tool_call_id, arguments, llm, context, result_callback
        ):
            await self.handle_transition(function_name)
            await result_callback("Acknowledged")

        # Get all available function names across all nodes safely
        all_functions = self.flow.get_all_available_function_names()

        # Register each function
        for function_name in all_functions:
            # Skip if already registered
            if function_name in registered_handlers:
                continue

            # Check if this is a node function (doesn't match any node name)
            is_node_function = function_name not in self.flow.nodes

            if is_node_function:
                # Don't override existing node function handlers
                if not hasattr(
                    self.llm, "has_function_handler"
                ) or not self.llm.has_function_handler(function_name):
                    logger.debug(f"Found node function: {function_name}")
            else:
                # Register edge function handler
                self.llm.register_function(function_name, handle_edge_function)
                logger.debug(f"Registered edge function: {function_name}")

            registered_handlers.add(function_name)

    async def handle_transition(self, function_name: str):
        """Handle the execution of functions and potential node transitions.

        This method implements the core state transition logic of the conversation flow.
        It distinguishes between two types of functions:

        1. Edge Functions:
        - Function names that match existing node names
        - Trigger a transition to a new node with:
            * Pre-action execution
            * Context and tool updates (making new node's functions available)
            * Post-action execution

        2. Node Functions:
        - Function names that don't match any node names
        - Execute within the current node without changing state
        - Don't trigger context updates or actions
        - Must be registered with LLM before flow initialization

        The transition process for edge functions:
        1. Validates the function call against available functions
        2. Executes pre-actions of the new node
        3. Updates the LLM context with new messages
        4. Updates available tools for the new node (via LLMSetToolsFrame)
        5. Executes post-actions of the new node

        Args:
            function_name: Name of the function to execute

        Raises:
            RuntimeError: If handle_transition is called before initialization
        """
        if not self.initialized:
            raise RuntimeError("FlowManager must be initialized before handling transitions")

        available_functions = self.flow.get_available_function_names()

        if function_name not in available_functions:
            logger.warning(
                f"Received invalid function call '{function_name}' for node '{self.flow.current_node}'. "
                f"Available functions are: {available_functions}"
            )
            return

        # Attempt transition - returns new node ID for edge functions,
        # None for node functions
        new_node = self.flow.transition(function_name)

        # Only perform node transition logic for edge functions
        if new_node is not None:
            # Execute pre-actions before updating LLM context
            if self.flow.get_current_pre_actions():
                logger.debug(f"Executing pre-actions for node {new_node}")
                await self.action_manager.execute_actions(self.flow.get_current_pre_actions())

            # Update LLM context and tools
            current_messages = self.flow.get_current_messages()
            await self.task.queue_frame(LLMMessagesAppendFrame(messages=current_messages))
            await self.task.queue_frame(LLMSetToolsFrame(tools=self.flow.get_current_functions()))

            # Execute post-actions after updating LLM context
            if self.flow.get_current_post_actions():
                logger.debug(f"Executing post-actions for node {new_node}")
                await self.action_manager.execute_actions(self.flow.get_current_post_actions())

            logger.debug(f"Transition to node {new_node} complete")
        else:
            logger.debug(f"Node function {function_name} executed without transition")
