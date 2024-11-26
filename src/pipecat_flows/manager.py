#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from asyncio import iscoroutinefunction
from typing import Callable, Dict, List, Optional

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    TTSSpeakFrame,
)

from .state import FlowState


class FlowManager:
    """Manages conversation flows in a Pipecat pipeline.

    This manager handles the progression through a flow defined by nodes, where each node
    represents a state in the conversation. Each node has:
    - Messages for the LLM (system/user/assistant messages)
    - Available functions that can be called
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
        self.flow = FlowState(flow_config)
        self.initialized = False
        self.task = task
        self.llm = llm
        self.tts = tts
        self.action_handlers: Dict[str, Callable] = {}

        # Register built-in actions
        self.register_action("tts_say", self._handle_tts_action)
        self.register_action("end_conversation", self._handle_end_action)

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

    async def register_functions(self):
        """Register edge functions from the flow configuration with the LLM service.

        This method:
        1. Identifies functions defined in the flow configuration
        2. For node functions (names that don't match node names):
        - Expects them to be already registered with the LLM
        - Logs their presence but doesn't register them
        3. For edge functions (names that match node names):
        - Registers them with the LLM using handle_edge_function
        - These trigger state transitions when called

        Node functions should be registered with the LLM before flow initialization.
        Edge functions are automatically registered during initialization.
        """
        registered_handlers = set()

        async def handle_edge_function(
            function_name, tool_call_id, arguments, llm, context, result_callback
        ):
            await self.handle_transition(function_name)
            await result_callback("Acknowledged")

        # Register all functions from all nodes
        for node in self.flow.nodes.values():
            for function in node.functions:
                function_name = function["function"]["name"]

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
                    registered_handlers.add(function_name)
                else:
                    # Register edge function handler
                    self.llm.register_function(function_name, handle_edge_function)
                    registered_handlers.add(function_name)
                    logger.debug(f"Registered edge function: {function_name}")

    def register_action(self, action_type: str, handler: Callable):
        """Register a handler for a specific action type.

        Args:
            action_type: String identifier for the action (e.g., "tts_say")
            handler: Async or sync function that handles the action.
                    Should accept action configuration as parameter.
        """
        if not callable(handler):
            raise ValueError("Action handler must be callable")
        self.action_handlers[action_type] = handler

    async def _execute_actions(self, actions: Optional[List[dict]]) -> None:
        """Execute actions specified for the current node.

        Args:
            actions: List of action configurations to execute

        Note:
            Each action must have a 'type' field matching a registered handler
        """
        if not actions:
            return

        for action in actions:
            action_type = action["type"]
            if action_type in self.action_handlers:
                handler = self.action_handlers[action_type]
                try:
                    if iscoroutinefunction(handler):
                        await handler(action)
                    else:
                        handler(action)
                except Exception as e:
                    logger.warning(f"Error executing action {action_type}: {e}")
            else:
                logger.warning(f"No handler registered for action type: {action_type}")

    async def _handle_tts_action(self, action: dict):
        """Built-in handler for TTS actions that speak immediately.

        This handler attempts to use the TTS service directly to speak the text
        immediately, bypassing the pipeline queue. If no TTS service is available,
        it falls back to queueing the text through the pipeline.

        Args:
            action: Dictionary containing the action configuration.
                Must include a 'text' key with the text to speak.
        """
        if self.tts:
            # Direct call to TTS service to speak text immediately
            await self.tts.say(action["text"])
        else:
            # Fall back to queued TTS if no direct service available
            await self.task.queue_frame(TTSSpeakFrame(text=action["text"]))

    async def _handle_end_action(self, action: dict):
        """Built-in handler for ending the conversation.

        This handler queues an EndFrame to terminate the conversation. If the action
        includes a 'text' key, it will queue that text to be spoken before ending.

        Args:
            action: Dictionary containing the action configuration.
                Optional 'text' key for a goodbye message.
        """
        if action.get("text"):  # Optional goodbye message
            await self.task.queue_frame(TTSSpeakFrame(text=action["text"]))
        await self.task.queue_frame(EndFrame())

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

        # Only perform node transition logic if we got a new node
        # (meaning it was an edge function, not a node function)
        if new_node is not None:
            # Execute pre-actions before updating LLM context
            if self.flow.get_current_pre_actions():
                logger.debug(f"Executing pre-actions for node {new_node}")
                await self._execute_actions(self.flow.get_current_pre_actions())

            # Update LLM context and tools
            current_messages = self.flow.get_current_messages()
            await self.task.queue_frame(LLMMessagesAppendFrame(messages=current_messages))
            await self.task.queue_frame(LLMSetToolsFrame(tools=self.flow.get_current_functions()))

            # Execute post-actions after updating LLM context
            if self.flow.get_current_post_actions():
                logger.debug(f"Executing post-actions for node {new_node}")
                await self._execute_actions(self.flow.get_current_post_actions())

            logger.debug(f"Transition to node {new_node} complete")
        else:
            logger.debug(f"Node function {function_name} executed without transition")
