#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from .formats import LLMFormatParser, LLMProvider


@dataclass
class NodeConfig:
    """Configuration for a single node in the flow.

    A node represents a state in the conversation flow, containing all the
    information needed for that particular point in the conversation.

    Attributes:
        messages: List of message dicts in provider-specific format
        functions: List of function definitions in provider-specific format
        pre_actions: Optional list of actions to execute before LLM inference
        post_actions: Optional list of actions to execute after LLM inference
    """

    messages: List[dict]
    functions: List[dict]
    pre_actions: Optional[List[dict]] = None
    post_actions: Optional[List[dict]] = None


class FlowState:
    """Manages the state and transitions between nodes in a conversation flow.

    This class handles the state machine logic for conversation flows, where each node
    represents a distinct state with its own messages, available functions, and optional
    pre- and post-actions. It manages transitions between nodes based on function calls
    and handles both node and edge functions.

    Attributes:
        nodes: Dictionary mapping node IDs to their configurations
        current_node: ID of the currently active node
        provider: LLM provider type for format parsing
    """

    def __init__(self, flow_config: dict, llm):
        """Initialize the conversation flow.

        Args:
            flow_config: Dictionary containing the complete flow configuration,
                        must include 'initial_node' and 'nodes' keys
            llm: LLM service instance for determining provider type

        Raises:
            ValueError: If required configuration keys are missing
        """
        self.nodes: Dict[str, NodeConfig] = {}
        self.current_node: str = flow_config["initial_node"]
        self.provider = LLMFormatParser.get_provider(llm)
        self._load_config(flow_config)

    def _load_config(self, config: dict):
        """Load and validate the flow configuration.

        Args:
            config: Dictionary containing the flow configuration

        Raises:
            ValueError: If required configuration keys are missing
        """
        if "initial_node" not in config:
            raise ValueError("Flow config must specify 'initial_node'")
        if "nodes" not in config:
            raise ValueError("Flow config must specify 'nodes'")

        for node_id, node_config in config["nodes"].items():
            self.nodes[node_id] = NodeConfig(
                messages=node_config["messages"],
                functions=node_config["functions"],
                pre_actions=node_config.get("pre_actions"),
                post_actions=node_config.get("post_actions"),
            )

    def get_current_messages(self) -> List[dict]:
        """Get the messages for the current node.

        Returns:
            List of message dictionaries for the current node in provider-specific format
        """
        messages = self.nodes[self.current_node].messages
        return messages

    def get_current_functions(self) -> List[dict]:
        """Get the available functions for the current node.

        Returns:
            List of function definitions available for the current node in provider-specific
                format
        """
        functions = self.nodes[self.current_node].functions

        if self.provider == LLMProvider.GEMINI:
            # For Gemini, combine all function declarations into a single tools object
            all_declarations = []
            for func in functions:
                if "function_declarations" in func:
                    all_declarations.extend(func["function_declarations"])
            return [{"function_declarations": all_declarations}] if all_declarations else []

        return functions

    def get_current_pre_actions(self) -> Optional[List[dict]]:
        """Get the pre-actions for the current node.

        Pre-actions are executed before updating the LLM context when
        transitioning to this node.

        Returns:
            List of pre-actions to execute, or None if no pre-actions
        """
        return self.nodes[self.current_node].pre_actions

    def get_current_post_actions(self) -> Optional[List[dict]]:
        """Get the post-actions for the current node.

        Post-actions are executed after updating the LLM context when
        transitioning to this node.

        Returns:
            List of post-actions to execute, or None if no post-actions
        """
        return self.nodes[self.current_node].post_actions

    def get_available_function_names(self) -> Set[str]:
        """Get the names of available functions for the current node.

        Returns:
            Set of function names that can be called from the current node
        """
        functions = self.nodes[self.current_node].functions

        if self.provider == LLMProvider.GEMINI:
            # Flatten Gemini's nested function declarations
            flattened = []
            for func in functions:
                if "function_declarations" in func:
                    flattened.extend(func["function_declarations"])
            functions = flattened

        names = {LLMFormatParser.get_function_name(self.provider, f) for f in functions}
        logger.debug(f"Available function names for node {self.current_node}: {names}")
        return names

    def get_function_name_from_call(self, function_call: Dict[str, Any]) -> str:
        """Extract function name from a function call.

        Args:
            function_call: Function call in provider-specific format

        Returns:
            Function name as string
        """
        return LLMFormatParser.get_function_name(self.provider, function_call)

    def get_all_available_function_names(self) -> Set[str]:
        """Get all function names across all nodes without modifying state.

        Returns:
            Set of all unique function names available in any node
        """
        all_functions = set()
        current = self.current_node  # Store current node

        try:
            for node_id in self.nodes.keys():
                self.current_node = node_id
                all_functions.update(self.get_available_function_names())
        finally:
            self.current_node = current  # Ensure we restore the original node

        return all_functions

    def get_function_args_from_call(self, function_call: Dict[str, Any]) -> dict:
        """Extract function arguments from a function call.

        Args:
            function_call: Function call in provider-specific format

        Returns:
            Dictionary of function arguments
        """
        return LLMFormatParser.get_function_args(self.provider, function_call)

    def transition(self, function_name: str) -> Optional[str]:
        """Attempt to transition to a new node based on a function call.

        This method handles two types of functions:
        1. Edge Functions: Functions whose names match node names, triggering
        a transition to a new node in the graph.
        2. Node Functions: Functions that execute within the current node without
        triggering a state change.

        Args:
            function_name: Name of the function that was called

        Returns:
            str | None: The ID of the new node if a transition occurred (edge function),
                    or None if no transition should occur (node function or invalid function)
        """
        available_functions = self.get_available_function_names()
        logger.debug(f"Attempting transition from {self.current_node} to {function_name}")

        if function_name not in available_functions:
            logger.warning(f"Function {function_name} not available in current node")
            return None

        # Only transition if the function name matches a node name (edge function)
        if function_name in self.nodes:
            previous_node = self.current_node
            self.current_node = function_name
            logger.info(f"Transitioned from {previous_node} to node: {self.current_node}")
            return self.current_node
        else:
            # Node function - no transition needed
            logger.info(f"Executed node function: {function_name}")
            return None

    def get_current_node(self) -> str:
        """Get the current node ID."""
        return self.current_node
