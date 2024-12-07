#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type definitions for the conversation flow system.

This module defines the core types used throughout the flow system:
- FlowResult: Function return type
- FlowArgs: Function argument type
- NodeConfig: Node configuration type
- FlowConfig: Complete flow configuration type

These types provide structure and validation for flow configurations
and function interactions.
"""

from typing import Any, Dict, List, TypedDict


class FlowResult(TypedDict, total=False):
    """Base type for function results.

    Example:
        {
            "status": "success",
            "data": {"processed": True},
            "error": None  # Optional error message
        }
    """

    status: str
    error: str


FlowArgs = Dict[str, Any]
"""Type alias for function handler arguments.

Example:
    {
        "user_name": "John",
        "age": 25,
        "preferences": {"color": "blue"}
    }
"""


class NodeConfigRequired(TypedDict):
    """Required fields for node configuration."""

    messages: List[dict]
    functions: List[dict]


class NodeConfig(NodeConfigRequired, total=False):
    """Configuration for a single node in the flow.

    Required fields:
        messages: List of message dicts in provider-specific format
        functions: List of function definitions in provider-specific format

    Optional fields:
        pre_actions: Actions to execute before LLM inference
        post_actions: Actions to execute after LLM inference

    Example:
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are handling orders..."
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "process_order",
                        "description": "Process the order",
                        "parameters": {...}
                    }
                }
            ],
            "pre_actions": [
                {
                    "type": "tts_say",
                    "text": "Processing your order..."
                }
            ],
            "post_actions": [
                {
                    "type": "update_db",
                    "user_id": 123,
                    "data": {"status": "completed"}
                }
            ]
        }
    """

    pre_actions: List[Dict[str, Any]]
    post_actions: List[Dict[str, Any]]


class FlowConfig(TypedDict):
    """Configuration for the entire conversation flow.

    Attributes:
        initial_node: Name of the starting node
        nodes: Dictionary mapping node names to their configurations

    Example:
        {
            "initial_node": "greeting",
            "nodes": {
                "greeting": {
                    "messages": [...],
                    "functions": [...],
                    "pre_actions": [...]
                },
                "process_order": {
                    "messages": [...],
                    "functions": [...],
                    "post_actions": [...]
                }
            }
        }
    """

    initial_node: str
    nodes: Dict[str, NodeConfig]
