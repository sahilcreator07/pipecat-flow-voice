#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from enum import Enum
from typing import Any, Dict

from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.openai import OpenAILLMService


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMFormatParser:
    """Handles parsing of LLM-specific formats without format conversion.

    This class provides static methods to extract information from different
    LLM provider formats without converting between them. This approach
    maintains the original format while providing a consistent way to
    access the needed information.
    """

    @staticmethod
    def get_provider(llm) -> LLMProvider:
        """Determine provider from LLM instance.

        Args:
            llm: LLM service instance

        Returns:
            LLMProvider enum value

        Raises:
            ValueError: If LLM type is not supported
        """
        if isinstance(llm, OpenAILLMService):
            return LLMProvider.OPENAI
        elif isinstance(llm, AnthropicLLMService):
            return LLMProvider.ANTHROPIC
        raise ValueError(f"Unsupported LLM type: {type(llm)}")

    @staticmethod
    def get_function_name(provider: LLMProvider, function_def: Dict[str, Any]) -> str:
        """Extract function name from provider-specific function definition.

        Args:
            provider: LLM provider type
            function_def: Function definition in provider-specific format

        Returns:
            Function name as string

        Raises:
            ValueError: If provider is not supported
        """
        if provider == LLMProvider.OPENAI:
            # OpenAI format: {"type": "function", "function": {"name": "func_name", ...}}
            return function_def["function"]["name"]
        elif provider == LLMProvider.ANTHROPIC:
            # Anthropic format: {"name": "func_name", ...}
            return function_def["name"]
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_function_args(provider: LLMProvider, function_call: Dict[str, Any]) -> dict:
        """Extract function arguments from provider-specific function call.

        Args:
            provider: LLM provider type
            function_call: Function call in provider-specific format

        Returns:
            Dictionary of function arguments

        Raises:
            ValueError: If provider is not supported
        """
        if provider == LLMProvider.OPENAI:
            # OpenAI format: {"arguments": {...}}
            return function_call.get("arguments", {})
        elif provider == LLMProvider.ANTHROPIC:
            # Anthropic format: {"arguments": {...}}
            return function_call.get("arguments", {})
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_message_content(provider: LLMProvider, message: Dict[str, Any]) -> str:
        """Extract message content from provider-specific format.

        Args:
            provider: LLM provider type
            message: Message in provider-specific format

        Returns:
            Message content as string

        Raises:
            ValueError: If provider is not supported
        """
        if provider == LLMProvider.OPENAI:
            return message["content"]
        elif provider == LLMProvider.ANTHROPIC:
            if isinstance(message["content"], list):
                return " ".join(
                    item["text"] for item in message["content"] if item["type"] == "text"
                )
            return message["content"]
        raise ValueError(f"Unsupported provider: {provider}")
