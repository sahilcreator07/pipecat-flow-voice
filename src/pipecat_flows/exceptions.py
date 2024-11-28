#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


class FlowError(Exception):
    """Base exception for all flow-related errors."""

    pass


class FlowInitializationError(FlowError):
    """Raised when flow initialization fails."""

    pass


class FlowTransitionError(FlowError):
    """Raised when a state transition fails."""

    pass


class InvalidFunctionError(FlowError):
    """Raised when an invalid or unavailable function is called."""

    pass


class ActionError(FlowError):
    """Raised when an action execution fails."""

    pass
