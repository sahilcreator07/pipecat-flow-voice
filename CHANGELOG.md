# Changelog

All notable changes to **Pipecat Flows** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.11] - 2025-01-19

### Changed

- Updated `FlowManager` to more predictably handle function calls:

  - Edge functions (which transition to a new node) now result in an LLM
    completion after both the function call and messages are added to the
    LLM's context.
  - Node functions (which execute a function call without transitioning nodes)
    result in an LLM completion upon the function call result returning.
  - This change also improves the reliability of the pre- and post-action
    execution timing.

- Breaking changes:

  - The FlowManager has a new required arg, `context_aggregator`.
  - Pipecat's minimum version has been updated to 0.0.53 in order to use the
    new `FunctionCallResultProperties` frame.

- Updated all examples to align with the new changes.

## [0.0.10] - 2024-12-21

### Changed

- Nodes now have two message types to better delineate defining the role or
  persona of the bot from the task it needs to accomplish. The message types are:

  - `role_messages`, which defines the personality or role of the bot
  - `task_messages`, which defines the task to be completed for a given node

- `role_messages` can be defined for the initial node and then inherited by
  subsequent nodes. You can treat this as an LLM "system" message.

- Simplified FlowManager initialization by removing the need for manual context
  setup in both static and dynamic flows. Now, you need to create a `FlowManager`
  and initialize it to start the flow.
- All examples have been updated to align with the API changes.

### Fixed

- Fixed an issue where importing the Flows module would require OpenAI,
  Anthropic, and Google LLM modules.

## [0.0.9] - 2024-12-08

### Changed

- Fixed function handler registration in FlowManager to handle `__function__:` tokens
  - Previously, the handler string was used directly, causing "not callable" errors
  - Now correctly looks up and uses the actual function object from the main module
  - Supports both direct function references and function names exported from the Flows editor

## [0.0.8] - 2024-12-07

### Changed

- Improved type safety in FlowManager by requiring keyword arguments for initialization
- Enhanced error messages for LLM service type validation

## [0.0.7] - 2024-12-06

### Added

- New `transition_to` field for static flows
  - Combines function handlers with state transitions
  - Supports all LLM providers (OpenAI, Anthropic, Gemini)
  - Static examples updated to use this new transition

### Changed

- Static flow transitions now use `transition_to` instead of matching function names
  - Before: Function name had to match target node name
  - After: Function explicitly declares target via `transition_to`

### Fixed

- Duplicate LLM responses during transitions

## [0.0.6] - 2024-12-02

### Added

- New FlowManager supporting both static and dynamic conversation flows
  - Static flows: Configuration-driven with predefined paths
  - Dynamic flows: Runtime-determined conversation paths
  - Documentation: [Guide](https://docs.pipecat.ai/guides/pipecat-flow) and [Reference](https://docs.pipecat.ai/api-reference/utilities/flows/pipecat-flows)
- Provider-specific examples demonstrating dynamic flows:
  - OpenAI: `insurance_openai.py`
  - Anthropic: `insurance_anthropic.py`
  - Gemini: `insurance_gemini.py`
- Type safety improvements:
  - `FlowArgs`: Type-safe function arguments
  - `FlowResult`: Type-safe function returns

### Changed

- Simplified function handling:
  - Automatic LLM function registration
  - Optional handlers for edge nodes
- Updated all examples to use unified FlowManager interface

## [0.0.5] - 2024-11-27

### Added

- Added LLM support for:

  - Anthropic
  - Google Gemini

- Added `LLMFormatParser`, a format parser to handle LLM provider-specific
  messages and function call formats

- Added new examples:

  - movie_explorer_anthropic.py (Claude 3.5)
  - movie_explorer_gemini.py (Gemini 1.5 Flash)
  - travel_planner_gemini.py (Gemini 1.5 Flash)

## [0.0.4] - 2024-11-26

### Added

- New example `movie_explorer.py` demonstrating:
  - Real API integration with TMDB
  - Node functions for API calls
  - Edge functions for state transitions
  - Proper function registration pattern

### Changed

- Renamed function types to use graph terminology:

  - "Terminal functions" are now "node functions" (operations within a state)
  - "Transitional functions" are now "edge functions" (transitions between states)

- Updated function registration process:

  - Node functions must be registered directly with the LLM before flow initialization
  - Edge functions are automatically registered by FlowManager during initialization
  - LLM instance is now required in FlowManager constructor

- Added flexibility to node naming with the Editor:
  - Start nodes can now use any descriptive name (e.g., "greeting")
  - End nodes conventionally use "end" but support custom names
  - Flow configuration's `initial_node` property determines the starting state

### Updated

- All examples updated to use new function registration pattern
- Documentation updated to reflect new terminology and patterns
- Editor updated to support flexible node naming

## [0.0.3] - 2024-11-25

### Added

- Added an `examples` directory which contains five different examples
  showing how to build a conversation flow with Pipecat Flows.

- Added a new editor example called `patient_intake.json` which demonstrates
  a patient intake conversation flow.

### Changed

- `pipecat-ai-flows` now includes `pipecat-ai` as a dependency, making it
  easier to get started with a fresh installation.

### Fixed

- Fixed an issue where terminal functions were updating the LLM context and
  tools. Now, only transitional functions update the LLM context and tools.

## [0.0.2] - 2024-11-22

### Fixed

- Fixed an issue where `pipecat-ai` was mistakenly added as a dependency

## [0.0.1] - 2024-11-18

### Added

- Initial public beta release.

- Added conversation flow management system through `FlowState` and `FlowManager` classes.
  This system enables developers to create structured, multi-turn conversations using
  a node-based state machine. Each node can contain:
  - Multiple LLM context messages (system/user/assistant)
  - Available functions for that state
  - Pre- and post-actions for state transitions
  - Support for both terminal functions (stay in same node) and transitional functions
  - Built-in handlers for immediate TTS feedback and conversation end
- Added `NodeConfig` dataclass for defining conversation states, supporting:
  - Multiple messages per node for complex prompt building
  - Function definitions for available actions
  - Optional pre- and post-action hooks
  - Clear separation between node configuration and state management
