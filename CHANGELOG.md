# Changelog

All notable changes to **Pipecat Flows** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - TBD

### Changed

- Terminal and transitional functions now use graph terms—node and edge
  functions, respectively. Updates are made throughout the codebase to
  reflect this terminology change.

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
