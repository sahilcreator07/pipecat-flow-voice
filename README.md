<h1><div align="center">
 <img alt="pipecat" width="300px" height="auto" src="https://github.com/pipecat-ai/pipecat-flows/blob/main/pipecat-flows.png">
</div></h1>

Pipecat's conversation flow system allows you to create structured, multi-turn conversations by defining your flow in JSON and processing it through the `FlowManager`. The system treats conversations as a series of connected nodes, where each node represents a distinct state with specific behaviors and options.

To learn more about building with Pipecat Flows, [check out the guide](https://docs.pipecat.ai/guides/pipecat-flows).

## 1. Pipecat AI Flows Package

A Python package for managing conversation flows in Pipecat AI applications.

### Installation

```bash
pip install pipecat-ai-flows
```

### Features

- State machine management for conversation flows
- Pre and post actions for state transitions
- Terminal and transitional functions
- Flexible node configuration

### Basic Usage

```python
from pipecat_flows import FlowManager

# Initialize flow management
flow_manager = FlowManager(flow_config, task, tts_service)  # Create flow manager
await flow_manager.register_functions(llm_service)          # Register all possible functions

# Initialize the flow
await flow_manager.initialize(initial_messages)
```

## 2. Pipecat Flows Editor

A visual editor for creating and managing Pipecat conversation flows.

### Features

- Visual flow creation and editing
- Import/export of flow configurations
- Support for terminal and transitional functions
- Merge node support for complex flows
- Real-time validation

### Getting Started

#### Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)

#### Installation

Clone the repository

```bash
git clone git@github.com:pipecat-ai/pipecat-flows.git
```

Navigate to project directory

```bash
cd pipecat-flows
```

Install dependencies

```bash
npm install
```

Start development server

```bash
npm run dev
```

Open the page in your browser at: http://localhost:8080.

#### Usage

1. Create a new flow using the toolbar buttons
2. Add nodes by right-clicking in the canvas
3. Connect nodes by dragging from outputs to inputs
4. Edit node properties in the side panel
5. Export your flow configuration using the toolbar

#### Examples

The `examples/` directory contains sample flow configurations:

- `food_ordering.json` - A restaurant order flow demonstrating terminal and transitional functions, merge nodes, and actions.

To use an example:
1. Open the editor
2. Click "Import Flow"
3. Select an example JSON file

See the [examples directory](examples/) for the complete files and documentation.

### Development

#### Available Scripts

- `npm start` - Start production server
- `npm run dev` - Start development server
- `npm run lint` - Check for linting issues
- `npm run lint:fix` - Fix linting issues
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting
- `npm run docs` - Generate documentation
- `npm run docs:serve` - Serve documentation locally

#### Documentation

The Pipecat Flows Editor project uses JSDoc for documentation. To generate and view the documentation:

Generate documentation:

```bash
npm run docs
```

Serve documentation locally:

```bash
npm run docs:serve
```

View in browser by opening: http://localhost:8080

## Contributing

TBD

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Pipecat Flows Guide](https://docs.pipecat.ai/guides/pipecat-flows)

➡️ [Reach us on X](https://x.com/pipecat_ai)
