<h1><div align="center">
 <img alt="pipecat" width="500px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat-flows/main/pipecat-flows.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-flows)](https://pypi.org/project/pipecat-ai-flows) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat's conversation flow system allows you to create structured, multi-turn conversations by defining your flow in JSON and processing it through the `FlowManager`. The system treats conversations as a series of connected nodes, where each node represents a distinct state with specific behaviors and options.

Pipecat Flows is comprised of:

- A [python module](#pipecat-flows-package) for building conversation flows with Pipecat
- A [visual editor](#pipecat-flows-editor) for visualizing conversations and exporting into flow_configs

To learn more about building with Pipecat Flows, [check out the guide](https://docs.pipecat.ai/guides/pipecat-flows).

## Pipecat Flows Package

A Python package for managing conversation flows in Pipecat applications.

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

# Initialize context and tools
initial_tools = flow_config["nodes"]["start"]["functions"]  # Available functions for starting state
context = OpenAILLMContext(messages, initial_tools)        # Create LLM context with initial state
context_aggregator = llm.create_context_aggregator(context)

# Create your pipeline: No new processors are required
pipeline = Pipeline(
    [
        transport.input(),  # Transport user input
        stt,  # STT
        context_aggregator.user(),  # User responses
        llm,  # LLM
        tts,  # TTS
        transport.output(),  # Transport bot output
        context_aggregator.assistant(),  # Assistant spoken responses
    ]
)

# Create the Pipecat task
task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

# Initialize flow management
flow_manager = FlowManager(flow_config, task, tts_service)  # Create flow manager
await flow_manager.register_functions(llm_service)          # Register all possible functions

# Initialize with starting messages
@transport.event_handler("on_first_participant_joined")
async def on_first_participant_joined(transport, participant):
    await transport.capture_participant_transcription(participant["id"])
    # Initialize the flow processor
    await flow_manager.initialize(messages)
    # Kick off the conversation using the context aggregator
    await task.queue_frames([context_aggregator.user().get_context_frame()])
```

## Pipecat Flows Editor

A visual editor for creating and managing Pipecat conversation flows.

![Food ordering flow example](https://raw.githubusercontent.com/pipecat-ai/pipecat-flows/main/images/food-ordering-flow.png)

### Features

- Visual flow creation and editing
- Import/export of flow configurations
- Support for terminal and transitional functions
- Merge node support for complex flows
- Real-time validation

### Online Editor

The editor is available online at [flows.pipecat.ai](https://flows.pipecat.ai).

### Local Development

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
cd pipecat-flows/editor
```

Install dependencies

```bash
npm install
```

Start development server

```bash
npm run dev
```

Open the page in your browser: http://localhost:5173.

#### Usage

1. Create a new flow using the toolbar buttons
2. Add nodes by right-clicking in the canvas
3. Connect nodes by dragging from outputs to inputs
4. Edit node properties in the side panel
5. Export your flow configuration using the toolbar

#### Examples

The `editor/examples/` directory contains sample flow configurations:

- `food_ordering.json` - A restaurant order flow demonstrating terminal and transitional functions, merge nodes, and actions. Shows how to handle branching paths (pizza vs sushi) that merge back to a common endpoint.
- `movie_booking.json` - A movie ticket booking flow showcasing date-based branching (today vs tomorrow), terminal functions for movie and showtime selection, and proper handling of sequential choices. Demonstrates how to manage time-dependent options and multiple selection criteria.
- `restaurant_reservation.json` - A comprehensive reservation system for an upscale restaurant that handles party size, date/time selection, availability checking, seating preferences, and special requests. Features a verification step with revision capability and demonstrates proper error handling for unavailable times.
- `travel_planner.json` - A vacation planning assistant that showcases parallel paths (beach vs mountain) merging into a common booking flow, multiple data collection points, array-based activity selection, and a verification system with revision capabilities. Demonstrates how to handle complex, multi-step planning processes.

To use an example:

1. Open the editor
2. Click "Import Flow"
3. Select an example JSON file

See the [examples directory](editor/examples/) for the complete files and documentation.

### Development

#### Available Scripts

- `npm start` - Start production server
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run preview:prod` - Preview production build with base path
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

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or adding new features, here's how you can help:

- **Found a bug?** Open an [issue](https://github.com/pipecat-ai/pipecat-flows/issues)
- **Have a feature idea?** Start a [discussion](https://discord.gg/pipecat)
- **Want to contribute code?** Check our [CONTRIBUTING.md](CONTRIBUTING.md) guide
- **Documentation improvements?** [Docs](https://github.com/pipecat-ai/docs) PRs are always welcome

Before submitting a pull request, please check existing issues and PRs to avoid duplicates.

We aim to review all contributions promptly and provide constructive feedback to help get your changes merged.

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Pipecat Flows Guide](https://docs.pipecat.ai/guides/pipecat-flows)

➡️ [Reach us on X](https://x.com/pipecat_ai)
