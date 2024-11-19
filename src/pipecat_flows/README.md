<h1><div align="center">
 <img alt="pipecat" width="500px" height="auto" src="https://github.com/pipecat-ai/pipecat-flows/blob/main/pipecat-flows.png">
</div></h1>

[Pipecat](https://www.pipecat.ai)'s conversation flow system allows you to create structured, multi-turn conversations by defining your flow in JSON and processing it through the `FlowManager`. The system treats conversations as a series of connected nodes, where each node represents a distinct state with specific behaviors and options.

Pipecat Flows is comprised of:

- This python module for building conversation flows with Pipecat
- A visual editor for visualizing conversations and exporting into flow_configs

To learn more about building with Pipecat Flows, [check out the guide](https://docs.pipecat.ai/guides/pipecat-flows).

To learn about building flows with the visual editor, check out the [README on GitHub](https://github.com/pipecat-ai/pipecat-flows?tab=readme-ov-file#pipecat-flows-editor).

## Installation

```bash
pip install pipecat-ai-flows
```

## Features

- State machine management for conversation flows
- Pre and post actions for state transitions
- Terminal and transitional functions
- Flexible node configuration

## Basic Usage

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
