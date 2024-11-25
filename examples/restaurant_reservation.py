#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from runner import configure

from pipecat_flows import FlowManager

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Flow Configuration - Restaurant Reservation System
#
# This configuration defines a restaurant reservation system with the following states:
#
# 1. start
#    - Initial state collecting party size information
#    - Functions:
#      * record_party_size (node function, validates 1-12 people)
#      * get_date_time (transitions to date/time selection)
#    - Pre-action: Welcome message from La Maison
#    - Expected flow: Greet -> Ask party size -> Record -> Transition to date/time
#
# 2. get_date_time
#    - Collects preferred reservation date and time
#    - Operating hours: Tue-Sun, 5 PM - 10 PM (last seating 9 PM)
#    - Functions:
#      * record_datetime (node function, collects date and time)
#      * check_availability (transitions to availability check)
#    - Expected flow: Ask preferences -> Record details -> Check availability
#
# 3. check_availability
#    - Verifies if requested time slot is available
#    - Functions:
#      * get_seating_preferences (transitions to seating if available)
#      * try_alternative_time (transitions to alternative time if unavailable)
#    - Expected flow: Check slot -> Either proceed or suggest alternatives
#
# 4. try_alternative_time
#    - Handles unavailable time slots
#    - Functions:
#      * get_date_time (returns to date/time selection)
#    - Expected flow: Apologize -> Suggest alternatives -> Return to date/time
#
# 5. get_seating_preferences
#    - Collects seating area preference
#    - Options: main dining room, outdoor terrace, bar area
#    - Functions:
#      * record_seating (node function, records seating choice)
#      * special_requests (transitions to special requests)
#    - Expected flow: Present options -> Record choice -> Transition to requests
#
# 6. special_requests
#    - Collects any special requirements or dietary needs
#    - Functions:
#      * record_requests (node function, records requests or empty string)
#      * verify_reservation (transitions to verification)
#    - Expected flow: Ask about requests -> Record details -> Transition to verify
#
# 7. verify_reservation
#    - Reviews all reservation details with guest
#    - Functions:
#      * confirm_reservation (transitions to confirmation if correct)
#      * revise_reservation (transitions to revision if changes needed)
#    - Expected flow: Review details -> Get confirmation -> Proceed or revise
#
# 8. revise_reservation
#    - Handles reservation modifications
#    - Functions:
#      * get_date_time (returns to date/time while keeping party size)
#    - Expected flow: Acknowledge changes needed -> Return to date/time
#
# 9. confirm_reservation
#    - Finalizes the reservation
#    - Functions:
#      * end (transitions to end)
#    - Pre-action: Confirmation message
#    - Expected flow: Thank guest -> Provide final details -> End
#
# 10. end
#    - Final state that closes the conversation
#    - No functions available
#    - Post-action: Ends conversation

flow_config = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "messages": [
                {
                    "role": "system",
                    "content": "For this step, warmly greet the customer and ask how many people are in their party (maximum 12). Once they tell you the party size, use record_party_size to save it and proceed to get_date_time.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_party_size",
                        "description": "Record the number of people in the party",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "size": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 12,
                                    "description": "Number of people in the party",
                                }
                            },
                            "required": ["size"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_date_time",
                        "description": "Proceed to date and time selection",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [
                {
                    "type": "tts_say",
                    "text": "Welcome to La Maison. I'll help you make a reservation.",
                }
            ],
        },
        "get_date_time": {
            "messages": [
                {
                    "role": "system",
                    "content": "Ask for their preferred date and time. The restaurant is open Tuesday through Sunday, 5 PM to 10 PM (last seating at 9 PM). Use record_datetime to collect their preference, then use check_availability to verify the slot.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_datetime",
                        "description": "Record the requested date and time",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "date": {
                                    "type": "string",
                                    "format": "date",
                                    "description": "Requested date (YYYY-MM-DD)",
                                },
                                "time": {
                                    "type": "string",
                                    "pattern": "^([0-1][0-9]|2[0-3]):[0-5][0-9]$",
                                    "description": "Requested time (HH:MM in 24-hour format)",
                                },
                            },
                            "required": ["date", "time"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "check_availability",
                        "description": "Check if the requested time slot is available",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "check_availability": {
            "messages": [
                {
                    "role": "system",
                    "content": "Check if the requested time slot is available. If available, use get_seating_preferences to proceed. If not available, suggest alternative times (30 minutes before or after) and use try_alternative_time.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_seating_preferences",
                        "description": "Proceed to seating preferences",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "try_alternative_time",
                        "description": "Return to date/time selection with suggested alternatives",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "try_alternative_time": {
            "messages": [
                {
                    "role": "system",
                    "content": "Apologize that their preferred time isn't available and suggest alternatives. Use get_date_time to let them select a new time.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_date_time",
                        "description": "Return to date and time selection",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        "get_seating_preferences": {
            "messages": [
                {
                    "role": "system",
                    "content": "Ask about their seating preferences. Options are: main dining room, outdoor terrace (weather permitting), or bar area. Use record_seating to save their preference and proceed to special_requests.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_seating",
                        "description": "Record seating preference",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "seating": {
                                    "type": "string",
                                    "enum": ["main dining room", "outdoor terrace", "bar area"],
                                    "description": "Preferred seating area",
                                }
                            },
                            "required": ["seating"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "special_requests",
                        "description": "Proceed to special requests",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "special_requests": {
            "messages": [
                {
                    "role": "system",
                    "content": "Ask if they have any special requests or dietary requirements. Use record_requests to save any requirements, then proceed to verify_reservation. If they have none, you can still use record_requests with an empty string.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_requests",
                        "description": "Record any special requests",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "requests": {
                                    "type": "string",
                                    "description": "Special requests or dietary requirements",
                                }
                            },
                            "required": ["requests"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "verify_reservation",
                        "description": "Proceed to reservation verification",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "verify_reservation": {
            "messages": [
                {
                    "role": "system",
                    "content": "Review all reservation details with the guest (party size, date, time, seating preference, and any special requests). Ask them to confirm everything is correct. Use confirm_reservation if they approve, or revise_reservation to make changes.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "confirm_reservation",
                        "description": "Confirm the reservation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "revise_reservation",
                        "description": "Return to date/time selection to revise details",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "revise_reservation": {
            "messages": [
                {
                    "role": "system",
                    "content": "Acknowledge their need to make changes and use get_date_time to restart the reservation process while maintaining their party size.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_date_time",
                        "description": "Return to date/time selection",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        "confirm_reservation": {
            "messages": [
                {
                    "role": "system",
                    "content": "Thank them for their reservation, confirm all details one final time, and provide any relevant information (e.g., parking, dress code). Then use end to complete the conversation.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "end",
                        "description": "End the conversation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "pre_actions": [
                {"type": "tts_say", "text": "Wonderful! Your reservation is confirmed."}
            ],
        },
        "end": {
            "messages": [{"role": "system", "content": "Say goodbye and end the conversation."}],
            "functions": [],
            "post_actions": [{"type": "end_conversation"}],
        },
    },
}


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-helios-en")
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Get initial tools from the first node
        initial_tools = flow_config["nodes"]["start"]["functions"]

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": "You are a restaurant reservation assistant for La Maison, an upscale French restaurant. You must ALWAYS use one of the available functions to progress the conversation. This is a phone conversations and your responses will be converted to audio. Avoid outputting special characters and emojis.",
            }
        ]

        context = OpenAILLMContext(messages, initial_tools)
        context_aggregator = llm.create_context_aggregator(context)

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

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        # Initialize flow manager
        flow_manager = FlowManager(flow_config, task, tts)

        # Register functions with LLM service
        await flow_manager.register_functions(llm)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Initialize the flow processor
            await flow_manager.initialize(messages)
            # Kick off the conversation using the context aggregator
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
