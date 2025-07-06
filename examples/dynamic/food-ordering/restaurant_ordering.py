#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

# run with: PYTHONPATH=src python examples/dynamic/restaurant_ordering.py
"""Restaurant Food Ordering Example using Pipecat Dynamic Flows.

This example demonstrates a structured, node-based conversation for ordering food at a restaurant.
- Friendly introduction and name collection
- Menu presentation (with dietary options)
- Handles price queries
- Allows multiple items to be ordered
- Summarizes and confirms the order, provides delivery details
- No API calls; uses static menu
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import TypedDict, List, Dict, Any

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowArgs, FlowManager, FlowResult, FlowsFunctionSchema, NodeConfig
import inflect
import difflib

# Add examples directory to Python path for runner import
examples_dir = Path(__file__).parent.parent.parent
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))
from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# --- Sample Menu ---
SAMPLE_MENU = [
    {"name": "Margherita Pizza", "price": 12.0, "dietary": ["vegetarian"]},
    {"name": "Pepperoni Pizza", "price": 14.0, "dietary": []},
    {"name": "Pasta Primavera", "price": 13.0, "dietary": ["vegetarian"]},
    {"name": "Grilled Chicken Salad", "price": 11.0, "dietary": ["gluten-free"]},
    {"name": "Caesar Salad", "price": 10.0, "dietary": ["vegetarian"]},
    {"name": "Vegan Burger", "price": 15.0, "dietary": ["vegan"]},
    {"name": "Gluten-Free Brownie", "price": 6.0, "dietary": ["gluten-free", "vegetarian"]},
    {"name": "Lemonade", "price": 4.0, "dietary": ["vegan", "gluten-free"]},
]

# --- Type Definitions ---
class OrderItem(TypedDict):
    name: str
    quantity: int
    price: float
    dietary: List[str]

class UserInfo(TypedDict):
    name: str

class OrderSummary(FlowResult):
    items: List[OrderItem]
    total: float
    user_name: str
    estimated_time: str

# --- Node Handlers ---
async def collect_name(args: FlowArgs, flow_manager: FlowManager) -> tuple[UserInfo, NodeConfig]:
    user_name = args["name"]
    flow_manager.state["user_name"] = user_name
    return UserInfo(name=user_name), create_menu_node()

async def handle_menu_query(args: FlowArgs, flow_manager: FlowManager) -> tuple[FlowResult, NodeConfig]:
    # Handles menu queries, price, dietary, etc.
    query = args.get("query", "")
    # For simplicity, just echo menu or price
    if "price" in query.lower():
        item_name = args.get("item_name", "")
        for item in SAMPLE_MENU:
            if item_name.lower() in item["name"].lower():
                return FlowResult(message=f"The price of {item['name']} is ${item['price']:.2f}.",), create_menu_node()
        return FlowResult(message="Sorry, I couldn't find that item on the menu."), create_menu_node()
    if "dietary" in query.lower() or "vegan" in query.lower() or "gluten" in query.lower() or "vegetarian" in query.lower():
        dietary = args.get("dietary", "")
        filtered = [item for item in SAMPLE_MENU if dietary in item["dietary"]]
        if filtered:
            items = ", ".join([item["name"] for item in filtered])
            return FlowResult(message=f"Here are our {dietary} options: {items}.",), create_menu_node()
        else:
            return FlowResult(message=f"Sorry, we have no {dietary} options."), create_menu_node()
    # Default: show menu (without prices)
    menu_str = "\n".join([
        f"- {item['name']} [{', '.join(item['dietary']) if item['dietary'] else 'regular'}]"
        for item in SAMPLE_MENU
    ])
    return FlowResult(message=f"Here is our menu:\n{menu_str}"), create_menu_node()

async def add_order_item(args: FlowArgs, flow_manager: FlowManager) -> tuple[OrderItem, NodeConfig]:
    item_name = args["item_name"]
    quantity = args.get("quantity", 1)
    # Try exact or partial match first
    matched_item = None
    for item in SAMPLE_MENU:
        if item_name.lower() in item["name"].lower():
            matched_item = item
            break
    # If no match, use fuzzy matching
    if not matched_item:
        menu_names = [item["name"] for item in SAMPLE_MENU]
        close_matches = difflib.get_close_matches(item_name, menu_names, n=1, cutoff=0.6)
        if close_matches:
            for item in SAMPLE_MENU:
                if item["name"] == close_matches[0]:
                    matched_item = item
                    break
            # Suggest the closest match to the user
            suggestion = close_matches[0]
            return OrderItem(name=suggestion, quantity=0, price=0.0, dietary=[]), create_add_more_node()
        else:
            # No close match found
            return OrderItem(name=item_name, quantity=0, price=0.0, dietary=[]), create_add_more_node()
    if matched_item:
        order_item = OrderItem(name=matched_item["name"], quantity=quantity, price=matched_item["price"] * quantity, dietary=matched_item["dietary"])
        if "order" not in flow_manager.state:
            flow_manager.state["order"] = []
        flow_manager.state["order"].append(order_item)
        return order_item, create_add_more_node()
    return OrderItem(name=item_name, quantity=quantity, price=0.0, dietary=[]), create_add_more_node()

async def handle_add_more(args: FlowArgs, flow_manager: FlowManager) -> tuple[FlowResult, NodeConfig]:
    if args.get("add_more", False):
        return FlowResult(message="What else would you like to order?"), create_menu_node()
    return FlowResult(message="Let's review your order."), create_summary_node(flow_manager)

async def finalize_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[OrderSummary, dict]:
    user_name = flow_manager.state.get("user_name", "Guest")
    order = flow_manager.state.get("order", [])
    total = sum(item["price"] for item in order)
    estimated_time = "20 minutes"
    if order:
        items_str = ", ".join([f"{item['quantity']} {item['name']}" for item in order])
        summary_message = (
            f"Thank you for ordering with KitchenAssistant, {user_name}! Your order of {items_str} totals ${total:.2f}. "
            f"Your food will be ready in about {estimated_time}. We appreciate your order. Have a wonderful meal!"
        )
    else:
        summary_message = "You haven't added any items to your order yet."
    flow_manager.state["final_summary_message"] = summary_message
    return OrderSummary(items=order, total=total, user_name=user_name, estimated_time=estimated_time), create_end_node()

async def confirm_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[OrderSummary, dict]:
    user_name = flow_manager.state.get("user_name", "Guest")
    order = flow_manager.state.get("order", [])
    total = sum(item["price"] for item in order)
    estimated_time = "20 minutes"
    if order:
        items_str = ", ".join([f"{item['quantity']} {item['name']}" for item in order])
        summary_message = (
            f"Here's your order, {user_name}: {items_str}. The total is ${total:.2f}. "
            f"Would you like to add or change anything, or should I place your order now?"
        )
    else:
        summary_message = "You haven't added any items to your order yet."
    flow_manager.state["review_summary_message"] = summary_message
    return OrderSummary(items=order, total=total, user_name=user_name, estimated_time=estimated_time), create_review_order_node(flow_manager)

# --- Node Creators ---
def create_initial_node() -> NodeConfig:
    return {
        "name": "introduction",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "Always start the conversation with 'Hello! I'm KitchenAssistant, your friendly restaurant ordering bot. ' "
                    "I'm here to help you place your order quickly and easily. "
                    "To get started, may I know your name?"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="collect_name",
                description="Collect the user's name",
                properties={"name": {"type": "string", "description": "User's name"}},
                required=["name"],
                handler=collect_name,
            )
        ],
    }

def create_menu_node() -> NodeConfig:
    def get_user_name(flow_manager=None):
        if flow_manager and hasattr(flow_manager, 'state'):
            return flow_manager.state.get("user_name", "there")
        return "there"
    user_name = get_user_name()
    # Spoken menu string
    menu_str = "\n".join([
        f"- {item['name']} ({', '.join(item['dietary']) if item['dietary'] else 'regular'})"
        for item in SAMPLE_MENU
    ])
    return {
        "name": "menu",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    f"Thank you, {{user_name}}! I'll now read our menu card. "
                    f"If you have any questions about ingredients, dietary options, or prices, just ask. "
                    f"Here is our menu: {menu_str}. What would you like to order?"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="handle_menu_query",
                description="Handle menu, price, or dietary queries",
                properties={
                    "query": {"type": "string", "description": "User's question or request"},
                    "item_name": {"type": "string", "description": "Name of the menu item", "default": ""},
                    "dietary": {"type": "string", "description": "Dietary preference (vegan, vegetarian, gluten-free)", "default": ""},
                },
                required=["query"],
                handler=handle_menu_query,
            ),
            FlowsFunctionSchema(
                name="add_order_item",
                description="Add an item to the order",
                properties={
                    "item_name": {"type": "string", "description": "Name of the menu item"},
                    "quantity": {"type": "integer", "description": "Quantity", "default": 1},
                },
                required=["item_name"],
                handler=add_order_item,
            ),
        ],
    }

def create_add_more_node() -> NodeConfig:
    return {
        "name": "add_more",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "Would you like to add anything else to your order, or are you ready to finish? "
                    "If the user says a food or drink item, call the add_order_item function. "
                    "If the user says they're done, call the confirm_order function to place the order and finish. "
                    "Examples: 'Add a lemonade', 'I'd like another pizza', 'No, I'm done', 'That's all for now', 'Finish my order'. "
                    "If the user says something that is not an exact menu item, suggest the closest menu item. "
                    "Always be friendly and helpful!"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="confirm_order",
                description="Confirm and place the order",
                properties={},
                required=[],
                handler=confirm_order,
            ),
            FlowsFunctionSchema(
                name="add_order_item",
                description="Add an item to the order",
                properties={
                    "item_name": {"type": "string", "description": "Name of the menu item"},
                    "quantity": {"type": "integer", "description": "Quantity", "default": 1},
                },
                required=["item_name"],
                handler=add_order_item,
            ),
            FlowsFunctionSchema(
                name="handle_menu_query",
                description="Handle menu, price, or dietary queries",
                properties={
                    "query": {"type": "string", "description": "User's question or request"},
                    "item_name": {"type": "string", "description": "Name of the menu item", "default": ""},
                    "dietary": {"type": "string", "description": "Dietary preference (vegan, vegetarian, gluten-free)", "default": ""},
                },
                required=["query"],
                handler=handle_menu_query,
            ),
        ],
    }

def create_summary_node(flow_manager: FlowManager) -> NodeConfig:
    order = flow_manager.state.get("order", [])
    user_name = flow_manager.state.get("user_name", "there")
    total = sum(item["price"] for item in order)

    def item_phrase(item):
        qty = item["quantity"]
        name = item["name"]
        qty_str = p.number_to_words(qty) if qty > 1 else ("one" if qty == 1 else "zero")
        return f"{qty_str} {name}" if qty == 1 else f"{qty_str} {name}s"

    if order:
        items_str = p.join([item_phrase(item) for item in order])
        summary_sentence = f"Here's your order, {user_name}: {items_str}. "
    else:
        summary_sentence = "You haven't added any items to your order yet. "

    total_words = p.number_to_words(int(total))
    summary_sentence += f"The total is {total_words} dollars. Would you like to confirm or make any changes?"

    return {
        "name": "summary",
        "task_messages": [
            {
                "role": "system",
                "content": summary_sentence,
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="confirm_order",
                description="Confirm and place the order",
                properties={},
                required=[],
                handler=confirm_order,
            ),
            FlowsFunctionSchema(
                name="add_order_item",
                description="Add an item to the order",
                properties={
                    "item_name": {"type": "string", "description": "Name of the menu item"},
                    "quantity": {"type": "integer", "description": "Quantity", "default": 1},
                },
                required=["item_name"],
                handler=add_order_item,
            ),
            FlowsFunctionSchema(
                name="handle_menu_query",
                description="Handle menu, price, or dietary queries",
                properties={
                    "query": {"type": "string", "description": "User's question or request"},
                    "item_name": {"type": "string", "description": "Name of the menu item", "default": ""},
                    "dietary": {"type": "string", "description": "Dietary preference (vegan, vegetarian, gluten-free)", "default": ""},
                },
                required=["query"],
                handler=handle_menu_query,
            ),
        ],
    }

def create_end_node() -> NodeConfig:
    def get_final_summary():
        return "Thank you for ordering with KitchenAssistant! Your food will be ready in about 20 minutes. We appreciate your order. Have a wonderful meal!"
    return {
        "name": "end",
        "task_messages": [
            {
                "role": "system",
                "content": get_final_summary(),
            }
        ],
        "post_actions": [{"type": "end_conversation"}],
    }

def create_review_order_node(flow_manager: FlowManager) -> dict:
    summary_message = flow_manager.state.get(
        "review_summary_message",
        "Would you like to add or change anything, or should I place your order now?"
    )
    return {
        "name": "review_order",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    f"{summary_message} "
                    "If the user says anything like 'yes', 'that's correct', 'place my order', 'order', 'go ahead', or 'no more', "
                    "call the finalize_order function to place the order and finish. "
                    "If the user wants to add or change items, call add_order_item. "
                    "If the user asks about their order, call handle_menu_query. "
                    "Examples: 'Yes, that's correct', 'Place my order', 'Order', 'No more', 'Add a lemonade', 'Change my pizza', 'What's my order?'. "
                    "Always be friendly and helpful!"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="finalize_order",
                description="Place the order and finish",
                properties={},
                required=[],
                handler=finalize_order,
            ),
            FlowsFunctionSchema(
                name="add_order_item",
                description="Add an item to the order",
                properties={
                    "item_name": {"type": "string", "description": "Name of the menu item"},
                    "quantity": {"type": "integer", "description": "Quantity", "default": 1},
                },
                required=["item_name"],
                handler=add_order_item,
            ),
            FlowsFunctionSchema(
                name="handle_menu_query",
                description="Handle menu, price, or dietary queries",
                properties={
                    "query": {"type": "string", "description": "User's question or request"},
                    "item_name": {"type": "string", "description": "Name of the menu item", "default": ""},
                    "dietary": {"type": "string", "description": "Dietary preference (vegan, vegetarian, gluten-free)", "default": ""},
                },
                required=["query"],
                handler=handle_menu_query,
            ),
        ],
    }

# --- Main Entrypoint ---
async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Restaurant Ordering Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-2-hermes-en")
        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-exp")

        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await flow_manager.initialize(create_initial_node())

        runner = PipelineRunner()
        await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main()) 